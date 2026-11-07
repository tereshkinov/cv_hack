"""Rebuild provisional identities using boundary evidence and bounded graph search.

This is a deterministic beam search, not a certified global optimum. All source
references survive. Unresolved alternatives are deferred by consensus.
"""

from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, median

from .scoring import appearance_quality, boundary_spatial_input, decide
from .topology import plausible_recovery


@dataclass(frozen=True)
class FragmentWindow:
    fragment_id: str
    camera_id: str
    observations: tuple
    split_reason: str = "local_track"

    @property
    def references(self):
        return tuple(o.reference for o in self.observations)

    @property
    def start_s(self):
        return self.observations[0].timestamp_s

    @property
    def end_s(self):
        return self.observations[-1].timestamp_s

    def head(self, seconds=2):
        return [o for o in self.observations if o.timestamp_s - self.start_s <= seconds]

    def tail(self, seconds=2):
        return [o for o in self.observations if self.end_s - o.timestamp_s <= seconds]


def fragments_from_observations(observations, config, exits):
    from .engine import _cosine_similarity

    grouped = defaultdict(list)
    events = defaultdict(list)
    for o in observations:
        grouped[o.local_key].append(o)
    for e in exits:
        events[(e.camera_id, e.local_track_id)].append(e.timestamp_s)
    result = []
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda o: (o.timestamp_s, o.frame_index))
        chunks = []
        current = []
        reason = "local_track"
        for i, o in enumerate(rows):
            split = None
            if current:
                previous = current[-1]
                gap = o.timestamp_s - previous.timestamp_s
                if gap > config.timeout_s:
                    split = "timeout_gap"
                elif any(
                    previous.timestamp_s <= t < o.timestamp_s for t in events[key]
                ):
                    split = "explicit_exit"
                elif i + 1 < len(rows) and not plausible_recovery(
                    previous.bbox_xyxy, o.bbox_xyxy, gap
                ):
                    future = rows[i + 1]
                    sustained_jump = not plausible_recovery(
                        previous.bbox_xyxy,
                        future.bbox_xyxy,
                        future.timestamp_s - previous.timestamp_s,
                    )
                    if sustained_jump and config.decision_mode in {
                        "geometry",
                        "geometry_visual",
                    }:
                        # Keep both geometry experiments independent of embeddings.
                        # The graph may later reconnect the pieces using boundary geometry.
                        split = "sustained_motion_jump"
                    elif sustained_jump:
                        # Appearance modes require an independent visual signal before
                        # interpreting an image-space discontinuity as ID contamination.
                        old = [x for x in current[-10:] if x.embedding is not None]
                        new = [x for x in rows[i : i + 5] if x.embedding is not None]
                        spaces = {
                            (x.embedding_model_id, len(x.embedding)) for x in old + new
                        }
                        if len(old) >= 2 and len(new) >= 2 and len(spaces) == 1:
                            similarity = median(
                                _cosine_similarity(a.embedding, b.embedding)
                                for a in old[-2:]
                                for b in new[:2]
                            )
                            if similarity < 0.4:
                                split = "sustained_motion_and_appearance_jump"
            if split:
                chunks.append((current, reason))
                current = []
                reason = split
            current.append(o)
        if current:
            chunks.append((current, reason))
        for chunk, why in chunks:
            result.append(
                FragmentWindow(f"f{len(result) + 1}", key[0], tuple(chunk), why)
            )
    return sorted(result, key=lambda f: (f.start_s, f.camera_id, f.fragment_id))


def _gallery(rows, confidence):
    result = []
    for o in sorted(rows, key=lambda x: x.timestamp_s):
        if (
            o.embedding is not None
            and o.detection_confidence >= confidence
            and (not result or o.timestamp_s - result[-1].timestamp_s >= 0.2 - 1e-9)
        ):
            result.append(o)
    if len(result) > 8:
        result = [result[round(i * (len(result) - 1) / 7)] for i in range(8)]
    return result


def boundary_evidence(engine, left, right):
    """Default providers over existing embeddings and native floor evidence.

    Future providers may replace this function through engine.boundary_provider;
    return (score, metadata), or (None, refusal metadata), without assigning IDs.
    """
    from .engine import _cosine_similarity

    cfg = engine.config
    if left.start_s > right.start_s:
        left, right = right, left
    overlap = min(left.end_s, right.end_s) - max(left.start_s, right.start_s)
    if overlap >= 0:
        if left.camera_id == right.camera_id:
            return None, {"reason": "same_camera_conflict"}
        relation = engine.graph.relation(left.camera_id, right.camera_id, 0)
        if relation != "overlap":
            return None, {"reason": "simultaneous_nonoverlap"}
        start, end = max(left.start_s, right.start_s), min(left.end_s, right.end_s)
        a = [
            o for o in left.observations if start - 0.25 <= o.timestamp_s <= end + 0.25
        ]
        b = [
            o for o in right.observations if start - 0.25 <= o.timestamp_s <= end + 0.25
        ]
        mode = "simultaneous_overlap"
    else:
        gap = right.start_s - left.end_s
        if gap > cfg.timeout_s:
            return None, {"reason": "outside_timeout"}
        relation = engine.graph.relation(left.camera_id, right.camera_id, gap)
        if relation is None:
            return None, {"reason": "no_camera_edge"}
        a, b = left.tail(cfg.boundary_window_s), right.head(cfg.boundary_window_s)
        mode = (
            "same_camera_recovery"
            if left.camera_id == right.camera_id
            else "sequential_transition"
        )
        if (
            relation == "same_camera"
            and gap <= 4
            and not plausible_recovery(a[-1].bbox_xyxy, b[0].bbox_xyxy, gap)
        ):
            return None, {"reason": "implausible_image_displacement"}
    ga, gb = (
        _gallery(a, cfg.min_embedding_confidence),
        _gallery(b, cfg.min_embedding_confidence),
    )
    # Bounded expansion for sparse embeddings, without taking the entire history.
    if not ga and overlap < 0:
        ga = _gallery(left.tail(4), cfg.min_embedding_confidence)
    if not gb and overlap < 0:
        gb = _gallery(right.head(4), cfg.min_embedding_confidence)
    if cfg.decision_mode in {"geometry", "geometry_visual"}:
        from .geometry_policy import decide_geometry

        appearance = None
        aq = 0.0
        if (
            cfg.decision_mode == "geometry_visual"
            and ga
            and gb
            and len({(o.embedding_model_id, len(o.embedding)) for o in ga + gb}) == 1
        ):
            matrix = [
                [_cosine_similarity(x.embedding, y.embedding) for y in gb] for x in ga
            ]
            appearance = fmean(
                [
                    median(max(row) for row in matrix),
                    median(max(row[j] for row in matrix) for j in range(len(gb))),
                ]
            )
            aq = min(
                median(appearance_quality(o, engine.cameras[o.camera_id]) for o in g)
                for g in [ga, gb]
            )
        _, _, spatial_details, boundary = boundary_spatial_input(engine, a, b, mode)
        threshold = (
            cfg.tracklet_threshold
            if cfg.tracklet_threshold is not None
            else cfg.similarity_threshold
        )
        decision = decide_geometry(
            boundary,
            threshold,
            appearance=appearance,
            appearance_reliability=aq,
            visual_confirmation_weight=(
                cfg.geometry_visual_weight
                if cfg.decision_mode == "geometry_visual"
                else 0.0
            ),
        )
        if decision.score is None:
            return None, {
                "mode": mode,
                "reason": decision.reason,
                "boundary_geometry": boundary.as_dict() if boundary else None,
                "spatial_details": spatial_details,
                **decision.export(),
            }
        return decision.score, {
            "mode": mode,
            "reason": decision.reason,
            "support_count": boundary.support_count if boundary else 0,
            "boundary_geometry": boundary.as_dict() if boundary else None,
            "spatial_details": spatial_details,
            **decision.export(),
        }
    if not ga or not gb:
        return None, {"reason": "missing_appearance", "mode": mode}
    if len({(o.embedding_model_id, len(o.embedding)) for o in ga + gb}) != 1:
        return None, {"reason": "incompatible_appearance", "mode": mode}
    matrix = [[_cosine_similarity(x.embedding, y.embedding) for y in gb] for x in ga]
    appearance = fmean(
        [
            median(max(row) for row in matrix),
            median(max(row[j] for row in matrix) for j in range(len(gb))),
        ]
    )
    aq = min(
        median(appearance_quality(o, engine.cameras[o.camera_id]) for o in g)
        for g in [ga, gb]
    )
    geo, gq, spatial_details, boundary = boundary_spatial_input(engine, a, b, mode)
    threshold = (
        cfg.tracklet_threshold
        if cfg.tracklet_threshold is not None
        else cfg.similarity_threshold
    )
    evidence = decide(
        appearance,
        aq,
        geo,
        gq,
        engine.graph.bonus(relation, cfg.adjacency_bonus),
        threshold,
        geometry_preweighted=True,
        spatial_details=spatial_details,
    )
    score = evidence.score
    # One-frame fragments can recover a supported episode, but a lucky pair is insufficient.
    if min(len(ga), len(gb)) < 2 and not (
        appearance >= 0.88 or (geo is not None and geo > 0.03 and gq >= 0.5)
    ):
        score = min(score, threshold - 0.05)
    return score, {
        "mode": mode,
        "reason": evidence.reason,
        "support_count": min(len(ga), len(gb)),
        "boundary_geometry": boundary.as_dict() if boundary else None,
        **evidence.export(),
    }


def rebuild_fragments(engine, observations, assignments, tracks, exits):
    cfg = engine.config
    nodes = fragments_from_observations(observations, cfg, exits)
    n = len(nodes)
    threshold = (
        cfg.tracklet_threshold
        if cfg.tracklet_threshold is not None
        else cfg.similarity_threshold
    )
    refs = {
        (a["camera_id"], a["frame_index"], a["local_track_id"]): a for a in assignments
    }
    forbidden = set()
    constraint_log = []
    weights = {}
    candidates = []
    evidence_log = []

    def valid(group):
        group = set(group)
        if any(i in group and j in group for i, j in forbidden):
            return False
        ordered = sorted((nodes[i].start_s, nodes[i].end_s) for i in group)
        end = ordered[0][1]
        for start, stop in ordered[1:]:
            if start - end > cfg.timeout_s:
                return False
            end = max(end, stop)
        return True

    provider = getattr(engine, "boundary_provider", None) or (
        lambda a, b: boundary_evidence(engine, a, b)
    )
    for i, a in enumerate(nodes):
        for j in range(i + 1, n):
            b = nodes[j]
            overlap = max(a.start_s, b.start_s) <= min(a.end_s, b.end_s)
            conflict = overlap and (
                a.camera_id == b.camera_id
                or engine.graph.relation(a.camera_id, b.camera_id, 0) != "overlap"
            )
            for e in exits:
                if (
                    any(
                        o.local_key == (e.camera_id, e.local_track_id)
                        and o.timestamp_s <= e.timestamp_s
                        for o in a.observations
                    )
                    and b.end_s > e.timestamp_s
                ):
                    conflict = True
                if (
                    any(
                        o.local_key == (e.camera_id, e.local_track_id)
                        and o.timestamp_s <= e.timestamp_s
                        for o in b.observations
                    )
                    and a.end_s > e.timestamp_s
                ):
                    conflict = True
            if conflict:
                forbidden.add((i, j))
                constraint_log.append(
                    {
                        "left": a.fragment_id,
                        "right": b.fragment_id,
                        "reason": "camera_or_exit_conflict",
                    }
                )
                continue
            score, details = provider(a, b)
            if score is None:
                if overlap and cfg.decision_mode in {"geometry", "geometry_visual"}:
                    # A simultaneous multi-camera bundle needs direct spatial
                    # support. It must not be created transitively through a
                    # later fragment when overlap geometry is unavailable.
                    forbidden.add((i, j))
                    constraint_log.append(
                        {
                            "left": a.fragment_id,
                            "right": b.fragment_id,
                            "reason": "simultaneous_geometry_unavailable",
                        }
                    )
                continue
            if (
                details.get("hard_reject")
                or (
                    details.get("reason") == "spatial_contradiction"
                    and details.get("geometry_quality", 0) >= 0.6
                )
            ):
                forbidden.add((i, j))
                constraint_log.append(
                    {
                        "left": a.fragment_id,
                        "right": b.fragment_id,
                        "reason": "geometry_physical_contradiction"
                        if details.get("hard_reject")
                        else "reliable_spatial_contradiction",
                    }
                )
                continue
            if (
                overlap
                and cfg.decision_mode in {"geometry", "geometry_visual"}
                and score < threshold
            ):
                forbidden.add((i, j))
                constraint_log.append(
                    {
                        "left": a.fragment_id,
                        "right": b.fragment_id,
                        "reason": "simultaneous_geometry_not_supported",
                    }
                )
            weights[(i, j)] = score - threshold
            evidence_log.append(
                {
                    "left": a.fragment_id,
                    "right": b.fragment_id,
                    "score": score,
                    "threshold": threshold,
                    **details,
                }
            )
            if score >= threshold:
                candidates.append((i, j, score - threshold))
    initial = tuple(range(n))
    if cfg.fragment_mode == "boundary":
        # Preserve provisional groups for a merge-only ablation. Rebuild starts clean.
        groups = {}
        labels = list(initial)
        for i, node in enumerate(nodes):
            gids = {refs[r]["global_track_id"] for r in node.references}
            if len(gids) == 1:
                gid = next(iter(gids))
                groups.setdefault(gid, i)
                labels[i] = groups[gid]
        initial = tuple(labels)

    def objective(labels):
        return sum(w for (i, j), w in weights.items() if labels[i] == labels[j])

    states = {initial: objective(initial)}
    truncated = 0
    for i, j, _ in sorted(
        candidates,
        key=lambda e: (-e[2], nodes[e[0]].fragment_id, nodes[e[1]].fragment_id),
    ):
        next_states = dict(states)
        for labels, score in states.items():
            x, y = labels[i], labels[j]
            if x == y:
                continue
            group = [k for k in range(n) if labels[k] in {x, y}]
            if not valid(group):
                continue
            root = min(x, y)
            merged = tuple(root if label in {x, y} else label for label in labels)
            next_states[merged] = objective(merged)
        ranked = sorted(next_states.items(), key=lambda s: (-s[1], s[0]))
        truncated += int(len(ranked) > cfg.fragment_beam_width)
        states = dict(ranked[: cfg.fragment_beam_width])
    ranked = sorted(states.items(), key=lambda s: (-s[1], s[0]))
    best = ranked[0][1] if ranked else 0
    plausible = [
        labels for labels, score in ranked if best - score < cfg.ambiguity_margin + 1e-9
    ]
    labels = list(initial)
    accepted = []
    for i, j, _ in sorted(candidates, key=lambda e: (-e[2], e[0], e[1])):
        if not all(s[i] == s[j] for s in plausible):
            continue
        x, y = labels[i], labels[j]
        if x == y:
            continue
        group = [k for k, label in enumerate(labels) if label in {x, y}]
        if valid(group):
            labels = [min(x, y) if label in {x, y} else label for label in labels]
            accepted.append((i, j))
    grouped = defaultdict(list)
    for i, label in enumerate(labels):
        grouped[label].append(i)
    ordered = sorted(
        grouped.values(),
        key=lambda g: (
            min(nodes[i].start_s for i in g),
            min(nodes[i].fragment_id for i in g),
        ),
    )
    mapping = {}
    exports = []
    max_time = max((o.timestamp_s for o in observations), default=0)
    for number, group in enumerate(ordered, 1):
        gid = f"g{number}"
        rows = [o for i in group for o in nodes[i].observations]
        start, end = min(o.timestamp_s for o in rows), max(o.timestamp_s for o in rows)
        explicit = any(
            o.local_key == (e.camera_id, e.local_track_id)
            and o.timestamp_s <= e.timestamp_s
            for o in rows
            for e in exits
        )
        exports.append(
            {
                "global_track_id": gid,
                "start_s": start,
                "last_seen_s": end,
                "end_s": end,
                "status": "closed",
                "camera_ids": sorted({o.camera_id for o in rows}),
                "close_reason": "explicit_exit"
                if explicit
                else "timeout"
                if max_time - end > cfg.timeout_s
                else "end_of_video",
                "fragment_ids": [nodes[i].fragment_id for i in group],
            }
        )
        for i in group:
            for r in nodes[i].references:
                mapping[r] = (gid, nodes[i].fragment_id)
    revised = []
    for a in assignments:
        gid, fid = mapping[(a["camera_id"], a["frame_index"], a["local_track_id"])]
        row = {
            **a,
            "provisional_global_track_id": a["global_track_id"],
            "global_track_id": gid,
            "fragment_id": fid,
            "reason": f"fragment_{cfg.fragment_mode}",
        }
        for field in ["match_score", "decision_evidence", "gallery_updated"]:
            if field in row:
                row["provisional_" + field] = row.pop(field)
        revised.append(row)
    accepted_set = {
        frozenset((nodes[i].fragment_id, nodes[j].fragment_id)) for i, j in accepted
    }
    for row in evidence_log:
        row["selected"] = frozenset((row["left"], row["right"])) in accepted_set
    return (
        revised,
        exports,
        {
            "mode": cfg.fragment_mode,
            "nodes": n,
            "candidate_edges": len(candidates),
            "selected_edges": len(accepted),
            "beam_width": cfg.fragment_beam_width,
            "truncated_steps": truncated,
            "near_optimal_alternatives": len(plausible),
            "best_objective": best,
            "optimality_certified": False,
            "constraints": constraint_log,
            "fragments": [
                {
                    "fragment_id": f.fragment_id,
                    "camera_id": f.camera_id,
                    "start_s": f.start_s,
                    "end_s": f.end_s,
                    "split_reason": f.split_reason,
                    "references": [list(r) for r in f.references],
                }
                for f in nodes
            ],
            "edges": evidence_log,
        },
    )
