"""Conservative offline reconciliation of provisional global tracks using multiple frames."""

from collections import defaultdict
from statistics import fmean, median

from .topology import plausible_recovery


def reconcile(engine, observations, assignments, tracks, exits):
    from .engine import _cosine_similarity

    config = engine.config
    track_by_id = {t["global_track_id"]: t for t in tracks}
    original_ids = list(track_by_id)
    parent = {gid: gid for gid in original_ids}
    evidence = {}
    log = []
    by_reference = {o.reference: o for o in observations}
    rows = defaultdict(list)
    for a in assignments:
        key = (a["camera_id"], a["frame_index"], a["local_track_id"])
        rows[a["global_track_id"]].append(by_reference[key])
    exit_keys = {(e.camera_id, e.local_track_id) for e in exits}
    # Never bridge a declared exit, including an earlier segment of a reused local ID.
    blocked = {
        gid for gid, obs in rows.items() if any(o.local_key in exit_keys for o in obs)
    }

    def find(gid):
        while parent[gid] != gid:
            gid = parent[gid]
        return gid

    def group_data():
        groups = defaultdict(list)
        for gid in original_ids:
            groups[find(gid)].extend(rows[gid])
        return groups

    def gallery(obs):
        samples = sorted(
            (
                o
                for o in obs
                if o.embedding is not None
                and o.detection_confidence >= config.min_embedding_confidence
            ),
            key=lambda o: (o.timestamp_s, o.frame_index, o.local_track_id),
        )
        distinct = []
        for o in samples:
            if not distinct or o.timestamp_s - distinct[-1].timestamp_s >= 0.2 - 1e-9:
                distinct.append(o)
        # Uniform temporal sampling bounds pairwise cost without keeping only the first view.
        if len(distinct) > 32:
            distinct = [
                distinct[round(i * (len(distinct) - 1) / 31)] for i in range(32)
            ]
        return distinct

    def score_groups(left, right):
        lcam = {o.camera_id for o in left}
        rcam = {o.camera_id for o in right}
        # A merge must never put two independently visible local tracks on one camera.
        for common in lcam & rcam:
            la = defaultdict(list)
            rb = defaultdict(list)
            for o in left:
                if o.camera_id == common:
                    la[o.local_track_id].append(o.timestamp_s)
            for o in right:
                if o.camera_id == common:
                    rb[o.local_track_id].append(o.timestamp_s)
            if any(
                max(min(a), min(b)) <= min(max(a), max(b))
                for a in la.values()
                for b in rb.values()
            ):
                return None
        scores = []
        for ca in sorted(lcam):
            a = [o for o in left if o.camera_id == ca]
            for cb in sorted(rcam):
                b = [o for o in right if o.camera_id == cb]
                ast, aend = min(o.timestamp_s for o in a), max(o.timestamp_s for o in a)
                bst, bend = min(o.timestamp_s for o in b), max(o.timestamp_s for o in b)
                gap = max(0.0, ast - bend, bst - aend)
                if gap > config.timeout_s:
                    continue
                relation = None
                if ca == cb:
                    earlier, later = (a, b) if aend < bst else (b, a)
                    tail = max(earlier, key=lambda o: o.timestamp_s)
                    head = min(later, key=lambda o: o.timestamp_s)
                    if head.timestamp_s > tail.timestamp_s and plausible_recovery(
                        tail.bbox_xyxy, head.bbox_xyxy, gap
                    ):
                        relation = "same_camera"
                elif cb in engine._overlaps[ca]:
                    relation = "overlap"
                elif aend < bst:
                    relation = engine.graph.relation(ca, cb, bst - aend)
                elif bend < ast:
                    relation = engine.graph.relation(cb, ca, ast - bend)
                if relation is None:
                    continue
                ga, gb = gallery(a), gallery(b)
                if min(len(ga), len(gb)) < config.tracklet_min_samples:
                    continue
                # Refuse incompatible feature spaces, even if dimensions happen to coincide.
                spaces = {(o.embedding_model_id, len(o.embedding)) for o in ga + gb}
                if len(spaces) != 1:
                    continue
                matrix = [
                    [_cosine_similarity(x.embedding, y.embedding) for y in gb]
                    for x in ga
                ]
                # Both views must have consistent support; a single lucky pair is insufficient.
                score = fmean(
                    [
                        median(max(row) for row in matrix),
                        median(max(row[j] for row in matrix) for j in range(len(gb))),
                    ]
                )
                geo = engine.floor.between(a, b) if engine.floor else None
                if config.decision_mode == "balanced":
                    from .scoring import appearance_quality, decide, geometry_quality

                    aq = min(
                        median(
                            appearance_quality(o, engine.cameras[o.camera_id])
                            for o in samples
                        )
                        for samples in [ga, gb]
                    )
                    gq = min(
                        geometry_quality(engine.floor, a),
                        geometry_quality(engine.floor, b),
                    )
                    threshold_value = (
                        config.tracklet_threshold
                        if config.tracklet_threshold is not None
                        else config.similarity_threshold
                    )
                    decision = decide(
                        score,
                        aq,
                        geo,
                        gq,
                        engine.graph.bonus(relation, config.adjacency_bonus),
                        threshold_value,
                    )
                    scores.append(decision.score)
                    engine.decision_log.append(
                        {
                            "phase": "offline",
                            "left_reference": list(a[0].reference),
                            "right_reference": list(b[0].reference),
                            **decision.export(),
                        }
                    )
                    continue
                if geo is not None and score >= 0.4:
                    engine.floor.stats["offline_evaluations"] += 1
                    score += geo
                scores.append(
                    min(
                        1.0,
                        score + engine.graph.bonus(relation, config.adjacency_bonus),
                    )
                )
        return max(scores) if scores else None

    threshold = (
        config.tracklet_threshold
        if config.tracklet_threshold is not None
        else config.similarity_threshold
    )
    while True:
        groups = group_data()
        ids = sorted(groups, key=lambda g: (min(o.timestamp_s for o in groups[g]), g))
        candidates = []
        for i, a in enumerate(ids):
            if any(find(g) == a for g in blocked):
                continue
            for b in ids[i + 1 :]:
                if any(find(g) == b for g in blocked):
                    continue
                score = score_groups(groups[a], groups[b])
                if score is not None:
                    candidates.append((score, a, b))
        neighbors = defaultdict(list)
        for score, a, b in candidates:
            neighbors[a].append((score, b))
            neighbors[b].append((score, a))
        for options in neighbors.values():
            options.sort(key=lambda pair: (-pair[0], pair[1]))

        def confident(a, b, neighbors):
            options = neighbors[a]
            return options[0][1] == b and (
                len(options) == 1
                or options[0][0] - options[1][0] >= config.ambiguity_margin
            )

        selected = next(
            (
                (score, a, b)
                for score, a, b in sorted(candidates, key=lambda e: (-e[0], e[1], e[2]))
                if score >= threshold
                and confident(a, b, neighbors)
                and confident(b, a, neighbors)
            ),
            None,
        )
        if selected is None:
            break
        score, a, b = selected
        parent[b] = a
        for gid in original_ids:
            if find(gid) == a:
                evidence[gid] = score
        log.append(
            {
                "kept_id": a,
                "merged_id": b,
                "score": round(score, 8),
                "threshold": threshold,
                "reason": "offline_multiframe_reid_floor"
                if engine.floor
                else "offline_multiframe_reid",
            }
        )

    revised = []
    for row in assignments:
        row = dict(row)
        old = row["global_track_id"]
        new = find(old)
        if new != old:
            row["global_track_id"] = new
            row["reason"] = "offline_multiframe_reid"
            row["match_score"] = round(evidence[old], 8)
        revised.append(row)
    exports = []
    for gid, obs in group_data().items():
        members = [track_by_id[k] for k in original_ids if find(k) == gid]
        if len(members) == 1:
            exports.append(members[0])
            continue
        latest = max(members, key=lambda t: (t["last_seen_s"], t["global_track_id"]))
        exports.append(
            {
                "global_track_id": gid,
                "start_s": min(o.timestamp_s for o in obs),
                "last_seen_s": max(o.timestamp_s for o in obs),
                "end_s": max(o.timestamp_s for o in obs),
                "status": "closed",
                "camera_ids": sorted({o.camera_id for o in obs}),
                "close_reason": latest["close_reason"],
            }
        )
    return revised, sorted(exports, key=lambda t: t["global_track_id"]), log
