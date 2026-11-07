"""Geometry-first offline identities. Appearance only breaks spatial ambiguity.

Local silhouette tracks are the units of continuity. The existing constrained
fragment solver makes joint decisions and defers near-equivalent alternatives.
"""

from collections import defaultdict
from dataclasses import replace
from math import exp
from statistics import median

from src.floor_mesh import overlap_report

from .fragments import _gallery, fragments_from_observations, rebuild_fragments


def _visual(engine, a, b):
    from .engine import _cosine_similarity

    ga = _gallery(a.tail(), engine.config.min_embedding_confidence)
    gb = _gallery(b.head(), engine.config.min_embedding_confidence)
    if (
        not ga
        or not gb
        or len({(o.embedding_model_id, len(o.embedding)) for o in ga + gb}) != 1
    ):
        return None
    scores = [[_cosine_similarity(x.embedding, y.embedding) for y in gb] for x in ga]
    return (
        median(max(row) for row in scores)
        + median(max(row[j] for row in scores) for j in range(len(gb)))
    ) / 2


def _partial_image_support(engine, a, b):
    """A horizontal bearing for a short cropped/shrunken silhouette, not a floor point."""
    if a.camera_id != b.camera_id or not 0 < b.start_s - a.end_s <= 1.5:
        return None
    left, right = a.tail(1), b.head(1)
    if min(len(left), len(right)) < 3:
        return None
    h = engine.cameras[a.camera_id].height
    height = lambda o: o.bbox_xyxy[3] - o.bbox_xyxy[1]
    ha, hb = median(height(o) for o in left), median(height(o) for o in right)
    cropped = bool(h and any(o.bbox_xyxy[3] >= h * 0.985 for o in left + right))
    if not cropped and max(ha, hb) / max(min(ha, hb), 1) > 1.6:
        cropped = True
    if not cropped:
        return None
    x = lambda o: (o.bbox_xyxy[0] + o.bbox_xyxy[2]) / 2
    width = median(o.bbox_xyxy[2] - o.bbox_xyxy[0] for o in left + right)
    dt = left[-1].timestamp_s - left[0].timestamp_s
    velocity = (x(left[-1]) - x(left[0])) / max(dt, 0.01)
    expected = x(left[-1]) + velocity * (b.start_s - a.end_s)
    discrepancy = abs(x(right[0]) - expected) / max(width, 1)
    q = 0.35 * min(
        median(o.detection_confidence for o in left),
        median(o.detection_confidence for o in right),
    )
    return (
        exp(-0.5 * discrepancy**2),
        q,
        discrepancy,
        "partial_image_bearing",
        min(len(left), len(right)),
    )


def _geometry(engine, a, b):
    floor = engine.floor
    if a.start_s > b.start_s:
        a, b = b, a
    overlap = min(a.end_s, b.end_s) - b.start_s
    mode = (
        "simultaneous_overlap"
        if overlap >= 0
        else "same_camera_recovery"
        if a.camera_id == b.camera_id
        else "sequential_transition"
    )
    details = {"mode": mode, "appearance_used": False}
    if overlap >= 0 and a.camera_id == b.camera_id:
        return None, details | {"reason": "same_camera_conflict"}
    boundary = (
        engine.boundary_geometry.compare(a, b, mode)
        if engine.boundary_geometry
        else None
    )
    evidence = _partial_image_support(engine, a, b)
    if floor and evidence is None:
        if overlap >= 0:
            # Only compare simultaneous measurements. Never compare an entire history to its mean.
            distances = []
            qualities = []
            right = [
                (o, floor.projection(o))
                for o in b.observations
                if o.timestamp_s <= a.end_s + 0.3
            ]
            for o in a.observations:
                if o.timestamp_s < b.start_s - 0.3:
                    continue
                p = floor.projection(o)
                options = [
                    (abs(o.timestamp_s - x.timestamp_s), q)
                    for x, q in right
                    if q.available
                ]
                if not p.available or not options:
                    continue
                dt, q = min(options, key=lambda item: item[0])
                if dt > 0.3:
                    continue
                from math import hypot

                distances.append(
                    hypot(p.point[0] - q.point[0], p.point[1] - q.point[1])
                    / max(floor.radius + floor.speed * dt, 0.001)
                )
                qualities.append(min(p.reliability, q.reliability))
            if distances:
                discrepancy = median(distances)
                evidence = (
                    exp(-0.5 * discrepancy**2),
                    median(qualities),
                    discrepancy,
                    "simultaneous_floor",
                    len(distances),
                )
        elif b.start_s - a.end_s <= 4:
            spatial = floor.between_evidence(a.tail(), b.head())
            if spatial.available and spatial.discrepancy is not None:
                evidence = (
                    exp(-0.5 * spatial.discrepancy**2),
                    spatial.reliability,
                    spatial.discrepancy,
                    spatial.reason,
                    2,
                )
    # Route timing is useful in blind corridors. Image motion supports cropped feet
    # on the same camera; neither invents a floor coordinate for display.
    if evidence is None and boundary and boundary.available:
        evidence = (
            boundary.value,
            boundary.reliability,
            boundary.discrepancy,
            boundary.reason,
            boundary.support_count,
        )
    if (
        evidence is None
        and floor
        and 4 < b.start_s - a.end_s <= engine.config.timeout_s
    ):
        left = [floor.projection(o) for o in a.tail()]
        right = [floor.projection(o) for o in b.head()]
        left = [p for p in left if p.available]
        right = [p for p in right if p.available]
        if left and right:
            from math import hypot

            distance = hypot(
                left[-1].point[0] - right[0].point[0],
                left[-1].point[1] - right[0].point[1],
            )
            # Reachability only: no minute-long extrapolation of a stale velocity.
            radius = max(floor.radius, floor.speed * (b.start_s - a.end_s))
            discrepancy = distance / radius
            evidence = (
                exp(-0.5 * discrepancy**2),
                0.22,
                discrepancy,
                "blind_gap_reachable",
                1,
            )
    if evidence is None:
        return None, details | {"reason": "geometry_unavailable"}
    value, quality, discrepancy, reason, support = evidence
    details |= {
        "geometry_compatibility": value,
        "geometry_quality": quality,
        "discrepancy": discrepancy,
        "reason": reason,
        "support_count": support,
        "boundary_geometry": boundary.as_dict() if boundary else None,
    }
    # The residual of interpolation at its own anchors is not an accuracy measure.
    if floor and reason not in {
        "partial_image_bearing",
        "same_camera_position_only",
        "same_camera_bidirectional",
        "same_camera_one_sided_motion",
    }:
        factors = []
        for cid in [a.camera_id, b.camera_id]:
            checks = floor.calibration_checks.get(cid, {})
            error = checks.get("rmse_plan_units")
            factors.append(
                0.8 if error is None else 1 / (1 + error / max(floor.radius, 1e-6))
            )
        quality *= min(factors)
        details["geometry_quality"] = quality
        details["calibration_validation"] = {
            cid: floor.calibration_checks.get(cid) for cid in [a.camera_id, b.camera_id]
        }
    if quality < 0.15:
        return None, details | {"reason": "geometry_too_uncertain"}
    if discrepancy is not None and discrepancy >= 3 and quality >= 0.6:
        return 0.2, details | {"reason": "spatial_contradiction"}
    score = 0.55 + 0.35 * value - 0.10 * (1 - quality)
    return score, details


def associate_geometry(engine, observations, exits):
    observations = sorted(
        observations,
        key=lambda o: (o.timestamp_s, o.camera_id, o.local_track_id, o.frame_index),
    )
    exits = sorted(exits, key=lambda e: (e.timestamp_s, e.camera_id, e.local_track_id))
    if engine.floor:
        patches = overlap_report(engine.floor.project)
        for patch in patches:
            a, b = patch["camera_a"], patch["camera_b"]
            if a in engine.graph.overlaps and b in engine.graph.overlaps:
                engine.graph.overlaps[a].add(b)
                engine.graph.overlaps[b].add(a)
    else:
        patches = []
    cfg = engine.config
    engine.config = replace(cfg, fragment_mode="rebuild")
    try:
        nodes = fragments_from_observations(observations, engine.config, exits)
        cache = {}
        neighbors = defaultdict(list)
        for i, a in enumerate(nodes):
            for b in nodes[i + 1 :]:
                if b.start_s - a.end_s > cfg.timeout_s:
                    continue
                overlap = b.start_s <= a.end_s
                relation = engine.graph.relation(
                    a.camera_id, b.camera_id, 0 if overlap else b.start_s - a.end_s
                )
                if overlap and relation != "overlap":
                    continue
                if not overlap and relation is None:
                    continue
                score, details = _geometry(engine, a, b)
                cache[a.fragment_id, b.fragment_id] = (score, details)
                if score is not None and details["reason"] != "spatial_contradiction":
                    neighbors[a.fragment_id].append((score, b.fragment_id))
                    neighbors[b.fragment_id].append((score, a.fragment_id))

        def provider(a, b):
            score, details = cache.get(
                (a.fragment_id, b.fragment_id), (None, {"reason": "no_geometry"})
            )
            details = dict(details)
            if score is None or details.get("reason") == "spatial_contradiction":
                return score, details
            # Compare competitors whose intervals actually overlap. Sequential visits
            # minutes apart do not make each other's geometry ambiguous.
            competitors = []
            byid = {n.fragment_id: n for n in nodes}
            for source, target in [(a, b), (b, a)]:
                competitors.extend(
                    s
                    for s, other in neighbors[source.fragment_id]
                    if other != target.fragment_id
                    and max(byid[other].start_s, target.start_s)
                    <= min(byid[other].end_s, target.end_s)
                )
            ambiguous = (
                any(abs(score - s) < 0.08 for s in competitors)
                or details.get("geometry_quality", 0) < 0.30
            )
            if ambiguous:
                similarity = _visual(engine, a, b)
                details["appearance_attempted"] = True
                details["appearance_used"] = similarity is not None
                details["appearance"] = similarity
                details["geometry_ambiguous"] = True
                if similarity is None:
                    threshold = (
                        cfg.tracklet_threshold
                        if cfg.tracklet_threshold is not None
                        else cfg.similarity_threshold
                    )
                    return min(score, threshold - 0.04), details | {
                        "reason": "unresolved_geometry"
                    }
                if details.get("geometry_quality", 0) < 0.30:
                    threshold = (
                        cfg.tracklet_threshold
                        if cfg.tracklet_threshold is not None
                        else cfg.similarity_threshold
                    )
                    if similarity < max(0.80, cfg.similarity_threshold):
                        return threshold - 0.04, details | {
                            "reason": "weak_appearance_for_uncertain_geometry"
                        }
                    # A large reachable region is permission to consider a candidate,
                    # not strong identity evidence. Bound its margin conservatively.
                    score = threshold + 0.04 + min(0.10, (similarity - 0.80) * 0.5)
                else:
                    score += max(-0.12, min(0.12, (similarity - 0.70) * 0.5))
            return score, details

        engine.boundary_provider = provider
        seeds = [
            {
                "camera_id": o.camera_id,
                "frame_index": o.frame_index,
                "local_track_id": o.local_track_id,
                "global_track_id": f"local:{o.camera_id}:{o.local_track_id}",
                "timestamp_s": o.timestamp_s,
                "reason": "local_silhouette",
            }
            for o in observations
        ]
        assignments, tracks, report = rebuild_fragments(
            engine, observations, seeds, [], exits
        )
        report["policy"] = "geometry_first"
        report["calibrated_overlap_patches"] = patches
        report["appearance_tiebreak_edges"] = sum(
            bool(e.get("appearance_used")) for e in report.get("edges", [])
        )
        engine.fragment_report = report
        for row in assignments:
            row["reason"] = "geometry_first"
        if cfg.confirm_tracks:
            from .confirmation import annotate_confirmation

            assignments, tracks, engine.confirmation_report = annotate_confirmation(
                observations, assignments, tracks, cfg, engine.floor
            )
        return assignments, tracks
    finally:
        engine.config = cfg
