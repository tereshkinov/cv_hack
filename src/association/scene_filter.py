"""Auditable floor admission and conservative stationary-artifact candidates.

Scores below are heuristic evidence, not calibrated probabilities. Raw inputs
are never erased; rejected observations get diagnostic IDs in the final export.
"""

from collections import defaultdict
from itertools import pairwise
from math import hypot, isfinite
from statistics import median

import cv2
import numpy as np


def validate_floor_filter(value):
    if not isinstance(value, dict) or value.get("mode", "mesh") not in {
        "mesh",
        "polygon",
        "off",
    }:
        raise ValueError("Зона пола: mesh, polygon или off")
    allowed = value.get("allowed_polygon", [])
    excluded = value.get("excluded_polygons", [])
    if not isinstance(excluded, list) or len(excluded) > 32:
        raise ValueError("Не более 32 запрещённых областей")
    polygons = ([allowed] if allowed else []) + excluded
    for polygon in polygons:
        if not isinstance(polygon, list) or not 3 <= len(polygon) <= 128:
            raise ValueError("Полигон должен содержать 3–128 точек кадра")
        for point in polygon:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or not isfinite(x)
                    or not 0 <= x <= 10_000_000
                    for x in point
                )
            ):
                raise ValueError("Точки зоны: конечные неотрицательные x, y")
        p = np.asarray(polygon, dtype=np.float32)
        if abs(cv2.contourArea(p)) < 1e-3 or len({tuple(q) for q in polygon}) != len(
            polygon
        ):
            raise ValueError("Полигон зоны вырожден или содержит повторные точки")

        # Reject self-crossings/touching non-adjacent edges; concavity is allowed.
        def orient(a, b, c):
            return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

        def intersects(a, b, c, d):
            if max(min(a[0], b[0]), min(c[0], d[0])) > min(
                max(a[0], b[0]), max(c[0], d[0])
            ) or max(min(a[1], b[1]), min(c[1], d[1])) > min(
                max(a[1], b[1]), max(c[1], d[1])
            ):
                return False
            return (
                orient(a, b, c) * orient(a, b, d) <= 0
                and orient(c, d, a) * orient(c, d, b) <= 0
            )

        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if j == i + 1 or (i == 0 and j == len(p) - 1):
                    continue
                if intersects(p[i], p[(i + 1) % len(p)], p[j], p[(j + 1) % len(p)]):
                    raise ValueError("Граница зоны пересекает сама себя")
    if value.get("mode") == "polygon" and not allowed:
        raise ValueError("Сначала завершите разрешённый полигон")
    return value


def _distance(polygon, point):
    return cv2.pointPolygonTest(np.asarray(polygon, dtype=np.float32), point, True)


def select_floor_observations(observations, project, cameras):
    zones, unavailable = {}, []
    for camera in project.get("cameras", []):
        cid = camera["id"]
        settings = validate_floor_filter(camera.get("floor_filter", {}))
        mode = settings.get("mode", "mesh")
        calibration = camera.get("calibration") or {}
        polygons = []
        if mode == "polygon":
            polygons = [settings["allowed_polygon"]]
        elif mode == "mesh":
            points = calibration.get("video_points", camera.get("video_points", []))
            if points and calibration:
                if calibration.get("triangles"):
                    polygons = [
                        [points[i] for i in tri] for tri in calibration["triangles"]
                    ]
                else:
                    polygons = [
                        cv2.convexHull(np.asarray(points, dtype=np.float32))
                        .reshape(-1, 2)
                        .tolist()
                    ]
            else:
                unavailable.append(cid)
        zones[cid] = (polygons, settings.get("excluded_polygons", []))
    accepted, rejected, last_inside = [], [], {}
    for o in sorted(
        observations,
        key=lambda o: (o.timestamp_s, o.camera_id, o.local_track_id, o.frame_index),
    ):
        polygons, exclusions = zones.get(o.camera_id, ([], []))
        point = ((o.bbox_xyxy[0] + o.bbox_xyxy[2]) / 2, o.bbox_xyxy[3])
        if any(_distance(p, point) >= 0 for p in exclusions):
            rejected.append((o, "excluded_polygon"))
            continue
        distance = max((_distance(p, point) for p in polygons), default=0)
        if distance >= -1e-5:
            last_inside[o.local_key] = o.timestamp_s
            accepted.append(o)
            continue
        camera = cameras[o.camera_id]
        # Grace is measured from the last strictly inside observation; it cannot
        # perpetuate itself through a long chain of outside detections.
        margin = max(3, 0.015 * hypot(camera.width or 0, camera.height or 0))
        if (
            o.timestamp_s - last_inside.get(o.local_key, float("-inf")) <= 1
            and distance >= -margin
        ):
            accepted.append(o)
        else:
            rejected.append((o, "outside_floor_zone"))
    return (
        accepted,
        rejected,
        {
            "unavailable_zone_cameras": unavailable,
            "kept_observations": len(accepted),
            "rejected_observations": len(rejected),
        },
    )


def _statistics(rows):
    boxes = np.asarray([o.bbox_xyxy for o in rows])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    sizes = boxes[:, 2:] - boxes[:, :2]
    center, size = np.median(centers, axis=0), np.median(sizes, axis=0)
    jitter = float(np.quantile(np.linalg.norm(centers - center, axis=1), 0.9))
    size_jitter = float(
        np.quantile(np.max(np.abs(sizes - size) / np.maximum(size, 1), axis=1), 0.9)
    )
    static = (
        len(rows) >= 3
        and jitter <= max(3, float(size[0]) * 0.08)
        and size_jitter <= 0.10
    )
    times = sorted({round(o.timestamp_s, 6) for o in rows})
    steps = [b - a for a, b in pairwise(times)]
    cadence = median(steps) if steps else 0
    flicker = bool(cadence and any(d > max(0.5, 2.5 * cadence) for d in steps))
    return {
        "center": center.tolist(),
        "size": size.tolist(),
        "jitter_px": jitter,
        "size_jitter": size_jitter,
        "static": bool(static),
        "flicker": flicker,
        "mean_confidence": sum(o.detection_confidence for o in rows) / len(rows),
    }


def finish_scene_filter(observations, assignments, tracks, rejected, zone_report):
    by_ref = {o.reference: o for o in observations}
    grouped = defaultdict(list)
    for row in assignments:
        grouped[row["global_track_id"]].append(
            by_ref[(row["camera_id"], row["frame_index"], row["local_track_id"])]
        )
    stats = {}
    for t in tracks:
        rows = grouped[t["global_track_id"]]
        if (
            len({o.camera_id for o in rows}) == 1
            and t["last_seen_s"] - t["start_s"] <= 3
        ):
            stats[t["global_track_id"]] = _statistics(rows)
    decisions = []
    for track in tracks:
        gid = track["global_track_id"]
        if gid not in stats or not stats[gid]["static"]:
            continue
        s = stats[gid]
        repeats = []
        for other in tracks:
            oid = other["global_track_id"]
            if (
                oid == gid
                or oid not in stats
                or not stats[oid]["static"]
                or other["camera_ids"] != track["camera_ids"]
            ):
                continue
            separation = max(track["start_s"], other["start_s"]) - min(
                track["last_seen_s"], other["last_seen_s"]
            )
            q = stats[oid]
            if (
                separation >= 0.5
                and hypot(*(np.asarray(s["center"]) - q["center"]))
                <= max(4, s["size"][0] * 0.10)
                and max(abs(a - b) / max(a, 1) for a, b in zip(s["size"], q["size"]))
                < 0.15
            ):
                repeats.append(oid)
        # Isolated, coherent stationary people are deliberately retained. Repeated
        # hotspots need a second independent cue (weak detector or flicker).
        low = s["mean_confidence"] < 0.45
        score = (
            0.10
            + 0.20
            + 0.10
            + (0.30 if len(repeats) >= 2 else 0)
            + (0.20 if low else 0)
            + (0.20 if s["flicker"] else 0)
        )
        suppress = score >= 0.75 and (low or s["flicker"])
        decision = dict(
            global_track_id=gid,
            score=round(min(score, 1), 3),
            threshold=0.75,
            probability_calibrated=False,
            repeated_tracks=repeats,
            suppressed=suppress,
            **s,
        )
        decisions.append(decision)
        track["artifact_evidence"] = decision
        if suppress:
            track.update(
                scene_state="suppressed",
                scene_reason="stationary_artifact_candidate",
                confirmation_state="suppressed",
                confirmed_at_s=None,
                confirmation_reason="scene_filter",
            )
    metadata = {t["global_track_id"]: t for t in tracks}
    for a in assignments:
        track = metadata[a["global_track_id"]]
        if track.get("scene_state") == "suppressed":
            a.update(
                scene_state="suppressed",
                scene_reason=track["scene_reason"],
                confirmation_state="suppressed",
                confirmed_at_s=None,
                observed_state="suppressed",
            )
    # Separate rejected runs at an inside observation or temporal gap. These are
    # diagnostic detections, not globally associated identities.
    groups = defaultdict(list)
    for o, reason in rejected:
        groups[(o.local_key, reason)].append(o)
    rejected_runs = []
    accepted_times = defaultdict(list)
    for a in assignments:
        accepted_times[(a["camera_id"], a["local_track_id"])].append(a["timestamp_s"])
    for (key, reason), rows in sorted(groups.items()):
        current = []
        for o in rows:
            if current and (
                o.timestamp_s - current[-1].timestamp_s > 1
                or any(
                    current[-1].timestamp_s < t < o.timestamp_s
                    for t in accepted_times[key]
                )
            ):
                rejected_runs.append((current, reason))
                current = []
            current.append(o)
        if current:
            rejected_runs.append((current, reason))
    for i, (rows, reason) in enumerate(rejected_runs, 1):
        gid = f"filtered{i}"
        tracks.append(
            {
                "global_track_id": gid,
                "start_s": rows[0].timestamp_s,
                "last_seen_s": rows[-1].timestamp_s,
                "end_s": rows[-1].timestamp_s,
                "status": "closed",
                "close_reason": "scene_filter",
                "camera_ids": sorted({o.camera_id for o in rows}),
                "scene_state": "suppressed",
                "scene_reason": reason,
                "confirmation_state": "suppressed",
                "confirmed_at_s": None,
                "confirmation_reason": "scene_filter",
            }
        )
        for o in rows:
            assignments.append(
                {
                    "camera_id": o.camera_id,
                    "frame_index": o.frame_index,
                    "local_track_id": o.local_track_id,
                    "timestamp_s": o.timestamp_s,
                    "global_track_id": gid,
                    "reason": reason,
                    "scene_state": "suppressed",
                    "scene_reason": reason,
                    "confirmation_state": "suppressed",
                    "confirmed_at_s": None,
                    "observed_state": "suppressed",
                }
            )
    assignments.sort(
        key=lambda a: (a["timestamp_s"], a["camera_id"], a["local_track_id"])
    )
    tracks.sort(key=lambda t: (t["start_s"], t["global_track_id"]))
    return (
        assignments,
        tracks,
        dict(
            enabled=True,
            **zone_report,
            stationary_decisions=decisions,
            suppressed_tracks=sum(t.get("scene_state") == "suppressed" for t in tracks),
            stationary_suppressed=sum(d["suppressed"] for d in decisions),
        ),
    )
