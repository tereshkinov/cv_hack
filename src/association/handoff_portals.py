"""Explicit image-space transfer zones for uncertain cross-camera floor mapping.

A portal is a configured topological connection, not a metric floor transform.
It does not move anchors or infer speed from the discrepancy between views.
"""

from math import exp, hypot, isfinite, sqrt
from statistics import median

import cv2
import numpy as np

from .scene_filter import validate_floor_filter


def validate_portals(values, camera_ids):
    if not isinstance(values, list) or len(values) > 100:
        raise ValueError("Не более 100 переходов между камерами")
    ids = set()
    for portal in values:
        if not isinstance(portal, dict):
            raise TypeError("Описание перехода должно быть объектом")
        pid = portal.get("id")
        if not isinstance(pid, str) or not pid or len(pid) > 80 or pid in ids:
            raise ValueError("У переходов должны быть уникальные ID")
        ids.add(pid)
        source, target = portal.get("source_camera"), portal.get("target_camera")
        if source not in camera_ids or target not in camera_ids or source == target:
            raise ValueError("Некорректная пара камер перехода")
        if not isinstance(portal.get("enabled", True), bool):
            raise TypeError("enabled должен быть boolean")
        for key in ["source_polygon", "target_polygon"]:
            polygon = portal.get(key)
            validate_floor_filter({"mode": "polygon", "allowed_polygon": polygon})
            if any(not 0 <= x <= 1 for p in polygon for x in p):
                raise ValueError("Зоны перехода задаются долями ширины и высоты кадра")
        for key in ["source_direction", "target_direction"]:
            v = portal.get(key)
            if (
                not isinstance(v, list)
                or len(v) != 2
                or any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or not isfinite(x)
                    for x in v
                )
                or hypot(*v) < 1e-6
            ):
                raise ValueError(
                    "Нужно ненулевое направление движения для каждой стороны перехода"
                )
        for key, default, low, high in [
            ("min_gap_s", -0.3, -2, 0),
            ("max_gap_s", 5, 0, 30),
            ("expected_gap_s", 1.5, 0, 30),
            ("timing_sigma_s", 1.2, 0.1, 10),
            ("reliability", 0.7, 0.1, 0.85),
            ("min_motion_span_s", 0.4, 0.4, 1.5),
        ]:
            x = portal.get(key, default)
            if (
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not isfinite(x)
                or not low <= x <= high
            ):
                raise ValueError("Некорректный параметр перехода: " + key)
        if (
            not portal.get("min_gap_s", -0.3)
            <= portal.get("expected_gap_s", 1.5)
            <= portal.get("max_gap_s", 5)
        ):
            raise ValueError("Ожидаемое время должно лежать в пределах перехода")
    return values


def _motion(rows, camera, boundary):
    if not camera.width or not camera.height:
        return None
    points = []
    for o in sorted(rows, key=lambda o: o.timestamp_s):
        if points and o.timestamp_s - points[-1][0] < 0.12:
            continue
        x1, _y1, x2, y2 = o.bbox_xyxy
        if y2 >= camera.height * 0.985:
            continue
        points.append(
            (
                o.timestamp_s,
                (x1 + x2) / 2 / camera.width,
                y2 / camera.height,
                o.detection_confidence,
            )
        )
    if len(points) < 3 or points[-1][0] - points[0][0] < 0.4 - 1e-8:
        return None
    times = np.array([p[0] for p in points])
    xy = np.array([p[1:3] for p in points])
    velocities = [
        (xy[j] - xy[i]) / (times[j] - times[i])
        for i in range(len(points))
        for j in range(i + 1, len(points))
        if times[j] - times[i] >= 0.25
    ]
    if not velocities:
        return None
    velocity = np.median(velocities, axis=0)
    t0 = times[0]
    origin = np.median(xy - (times - t0)[:, None] * velocity, axis=0)
    jitter = float(
        np.median(
            np.linalg.norm(xy - origin - (times - t0)[:, None] * velocity, axis=1)
        )
    )
    time = times[-1] if boundary == "tail" else times[0]
    near = [p for p in points if abs(p[0] - time) <= 0.35]
    endpoint = np.median([p[1:3] for p in near], axis=0)
    return {
        "endpoint": endpoint,
        "velocity": velocity,
        "jitter": jitter,
        "span": float(times[-1] - times[0]),
        "samples": len(points),
        "confidence": median(p[3] for p in points),
    }


def portal_evidence(engine, a, b):
    portals = getattr(engine, "scene_project", {}).get("handoff_portals", [])
    if not portals or a.camera_id == b.camera_id:
        return None, None
    gap = b.start_s - a.end_s
    candidates = []
    for portal in portals:
        if not portal.get("enabled", True) or (
            portal["source_camera"],
            portal["target_camera"],
        ) != (a.camera_id, b.camera_id):
            continue
        if (
            not portal.get("min_gap_s", -0.3) - 1e-8
            <= gap
            <= portal.get("max_gap_s", 5) + 1e-8
        ):
            continue
        left = _motion(a.tail(1.5), engine.cameras[a.camera_id], "tail")
        right = _motion(b.head(1.5), engine.cameras[b.camera_id], "head")
        if not left or not right:
            continue
        minimum_span = portal.get("min_motion_span_s", 0.4)
        if min(left["span"], right["span"]) < minimum_span - 1e-8:
            continue
        fits = []
        for motion, polygon, direction in [
            (left, portal["source_polygon"], portal["source_direction"]),
            (right, portal["target_polygon"], portal["target_direction"]),
        ]:
            if (
                cv2.pointPolygonTest(
                    np.asarray(polygon, dtype=np.float32),
                    tuple(map(float, motion["endpoint"])),
                    False,
                )
                < 0
            ):
                break
            v = motion["velocity"]
            d = np.asarray(direction, dtype=float)
            d /= np.linalg.norm(d)
            projected = float(np.dot(v, d))
            speed = float(np.linalg.norm(v))
            displacement = projected * motion["span"]
            # Coherent approach/departure, not a stationary object in a gate.
            if displacement < max(0.004, 2 * motion["jitter"]) or speed < 0.003:
                break
            cosine = projected / speed
            if cosine < 0.25:
                break
            fits.append(cosine)
        if len(fits) != 2:
            continue
        timing = (gap - portal.get("expected_gap_s", 1.5)) / portal.get(
            "timing_sigma_s", 1.2
        )
        discrepancy = sqrt(timing**2 + 0.3 * sum((1 - c) ** 2 for c in fits))
        quality = portal.get("reliability", 0.7) * min(
            left["confidence"], right["confidence"]
        )
        details = {
            "reason": "handoff_trajectory_portal",
            "portal_id": portal["id"],
            "evidence_window": "image_transfer_zones",
            "gap_s": gap,
            "expected_gap_s": portal.get("expected_gap_s", 1.5),
            "timing_sigma_s": portal.get("timing_sigma_s", 1.2),
            "direction_cosines": fits,
            "source_point": left["endpoint"].tolist(),
            "target_point": right["endpoint"].tolist(),
            "left_samples": left["samples"],
            "right_samples": right["samples"],
            "left_span_s": left["span"],
            "right_span_s": right["span"],
            "geometry_quality": quality,
            "discrepancy": discrepancy,
            "coordinate_units": "normalized_image",
            "metric_distance_used": False,
            "lookahead_window_s": 1.5,
            "min_motion_span_s": minimum_span,
        }
        candidates.append(
            (
                (
                    exp(-0.5 * discrepancy**2),
                    quality,
                    discrepancy,
                    details["reason"],
                    min(left["samples"], right["samples"]),
                ),
                details,
            )
        )
    return (
        max(candidates, key=lambda e: e[0][0] * e[0][1]) if candidates else (None, None)
    )
