"""Validated, bidirectional piecewise affine floor calibration.

Triangles interpolate measured anchors. No fitted/global lens model is assumed.
Outside the measured hull, the default is refusal, never an invented position.
"""

from math import hypot

import cv2
import numpy as np

METHOD = "piecewise_affine_floor_mesh"


def _points(value, name):
    p = np.asarray(value, dtype=float)
    if (
        p.ndim != 2
        or p.shape[1] != 2
        or not 4 <= len(p) <= 128
        or not np.isfinite(p).all()
    ):
        raise ValueError(f"{name}: нужны от 4 до 128 конечных пар координат")
    span = max(float(np.ptp(p, axis=0).max()), 1e-9)
    if min(np.ptp(p, axis=0)) < span * 1e-7:
        raise ValueError(f"{name}: вырожденная область")
    distances = np.linalg.norm(p[:, None] - p[None, :], axis=2) + np.eye(len(p)) * span
    if distances.min() < span * 1e-6:
        raise ValueError(f"{name}: совпадающие точки")
    return p


def _signed_area(p):
    a, b, c = p
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def calibrate_mesh(video_points, plan_points):
    source, target = _points(video_points, "Кадр"), _points(plan_points, "Схема")
    if len(source) != len(target):
        raise ValueError("Число пар на кадре и схеме должно совпадать")
    # Triangulate the plan; normalized coordinates avoid Subdiv's integer rectangle limits.
    base = target.min(axis=0)
    span = float(np.ptp(target, axis=0).max())
    normalized = (target - base) / span * 1000 + 10
    subdiv = cv2.Subdiv2D((0, 0, 1024, 1024))
    for p in normalized:
        subdiv.insert(tuple(map(float, p)))
    triangles = set()
    for tri in subdiv.getTriangleList():
        vertices = np.asarray(tri).reshape(3, 2)
        indices = [
            int(np.argmin(np.linalg.norm(normalized - p, axis=1))) for p in vertices
        ]
        if len(set(indices)) == 3 and all(
            np.linalg.norm(normalized[i] - p) < 0.01 for i, p in zip(indices, vertices)
        ):
            triangles.add(tuple(sorted(indices)))
    triangles = sorted(triangles)
    if not triangles or {i for t in triangles for i in t} != set(range(len(source))):
        raise ValueError("Не удалось построить сетку: проверьте распределение точек")
    signs = []
    for t in triangles:
        a, b = _signed_area(source[list(t)]), _signed_area(target[list(t)])
        if abs(a) < np.ptp(source, axis=0).max() ** 2 * 1e-8 or abs(b) < span**2 * 1e-8:
            raise ValueError("Слишком узкий или вырожденный треугольник")
        signs.append(a * b > 0)
    if len(set(signs)) > 1:
        raise ValueError("Сетка складывается: перепутаны соответствующие точки")
    # Consistent local orientation is necessary but not sufficient for an injective mesh.
    for i, t in enumerate(triangles):
        for u in triangles[i + 1 :]:
            for points in (source, target):
                a = cv2.convexHull(points[list(t)].astype(np.float32))
                b = cv2.convexHull(points[list(u)].astype(np.float32))
                intersection, _ = cv2.intersectConvexConvex(a, b)
                if intersection > max(cv2.contourArea(a), cv2.contourArea(b)) * 1e-5:
                    raise ValueError("Треугольники сетки перекрываются; проверьте пары")
    return {
        "method": METHOD,
        "video_points": source.tolist(),
        "plan_points": target.tolist(),
        "triangles": [list(t) for t in triangles],
        "max_anchor_error_m": 0.0,
        "coverage": "measured_convex_hull",
        "anchor_error_is_validation": False,
    }


def map_point(calibration, point, *, inverse=False, extrapolate=False):
    """Return (point|None, quality, reason). Small extrapolation is explicitly marked."""
    source = np.asarray(
        calibration["plan_points" if inverse else "video_points"], dtype=float
    )
    target = np.asarray(
        calibration["video_points" if inverse else "plan_points"], dtype=float
    )
    p = np.asarray(point, dtype=float)
    if p.shape != (2,) or not np.isfinite(p).all():
        return None, 0.0, "invalid_point"
    best = None
    for t in calibration["triangles"]:
        s, q = source[t], target[t]
        weights = np.linalg.solve(np.vstack((s.T, np.ones(3))), np.r_[p, 1])
        distance = float(max(0.0, -weights.min()))
        if best is None or distance < best[0]:
            best = (distance, weights @ q)
        if distance < 1e-8:
            return tuple(map(float, weights @ q)), 1.0, "mesh_interpolated"
    # Barycentric limit and pixel distance both bound extrapolation. It is never used for rendering.
    if best and extrapolate and best[0] <= 0.10:
        hull = cv2.convexHull(source.astype(np.float32))
        outside = abs(cv2.pointPolygonTest(hull, tuple(map(float, p)), True))
        if outside <= 0.025 * hypot(*np.ptp(source, axis=0)):
            return tuple(map(float, best[1])), 0.25, "mesh_extrapolated"
    return None, 0.0, "outside_mesh"


def project_calibration(calibration, points):
    if calibration.get("method") != METHOD:
        from src.app.geometry import project

        return project(calibration["matrix"], points)
    result = []
    for p in points:
        q, _, reason = map_point(calibration, p)
        if q is None:
            raise ValueError(reason)
        result.append(q)
    return np.asarray(result)


def overlap_report(project):
    """Overlap of marked floor patches, not proof that walls do not occlude the view."""
    patches = []
    for c in project.get("cameras", []):
        anchors = c.get("calibration", {}) or {}
        if not anchors:
            continue
        points = anchors.get("plan_points", c.get("plan_points", []))
        if len(points) >= 3:
            patches.append(
                (c["id"], cv2.convexHull(np.asarray(points, dtype=np.float32)))
            )
    result = []
    for i, (a, pa) in enumerate(patches):
        for b, pb in patches[i + 1 :]:
            area, polygon = cv2.intersectConvexConvex(pa, pb)
            if area > 1e-7:
                result.append(
                    {
                        "camera_a": a,
                        "camera_b": b,
                        "area_plan_units2": float(area),
                        "polygon": polygon.reshape(-1, 2).tolist(),
                        "source": "calibrated_floor_patches",
                    }
                )
    return result


def validation_report(calibration, checks):
    """Re-evaluate independent points after every edit; never trust stored errors."""
    errors = []
    unavailable = 0
    for check in checks:
        try:
            predicted = project_calibration(calibration, [check["video_point"]])[0]
            errors.append(
                float(np.linalg.norm(predicted - np.asarray(check["plan_point"])))
            )
        except (ValueError, KeyError):
            unavailable += 1
    return {
        "count": len(checks),
        "usable": len(errors),
        "outside_coverage": unavailable,
        "rmse_plan_units": float(np.sqrt(np.mean(np.square(errors))))
        if errors
        else None,
        "max_error_plan_units": max(errors) if errors else None,
    }
