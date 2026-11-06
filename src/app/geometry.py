"""Pixel-to-floor homography with explicit degeneracy checks."""

from itertools import combinations

import cv2
import numpy as np


def points4(value, name):
    points = np.asarray(value, dtype=np.float64)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError(f"{name}: нужны ровно четыре конечные пары координат")
    span = np.ptp(points, axis=0)
    if min(span) <= 1e-9:
        raise ValueError(f"{name}: точки не должны лежать на одной прямой")
    normalized = (points - points.mean(axis=0)) / max(span)
    for a, b, c in combinations(normalized, 3):
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) < 1e-5:
            raise ValueError(
                f"{name}: три точки совпадают или почти лежат на одной прямой"
            )
    return points


def project(matrix, points):
    h = np.asarray(matrix, dtype=np.float64)
    p = np.asarray(points, dtype=np.float64)
    if (
        h.shape != (3, 3)
        or p.ndim != 2
        or p.shape[1] != 2
        or not np.isfinite(h).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Некорректная матрица или точки")
    q = np.column_stack([p, np.ones(len(p))]) @ h.T
    if (np.abs(q[:, 2]) < 1e-10).any():
        raise ValueError("Точка близка к горизонту проекции; координаты ненадёжны")
    result = q[:, :2] / q[:, 2, None]
    if not np.isfinite(result).all():
        raise ValueError("Неконечные координаты проекции")
    return result


def calibrate(video_points, plan_points):
    source = points4(video_points, "Кадр")
    target = points4(plan_points, "Схема")
    matrix, _ = cv2.findHomography(source, target, method=0)
    if (
        matrix is None
        or not np.isfinite(matrix).all()
        or abs(np.linalg.det(matrix)) < 1e-14
    ):
        raise ValueError("Не удалось вычислить устойчивую привязку")
    # A horizon through the marked floor patch usually signals crossed/misnumbered pairs.
    denominators = np.column_stack([source, np.ones(4)]) @ matrix[2]
    if min(denominators) * max(denominators) <= 0:
        raise ValueError(
            "Пары точек задают пересекающую горизонт область. Проверьте нумерацию"
        )
    fitted = project(matrix, source)
    error = float(np.max(np.linalg.norm(fitted - target, axis=1)))
    if error > 1e-5 * max(1, float(np.ptp(target, axis=0).max())):
        raise ValueError("Привязка нестабильна; выберите более разнесённые точки")
    return {
        "matrix": matrix.tolist(),
        "max_anchor_error_m": error,
        "method": "four_point_floor_homography",
        "video_points": source.tolist(),
        "plan_points": target.tolist(),
    }
