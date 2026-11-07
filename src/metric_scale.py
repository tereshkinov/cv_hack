"""Approximate physical scale without changing user calibration coordinates."""

from math import hypot, isfinite
from statistics import median


def scale_info(project):
    if project.get("coordinate_units") == "meters":
        return {
            "meters_per_unit": 1.0,
            "uncertainty_fraction": 0.0,
            "approximate": False,
            "source": "metric_plan",
        }
    value = project.get("metric_scale")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("metric_scale must be an object")
    scale = value.get("meters_per_unit")
    uncertainty = value.get("uncertainty_fraction", 0.15)
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not isfinite(scale)
        or not 0.001 <= scale <= 1000
    ):
        raise ValueError("Масштаб: от 0.001 до 1000 м/ед.")
    if (
        isinstance(uncertainty, bool)
        or not isinstance(uncertainty, (int, float))
        or not isfinite(uncertainty)
        or not 0 <= uncertainty <= 0.5
    ):
        raise ValueError("Неопределённость масштаба: от 0 до 0.5")
    return {
        "meters_per_unit": float(scale),
        "uncertainty_fraction": float(uncertainty),
        "approximate": bool(value.get("approximate", True)),
        "source": value.get("source", "explicit_scale"),
    }


def fit_scale(measurements):
    """Robust scalar fit. Endpoints are supplied, never inferred from room area."""
    ratios = []
    for m in measurements:
        length = hypot(m["a"][0] - m["b"][0], m["a"][1] - m["b"][1])
        if (
            length <= 0
            or not isfinite(length)
            or not isfinite(m["distance_m"])
            or m["distance_m"] <= 0
        ):
            raise ValueError("Ненулевая длина и положительный размер обязательны")
        ratios.append(m["distance_m"] / length)
    if not ratios:
        raise ValueError("Нужен минимум один измеренный отрезок")
    scale = median(ratios)
    residuals = [
        dict(
            m,
            predicted_m=hypot(m["a"][0] - m["b"][0], m["a"][1] - m["b"][1]) * scale,
            relative_error=scale / r - 1,
        )
        for m, r in zip(measurements, ratios)
    ]
    return {
        "meters_per_unit": scale,
        "approximate": True,
        "uncertainty_fraction": max(
            0.15, max(abs(m["relative_error"]) for m in residuals)
        ),
        "source": "user_office_dimensions_2026-11-07",
        "measurements": residuals,
    }


def annotate_speeds(paths, project):
    """Windowed floor displacement speed; no speed across missing observations."""
    info = scale_info(project)
    for points in paths.values():
        history = []
        for p in points:
            # Cached projection exports must not retain speeds from an older scale.
            for key in (
                "speed_m_s",
                "speed_uncertainty_m_s",
                "speed_approximate",
                "speed_window_s",
            ):
                p.pop(key, None)
            if (
                info is None
                or p.get("x_m") is None
                or p.get("status") == "disagreement"
            ):
                history = []
                continue
            if history and (
                p["timestamp_s"] - history[-1]["timestamp_s"] > 0.5
                or {s["camera_id"] for s in p.get("sources", [])}
                != {s["camera_id"] for s in history[-1].get("sources", [])}
            ):
                history = []
            history = [q for q in history if p["timestamp_s"] - q["timestamp_s"] <= 1.5]
            history.append(p)
            if len(history) < 3 or p["timestamp_s"] - history[0]["timestamp_s"] < 0.6:
                continue
            # Fit in the original plane; convert only derived distances/speeds.
            t0 = history[0]["timestamp_s"]
            times = [q["timestamp_s"] - t0 for q in history]
            tm = sum(times) / len(times)
            denom = sum((t - tm) ** 2 for t in times)
            velocities = []
            error = 0
            for axis in ("x_m", "y_m"):
                values = [q[axis] for q in history]
                mean = sum(values) / len(values)
                v = sum((t - tm) * (x - mean) for t, x in zip(times, values)) / denom
                velocities.append(v)
                error += sum(
                    (x - mean - v * (t - tm)) ** 2 for t, x in zip(times, values)
                ) / len(values)
            speed = hypot(*velocities) * info["meters_per_unit"]
            sigma = (error / denom) ** 0.5 * info["meters_per_unit"] + speed * info[
                "uncertainty_fraction"
            ]
            p.update(
                speed_m_s=speed,
                speed_uncertainty_m_s=sigma,
                speed_approximate=info["approximate"],
                speed_window_s=times[-1],
            )
