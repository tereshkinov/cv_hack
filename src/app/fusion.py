"""Fuse simultaneous, calibrated floor observations without inventing missing positions."""

from math import exp, hypot
from statistics import fmean


def fuse_positions(samples, max_distance_m=2.0):
    """Input: x/y/confidence/camera_id. Output keeps each source for diagnostics."""
    if not samples:
        return None
    conflict = any(
        hypot(a["x_m"] - b["x_m"], a["y_m"] - b["y_m"]) > max_distance_m
        for i, a in enumerate(samples)
        for b in samples[i + 1 :]
    )
    if conflict:
        return {"status": "disagreement", "sources": samples, "x_m": None, "y_m": None}
    weights = [max(0.01, s["confidence"]) ** 2 for s in samples]
    total = sum(weights)
    return {
        "status": "fused" if len(samples) > 1 else "single",
        "sources": samples,
        "x_m": sum(s["x_m"] * w for s, w in zip(samples, weights, strict=True)) / total,
        "y_m": sum(s["y_m"] * w for s, w in zip(samples, weights, strict=True)) / total,
        "confidence": fmean(s["confidence"] for s in samples),
    }


def smooth_paths(paths, tau_s=0.25, max_gap_s=0.5):
    for points in paths.values():
        previous = None
        for point in points:
            if point["x_m"] is None:
                previous = None
                continue
            x, y = point["x_m"], point["y_m"]
            if (
                previous is not None
                and 0 < point["timestamp_s"] - previous["timestamp_s"] <= max_gap_s
            ):
                dt = point["timestamp_s"] - previous["timestamp_s"]
                alpha = 1 - exp(-dt / tau_s) if tau_s > 0 else 1
                x = previous["smooth_x_m"] + alpha * (x - previous["smooth_x_m"])
                y = previous["smooth_y_m"] + alpha * (y - previous["smooth_y_m"])
            point["smooth_x_m"], point["smooth_y_m"] = x, y
            previous = point
    return paths
