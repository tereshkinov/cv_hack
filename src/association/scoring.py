"""Experimental reliability-weighted evidence. Scores are not probabilities."""

from dataclasses import asdict, dataclass
from statistics import fmean


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def appearance_quality(observation, camera=None):
    """Available metadata only: confidence/size/cropping, not measured lighting."""
    x1, y1, x2, y2 = observation.bbox_xyxy
    size = min(1.0, (y2 - y1) / 160) * min(1.0, (x2 - x1) / 48)
    cropped = bool(
        camera
        and (
            (camera.height and y2 >= 0.985 * camera.height)
            or y1 <= 1
            or x1 <= 1
            or (camera.width and x2 >= 0.985 * camera.width)
        )
    )
    return clamp(
        observation.detection_confidence * (0.5 + 0.5 * size) * (0.7 if cropped else 1),
        0.15,
        1,
    )


def geometry_quality(floor, observations):
    """Adapter to agent 2's unchanged signed evidence API; heuristic reliability."""
    if not floor or not observations:
        return 0.0
    samples = observations[:: max(1, len(observations) // 16)]
    weights = []
    for observation in samples:
        if floor.point(observation) is None:
            weights.append(0.0)
            continue
        anchors = floor.calibrations[observation.camera_id].get("video_points", [])
        x1, _, x2, y = observation.bbox_xyxy
        x = (x1 + x2) / 2
        # Extrapolation outside anchor bounds is weaker support, never proof of absence.
        inside = not anchors or (
            min(a[0] for a in anchors) <= x <= max(a[0] for a in anchors)
            and min(a[1] for a in anchors) <= y <= max(a[1] for a in anchors)
        )
        weights.append(0.85 if inside else 0.45)
    return fmean(weights)


@dataclass(frozen=True)
class Evidence:
    appearance: float
    appearance_quality: float
    geometry: float | None
    geometry_quality: float
    topology: float
    threshold: float
    score: float
    accepted: bool
    trusted: bool
    reason: str

    def export(self):
        return {
            k: round(v, 6) if isinstance(v, float) else v
            for k, v in asdict(self).items()
        }


def decide(
    appearance,
    appearance_reliability,
    geometry,
    geometry_reliability,
    topology,
    threshold,
):
    aq = 0.5 + 0.5 * clamp(appearance_reliability)
    gq = clamp(geometry_reliability) if geometry is not None else 0.0
    # Reduce duplicate support from adjacency when floor geometry is available.
    graph = topology * (1 - 0.5 * gq)
    spatial = (
        (geometry or 0) * gq * (1.5 if geometry is not None and geometry > 0 else 1)
    )
    score = threshold + (appearance - threshold) * aq + graph + spatial
    reason = "supported" if score >= threshold else "insufficient_support"
    if appearance < 0.45:
        score = min(score, threshold - 0.10)
        reason = "appearance_too_weak"
    elif geometry is not None and geometry < -0.10 and gq >= 0.6:
        # Strong trustworthy contradiction cannot be cancelled by matching clothes.
        score = min(score, threshold - 0.04)
        reason = "spatial_contradiction"
    score = clamp(score, -1, 1)
    accepted = score >= threshold
    trusted = (
        accepted
        and score - threshold >= 0.06
        and appearance >= threshold
        and appearance_reliability >= 0.55
        and (geometry is None or geometry >= -0.02)
    )
    return Evidence(
        appearance,
        clamp(appearance_reliability),
        geometry,
        gq,
        graph,
        threshold,
        score,
        accepted,
        trusted,
        reason,
    )
