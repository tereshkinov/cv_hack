"""Experimental reliability-weighted evidence. Scores are not probabilities."""

from dataclasses import asdict, dataclass
from math import exp


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


def spatial_input(evidence):
    """Agent 2 already weights projection/sync/fade; apply only missing motion weight.

    Simultaneous comparisons already include their full pair reliability.
    For predictions, region reliability also contains motion confidence, which
    the legacy signed adjustment does not include. Remove the already applied
    time fade before using that remaining factor.
    """
    if evidence is None:
        return None, 0.0
    adjustment = evidence.adjustment
    if adjustment is not None and evidence.region is not None:
        remaining = min(
            1.0, evidence.region.reliability / max(exp(-evidence.region.dt_s / 4), 1e-9)
        )
        adjustment *= remaining
    return adjustment, evidence.reliability if adjustment is not None else 0.0


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
    geometry_preweighted: bool = False
    spatial_details: dict | None = None

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
    *,
    geometry_preweighted=False,
    spatial_details=None,
):
    aq = 0.5 + 0.5 * clamp(appearance_reliability)
    gq = clamp(geometry_reliability) if geometry is not None else 0.0
    # Reduce duplicate support from adjacency when floor geometry is available.
    graph = topology * (1 - 0.5 * gq)
    spatial = (
        (geometry or 0)
        * (1 if geometry_preweighted else gq)
        * (1.5 if geometry is not None and geometry > 0 else 1)
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
        geometry_preweighted,
        spatial_details,
    )
