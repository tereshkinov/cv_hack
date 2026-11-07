"""Geometry-first boundary decisions with optional appearance confirmation.

The policy intentionally consumes already computed boundary evidence.  It does
not project detections, invent camera gates, or assign identities by itself.
Missing geometry is neutral and cannot be replaced by appearance; trustworthy
physical contradictions are hard refusals.
"""

from dataclasses import asdict, dataclass

from .scoring import clamp


@dataclass(frozen=True)
class GeometryDecision:
    geometry_value: float | None
    geometry_quality: float
    geometry_score: float | None
    appearance: float | None
    appearance_quality: float
    visual_adjustment: float
    score: float | None
    threshold: float
    accepted: bool
    hard_reject: bool
    reason: str

    def export(self):
        return {
            key: round(value, 6) if isinstance(value, float) else value
            for key, value in asdict(self).items()
        }


def decide_geometry(
    boundary,
    threshold,
    *,
    appearance=None,
    appearance_reliability=0.0,
    visual_confirmation_weight=0.0,
):
    """Make geometry primary and use appearance only near its decision boundary.

    Boundary ``value`` is normalized compatibility in [0, 1]. Reliability pulls
    weak evidence halfway towards the neutral value 0.5 instead of erasing it;
    this keeps a provisional but explicit route useful while exposing its low
    confidence. Appearance can move the result by at most the configured small
    weight and can never rescue missing or physically impossible geometry.
    """

    if boundary is None or not boundary.available or boundary.value is None:
        reason = boundary.reason if boundary is not None else "missing_geometry"
        return GeometryDecision(
            None,
            0.0,
            None,
            appearance,
            clamp(appearance_reliability),
            0.0,
            None,
            threshold,
            False,
            False,
            reason,
        )

    value = clamp(boundary.value)
    quality = clamp(boundary.reliability)
    reason = str(boundary.reason)
    observed_time = getattr(boundary, "time_delta_s", None)
    hard_reject = reason in {
        "same_camera_time_overlap",
        "cross_camera_mode_mismatch",
        "same_camera_mode_mismatch",
        "overlap_not_configured",
        "transition_not_configured",
    }
    hard_reject = hard_reject or (
        observed_time is not None
        and observed_time < -1e-6
        and "simultaneous_overlap" not in str(getattr(boundary, "mode", ""))
    )
    if (
        "spatial_contradiction" in reason
        or "route_time_contradiction" in reason
        or "physically_too_fast" in reason
    ) and quality >= 0.45:
        hard_reject = True

    # Even low-reliability explicit geometry retains half of its displacement
    # from neutral. The reliability controls confidence, not route existence.
    geometry_score = 0.5 + (value - 0.5) * (0.5 + 0.5 * quality)
    aq = clamp(appearance_reliability)
    visual_adjustment = 0.0
    if appearance is not None and visual_confirmation_weight > 0:
        visual_adjustment = (
            (clamp(appearance, -1.0, 1.0) - 0.5)
            * 2
            * clamp(visual_confirmation_weight, 0.0, 0.2)
            * aq
        )
    score = clamp(geometry_score + visual_adjustment, 0.0, 1.0)
    if hard_reject:
        score = clamp(min(score, threshold - 0.20), 0.0, 1.0)
    accepted = not hard_reject and score >= threshold
    decision_reason = (
        "physical_contradiction"
        if hard_reject
        else "geometry_supported"
        if accepted and not visual_adjustment
        else "geometry_visual_supported"
        if accepted
        else "insufficient_geometry"
    )
    return GeometryDecision(
        value,
        quality,
        geometry_score,
        appearance,
        aq,
        visual_adjustment,
        score,
        threshold,
        accepted,
        hard_reject,
        decision_reason,
    )
