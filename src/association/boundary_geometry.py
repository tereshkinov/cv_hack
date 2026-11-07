"""Geometry evidence for linking the end and start of track fragments.

The provider deliberately does not assign global IDs.  It returns a bounded
compatibility value plus diagnostics that an association policy can combine
with appearance and lifecycle constraints.

For ``same_camera_recovery`` it works without floor calibration: motion and
box scale are compared in normalized image coordinates.  Cross-camera modes
remain conservative and require floor projections from :class:`FloorMatcher`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, hypot, isfinite, log, sqrt
from statistics import median
from typing import Literal

from .geolocation import FloorMatcher

BoundaryMode = Literal[
    "same_camera_recovery",
    "simultaneous_overlap",
    "sequential_transition",
]


@dataclass(frozen=True, slots=True)
class BoundaryGeometryEvidence:
    """Spatial compatibility on a documented ``[0, 1]`` scale.

    ``value`` is ``exp(-0.5 * discrepancy**2)``.  It has not been multiplied
    by ``reliability``; the integration owner must apply reliability exactly
    once.  Missing geometry is represented by ``available=False`` and is
    neutral, not evidence that fragments belong to different people.
    """

    available: bool
    value: float | None
    reliability: float
    support_count: int
    reason: str
    mode: BoundaryMode
    discrepancy: float | None
    coordinate_units: str
    time_delta_s: float | None
    expected_region: dict | None
    details: dict

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ImageSample:
    timestamp_s: float
    x: float
    y: float
    height: float
    quality: float
    cropped_feet: bool


@dataclass(frozen=True, slots=True)
class _ImageState:
    time_s: float
    x: float
    y: float
    height: float
    velocity: tuple[float, float] | None
    jitter: float
    reliability: float
    support_count: int
    time_span_s: float
    cropped_fraction: float


class BoundaryGeometryMatcher:
    """Compare fragment boundaries without making an ID decision."""

    def __init__(
        self,
        cameras,
        floor_matcher: FloorMatcher | None = None,
        *,
        window_s: float = 1.5,
        max_same_camera_gap_s: float = 4.0,
        independent_step_s: float = 0.12,
    ):
        if not isfinite(window_s) or window_s <= 0:
            raise ValueError("window_s must be positive and finite")
        if not isfinite(max_same_camera_gap_s) or max_same_camera_gap_s <= 0:
            raise ValueError("max_same_camera_gap_s must be positive and finite")
        if not isfinite(independent_step_s) or independent_step_s <= 0:
            raise ValueError("independent_step_s must be positive and finite")
        self.cameras = {camera.camera_id: camera for camera in cameras}
        self.floor = floor_matcher
        self.window_s = window_s
        self.max_same_camera_gap_s = max_same_camera_gap_s
        self.independent_step_s = independent_step_s

    def compare(
        self,
        left_fragment,
        right_fragment,
        mode: BoundaryMode,
    ) -> BoundaryGeometryEvidence:
        if mode not in {
            "same_camera_recovery",
            "simultaneous_overlap",
            "sequential_transition",
        }:
            raise ValueError(f"unsupported boundary mode: {mode}")
        left = self._observations(left_fragment)
        right = self._observations(right_fragment)
        if not left or not right:
            return self._missing(mode, "empty_fragment")

        left_cameras = {row.camera_id for row in left}
        right_cameras = {row.camera_id for row in right}
        if len(left_cameras) != 1 or len(right_cameras) != 1:
            return self._missing(mode, "fragment_has_multiple_cameras")
        left_camera = next(iter(left_cameras))
        right_camera = next(iter(right_cameras))

        topology_reason = self._topology_reason(left_camera, right_camera, mode)
        if topology_reason is not None:
            return self._missing(mode, topology_reason)

        left = self._tail(left)
        right = self._head(right)
        if mode == "same_camera_recovery":
            return self._same_camera(left, right, mode)
        return self._floor_boundary(left, right, mode)

    def _same_camera(self, left, right, mode) -> BoundaryGeometryEvidence:
        end_s = left[-1].timestamp_s
        start_s = right[0].timestamp_s
        gap_s = start_s - end_s
        if gap_s < -1e-9:
            return self._contradiction(
                mode,
                "same_camera_time_overlap",
                time_delta_s=gap_s,
                support_count=1,
            )
        if gap_s > self.max_same_camera_gap_s:
            return self._missing(mode, "same_camera_gap_too_long", gap_s)

        camera = self.cameras[left[0].camera_id]
        if not camera.width or not camera.height:
            return self._missing(mode, "missing_frame_dimensions", gap_s)
        left_samples = self._image_samples(left, camera.width, camera.height)
        right_samples = self._image_samples(right, camera.width, camera.height)
        if not left_samples or not right_samples:
            return self._missing(mode, "invalid_boundary_boxes", gap_s)

        left_state = self._image_state(left_samples, boundary="tail")
        right_state = self._image_state(right_samples, boundary="head")
        forward = self._directional_discrepancy(left_state, right_state, gap_s)
        reverse = self._directional_discrepancy(right_state, left_state, -gap_s)
        directional = []
        if left_state.velocity is not None:
            directional.append(forward)
        if right_state.velocity is not None:
            directional.append(reverse)

        static_distance = hypot(
            right_state.x - left_state.x,
            right_state.y - left_state.y,
        )
        position_radius = self._position_radius(left_state, right_state, gap_s)
        position_only = static_distance / position_radius
        scale_discrepancy = (
            abs(log(max(right_state.height, 1e-6) / max(left_state.height, 1e-6)))
            / 0.45
        )

        if directional:
            motion_discrepancy = median(directional)
            discrepancy = sqrt(motion_discrepancy**2 + 0.30 * scale_discrepancy**2)
            reason = (
                "same_camera_bidirectional"
                if len(directional) == 2
                else "same_camera_one_sided_motion"
            )
            direction_factor = 1.0 if len(directional) == 2 else 0.82
        else:
            discrepancy = sqrt(position_only**2 + 0.30 * scale_discrepancy**2)
            reason = "same_camera_position_only"
            direction_factor = 0.50

        support_count = min(left_state.support_count, right_state.support_count)
        reliability = min(left_state.reliability, right_state.reliability)
        reliability *= direction_factor * exp(-gap_s / 5.0)
        expected_x, expected_y = self._predict(left_state, max(0.0, gap_s))
        value = exp(-0.5 * discrepancy**2)
        return BoundaryGeometryEvidence(
            True,
            value,
            max(0.0, min(1.0, reliability)),
            support_count,
            reason,
            mode,
            discrepancy,
            "normalized_image",
            gap_s,
            {
                "center": [expected_x, expected_y],
                "radius": position_radius,
                "scale_height": left_state.height,
            },
            {
                "value_scale": "exp(-0.5 * discrepancy^2), reliability not applied",
                "left_support": left_state.support_count,
                "right_support": right_state.support_count,
                "left_time_span_s": left_state.time_span_s,
                "right_time_span_s": right_state.time_span_s,
                "left_cropped_fraction": left_state.cropped_fraction,
                "right_cropped_fraction": right_state.cropped_fraction,
                "forward_discrepancy": forward,
                "reverse_discrepancy": reverse,
                "position_only_discrepancy": position_only,
                "scale_discrepancy": scale_discrepancy,
            },
        )

    def _floor_boundary(self, left, right, mode) -> BoundaryGeometryEvidence:
        if self.floor is None:
            return self._missing(mode, "missing_floor_matcher")
        spatial = self.floor.between_evidence(left, right)
        if not spatial.available or spatial.discrepancy is None:
            return self._missing(
                mode,
                f"floor:{spatial.reason}",
                spatial.time_delta_s,
                reliability=spatial.reliability,
            )
        support_count = min(
            self._projected_support(left),
            self._projected_support(right),
        )
        return BoundaryGeometryEvidence(
            True,
            exp(-0.5 * spatial.discrepancy**2),
            spatial.reliability,
            support_count,
            f"floor:{spatial.reason}",
            mode,
            spatial.discrepancy,
            self.floor.coordinate_units,
            spatial.time_delta_s,
            spatial.region.as_dict() if spatial.region else None,
            {
                "value_scale": "exp(-0.5 * discrepancy^2), reliability not applied",
                "legacy_adjustment": spatial.adjustment,
            },
        )

    def _topology_reason(self, left_camera, right_camera, mode):
        left = self.cameras.get(left_camera)
        right = self.cameras.get(right_camera)
        if left is None or right is None:
            return "unknown_camera"
        if mode == "same_camera_recovery":
            return None if left_camera == right_camera else "same_camera_mode_mismatch"
        if left_camera == right_camera:
            return "cross_camera_mode_mismatch"
        if mode == "simultaneous_overlap":
            linked = (
                right_camera in left.overlap_with or left_camera in right.overlap_with
            )
            return None if linked else "overlap_not_configured"
        return (
            None if right_camera in left.transition_to else "transition_not_configured"
        )

    def _tail(self, observations):
        end_s = observations[-1].timestamp_s
        return tuple(
            row for row in observations if end_s - row.timestamp_s <= self.window_s
        )

    def _head(self, observations):
        start_s = observations[0].timestamp_s
        return tuple(
            row for row in observations if row.timestamp_s - start_s <= self.window_s
        )

    def _observations(self, fragment):
        if isinstance(fragment, dict):
            rows = fragment.get("observations", ())
        else:
            rows = getattr(fragment, "observations", fragment)
        return tuple(sorted(rows, key=lambda row: (row.timestamp_s, row.frame_index)))

    def _image_samples(self, observations, width, height):
        raw = []
        for row in observations:
            x1, y1, x2, y2 = row.bbox_xyxy
            box_height = y2 - y1
            if box_height <= 0 or x2 <= x1:
                continue
            cropped = y2 >= height * 0.985
            quality = max(0.05, min(1.0, row.detection_confidence))
            if cropped:
                quality *= 0.35
            raw.append(
                _ImageSample(
                    row.timestamp_s,
                    ((x1 + x2) / 2) / width,
                    y2 / height,
                    box_height / height,
                    quality,
                    cropped,
                )
            )
        return self._independent_samples(raw)

    def _independent_samples(self, samples):
        kept = []
        for sample in sorted(samples, key=lambda item: item.timestamp_s):
            if (
                not kept
                or sample.timestamp_s - kept[-1].timestamp_s >= self.independent_step_s
            ):
                kept.append(sample)
            elif sample.quality > kept[-1].quality:
                kept[-1] = sample
        return tuple(kept)

    def _image_state(self, samples, *, boundary):
        edge_time = (
            samples[-1].timestamp_s if boundary == "tail" else samples[0].timestamp_s
        )
        edge = [
            sample for sample in samples if abs(sample.timestamp_s - edge_time) <= 0.35
        ]
        x = median(sample.x for sample in edge)
        y = median(sample.y for sample in edge)
        height = median(sample.height for sample in edge)
        jitter = median(hypot(sample.x - x, sample.y - y) for sample in edge)
        velocity = self._velocity(samples)
        span = samples[-1].timestamp_s - samples[0].timestamp_s
        quality = median(sample.quality for sample in samples)
        support_factor = min(1.0, len(samples) / 3)
        span_factor = min(1.0, 0.45 + span / 0.8)
        cropped_fraction = sum(sample.cropped_feet for sample in samples) / len(samples)
        reliability = quality * support_factor * span_factor
        return _ImageState(
            edge_time,
            x,
            y,
            height,
            velocity,
            jitter,
            reliability,
            len(samples),
            span,
            cropped_fraction,
        )

    def _velocity(self, samples):
        if len(samples) < 2 or samples[-1].timestamp_s - samples[0].timestamp_s < 0.2:
            return None
        displacement = hypot(
            samples[-1].x - samples[0].x,
            samples[-1].y - samples[0].y,
        )
        # A standing person has no reliable direction. Treat a nearly static
        # boundary as position-only evidence; otherwise a chair or screen false
        # positive can look like an exceptionally clean motion continuation.
        stationary_limit = max(
            0.012,
            median(sample.height for sample in samples) * 0.05,
        )
        if displacement <= stationary_limit:
            return None
        slopes_x = []
        slopes_y = []
        for index, first in enumerate(samples):
            for second in samples[index + 1 :]:
                dt = second.timestamp_s - first.timestamp_s
                if dt < 0.18:
                    continue
                slopes_x.append((second.x - first.x) / dt)
                slopes_y.append((second.y - first.y) / dt)
        if not slopes_x:
            return None
        vx, vy = median(slopes_x), median(slopes_y)
        speed = hypot(vx, vy)
        if speed > 1.5:
            vx *= 1.5 / speed
            vy *= 1.5 / speed
        return (vx, vy)

    def _directional_discrepancy(self, source, target, dt):
        predicted_x, predicted_y = self._predict(source, dt)
        radius = self._position_radius(source, target, abs(dt))
        return hypot(target.x - predicted_x, target.y - predicted_y) / radius

    @staticmethod
    def _predict(state, dt):
        if state.velocity is None:
            return state.x, state.y
        return state.x + state.velocity[0] * dt, state.y + state.velocity[1] * dt

    @staticmethod
    def _position_radius(left, right, gap_s):
        body_scale = max(left.height, right.height) * 0.30
        jitter = max(left.jitter, right.jitter) * 2.5
        speed = hypot(*(left.velocity or (0.0, 0.0)))
        gap_expansion = gap_s * (0.035 + min(speed, 0.8) * 0.10)
        return max(0.025, body_scale, jitter) + gap_expansion

    def _projected_support(self, observations):
        times = [
            row.timestamp_s
            for row in observations
            if self.floor.projection(row).available
        ]
        support = 0
        last = None
        for time_s in sorted(times):
            if last is None or time_s - last >= self.independent_step_s:
                support += 1
                last = time_s
        return support

    @staticmethod
    def _missing(
        mode,
        reason,
        time_delta_s=None,
        *,
        reliability=0.0,
    ):
        return BoundaryGeometryEvidence(
            False,
            None,
            reliability,
            0,
            reason,
            mode,
            None,
            "unavailable",
            time_delta_s,
            None,
            {},
        )

    @staticmethod
    def _contradiction(mode, reason, *, time_delta_s, support_count):
        return BoundaryGeometryEvidence(
            True,
            0.0,
            1.0,
            support_count,
            reason,
            mode,
            10.0,
            "normalized_image",
            time_delta_s,
            None,
            {"hard_constraint": True},
        )
