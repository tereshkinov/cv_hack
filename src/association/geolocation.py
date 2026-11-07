"""Conservative floor-space evidence with explicit quality and refusal reasons."""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from math import exp, hypot, isfinite, sqrt
from statistics import median

from .kalman_motion import fit_floor_state


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    camera_id: str
    point: tuple[float, float] | None
    reliability: float
    reason: str
    coordinate_units: str

    @property
    def available(self) -> bool:
        return self.point is not None

    def as_dict(self) -> dict:
        return asdict(self) | {"available": self.available}


@dataclass(frozen=True, slots=True)
class MotionEstimate:
    position: tuple[float, float] | None
    velocity: tuple[float, float] | None
    speed_coordinate_units_s: float | None
    stationary: bool
    reliability: float
    sample_count: int
    time_span_s: float
    reason: str
    coordinate_units: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExpectedRegion:
    center: tuple[float, float]
    direction: tuple[float, float] | None
    along_radius: float
    across_radius: float
    dt_s: float
    reliability: float
    coordinate_units: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpatialEvidence:
    adjustment: float | None
    discrepancy: float | None
    reliability: float
    reason: str
    time_delta_s: float | None = None
    region: ExpectedRegion | None = None

    @property
    def available(self) -> bool:
        return self.adjustment is not None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["available"] = self.available
        return data


class FloorMatcher:
    """Project observations to the floor and expose soft, diagnostic evidence.

    Legacy ``point()``, ``predict()``, ``online()`` and ``between()`` keep returning
    a point, a numeric bonus, or ``None``. Their structured counterparts provide
    reliability, discrepancy, the expected region, and a refusal reason for tuning.
    """

    def __init__(self, project, cameras):
        self.project = project
        self.cameras = {camera.camera_id: camera for camera in cameras}
        self.project_cameras = {
            camera["id"]: camera for camera in project.get("cameras", [])
        }
        self.calibrations = {
            camera_id: camera["calibration"]
            for camera_id, camera in self.project_cameras.items()
            if camera.get("calibration")
        }
        room = project["room"]
        self.width = _positive_number(room["width_m"], "room.width_m")
        self.height = _positive_number(room["height_m"], "room.height_m")
        self.coordinate_units = project.get("coordinate_units", "meters")
        # Relative plans have no physical speed units. Normalize by plan diagonal.
        self.scale = (
            1.0
            if self.coordinate_units == "meters"
            else max(hypot(self.width, self.height) / 25, 1e-6)
        )
        processing = project.get("processing", {})
        requested_radius = processing.get("fusion_distance_m", 0.8 * self.scale)
        self.radius = max(
            0.15 * self.scale,
            min(
                _positive_number(requested_radius, "fusion_distance_m"),
                1.2 * self.scale,
            ),
        )
        self.speed = 2.0 * self.scale
        self.prediction_model = project.get("floor_prediction", "heuristic")
        if self.prediction_model not in ("heuristic", "kalman"):
            raise ValueError("floor_prediction must be heuristic or kalman")
        self._measurement_var = (0.55 * self.radius) ** 2
        self._acceleration_var = (0.65 * self.scale) ** 2
        self.sync_uncertainty_s = _bounded_number(
            processing.get("sync_uncertainty_s", 0.25),
            "sync_uncertainty_s",
            minimum=0.0,
            maximum=2.0,
        )
        self.sync_uncertainty_configured = "sync_uncertainty_s" in processing
        self.sync_verified = bool(processing.get("sync_verified", False))
        from src.floor_mesh import validation_report

        self.calibration_checks = {
            cid: validation_report(
                cal, self.project_cameras[cid].get("validation_points", [])
            )
            for cid, cal in self.calibrations.items()
        }
        self.cache: dict[tuple[str, int, str], tuple[float, float] | None] = {}
        self._projection_details: dict[tuple[str, int, str], ProjectionResult] = {}
        self.stats = Counter()

    def project_video_point(
        self, camera_id: str, video_point: tuple[float, float] | list[float]
    ) -> ProjectionResult:
        """Project an independent floor point without changing user calibration."""
        if camera_id not in self.cameras:
            return self._projection(camera_id, None, 0.0, "unknown_camera")
        calibration = self.calibrations.get(camera_id)
        if calibration is None:
            return self._projection(camera_id, None, 0.0, "missing_calibration")
        if not isinstance(video_point, (tuple, list)) or len(video_point) != 2:
            return self._projection(camera_id, None, 0.0, "invalid_video_point")
        x, y = video_point
        if not _finite_pair((x, y)):
            return self._projection(camera_id, None, 0.0, "invalid_video_point")
        if calibration.get("method") == "piecewise_affine_floor_mesh":
            from src.floor_mesh import map_point

            point, quality, reason = map_point(calibration, (x, y), extrapolate=True)
            if point is not None and not (
                0 <= point[0] <= self.width and 0 <= point[1] <= self.height
            ):
                point, quality, reason = None, 0.0, "outside_plan"
            return self._projection(camera_id, point, quality, reason)
        matrix = _matrix3(calibration.get("matrix"))
        if matrix is None:
            return self._projection(camera_id, None, 0.0, "invalid_matrix")

        denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
        anchors = calibration.get("video_points", [])
        anchor_denominators = [
            matrix[2][0] * a + matrix[2][1] * b + matrix[2][2]
            for point in anchors
            if isinstance(point, (tuple, list))
            and len(point) == 2
            and _finite_pair(point)
            for a, b in [point]
        ]
        anchor_denominator = (
            median(anchor_denominators) if anchor_denominators else denominator
        )
        reference = max(abs(anchor_denominator), 1e-9)
        if abs(denominator) <= reference * 1e-6:
            return self._projection(camera_id, None, 0.0, "near_horizon")
        if denominator * anchor_denominator <= 0:
            return self._projection(camera_id, None, 0.0, "opposite_horizon_side")

        px = (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator
        py = (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator
        if not _finite_pair((px, py)):
            return self._projection(camera_id, None, 0.0, "non_finite_projection")
        if not 0 <= px <= self.width or not 0 <= py <= self.height:
            return self._projection(camera_id, None, 0.0, "outside_plan")

        horizon_quality = min(1.0, abs(denominator) / max(reference * 0.2, 1e-9))
        edge_distance = min(px, py, self.width - px, self.height - py)
        edge_quality = min(1.0, 0.75 + edge_distance / max(self.radius * 4, 1e-9))
        return self._projection(
            camera_id,
            (px, py),
            max(0.05, min(horizon_quality, edge_quality)),
            "projected",
        )

    def projection(self, observation) -> ProjectionResult:
        key = observation.reference
        if key in self._projection_details:
            return self._projection_details[key]
        camera = self.cameras.get(observation.camera_id)
        if camera is None:
            result = self._projection(
                observation.camera_id, None, 0.0, "unknown_camera"
            )
        elif observation.camera_id not in self.calibrations:
            result = self._projection(
                observation.camera_id, None, 0.0, "missing_calibration"
            )
        else:
            x1, _, x2, feet_y = observation.bbox_xyxy
            if camera.height is not None and feet_y >= camera.height * 0.985:
                result = self._projection(
                    observation.camera_id, None, 0.0, "cropped_feet"
                )
            else:
                result = self.project_video_point(
                    observation.camera_id, ((x1 + x2) / 2, feet_y)
                )
                if result.available and camera.height:
                    clearance = max(0.0, (camera.height - feet_y) / camera.height)
                    crop_quality = min(1.0, clearance / 0.05)
                    result = replace(
                        result,
                        reliability=max(
                            0.05, min(result.reliability, 0.55 + 0.45 * crop_quality)
                        ),
                    )
        self._projection_details[key] = result
        self.cache[key] = result.point
        self.stats["projection_total"] += 1
        if result.available:
            self.stats["projected"] += 1
        else:
            self.stats["unavailable"] += 1
            self.stats[f"rejected:{result.reason}"] += 1
        return result

    def point(self, observation):
        """Backward-compatible point or ``None``."""
        return self.projection(observation).point

    def samples(self, observations):
        return [
            (observation.timestamp_s, *result.point)
            for observation in observations
            if (result := self.projection(observation)).point is not None
        ]

    def motion(self, history: Iterable[tuple[float, float, float]]) -> MotionEstimate:
        points = self._collapse_history(history)
        if not points:
            return MotionEstimate(
                None,
                None,
                None,
                False,
                0.0,
                0,
                0.0,
                "no_history",
                self.coordinate_units,
            )
        end = points[-1][0]
        tail = [point for point in points if end - point[0] <= 0.35]
        position = (median(p[1] for p in tail), median(p[2] for p in tail))
        span = end - points[0][0]
        if len(points) < 2 or span < 0.2:
            return MotionEstimate(
                position,
                None,
                None,
                False,
                0.25,
                len(points),
                span,
                "insufficient_history",
                self.coordinate_units,
            )

        pause_tail = [point for point in points if end - point[0] <= 0.7]
        pause_span = pause_tail[-1][0] - pause_tail[0][0]
        pause_spread = max(
            (hypot(p[1] - position[0], p[2] - position[1]) for p in pause_tail),
            default=0.0,
        )
        stationary_radius = self.radius * 0.35
        if pause_span >= 0.35 and pause_spread <= stationary_radius:
            return MotionEstimate(
                position,
                None,
                0.0,
                True,
                min(1.0, 0.55 + pause_span / 2),
                len(points),
                span,
                "stationary",
                self.coordinate_units,
            )

        older = [point for point in points if 0.4 <= end - point[0] <= 1.6]
        if not older:
            older = [points[0]] if span >= 0.2 else []
        if not older:
            return MotionEstimate(
                position,
                None,
                None,
                False,
                0.25,
                len(points),
                span,
                "insufficient_history",
                self.coordinate_units,
            )
        older_time = median(p[0] for p in older)
        elapsed = end - older_time
        if elapsed <= 1e-6:
            return MotionEstimate(
                position,
                None,
                None,
                False,
                0.0,
                len(points),
                span,
                "simultaneous_only",
                self.coordinate_units,
            )
        older_position = (median(p[1] for p in older), median(p[2] for p in older))
        dx, dy = position[0] - older_position[0], position[1] - older_position[1]
        displacement = hypot(dx, dy)
        jitter = median(hypot(p[1] - position[0], p[2] - position[1]) for p in tail)
        if displacement <= max(stationary_radius, jitter * 2.5):
            return MotionEstimate(
                position,
                None,
                0.0,
                True,
                min(0.85, 0.45 + span / 3),
                len(points),
                span,
                "stationary",
                self.coordinate_units,
            )
        vx, vy = dx / elapsed, dy / elapsed
        speed = hypot(vx, vy)
        clipped = speed > self.speed
        if clipped:
            vx *= self.speed / speed
            vy *= self.speed / speed
            speed = self.speed
        signal = displacement / max(jitter * 2.5, stationary_radius, 1e-9)
        reliability = min(1.0, 0.25 + span / 2, 0.35 + signal / 3)
        if clipped:
            reliability *= 0.6
        return MotionEstimate(
            position,
            (vx, vy),
            speed,
            False,
            reliability,
            len(points),
            span,
            "moving_speed_clipped" if clipped else "moving",
            self.coordinate_units,
        )

    def _fit_kalman(self, points):
        return fit_floor_state(
            points, self._measurement_var, self._acceleration_var, self.speed**2
        )

    def route_speed(self, observations, meters_per_unit):
        """Return a departure speed and confidence for the existing route model."""
        if self.prediction_model == "heuristic":
            if self.coordinate_units != "meters":
                return None, 0.0
            motion = self.motion(self.samples(observations))
            return motion.speed_coordinate_units_s, motion.reliability

        samples = self._detailed_samples(observations)
        state = self._fit_kalman(self._collapse_history(samples))
        if state is None or observations[-1].timestamp_s - state.time_s > 0.25:
            return None, 0.0
        speed = hypot(*state.velocity)
        if speed <= sqrt(state.velocity_var):
            return None, 0.0
        projection_quality = median(sample[3] for sample in samples)
        return speed * meters_per_unit, state.velocity_reliability * projection_quality

    def expected_region(
        self, history, time, *, time_uncertainty_s=0.0
    ) -> tuple[ExpectedRegion | None, str]:
        points = self._collapse_history(history)
        if not points:
            return None, "no_history"
        end = points[-1][0]
        dt = time - end
        if dt < -max(self.sync_uncertainty_s, 1e-6):
            return None, "target_before_history"
        dt = max(0.0, dt)
        if dt > 4:
            return None, "long_gap"
        if self.prediction_model == "kalman":
            state = self._fit_kalman(points)
            if state is not None:
                forecast = state.predict(end + dt, self._acceleration_var)
                speed = hypot(*forecast.velocity)
                direction = (
                    tuple(v / speed for v in forecast.velocity) if speed > 0 else None
                )
                # S includes the uncertainty of the new observation and of timing.
                across_var = forecast.position_var + self._measurement_var
                along_var = across_var + (speed * time_uncertainty_s) ** 2
                return (
                    ExpectedRegion(
                        forecast.position,
                        direction,
                        sqrt(along_var),
                        sqrt(across_var),
                        dt,
                        exp(-dt / 4),
                        self.coordinate_units,
                    ),
                    "kalman_prediction",
                )
            # Position-only evidence for a single sighting or a short history.
            radius = self.radius + self.speed * (dt * 0.35 + time_uncertainty_s)
            return (
                ExpectedRegion(
                    points[-1][1:],
                    None,
                    radius,
                    radius,
                    dt,
                    0.25 * exp(-dt / 4),
                    self.coordinate_units,
                ),
                "kalman_insufficient_history",
            )
        motion = self.motion(points)
        if motion.position is None:
            return None, motion.reason
        direction = None
        center = motion.position
        speed = 0.0
        if motion.velocity is not None and motion.speed_coordinate_units_s:
            speed = motion.speed_coordinate_units_s
            direction = (
                motion.velocity[0] / speed,
                motion.velocity[1] / speed,
            )
            ahead = min(dt, 2.0)
            center = (
                center[0] + motion.velocity[0] * ahead,
                center[1] + motion.velocity[1] * ahead,
            )
        uncertainty = self.speed * dt * 0.35
        across = self.radius + uncertainty
        along = across + (speed * dt * 0.8 if direction else 0.0)
        reliability = max(0.15, motion.reliability) * exp(-dt / 4)
        return (
            ExpectedRegion(
                center,
                direction,
                max(along, self.radius),
                max(across, self.radius),
                dt,
                reliability,
                self.coordinate_units,
            ),
            motion.reason,
        )

    def evaluate(
        self,
        history,
        time,
        point,
        *,
        time_uncertainty_s: float | None = None,
    ) -> SpatialEvidence:
        if not _finite_pair(point):
            return SpatialEvidence(None, None, 0.0, "invalid_target_point")
        time_uncertainty = (
            0.0 if time_uncertainty_s is None else max(0.0, time_uncertainty_s)
        )
        use_kalman = self.prediction_model == "kalman"
        region, motion_reason = self.expected_region(
            history, time, time_uncertainty_s=time_uncertainty
        )
        if use_kalman:
            self.stats[f"floor_prediction:{motion_reason}"] += 1
        if region is None:
            return SpatialEvidence(None, None, 0.0, motion_reason)
        expansion = 0.0 if use_kalman else self.speed * time_uncertainty
        along_radius = region.along_radius + expansion
        across_radius = region.across_radius + expansion
        dx, dy = point[0] - region.center[0], point[1] - region.center[1]
        if region.direction is not None:
            ux, uy = region.direction
            along = dx * ux + dy * uy
            across = dx * -uy + dy * ux
        else:
            along, across = dx, dy
        normalized_along = along / max(along_radius, 1e-9)
        normalized_across = across / max(across_radius, 1e-9)
        discrepancy = hypot(normalized_along, normalized_across)
        reliability = region.reliability
        if not self.sync_verified:
            reliability *= 0.75
        # Reliability is exposed to the scoring owner. Keep the legacy soft range
        # (+0.12/-0.20) and only fade it by prediction horizon for compatibility.
        adjustment = self.adjustment(discrepancy) * exp(-region.dt_s / 4)
        if (
            region.direction is not None
            and abs(normalized_across) >= 0.9
            and abs(normalized_across) <= 1.5
            and abs(normalized_along) <= 1.2
        ):
            adjustment = 0.0
            reason = "lateral_uncertain"
        elif discrepancy <= 1.0:
            reason = "expected_position"
        elif discrepancy <= 1.5:
            reason = "uncertain_position"
        else:
            reason = "spatial_contradiction"
        adjusted_region = replace(
            region,
            along_radius=along_radius,
            across_radius=across_radius,
            reliability=reliability,
        )
        return SpatialEvidence(
            adjustment,
            discrepancy,
            reliability,
            reason,
            region.dt_s,
            adjusted_region,
        )

    def predict(self, history, time, point):
        """Backward-compatible soft bonus or ``None``."""
        return self.evaluate(history, time, point).adjustment

    @staticmethod
    def adjustment(distance):
        # Close support boosts weak appearance; contradiction softly penalizes it.
        return 0.12 * exp(-0.5 * distance * distance) - 0.20 * min(
            1, max(0, (distance - 1.5) / 2)
        )

    def online_evidence(self, observation, history) -> SpatialEvidence:
        projection = self.projection(observation)
        if projection.point is None:
            return SpatialEvidence(None, None, 0.0, f"projection:{projection.reason}")
        evidence = self.evaluate(
            history,
            observation.timestamp_s,
            projection.point,
            time_uncertainty_s=(
                self.sync_uncertainty_s
                if self.sync_uncertainty_configured or self.prediction_model == "kalman"
                else 0.0
            ),
        )
        if evidence.adjustment is None:
            return replace(
                evidence,
                reliability=evidence.reliability * projection.reliability,
            )
        return replace(
            evidence,
            adjustment=evidence.adjustment * projection.reliability,
            reliability=evidence.reliability * projection.reliability,
        )

    def online(self, observation, history):
        """Backward-compatible soft bonus or ``None``."""
        return self.online_evidence(observation, history).adjustment

    def between_evidence(
        self,
        left,
        right,
        *,
        time_uncertainty_s: float | None = None,
    ) -> SpatialEvidence:
        """Compare simultaneous views or predict across sequential boundaries."""
        tolerance = (
            self.sync_uncertainty_s
            if time_uncertainty_s is None
            else max(0.0, time_uncertainty_s)
        )
        a = sorted(self._detailed_samples(left))
        b = sorted(self._detailed_samples(right))
        if not a or not b:
            missing = "both" if not a and not b else "left" if not a else "right"
            return SpatialEvidence(None, None, 0.0, f"missing_projection:{missing}")

        comparisons = []
        times = [sample[0] for sample in b]
        step = max(1, len(a) // 32)
        for time, x, y, reliability, _ in a[::step]:
            index = bisect_left(times, time)
            choices = b[max(0, index - 1) : index + 1]
            if not choices:
                continue
            other = min(choices, key=lambda sample: abs(sample[0] - time))
            delta = abs(other[0] - time)
            if delta > tolerance + 1e-9:
                continue
            denominator = self.radius + self.speed * (delta + tolerance)
            discrepancy = hypot(other[1] - x, other[2] - y) / denominator
            sync_reliability = (
                1.0
                if tolerance <= 1e-9 and delta <= 1e-9
                else exp(-delta / max(tolerance, 1e-9))
            )
            if not self.sync_verified:
                sync_reliability *= 0.75
            pair_reliability = min(reliability, other[3]) * sync_reliability
            comparisons.append(
                (
                    self.adjustment(discrepancy) * pair_reliability,
                    discrepancy,
                    pair_reliability,
                    delta,
                )
            )
        if comparisons:
            return SpatialEvidence(
                median(value[0] for value in comparisons),
                median(value[1] for value in comparisons),
                median(value[2] for value in comparisons),
                "simultaneous_overlap",
                median(value[3] for value in comparisons),
            )

        earlier, later = (a, b) if a[-1][0] < b[0][0] else (b, a)
        if later[0][0] <= earlier[-1][0]:
            return SpatialEvidence(
                None,
                None,
                0.0,
                "sync_mismatch_or_overlapping_segments",
                abs(later[0][0] - earlier[-1][0]),
            )
        target = later[0]
        evidence = self.evaluate(
            [(time, x, y) for time, x, y, _, _ in earlier],
            target[0],
            (target[1], target[2]),
            time_uncertainty_s=tolerance,
        )
        projection_reliability = min(target[3], median(p[3] for p in earlier))
        adjustment = (
            evidence.adjustment * projection_reliability
            if evidence.adjustment is not None
            else None
        )
        return replace(
            evidence,
            adjustment=adjustment,
            reliability=evidence.reliability * projection_reliability,
            reason=(
                "sequential_prediction"
                if evidence.available and self.prediction_model == "heuristic"
                else evidence.reason
            ),
        )

    def between(self, left, right):
        """Backward-compatible soft bonus or ``None``."""
        return self.between_evidence(left, right).adjustment

    def projection_report(self, observations, validation_points=()) -> dict:
        """Summarize projection coverage and independent-point errors by camera."""
        per_camera = defaultdict(Counter)
        for observation in observations:
            result = self.projection(observation)
            counts = per_camera[observation.camera_id]
            counts["total"] += 1
            counts["projected" if result.available else "rejected"] += 1
            counts[f"reason:{result.reason}"] += 1

        validation_errors = defaultdict(list)
        validation_reasons = defaultdict(Counter)
        for item in validation_points:
            camera_id = item.get("camera_id")
            result = self.project_video_point(camera_id, item.get("video_point"))
            if result.point is None:
                validation_reasons[camera_id][result.reason] += 1
                continue
            expected = item.get("plan_point")
            if not isinstance(expected, (tuple, list)) or not _finite_pair(expected):
                validation_reasons[camera_id]["invalid_plan_point"] += 1
                continue
            validation_errors[camera_id].append(
                hypot(result.point[0] - expected[0], result.point[1] - expected[1])
            )

        cameras = {}
        camera_ids = sorted(
            set(self.cameras)
            | set(per_camera)
            | set(validation_errors)
            | set(validation_reasons)
        )
        for camera_id in camera_ids:
            counts = per_camera[camera_id]
            total = counts["total"]
            errors = validation_errors[camera_id]
            cameras[camera_id] = {
                "total_observations": total,
                "projected": counts["projected"],
                "rejected": counts["rejected"],
                "coverage": counts["projected"] / total if total else None,
                "reasons": {
                    key.removeprefix("reason:"): value
                    for key, value in sorted(counts.items())
                    if key.startswith("reason:")
                },
                "independent_validation": {
                    "count": len(errors),
                    "median_error_coordinate_units": median(errors) if errors else None,
                    "max_error_coordinate_units": max(errors) if errors else None,
                    "rejected_reasons": dict(validation_reasons[camera_id]),
                },
            }
        return {
            "coordinate_units": self.coordinate_units,
            "coordinate_units_are_metric": self.coordinate_units == "meters",
            # Metric labels alone do not prove calibration accuracy.
            "physical_speed_claimed": False,
            "sync_uncertainty_s": self.sync_uncertainty_s,
            "sync_verified": self.sync_verified,
            "cameras": cameras,
        }

    def _detailed_samples(self, observations):
        samples = []
        for observation in observations:
            result = self.projection(observation)
            if result.point is not None:
                samples.append(
                    (
                        observation.timestamp_s,
                        result.point[0],
                        result.point[1],
                        result.reliability,
                        observation.camera_id,
                    )
                )
        return samples

    def _collapse_history(self, history):
        points = sorted(
            (float(t), float(x), float(y))
            for t, x, y, *_ in history
            if _finite_pair((t, x)) and isfinite(float(y))
        )
        if not points:
            return []
        window = min(0.05, max(self.sync_uncertainty_s, 0.0))
        groups = [[points[0]]]
        for point in points[1:]:
            if point[0] - groups[-1][-1][0] <= window + 1e-9:
                groups[-1].append(point)
            else:
                groups.append([point])
        return [
            (
                median(point[0] for point in group),
                median(point[1] for point in group),
                median(point[2] for point in group),
            )
            for group in groups
        ]

    def _projection(self, camera_id, point, reliability, reason):
        return ProjectionResult(
            camera_id,
            point,
            reliability,
            reason,
            self.coordinate_units,
        )


def _matrix3(value):
    if not isinstance(value, list) or len(value) != 3:
        return None
    rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != 3:
            return None
        if not all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and isfinite(float(item))
            for item in row
        ):
            return None
        rows.append(tuple(float(item) for item in row))
    return tuple(rows)


def _finite_pair(value) -> bool:
    return (
        isinstance(value, (tuple, list))
        and len(value) >= 2
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and isfinite(float(item))
            for item in value[:2]
        )
    )


def _positive_number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive finite number")
    return float(value)


def _bounded_number(value, name, *, minimum, maximum):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
        or not minimum <= float(value) <= maximum
    ):
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return float(value)
