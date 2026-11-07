"""Metric office-route and travel-time evidence for fragment transitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from heapq import heappop, heappush
from math import exp, hypot, isfinite, sqrt


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    node_id: str
    x_m: float
    y_m: float
    distance_m: float
    expected_timestamp_s: float


@dataclass(frozen=True, slots=True)
class TravelTimeEvidence:
    """Compatibility of an observed transition with a walkable office route.

    ``value`` is ``exp(-0.5 * z_score**2)`` and does not include reliability.
    A caller must apply reliability exactly once when combining evidence.
    """

    available: bool
    value: float | None
    reliability: float
    support_count: int
    reason: str
    source_camera_id: str
    target_camera_id: str
    distance_m: float | None
    observed_time_s: float | None
    expected_time_s: float | None
    sigma_time_s: float | None
    residual_time_s: float | None
    z_score: float | None
    feasible_time_range_s: tuple[float, float] | None
    speed_m_s: float | None
    speed_source: str
    path_node_ids: tuple[str, ...]
    predicted_trajectory: tuple[TrajectoryPoint, ...]
    details: dict

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Node:
    node_id: str
    x_m: float
    y_m: float


class OfficeMotionModel:
    """Shortest-path walking model built from explicitly marked office routes.

    The project must contain ``motion_model`` with route nodes, edges, and camera
    entry/exit gates. Metric claims are accepted only when the plan itself is
    metric or ``meters_per_unit`` is explicitly configured.
    """

    def __init__(self, project: dict):
        config = project.get("motion_model")
        if config is None:
            raise ValueError("project.motion_model must be configured")
        if not isinstance(config, dict):
            raise TypeError("project.motion_model must be an object")
        units = project.get("coordinate_units", "meters")
        if units == "meters":
            self.meters_per_unit = 1.0
            scale_source = "metric_plan"
        else:
            self.meters_per_unit = _positive(
                config.get("meters_per_unit"),
                "motion_model.meters_per_unit",
            )
            scale_source = "explicit_meters_per_unit"

        self.nodes = self._parse_nodes(config.get("nodes", []))
        self.graph = {node_id: [] for node_id in self.nodes}
        self.edge_distances: dict[tuple[str, str], float] = {}
        self._parse_edges(config.get("edges", []))
        self.gates = self._parse_gates(config.get("camera_gates", {}))

        profile = config.get("walking_speed_m_s", {})
        if not isinstance(profile, dict):
            raise TypeError("motion_model.walking_speed_m_s must be an object")
        self.min_speed = _positive(profile.get("min", 0.6), "walking speed min")
        self.typical_speed = _positive(
            profile.get("typical", 1.35), "walking speed typical"
        )
        self.max_speed = _positive(profile.get("max", 2.2), "walking speed max")
        self.speed_sigma = _positive(profile.get("sigma", 0.35), "walking speed sigma")
        if not self.min_speed < self.typical_speed < self.max_speed:
            raise ValueError("walking speeds must satisfy min < typical < max")

        self.timing_sigma_s = _non_negative(
            config.get("timing_sigma_s", 0.75), "motion_model.timing_sigma_s"
        )
        self.sync_uncertainty_s = _non_negative(
            project.get("processing", {}).get("sync_uncertainty_s", 0.25),
            "processing.sync_uncertainty_s",
        )
        self.route_uncertainty_fraction = _bounded(
            config.get("route_uncertainty_fraction", 0.10),
            "motion_model.route_uncertainty_fraction",
            0.0,
            1.0,
        )
        self.max_dwell_s = _non_negative(
            config.get("max_dwell_s", 3.0), "motion_model.max_dwell_s"
        )
        self.distance_reliability = _bounded(
            config.get("distance_reliability", 0.8),
            "motion_model.distance_reliability",
            0.0,
            1.0,
        )
        self.scale_source = scale_source

    @classmethod
    def from_project(cls, project: dict) -> OfficeMotionModel | None:
        """Return no model when it was not configured; reject invalid claims."""
        if not isinstance(project.get("motion_model"), dict):
            return None
        return cls(project)

    def transition_evidence(
        self,
        source_camera_id: str,
        target_camera_id: str,
        departure_s: float,
        arrival_s: float,
        *,
        observed_speed_m_s: float | None = None,
        observed_speed_reliability: float = 0.0,
    ) -> TravelTimeEvidence:
        if not all(isfinite(value) for value in (departure_s, arrival_s)):
            return self._missing(
                source_camera_id,
                target_camera_id,
                "invalid_timestamps",
            )
        path = self._camera_path(source_camera_id, target_camera_id)
        if path is None:
            return self._missing(
                source_camera_id,
                target_camera_id,
                "route_or_camera_gate_missing",
                observed_time_s=arrival_s - departure_s,
            )
        distance_m, node_ids = path
        observed_time_s = arrival_s - departure_s

        speed, speed_source, speed_reliability, speed_sigma = self._speed(
            observed_speed_m_s,
            observed_speed_reliability,
        )
        expected_time_s = distance_m / speed
        route_sigma_s = distance_m * self.route_uncertainty_fraction / speed
        propagated_speed_sigma_s = distance_m * speed_sigma / (speed * speed)
        sigma_time_s = max(
            0.1,
            sqrt(
                self.timing_sigma_s**2
                + self.sync_uncertainty_s**2
                + route_sigma_s**2
                + propagated_speed_sigma_s**2
            ),
        )
        residual = observed_time_s - expected_time_s
        z_score = abs(residual) / sigma_time_s
        value = exp(-0.5 * z_score * z_score)
        feasible_min = max(0.0, distance_m / self.max_speed - self.sync_uncertainty_s)
        feasible_max = (
            distance_m / self.min_speed + self.max_dwell_s + self.sync_uncertainty_s
        )
        if observed_time_s < feasible_min:
            reason = "physically_too_fast"
        elif observed_time_s > feasible_max:
            reason = "physically_too_slow_or_stopped"
        elif z_score <= 1.0:
            reason = "route_time_expected"
        elif z_score <= 2.0:
            reason = "route_time_uncertain"
        else:
            reason = "route_time_contradiction"

        gate_specificity = 1 / sqrt(
            max(
                1,
                len(self.gates[source_camera_id]["exit"])
                * len(self.gates[target_camera_id]["entry"]),
            )
        )
        reliability = (
            self.distance_reliability * speed_reliability * max(0.45, gate_specificity)
        )
        trajectory = self._trajectory(node_ids, departure_s, speed)
        return TravelTimeEvidence(
            True,
            value,
            min(1.0, reliability),
            2 if speed_source == "observed_boundary_motion" else 1,
            reason,
            source_camera_id,
            target_camera_id,
            distance_m,
            observed_time_s,
            expected_time_s,
            sigma_time_s,
            residual,
            z_score,
            (feasible_min, feasible_max),
            speed,
            speed_source,
            node_ids,
            trajectory,
            {
                "value_scale": "exp(-0.5 * z_score^2), reliability not applied",
                "scale_source": self.scale_source,
                "route_uncertainty_fraction": self.route_uncertainty_fraction,
                "sync_uncertainty_s": self.sync_uncertainty_s,
                "gate_specificity": gate_specificity,
                "observed_speed_reliability": observed_speed_reliability,
            },
        )

    def distance_between_cameras(
        self, source_camera_id: str, target_camera_id: str
    ) -> float | None:
        path = self._camera_path(source_camera_id, target_camera_id)
        return path[0] if path else None

    def _speed(self, observed_speed, observed_reliability):
        reliability = _bounded(
            observed_reliability,
            "observed_speed_reliability",
            0.0,
            1.0,
        )
        if (
            observed_speed is not None
            and isfinite(observed_speed)
            and self.min_speed * 0.5 <= observed_speed <= self.max_speed * 1.5
            and reliability >= 0.25
        ):
            speed = min(self.max_speed, max(self.min_speed, observed_speed))
            sigma = max(0.15, self.speed_sigma * (1.15 - reliability))
            return speed, "observed_boundary_motion", 0.55 + 0.45 * reliability, sigma
        return self.typical_speed, "population_profile", 0.55, self.speed_sigma

    def _camera_path(self, source_camera_id, target_camera_id):
        source = self.gates.get(source_camera_id)
        target = self.gates.get(target_camera_id)
        if not source or not target or not source["exit"] or not target["entry"]:
            return None
        best = None
        for start in source["exit"]:
            for finish in target["entry"]:
                candidate = self._shortest_path(start, finish)
                if candidate is not None and (best is None or candidate[0] < best[0]):
                    best = candidate
        return best

    def _shortest_path(self, start, finish):
        queue = [(0.0, start, (start,))]
        best = {start: 0.0}
        while queue:
            distance, node_id, path = heappop(queue)
            if distance > best.get(node_id, float("inf")) + 1e-12:
                continue
            if node_id == finish:
                return distance, path
            for neighbor, edge_distance in self.graph[node_id]:
                candidate = distance + edge_distance
                if candidate + 1e-12 < best.get(neighbor, float("inf")):
                    best[neighbor] = candidate
                    heappush(queue, (candidate, neighbor, path + (neighbor,)))
        return None

    def _trajectory(self, node_ids, departure_s, speed):
        distance = 0.0
        rows = []
        for index, node_id in enumerate(node_ids):
            if index:
                distance += self.edge_distances[(node_ids[index - 1], node_id)]
            node = self.nodes[node_id]
            rows.append(
                TrajectoryPoint(
                    node_id,
                    node.x_m,
                    node.y_m,
                    distance,
                    departure_s + distance / speed,
                )
            )
        return tuple(rows)

    def _parse_nodes(self, values):
        if not isinstance(values, list) or not values:
            raise ValueError("motion_model.nodes must be a non-empty list")
        nodes = {}
        for value in values:
            if not isinstance(value, dict):
                raise TypeError("each motion-model node must be an object")
            node_id = _identifier(value.get("id"), "motion-model node id")
            point = value.get("point")
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError("motion-model node point must contain x and y")
            if node_id in nodes:
                raise ValueError(f"duplicate motion-model node: {node_id}")
            nodes[node_id] = _Node(
                node_id,
                _finite(point[0], "node x") * self.meters_per_unit,
                _finite(point[1], "node y") * self.meters_per_unit,
            )
        return nodes

    def _parse_edges(self, values):
        if not isinstance(values, list) or not values:
            raise ValueError("motion_model.edges must be a non-empty list")
        for value in values:
            if not isinstance(value, dict):
                raise TypeError("each motion-model edge must be an object")
            start = _identifier(value.get("a"), "edge.a")
            finish = _identifier(value.get("b"), "edge.b")
            if start not in self.nodes or finish not in self.nodes or start == finish:
                raise ValueError("motion-model edge references invalid nodes")
            if "distance_m" in value:
                distance = _positive(value["distance_m"], "edge.distance_m")
            else:
                a, b = self.nodes[start], self.nodes[finish]
                distance = hypot(b.x_m - a.x_m, b.y_m - a.y_m)
                if distance <= 0:
                    raise ValueError("motion-model edge must have positive length")
            self._add_edge(start, finish, distance)
            if value.get("bidirectional", True):
                self._add_edge(finish, start, distance)

    def _add_edge(self, start, finish, distance):
        self.graph[start].append((finish, distance))
        self.edge_distances[(start, finish)] = distance

    def _parse_gates(self, values):
        if not isinstance(values, dict):
            raise TypeError("motion_model.camera_gates must be an object")
        gates = {}
        for camera_id, value in values.items():
            camera_id = _identifier(camera_id, "camera gate id")
            if not isinstance(value, dict):
                raise TypeError("camera gate must be an object")
            both = value.get("both", [])
            entries = value.get("entry", both)
            exits = value.get("exit", both)
            entries = self._gate_nodes(entries, f"{camera_id}.entry")
            exits = self._gate_nodes(exits, f"{camera_id}.exit")
            gates[camera_id] = {"entry": entries, "exit": exits}
        return gates

    def _gate_nodes(self, values, name):
        if not isinstance(values, list):
            raise TypeError(f"{name} must be a list")
        result = tuple(_identifier(value, name) for value in values)
        if any(node_id not in self.nodes for node_id in result):
            raise ValueError(f"{name} references an unknown route node")
        return result

    @staticmethod
    def _missing(
        source_camera_id,
        target_camera_id,
        reason,
        *,
        observed_time_s=None,
    ):
        return TravelTimeEvidence(
            False,
            None,
            0.0,
            0,
            reason,
            source_camera_id,
            target_camera_id,
            None,
            observed_time_s,
            None,
            None,
            None,
            None,
            None,
            None,
            "unavailable",
            (),
            (),
            {},
        )


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value, name):
    result = _finite(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative(value, name):
    result = _finite(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _bounded(value, name, minimum, maximum):
    result = _finite(value, name)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return result
