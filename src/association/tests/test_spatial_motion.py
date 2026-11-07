import pytest

from src.association import Camera, Observation
from src.association.boundary_geometry import BoundaryGeometryMatcher
from src.association.spatial_motion import OfficeMotionModel


def project(*, units="meters", meters_per_unit=None):
    motion = {
        "nodes": [
            {"id": "a_exit", "point": [0, 0]},
            {"id": "corner", "point": [0, 4]},
            {"id": "b_entry", "point": [6, 4]},
        ],
        "edges": [
            {"a": "a_exit", "b": "corner"},
            {"a": "corner", "b": "b_entry"},
            {"a": "a_exit", "b": "b_entry", "distance_m": 20},
        ],
        "camera_gates": {
            "a": {"exit": ["a_exit"], "entry": ["a_exit"]},
            "b": {"entry": ["b_entry"], "exit": ["b_entry"]},
        },
        "walking_speed_m_s": {
            "min": 0.5,
            "typical": 1.25,
            "max": 2.5,
            "sigma": 0.25,
        },
        "timing_sigma_s": 0.5,
        "route_uncertainty_fraction": 0.05,
        "max_dwell_s": 2,
        "distance_reliability": 0.9,
    }
    if meters_per_unit is not None:
        motion["meters_per_unit"] = meters_per_unit
    return {
        "coordinate_units": units,
        "processing": {"sync_uncertainty_s": 0.2},
        "motion_model": motion,
    }


def observation(camera, local_id, timestamp_s, x=300):
    return Observation(
        camera,
        round(timestamp_s * 25),
        timestamp_s,
        local_id,
        (x - 30, 200, x + 30, 500),
        0.9,
    )


def test_shortest_walkable_path_predicts_arrival_and_trajectory():
    model = OfficeMotionModel(project())

    evidence = model.transition_evidence("a", "b", 10, 18)

    assert model.distance_between_cameras("a", "b") == pytest.approx(10)
    assert evidence.available
    assert evidence.reason == "route_time_expected"
    assert evidence.expected_time_s == pytest.approx(8)
    assert evidence.value == pytest.approx(1)
    assert evidence.path_node_ids == ("a_exit", "corner", "b_entry")
    assert [row.distance_m for row in evidence.predicted_trajectory] == pytest.approx(
        [0, 4, 10]
    )
    assert [row.expected_timestamp_s for row in evidence.predicted_trajectory] == (
        pytest.approx([10, 13.2, 18])
    )


def test_impossible_early_and_very_late_arrivals_are_contradictions():
    model = OfficeMotionModel(project())

    early = model.transition_evidence("a", "b", 0, 2)
    late = model.transition_evidence("a", "b", 0, 30)

    assert early.reason == "physically_too_fast"
    assert early.value < 0.1
    assert early.feasible_time_range_s[0] == pytest.approx(3.8)
    assert late.reason == "physically_too_slow_or_stopped"
    assert late.value < 0.1


def test_reliable_observed_speed_personalizes_expected_arrival():
    model = OfficeMotionModel(project())

    population = model.transition_evidence("a", "b", 0, 5)
    personalized = model.transition_evidence(
        "a",
        "b",
        0,
        5,
        observed_speed_m_s=2.0,
        observed_speed_reliability=0.9,
    )

    assert population.expected_time_s == pytest.approx(8)
    assert personalized.expected_time_s == pytest.approx(5)
    assert personalized.speed_source == "observed_boundary_motion"
    assert personalized.support_count == 2
    assert personalized.value > population.value
    assert personalized.reliability > population.reliability


def test_unreliable_or_implausible_speed_falls_back_to_population_profile():
    model = OfficeMotionModel(project())

    unreliable = model.transition_evidence(
        "a", "b", 0, 8, observed_speed_m_s=2.0, observed_speed_reliability=0.1
    )
    impossible = model.transition_evidence(
        "a", "b", 0, 8, observed_speed_m_s=20.0, observed_speed_reliability=1.0
    )

    assert unreliable.speed_source == "population_profile"
    assert impossible.speed_source == "population_profile"
    assert unreliable.speed_m_s == pytest.approx(1.25)


def test_relative_plan_requires_explicit_metric_scale():
    with pytest.raises((TypeError, ValueError), match="meters_per_unit"):
        OfficeMotionModel(project(units="relative"))

    scaled_project = project(units="relative", meters_per_unit=0.5)
    scaled_project["motion_model"]["edges"] = [
        {"a": "a_exit", "b": "corner"},
        {"a": "corner", "b": "b_entry"},
    ]
    model = OfficeMotionModel(scaled_project)

    assert model.distance_between_cameras("a", "b") == pytest.approx(5)
    assert model.scale_source == "explicit_meters_per_unit"


def test_missing_route_or_gate_is_neutral_not_a_different_person_claim():
    model = OfficeMotionModel(project())

    evidence = model.transition_evidence("a", "missing", 0, 8)

    assert not evidence.available
    assert evidence.value is None
    assert evidence.reason == "route_or_camera_gate_missing"
    assert evidence.reliability == 0


def test_boundary_provider_uses_route_time_without_floor_calibration():
    cameras = [
        Camera("a", width=1000, height=1000, transition_to=frozenset({"b"})),
        Camera("b", width=1000, height=1000),
    ]
    matcher = BoundaryGeometryMatcher(
        cameras,
        route_model=OfficeMotionModel(project()),
    )
    left = [observation("a", "old", t) for t in (0, 0.4, 0.8)]
    right = [observation("b", "new", t) for t in (8.8, 9.2, 9.6)]

    evidence = matcher.compare(left, right, "sequential_transition")

    assert evidence.available
    assert evidence.reason == "route:route_time_expected"
    assert evidence.value == pytest.approx(1)
    assert evidence.coordinate_units == "meters_and_seconds"
    assert evidence.support_count == 1
    assert evidence.expected_region["path_node_ids"] == [
        "a_exit",
        "corner",
        "b_entry",
    ]


def test_boundary_provider_keeps_camera_topology_as_a_hard_prerequisite():
    cameras = [
        Camera("a", width=1000, height=1000),
        Camera("b", width=1000, height=1000),
    ]
    matcher = BoundaryGeometryMatcher(
        cameras,
        route_model=OfficeMotionModel(project()),
    )

    evidence = matcher.compare(
        [observation("a", "old", 0)],
        [observation("b", "new", 8)],
        "sequential_transition",
    )

    assert not evidence.available
    assert evidence.reason == "transition_not_configured"


def test_boundary_provider_preserves_missing_route_diagnostic_without_floor_data():
    cameras = [
        Camera("a", width=1000, height=1000, transition_to=frozenset({"missing"})),
        Camera("missing", width=1000, height=1000),
    ]
    matcher = BoundaryGeometryMatcher(
        cameras,
        route_model=OfficeMotionModel(project()),
    )

    evidence = matcher.compare(
        [observation("a", "old", 0)],
        [observation("missing", "new", 8)],
        "sequential_transition",
    )

    assert not evidence.available
    assert evidence.reason == "route:route_or_camera_gate_missing"
