import pytest

from src.association.projected_route import endpoint_route
from src.association.spatial_motion import OfficeMotionModel


def model(bidirectional=True):
    return OfficeMotionModel(
        {
            "coordinate_units": "relative",
            "metric_scale": {"meters_per_unit": 4},
            "motion_model": {
                "meters_per_unit": 2,
                "nodes": [
                    {"id": "a", "point": [0, 0]},
                    {"id": "b", "point": [5, 0]},
                    {"id": "c", "point": [5, 5]},
                ],
                "edges": [
                    {"a": "a", "b": "b", "bidirectional": bidirectional},
                    {"a": "b", "b": "c", "bidirectional": bidirectional},
                ],
                "camera_gates": {},
            },
        }
    )


def test_uses_observed_endpoints_and_turns_instead_of_full_gate_distance():
    result = endpoint_route(model(), [4, 0], [5, 1], 1)
    assert result["distance_m"] == pytest.approx(8)
    assert result["first_vector"][0] > 0
    assert result["last_vector"][1] > 0


def test_direct_same_edge_shortcut_and_one_way_constraints():
    assert endpoint_route(model(False), [1, 0], [2, 0], 1)[
        "distance_m"
    ] == pytest.approx(4)
    assert endpoint_route(model(False), [2, 0], [1, 0], 1) is None
    assert endpoint_route(model(True), [2, 0], [1, 0], 1)[
        "distance_m"
    ] == pytest.approx(4)


def test_far_points_are_not_snapped_to_a_corridor():
    assert endpoint_route(model(), [2, 3], [5, 1], 1) is None
