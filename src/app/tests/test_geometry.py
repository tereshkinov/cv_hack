import copy

import numpy as np
import pytest

from src.app.geometry import calibrate, project
from src.app.server import default_project, validate_project


def test_four_anchors_and_independent_points_follow_known_perspective():
    truth = np.array([[0.02, 0.003, 2], [-0.001, 0.015, 1], [0.0002, 0.0001, 1.0]])
    image = [[30, 40], [1000, 80], [950, 800], [100, 750]]
    floor = project(truth, image)
    result = calibrate(image, floor.tolist())
    assert np.allclose(project(result["matrix"], image), floor, atol=1e-6)
    independent = [[300, 500], [600, 600], [250, 120]]
    assert np.allclose(
        project(result["matrix"], independent), project(truth, independent), atol=1e-5
    )


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[0, 0], [1, 0], [1, 0], [0, 1]],
        [[0, 0], [1, 0], [2, 0], [1, 1]],
        [[0, 0], [1, 0], [1, float("nan")], [0, 1]],
        [[0, 0], [1, 1], [2, 2]],
    ],
)
def test_degenerate_or_incomplete_points_are_rejected(points):
    with pytest.raises(ValueError):
        calibrate(points, [[0, 0], [10, 0], [10, 10], [0, 10]])


def test_point_on_projection_horizon_is_rejected():
    with pytest.raises(ValueError, match="горизонту"):
        project([[1, 0, 0], [0, 1, 0], [1, 0, -1]], [[1, 2]])


def test_import_does_not_trust_an_arbitrary_matrix():
    state = default_project()
    state["cameras"] = [
        {
            "id": "cam1",
            "name": "Test",
            "position": [1, 1],
            "height_m": 2.5,
            "yaw_deg": 0,
            "fov_deg": 60,
            "range_m": 5,
            "video_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
            "plan_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "calibration": {"matrix": [[9, 9, 9]]},
        }
    ]
    result = validate_project(state)
    assert np.allclose(
        project(result["cameras"][0]["calibration"]["matrix"], [[50, 50]]), [[5, 5]]
    )
    bad = copy.deepcopy(state)
    bad["walls"] = [{"id": "w", "a": [0, 0], "b": [100, 100]}]
    with pytest.raises(ValueError):
        validate_project(bad)


def test_plan_metadata_uses_explicit_units_and_rejects_external_assets():
    state = default_project()
    state["coordinate_units"] = "relative"
    state["plan_background"] = {"image_id": "a" * 20, "opacity": 0.85}
    assert validate_project(state)["coordinate_units"] == "relative"
    for bad in [
        {"image_id": "https://example.com/private.png"},
        {"image_id": "a" * 20, "opacity": 2},
    ]:
        state["plan_background"] = bad
        with pytest.raises(ValueError):
            validate_project(state)
