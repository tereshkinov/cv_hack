import copy

import numpy as np
import pytest

from src.app.geometry import calibrate
from src.app.server import validate_project
from src.association.tests.test_geolocation import setup
from src.floor_mesh import (
    calibrate_mesh,
    map_point,
    overlap_report,
    project_calibration,
)


def grid():
    plan = [[x, y] for y in range(3) for x in range(3)]
    # Deliberately nonlinear image distortion, measured correspondences.
    video = [[100 + 100 * x + 8 * y * y, 100 + 100 * y + 6 * x * x] for x, y in plan]
    return video, plan


def test_nonlinear_mesh_interpolates_anchors_and_independent_triangle_point_both_directions():
    video, plan = grid()
    cal = calibrate_mesh(video, plan)
    assert len(cal["triangles"]) == 8
    assert np.allclose(project_calibration(cal, video), plan)
    t = cal["triangles"][0]
    weights = np.array([0.2, 0.3, 0.5])
    p = weights @ np.array(video)[t]
    q = weights @ np.array(plan)[t]
    assert np.allclose(map_point(cal, p)[0], q)
    assert np.allclose(map_point(cal, q, inverse=True)[0], p)
    assert cal["anchor_error_is_validation"] is False


def test_mesh_refuses_unmeasured_regions_and_marks_bounded_extrapolation():
    video = [[0, 0], [100, 0], [100, 100], [0, 100]]
    plan = [[0, 0], [10, 0], [10, 10], [0, 10]]
    cal = calibrate_mesh(video, plan)
    assert map_point(cal, [-1, 50])[0] is None
    q, quality, reason = map_point(cal, [-1, 50], extrapolate=True)
    assert (
        np.allclose(q, [-0.1, 5]) and quality <= 0.25 and reason == "mesh_extrapolated"
    )
    assert map_point(cal, [-50, 50], extrapolate=True)[0] is None
    with pytest.raises(ValueError):
        project_calibration(cal, [[-1, 50]])


def test_mesh_rejects_duplicates_flips_and_mismatched_counts():
    video, plan = grid()
    bad = copy.deepcopy(video)
    bad[4] = bad[0]
    with pytest.raises(ValueError):
        calibrate_mesh(bad, plan)
    bad = copy.deepcopy(video)
    bad[4] = [1000, 1000]
    with pytest.raises(ValueError):
        calibrate_mesh(bad, plan)
    with pytest.raises(ValueError):
        calibrate_mesh(video[:-1], plan)
    with pytest.raises(ValueError):
        calibrate_mesh([[i, i] for i in range(9)], plan)


def test_mesh_import_rebuilds_triangles_and_keeps_legacy_four_point_calibration():
    from src.app.tests.test_geometry import (
        test_import_does_not_trust_an_arbitrary_matrix,
    )

    test_import_does_not_trust_an_arbitrary_matrix()
    from src.app.server import default_project

    p = default_project()
    video, plan = grid()
    p["cameras"] = [
        {
            "id": "a",
            "name": "a",
            "position": [1, 1],
            "height_m": 2,
            "yaw_deg": 0,
            "fov_deg": 60,
            "range_m": 5,
            "video_points": video,
            "plan_points": plan,
            "calibration_method": "mesh",
            "calibration": {"triangles": [[999]]},
        }
    ]
    result = validate_project(p)["cameras"][0]["calibration"]
    assert len(result["triangles"]) == 8
    assert np.allclose(project_calibration(result, video), plan)
    assert calibrate(video, plan, "mesh")["method"] == "piecewise_affine_floor_mesh"


def test_calibrated_patch_overlap_does_not_depend_on_camera_yaw():
    _, p = setup()
    p["cameras"][0]["calibration"] = calibrate_mesh(
        [[0, 0], [100, 0], [100, 100], [0, 100]], [[0, 0], [5, 0], [5, 5], [0, 5]]
    )
    p["cameras"][1]["calibration"] = calibrate_mesh(
        [[0, 0], [100, 0], [100, 100], [0, 100]], [[4, 0], [9, 0], [9, 5], [4, 5]]
    )
    overlap = overlap_report(p)
    assert len(overlap) == 1 and overlap[0]["area_plan_units2"] == pytest.approx(5)


def test_independent_validation_is_recomputed_and_not_used_as_training():
    from src.floor_mesh import validation_report

    video, plan = grid()
    cal = calibrate_mesh(video, plan)
    before = copy.deepcopy(cal)
    checks = [
        {
            "video_point": video[4],
            "plan_point": [plan[4][0] + 0.2, plan[4][1]],
            "error_plan_units": 999,
        }
    ]
    report = validation_report(cal, checks)
    assert report["rmse_plan_units"] == pytest.approx(0.2)
    assert cal == before
