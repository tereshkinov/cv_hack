from copy import deepcopy

import pytest

from src.association.geolocation import FloorMatcher
from src.metric_scale import annotate_speeds, fit_scale, scale_info

from .test_geolocation import setup


def test_scale_fit_resists_one_bad_measurement():
    measures = [
        {"a": [0, 0], "b": [d / 4, 0], "distance_m": d} for d in [4, 11.6, 23, 30]
    ]
    measures.append({"a": [0, 0], "b": [1, 0], "distance_m": 8})
    assert fit_scale(measures)["meters_per_unit"] == 4


@pytest.mark.parametrize("value", [0, -1, float("nan"), True, "4"])
def test_invalid_scale_rejected(value):
    with pytest.raises((ValueError, TypeError)):
        scale_info(
            {"coordinate_units": "relative", "metric_scale": {"meters_per_unit": value}}
        )


def test_metric_scale_does_not_change_projection_or_user_points():
    cameras, p = setup()
    original = deepcopy(p)
    p["coordinate_units"] = "relative"
    p["metric_scale"] = {"meters_per_unit": 4.2, "uncertainty_fraction": 0.15}
    before = FloorMatcher(original, cameras).project_video_point("a", [100, 100]).point
    floor = FloorMatcher(p, cameras)
    after = floor.project_video_point("a", [100, 100]).point
    assert before == after
    assert floor.speed == pytest.approx(2.5 / 4.2)
    assert p["cameras"] == original["cameras"]


def test_display_speed_converted_to_meters_and_stale_values_cleared():
    points = [
        {
            "timestamp_s": i * 0.2,
            "x_m": i * 0.05,
            "y_m": 0,
            "status": "single",
            "sources": [{"camera_id": "a"}],
        }
        for i in range(8)
    ]
    paths = {"g1": points}
    annotate_speeds(
        paths, {"coordinate_units": "relative", "metric_scale": {"meters_per_unit": 4}}
    )
    assert points[-1]["speed_m_s"] == pytest.approx(1)
    assert points[-1]["speed_uncertainty_m_s"] >= 0.15 - 1e-8
    annotate_speeds(paths, {"coordinate_units": "relative"})
    assert "speed_m_s" not in points[-1]


def test_no_speed_extrapolated_across_absence():
    points = [
        {"timestamp_s": t, "x_m": t, "y_m": 0, "status": "single"}
        for t in [0, 0.2, 0.4, 0.6, 5]
    ]
    annotate_speeds({"g1": points}, {"coordinate_units": "meters"})
    assert "speed_m_s" in points[-2] and "speed_m_s" not in points[-1]


def test_speed_history_resets_when_camera_set_changes():
    points = [
        {
            "timestamp_s": i * 0.2,
            "x_m": i * 0.2 + (100 if i >= 5 else 0),
            "y_m": 0,
            "status": "single",
            "sources": [{"camera_id": "a" if i < 5 else "b"}],
        }
        for i in range(10)
    ]
    annotate_speeds({"g": points}, {"coordinate_units": "meters"})
    assert points[4]["speed_m_s"] == pytest.approx(1)
    assert "speed_m_s" not in points[5]
    assert points[-1]["speed_m_s"] == pytest.approx(1)
