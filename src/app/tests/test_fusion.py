import pytest

from src.app.fusion import fuse_positions, smooth_paths


def test_simultaneous_sources_use_confidence_squared_weights():
    sources = [
        {"camera_id": "a", "x_m": 2.0, "y_m": 5.0, "confidence": 1.0},
        {"camera_id": "b", "x_m": 2.2, "y_m": 5.0, "confidence": 0.5},
    ]
    result = fuse_positions(sources)
    assert result["status"] == "fused" and result["x_m"] == pytest.approx(2.04)
    assert len(result["sources"]) == 2


def test_conflicting_positions_are_preserved_but_not_averaged():
    result = fuse_positions(
        [
            {"camera_id": "a", "x_m": 0, "y_m": 0, "confidence": 0.9},
            {"camera_id": "b", "x_m": 10, "y_m": 0, "confidence": 0.9},
        ],
        max_distance_m=2,
    )
    assert result["status"] == "disagreement" and result["x_m"] is None
    assert len(result["sources"]) == 2 and fuse_positions([]) is None


def test_smoothing_resets_after_absence_and_disagreement():
    points = [
        {"timestamp_s": 0, "x_m": 0, "y_m": 0},
        {"timestamp_s": 0.1, "x_m": 1, "y_m": 0},
        {"timestamp_s": 2, "x_m": 10, "y_m": 0},
        {"timestamp_s": 2.1, "x_m": None, "y_m": None},
        {"timestamp_s": 2.2, "x_m": 20, "y_m": 0},
    ]
    smooth_paths({"g1": points})
    assert 0 < points[1]["smooth_x_m"] < 1
    assert points[2]["smooth_x_m"] == 10 and points[4]["smooth_x_m"] == 20
