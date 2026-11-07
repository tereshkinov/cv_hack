from dataclasses import replace

import pytest

from src.association import Camera, Observation
from src.association.boundary_geometry import BoundaryGeometryMatcher
from src.association.geolocation import FloorMatcher


def observation(camera, local_id, time_s, x, *, y=500, height=200, confidence=0.9):
    return Observation(
        camera,
        round(time_s * 25),
        time_s,
        local_id,
        (x - 40, y - height, x + 40, y),
        confidence,
    )


def same_camera_matcher():
    return BoundaryGeometryMatcher([Camera("cam", width=1000, height=1000)])


def test_same_camera_boundary_recovers_smooth_motion_without_floor_calibration():
    matcher = same_camera_matcher()
    left = [observation("cam", "old", t, 100 + 100 * t) for t in (0, 0.4, 0.8)]
    right = [observation("cam", "new", t, 100 + 100 * t) for t in (1.2, 1.6, 2.0)]

    evidence = matcher.compare(left, right, "same_camera_recovery")

    assert evidence.available
    assert evidence.reason == "same_camera_bidirectional"
    assert evidence.coordinate_units == "normalized_image"
    assert evidence.support_count == 3
    assert evidence.value > 0.9
    assert evidence.reliability > 0.5


def test_same_camera_motion_separates_nearby_people_moving_differently():
    matcher = same_camera_matcher()
    left = [observation("cam", "old", t, 200 + 120 * t) for t in (0, 0.4, 0.8)]
    continuation = [
        observation("cam", "same", t, 200 + 120 * t) for t in (1.2, 1.6, 2.0)
    ]
    crossing_person = [
        observation("cam", "other", t, 520 - 120 * (t - 1.2)) for t in (1.2, 1.6, 2.0)
    ]

    same = matcher.compare(left, continuation, "same_camera_recovery")
    other = matcher.compare(left, crossing_person, "same_camera_recovery")

    assert same.value > 0.9
    assert other.value < 0.2
    assert other.details["forward_discrepancy"] > same.details["forward_discrepancy"]


def test_same_camera_position_only_is_available_but_less_reliable():
    matcher = same_camera_matcher()
    left = [observation("cam", "old", 1.0, 300)]
    right = [observation("cam", "new", 1.2, 305)]

    evidence = matcher.compare(left, right, "same_camera_recovery")

    assert evidence.available
    assert evidence.reason == "same_camera_position_only"
    assert evidence.value > 0.9
    assert evidence.reliability < 0.25
    assert evidence.support_count == 1


def test_stationary_sequence_has_no_invented_direction_or_strong_motion_support():
    matcher = same_camera_matcher()
    left = [observation("cam", "old", t, 300) for t in (0, 0.4, 0.8)]
    right = [observation("cam", "new", t, 302) for t in (1.2, 1.6, 2.0)]

    evidence = matcher.compare(left, right, "same_camera_recovery")

    assert evidence.available
    assert evidence.reason == "same_camera_position_only"
    assert evidence.value > 0.9
    assert evidence.reliability < 0.5
    assert evidence.details["forward_discrepancy"] == pytest.approx(
        evidence.details["position_only_discrepancy"]
    )


def test_same_camera_overlap_is_a_reliable_contradiction_and_long_gap_is_neutral():
    matcher = same_camera_matcher()
    left = [observation("cam", "old", t, 300) for t in (0, 1)]
    overlapping = [observation("cam", "other", t, 310) for t in (0.8, 1.2)]
    late = [observation("cam", "late", t, 310) for t in (6, 6.4)]

    contradiction = matcher.compare(left, overlapping, "same_camera_recovery")
    unavailable = matcher.compare(left, late, "same_camera_recovery")

    assert contradiction.available and contradiction.value == 0
    assert contradiction.reason == "same_camera_time_overlap"
    assert contradiction.details["hard_constraint"] is True
    assert not unavailable.available
    assert unavailable.reason == "same_camera_gap_too_long"


def test_cropped_feet_reduce_reliability_without_turning_missing_into_negative():
    matcher = same_camera_matcher()
    clean_left = [observation("cam", "old", t, 100 + 80 * t) for t in (0, 0.4, 0.8)]
    clean_right = [observation("cam", "new", t, 100 + 80 * t) for t in (1.2, 1.6, 2.0)]
    cropped_left = [
        replace(row, bbox_xyxy=(row.bbox_xyxy[0], 780, row.bbox_xyxy[2], 995))
        for row in clean_left
    ]
    cropped_right = [
        replace(row, bbox_xyxy=(row.bbox_xyxy[0], 780, row.bbox_xyxy[2], 995))
        for row in clean_right
    ]

    clean = matcher.compare(clean_left, clean_right, "same_camera_recovery")
    cropped = matcher.compare(cropped_left, cropped_right, "same_camera_recovery")

    assert cropped.available
    assert cropped.reliability < clean.reliability * 0.5
    assert cropped.details["left_cropped_fraction"] == 1


def test_cross_camera_modes_require_topology_and_floor_geometry():
    cameras = [
        Camera(
            "a",
            width=1000,
            height=1000,
            overlap_with=frozenset({"b"}),
            transition_to=frozenset({"b"}),
        ),
        Camera("b", width=1000, height=1000),
        Camera("c", width=1000, height=1000),
    ]
    project = {
        "coordinate_units": "relative",
        "room": {"width_m": 30, "height_m": 30},
        "cameras": [
            {
                "id": camera,
                "calibration": {"matrix": [[0.03, 0, 0], [0, 0.03, 0], [0, 0, 1]]},
            }
            for camera in ("a", "b")
        ],
        "processing": {"fusion_distance_m": 0.6, "sync_verified": True},
    }
    matcher = BoundaryGeometryMatcher(cameras, FloorMatcher(project, cameras))
    left = [observation("a", "left", t, 100) for t in (0, 0.4, 0.8)]
    right = [observation("b", "right", t, 105) for t in (0, 0.4, 0.8)]
    unlinked = [observation("c", "other", t, 105) for t in (0, 0.4, 0.8)]

    overlap = matcher.compare(left, right, "simultaneous_overlap")
    missing_topology = matcher.compare(left, unlinked, "simultaneous_overlap")

    assert overlap.available
    assert overlap.coordinate_units == "relative"
    assert overlap.value > 0.8
    assert overlap.support_count == 3
    assert not missing_topology.available
    assert missing_topology.reason == "overlap_not_configured"


def test_fragment_object_or_dict_uses_time_windows_not_frame_count():
    matcher = BoundaryGeometryMatcher(
        [Camera("cam", width=1000, height=1000)],
        window_s=0.6,
    )
    old_noise = [observation("cam", "old", t, 900) for t in (0, 0.2, 0.4)]
    left_tail = [
        observation("cam", "old", t, 200 + 50 * (t - 2)) for t in (2, 2.4, 2.8)
    ]
    right = [observation("cam", "new", t, 200 + 50 * (t - 2)) for t in (3.2, 3.6, 4.0)]

    evidence = matcher.compare(
        {"fragment_id": "a", "observations": old_noise + left_tail},
        {"fragment_id": "b", "observations": right},
        "same_camera_recovery",
    )

    assert evidence.available
    assert evidence.value > 0.85
    assert evidence.details["left_support"] == 2


def test_invalid_mode_is_rejected():
    matcher = same_camera_matcher()
    row = observation("cam", "id", 0, 100)
    with pytest.raises(ValueError, match="unsupported boundary mode"):
        matcher.compare([row], [row], "wrong")
