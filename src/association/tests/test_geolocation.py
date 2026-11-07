from dataclasses import replace

import pytest

from src.association import AssociationConfig, AssociationEngine, Camera, Observation
from src.association.geolocation import FloorMatcher


def setup():
    cameras = [
        Camera("a", width=1000, height=1000, overlap_with=frozenset({"b"})),
        Camera("b", width=1000, height=1000),
    ]
    project = {
        "coordinate_units": "meters",
        "room": {"width_m": 30, "height_m": 30},
        "cameras": [
            {
                "id": cid,
                "calibration": {"matrix": [[0.03, 0, 0], [0, 0.03, 0], [0, 0, 1]]},
            }
            for cid in ["a", "b"]
        ],
        "processing": {"fusion_distance_m": 0.6},
    }
    return cameras, project


def obs(cam, local, t, x, vector=(1.0, 0.0)):
    return Observation(
        cam, round(t * 25), t, local, (x - 10, 100, x + 10, 200), 0.9, vector, "test"
    )


def test_near_floor_recovers_view_change_but_far_similar_person_is_penalized():
    cams, project = setup()
    config = AssociationConfig(offline_reconcile=False, ambiguity_margin=0.03)
    rows = [obs("a", "first", 0, 100), obs("b", "next", 0.2, 110, (0.69, 0.724))]
    assert len(AssociationEngine(cams, config).run(rows)[1]) == 2
    engine = AssociationEngine(cams, config, project)
    assignments, tracks = engine.run(rows)
    assert len(tracks) == 1 and assignments[-1]["reason"].endswith("_floor")
    rows[-1] = obs("b", "next", 0.2, 900, (0.9, 0.436))
    assert len(AssociationEngine(cams, config).run(rows)[1]) == 1
    assert len(AssociationEngine(cams, config, project).run(rows)[1]) == 2


def test_geometry_disambiguates_similar_people_and_never_reuses_closed_id():
    cams, project = setup()
    config = AssociationConfig(offline_reconcile=False, timeout_s=15)
    rows = [
        obs("a", "one", 0, 100),
        obs("a", "two", 0, 800),
        obs("b", "seen", 0.2, 105),
    ]
    assignments, tracks = AssociationEngine(cams, config, project).run(rows)
    assert [r["global_track_id"] for r in assignments] == ["g1", "g2", "g1"]
    assert len(tracks) == 2
    rows = [rows[0], obs("b", "returned", 16, 105)]
    assert len(AssociationEngine(cams, config, project).run(rows)[1]) == 2


def test_missing_bad_or_cropped_calibration_is_neutral():
    cams, project = setup()
    matcher = FloorMatcher(project, cams)
    bad = replace(obs("a", "crop", 0, 100), bbox_xyxy=(0, 100, 30, 1000))
    assert matcher.point(bad) is None
    assert matcher.online(bad, [(0, 1, 1)]) is None
    del matcher.calibrations["b"]
    assert matcher.point(obs("b", "unknown", 1, 100)) is None
    assert matcher.point(obs("a", "outside", 1, 2000)) is None
    matcher.calibrations["a"]["matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    assert matcher.point(obs("a", "horizon", 2, 100)) is None


def test_direction_uses_history_uncertainty_expands_and_long_gap_neutral():
    cams, project = setup()
    matcher = FloorMatcher(project, cams)
    history = [(0, 1, 2), (0.5, 1.5, 2), (1, 2, 2)]
    forward = matcher.predict(history, 1.5, (2.5, 2))
    backwards = matcher.predict(history, 1.5, (-0.5, 2))
    assert forward > 0 and backwards < 0
    assert matcher.predict(history, 10, (20, 20)) is None
    assert matcher.predict([(0, 2, 2)], 0.2, (2, 2.1)) == pytest.approx(
        matcher.predict([(0, 2, 2)], 0.2, (2.1, 2))
    )


def test_offline_geometry_compares_simultaneous_positions_and_preserves_conflicts():
    cams, project = setup()
    matcher = FloorMatcher(project, cams)
    a = [obs("a", "one", t, 100) for t in [0, 0.3, 0.6, 0.9]]
    close = [obs("b", "other", t, 105, (0.6, 0.8)) for t in [0, 0.3, 0.6, 0.9]]
    far = [obs("b", "far", t, 900) for t in [0, 0.3, 0.6, 0.9]]
    assert matcher.between(a, close) > 0
    assert matcher.between(a, far) < 0
    cfg = AssociationConfig(similarity_threshold=0.9, tracklet_threshold=0.7)
    assert len(AssociationEngine(cams, cfg).run(a + close)[1]) == 2
    assert len(AssociationEngine(cams, cfg, project).run(a + close)[1]) == 1
    # Geolocation cannot merge two simultaneously visible local tracks on one camera.
    same = [replace(o, camera_id="a") for o in close]
    assert len(AssociationEngine(cams, cfg, project).run(a + same)[1]) == 2


def test_projection_diagnostics_distinguish_refusal_reasons_and_validate_holdout_points():
    cams, project = setup()
    matcher = FloorMatcher(project, cams)
    valid = obs("a", "valid", 0, 100)
    cropped = replace(obs("a", "cropped", 1, 100), bbox_xyxy=(90, 100, 110, 990))
    del matcher.calibrations["b"]
    missing = obs("b", "missing", 2, 100)

    report = matcher.projection_report(
        [valid, cropped, missing],
        [
            {
                "camera_id": "a",
                "video_point": [100, 200],
                "plan_point": [3.1, 6.0],
            }
        ],
    )

    assert matcher.projection(valid).reason == "projected"
    assert matcher.projection(cropped).reason == "cropped_feet"
    assert matcher.projection(missing).reason == "missing_calibration"
    assert report["cameras"]["a"]["coverage"] == 0.5
    assert report["cameras"]["a"]["reasons"] == {
        "cropped_feet": 1,
        "projected": 1,
    }
    validation = report["cameras"]["a"]["independent_validation"]
    assert validation["count"] == 1
    assert validation["max_error_coordinate_units"] == pytest.approx(0.1)


def test_projection_reports_horizon_outside_plan_and_invalid_matrix_without_crashing():
    cams, project = setup()
    matcher = FloorMatcher(project, cams)
    assert matcher.projection(obs("a", "outside", 0, 2000)).reason == "outside_plan"
    matcher.calibrations["a"]["matrix"] = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    assert matcher.projection(obs("a", "horizon", 1, 100)).reason == "near_horizon"
    matcher.calibrations["a"]["matrix"] = [[1, 0], [0, 1]]
    assert matcher.projection(obs("a", "matrix", 2, 100)).reason == "invalid_matrix"


def test_simultaneous_camera_points_do_not_create_velocity_and_pause_stops_motion():
    cams, project = setup()
    matcher = FloorMatcher(project, cams)
    empty = matcher.motion([])
    assert empty.position is None
    assert empty.velocity is None
    assert empty.stationary is False
    assert empty.reason == "no_history"

    simultaneous = matcher.motion([(0.0, 1, 1), (0.02, 5, 1)])
    assert simultaneous.velocity is None
    assert simultaneous.speed_coordinate_units_s is None
    assert simultaneous.sample_count == 1

    paused = matcher.motion(
        [(0, 0, 0), (0.5, 0.5, 0), (1.0, 1, 0), (1.3, 1, 0), (1.6, 1, 0)]
    )
    assert paused.stationary
    assert paused.velocity is None
    assert paused.reason == "stationary"


def test_expected_region_turns_expands_and_makes_lateral_motion_neutral():
    cams, project = setup()
    matcher = FloorMatcher(project, cams)
    turning = [(0, 0, 0), (0.5, 1, 0), (1.0, 1, 0), (1.5, 1, 1), (2.0, 1, 2)]
    motion = matcher.motion(turning)
    assert motion.velocity is not None
    assert abs(motion.velocity[1]) > abs(motion.velocity[0])

    near, _ = matcher.expected_region(turning, 2.2)
    later, _ = matcher.expected_region(turning, 3.0)
    assert near.direction is not None
    assert later.along_radius > near.along_radius
    assert later.across_radius > near.across_radius
    lateral = matcher.evaluate(
        turning,
        2.2,
        (
            near.center[0] - near.direction[1] * near.across_radius * 1.1,
            near.center[1] + near.direction[0] * near.across_radius * 1.1,
        ),
        time_uncertainty_s=0,
    )
    assert lateral.reason == "lateral_uncertain"
    assert lateral.adjustment == 0
    assert matcher.evaluate(turning, 10, (10, 10)).reason == "long_gap"


def test_relative_plan_never_labels_coordinate_speed_as_physical_meters():
    cams, project = setup()
    project["coordinate_units"] = "relative"
    project["room"] = {"width_m": 100, "height_m": 100}
    matcher = FloorMatcher(project, cams)
    estimate = matcher.motion([(0, 1, 1), (0.5, 2, 1), (1, 3, 1)])
    report = matcher.projection_report([])

    assert estimate.coordinate_units == "relative"
    assert estimate.speed_coordinate_units_s is not None
    assert report["coordinate_units"] == "relative"
    assert report["physical_speed_claimed"] is False


def test_sync_uncertainty_is_exposed_and_overlap_mismatch_stays_neutral():
    cams, project = setup()
    project["processing"]["sync_uncertainty_s"] = 0.1
    matcher = FloorMatcher(project, cams)
    left = [obs("a", "left", t, 100) for t in [0.0, 1.0]]
    right = [obs("b", "right", t, 100) for t in [0.6, 1.6]]
    mismatch = matcher.between_evidence(left, right)
    assert mismatch.adjustment is None
    assert mismatch.reason == "sync_mismatch_or_overlapping_segments"

    aligned = matcher.between_evidence(
        [obs("a", "aligned-a", 2.0, 100)],
        [obs("b", "aligned-b", 2.05, 100)],
    )
    assert aligned.reason == "simultaneous_overlap"
    assert 0 < aligned.reliability < 1
    assert aligned.time_delta_s == pytest.approx(0.05)
