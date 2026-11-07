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
