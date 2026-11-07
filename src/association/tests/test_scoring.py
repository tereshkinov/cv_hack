from dataclasses import replace

import pytest

from src.association import AssociationConfig, AssociationEngine
from src.association.scoring import decide
from src.association.tests.test_geolocation import obs, setup


def test_absence_is_neutral_but_reliable_contradiction_rejects_clothes_match():
    absent = decide(0.9, 0.9, None, 1, 0.04, 0.72)
    contradiction = decide(0.9, 0.9, -0.2, 0.85, 0.04, 0.72)
    assert absent.accepted
    assert not contradiction.accepted
    assert contradiction.reason == "spatial_contradiction"
    # An unavailable feature has no effect regardless of the supplied reliability.
    assert absent.score == decide(0.9, 0.9, None, 0, 0.04, 0.72).score


def test_geometry_cannot_replace_appearance_and_adjacency_is_not_double_counted():
    assert not decide(0.2, 0.9, 0.12, 1, 0.04, 0.72).accepted
    evidence = decide(0.7, 0.9, 0.12, 0.8, 0.04, 0.72)
    assert evidence.accepted and not evidence.trusted
    assert evidence.topology < 0.04
    assert (
        decide(0.8, 0.2, None, 0, 0, 0.72).score
        < decide(0.8, 0.9, None, 0, 0, 0.72).score
    )
    assert decide(0.7, 0.9, -0.2, 0.1, 0.04, 0.72).reason != "spatial_contradiction"


def test_uncertain_link_does_not_immediately_contaminate_gallery():
    cams, project = setup()
    cfg = AssociationConfig(decision_mode="balanced", offline_reconcile=False)
    engine = AssociationEngine(cams, cfg, project)
    new_view = (0.67, 0.7424)
    rows = [obs("a", "first", 0, 100)] + [
        obs("b", "next", t, 105, new_view) for t in [0.2, 0.4, 0.6]
    ]
    assignments, tracks = engine.run(rows)
    assert len(tracks) == 1
    assert assignments[1]["gallery_updated"] is False
    assert assignments[2]["gallery_updated"] is False
    assert assignments[3]["gallery_updated"] is True
    assert engine.gallery_updates_blocked == 2


def test_near_threshold_runner_up_is_considered_before_accepting_best():
    cams, _ = setup()
    engine = AssociationEngine(
        cams,
        AssociationConfig(
            decision_mode="balanced",
            offline_reconcile=False,
            similarity_threshold=0.72,
            ambiguity_margin=0.03,
        ),
    )
    rows = [obs("a", "one", 0, 100), obs("a", "two", 0, 800), obs("a", "next", 1, 450)]
    engine._appearance_score = lambda o, t: 0.74 if t.global_track_id == "g1" else 0.715
    assignments, _ = engine.run(rows)
    assert assignments[-1]["global_track_id"] == "g3"


def test_closed_episode_and_same_camera_collision_remain_separate():
    cams, project = setup()
    cfg = AssociationConfig(decision_mode="balanced", timeout_s=15)
    rows = [
        obs("a", "one", 0, 100),
        obs("a", "two", 0, 105),
        obs("b", "return", 16, 105),
    ]
    assignments, tracks = AssociationEngine(cams, cfg, project).run(rows)
    assert len(tracks) == 3
    assert len({a["global_track_id"] for a in assignments}) == 3


def test_missing_calibration_has_no_spatial_bonus_and_incompatible_vectors_rejected():
    cams, project = setup()
    project["cameras"] = []
    rows = [
        obs("a", "one", 0, 100),
        replace(obs("b", "other", 0.2, 105), embedding_model_id="other-space"),
    ]
    assignments, tracks = AssociationEngine(
        cams, AssociationConfig(decision_mode="balanced"), project
    ).run(rows)
    assert len(tracks) == 2
    assert all("decision_evidence" not in row for row in assignments)


def test_invalid_decision_mode():
    with pytest.raises(ValueError):
        AssociationConfig(decision_mode="unverified-probability")


def test_native_pair_reliability_is_not_applied_twice():
    from src.association.geolocation import SpatialEvidence
    from src.association.scoring import spatial_input

    spatial = SpatialEvidence(0.06, 0.5, 0.5, "simultaneous_overlap")
    geometry, reliability = spatial_input(spatial)
    decision = decide(
        0.72,
        1,
        geometry,
        reliability,
        0,
        0.72,
        geometry_preweighted=True,
        spatial_details=spatial.as_dict(),
    )
    assert decision.score == pytest.approx(0.81)
    assert decision.spatial_details["reason"] == "simultaneous_overlap"


def test_native_prediction_adds_motion_reliability_without_second_time_fade():
    from math import exp

    from src.association.geolocation import ExpectedRegion, SpatialEvidence
    from src.association.scoring import spatial_input

    region = ExpectedRegion((1, 1), None, 1, 1, 2, 0.25 * exp(-2 / 4), "relative")
    spatial = SpatialEvidence(
        0.12 * exp(-2 / 4), 0.1, region.reliability, "expected_position", 2, region
    )
    geometry, reliability = spatial_input(spatial)
    assert geometry == pytest.approx(0.12 * exp(-2 / 4) * 0.25)
    assert reliability == region.reliability
    missing = SpatialEvidence(None, None, 0, "projection:cropped_feet")
    assert spatial_input(missing) == (None, 0)


def test_balanced_engine_exports_native_projection_reasons():
    cams, project = setup()
    rows = [obs("a", "first", 0, 100), obs("b", "next", 0.2, 105)]
    engine = AssociationEngine(
        cams, AssociationConfig(decision_mode="balanced"), project
    )
    engine.run(rows)
    assert engine.decision_log[0]["geometry_preweighted"]
    assert engine.decision_log[0]["spatial_details"]["reason"] == "expected_position"
