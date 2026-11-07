from dataclasses import replace

import pytest

from src.association import AssociationConfig, AssociationEngine, Camera
from src.association.boundary_geometry import BoundaryGeometryEvidence
from src.association.fragments import fragments_from_observations
from src.association.geometry_policy import decide_geometry
from src.association.tests.test_geolocation import obs
from src.association.tests.test_reconcile import (
    route_cameras,
    route_project,
    segment,
)


def boundary(value, reliability, reason="route:route_time_expected", time_delta=8):
    return BoundaryGeometryEvidence(
        True,
        value,
        reliability,
        1,
        reason,
        "sequential_transition",
        0.0,
        "meters_and_seconds",
        time_delta,
        None,
        {},
    )


def without_appearance(rows):
    return [replace(row, embedding=None, embedding_model_id=None) for row in rows]


def test_geometry_policy_works_without_appearance_and_exposes_reliability():
    decision = decide_geometry(boundary(0.84, 0.25), 0.62)

    assert decision.accepted
    assert decision.appearance is None
    assert decision.geometry_score == pytest.approx(0.7125)
    assert decision.reason == "geometry_supported"


def test_visual_confirmation_is_bounded_and_cannot_replace_missing_geometry():
    geometry = boundary(0.63, 0.4)
    pure = decide_geometry(geometry, 0.62)
    confirmed = decide_geometry(
        geometry,
        0.62,
        appearance=0.9,
        appearance_reliability=0.8,
        visual_confirmation_weight=0.08,
    )
    missing = replace(geometry, available=False, value=None, reason="missing_calibration")
    missing_decision = decide_geometry(
        missing,
        0.62,
        appearance=1.0,
        appearance_reliability=1.0,
        visual_confirmation_weight=0.2,
    )

    assert not pure.accepted and confirmed.accepted
    assert confirmed.visual_adjustment <= 0.08
    assert missing_decision.score is None and not missing_decision.accepted


def test_negative_or_reliable_impossible_travel_is_a_hard_refusal():
    negative = decide_geometry(
        boundary(0.95, 0.2, "route:physically_too_fast", -0.2), 0.62
    )
    impossible = decide_geometry(
        boundary(0.8, 0.8, "route:physically_too_fast", 1.0),
        0.62,
        appearance=1.0,
        appearance_reliability=1.0,
        visual_confirmation_weight=0.2,
    )

    assert negative.hard_reject and not negative.accepted
    assert impossible.hard_reject and not impossible.accepted
    assert impossible.score <= 0.42


def test_geometry_fragment_rebuild_links_expected_route_without_embeddings():
    source = without_appearance(segment("cam1", "source", 0))
    target = without_appearance(segment("cam3", "target", 9))
    engine = AssociationEngine(
        route_cameras(),
        AssociationConfig(
            decision_mode="geometry",
            fragment_mode="rebuild",
            offline_reconcile=False,
            tracklet_threshold=0.62,
            ambiguity_margin=0.01,
        ),
        route_project(),
    )

    assignments, tracks = engine.run(source + target)

    assert len(tracks) == 1
    assert {row["global_track_id"] for row in assignments} == {"g1"}
    selected = [row for row in engine.fragment_report["edges"] if row["selected"]]
    assert len(selected) == 1
    assert selected[0]["appearance"] is None
    assert selected[0]["boundary_geometry"]["reason"] == "route:route_time_expected"


def test_geometry_fragment_rebuild_rejects_impossible_route_despite_appearance():
    source = segment("cam1", "source", 0)
    target = segment("cam3", "target", 2.0)
    engine = AssociationEngine(
        route_cameras(),
        AssociationConfig(
            decision_mode="geometry_visual",
            fragment_mode="rebuild",
            offline_reconcile=False,
            tracklet_threshold=0.62,
            geometry_visual_weight=0.2,
        ),
        route_project(),
    )

    _, tracks = engine.run(source + target)

    assert len(tracks) == 2
    assert any(
        constraint["reason"] == "geometry_physical_contradiction"
        for constraint in engine.fragment_report["constraints"]
    )


@pytest.mark.parametrize("mode", ["geometry", "geometry_visual"])
def test_geometry_modes_require_fragment_rebuild_to_avoid_provisional_visual_ids(mode):
    with pytest.raises(ValueError, match="fragment_mode"):
        AssociationConfig(decision_mode=mode, fragment_mode="off")


@pytest.mark.parametrize("mode", ["geometry", "geometry_visual", "geometry_ambiguity"])
def test_missing_simultaneous_geometry_cannot_merge_transitively_via_later_track(mode):
    cameras = [
        Camera(
            "a",
            width=1000,
            height=1000,
            overlap_with=frozenset({"b"}),
            transition_to=frozenset({"c"}),
        ),
        Camera(
            "b",
            width=1000,
            height=1000,
            transition_to=frozenset({"c"}),
        ),
        Camera("c", width=1000, height=1000),
    ]
    rows = segment("a", "A", 0) + segment("b", "B", 0) + segment("c", "C", 3)
    engine = AssociationEngine(
        cameras,
        AssociationConfig(
            decision_mode=mode,
            fragment_mode="rebuild",
            offline_reconcile=False,
        ),
    )

    def provider(left, right):
        if {left.camera_id, right.camera_id} == {"a", "b"}:
            return None, {"reason": "missing_calibration"}
        return 0.9, {"reason": "route_supported"}

    engine.boundary_provider = provider
    assignments, _ = engine.run(rows)
    ids = {
        local: {row["global_track_id"] for row in assignments if row["local_track_id"] == local}
        for local in ["A", "B", "C"]
    }

    assert ids["A"] != ids["B"]
    assert any(
        row["reason"] == "simultaneous_geometry_unavailable"
        for row in engine.fragment_report["constraints"]
    )


@pytest.mark.parametrize("mode", ["geometry", "geometry_visual"])
def test_geometry_fragmentation_does_not_depend_on_embeddings(mode):
    config = AssociationConfig(decision_mode=mode, fragment_mode="rebuild")
    prefix = [obs("a", "same", t, 100) for t in [0, 0.2, 0.4]]
    suffix = [obs("a", "same", t, 900, (0.0, 1.0)) for t in [0.6, 0.8, 1.0]]
    with_appearance = fragments_from_observations(prefix + suffix, config, [])
    without_appearance = fragments_from_observations(
        [replace(row, embedding=None, embedding_model_id=None) for row in prefix + suffix],
        config,
        [],
    )

    assert [row.references for row in with_appearance] == [
        row.references for row in without_appearance
    ]
    assert [row.split_reason for row in with_appearance] == [
        "local_track",
        "sustained_motion_jump",
    ]
