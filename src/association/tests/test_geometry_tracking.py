from dataclasses import replace

import pytest

from src.association import AssociationConfig, AssociationEngine, ExitEvent
from src.floor_mesh import calibrate_mesh

from .test_geolocation import obs, setup


def run(rows, cfg=None, project=None):
    cams, p = setup()
    engine = AssociationEngine(
        cams,
        cfg or AssociationConfig(decision_mode="geometry", tracklet_threshold=0.62),
        project or p,
    )
    assignments, tracks = engine.run(rows)
    return assignments, tracks, engine


def series(cam, local, x, vector=None, start=0):
    return [obs(cam, local, start + t, x, vector) for t in [0, 0.2, 0.4]]


def gid(assignments, cam, local):
    return {
        r["global_track_id"]
        for r in assignments
        if r["camera_id"] == cam and r["local_track_id"] == local
    }


def test_geometry_matches_unique_position_without_appearance_or_with_opposite_clothing():
    for a, b in [(None, None), ((1.0, 0.0), (0.0, 1.0))]:
        rows = series("a", "one", 100, a) + series("b", "other", 101, b)
        assignments, tracks, engine = run(rows)
        assert len(tracks) == 1 and len(assignments) == 6
        assert all(not e["appearance_used"] for e in engine.fragment_report["edges"])


def test_similar_clothing_cannot_override_geometric_contradiction():
    _assignments, tracks, engine = run(
        series("a", "one", 100, (1.0, 0.0)) + series("b", "other", 900, (1.0, 0.0))
    )
    assert len(tracks) == 2
    assert (
        engine.fragment_report["constraints"][0]["reason"]
        == "reliable_spatial_contradiction"
    )


def test_two_nearby_people_use_appearance_only_to_break_ambiguity():
    rows = (
        series("a", "one", 100, (1.0, 0.0))
        + series("a", "two", 105, (0.0, 1.0))
        + series("b", "first", 101, (1.0, 0.0))
        + series("b", "second", 104, (0.0, 1.0))
    )
    assignments, tracks, engine = run(rows)
    assert len(tracks) == 2
    assert gid(assignments, "a", "one") == gid(assignments, "b", "first")
    assert gid(assignments, "a", "two") == gid(assignments, "b", "second")
    assert any(e["appearance_used"] for e in engine.fragment_report["edges"])


def test_unresolved_close_people_are_deferred_without_embedding():
    rows = series("a", "one", 100) + series("a", "two", 105) + series("b", "first", 102)
    _assignments, tracks, engine = run(rows)
    assert len(tracks) == 3
    assert all(
        e["reason"] == "unresolved_geometry" for e in engine.fragment_report["edges"]
    )


def test_floor_projection_works_with_mesh_and_cropped_feet_are_not_invented():
    _cams, p = setup()
    for c in p["cameras"]:
        c["calibration"] = calibrate_mesh(
            [[0, 0], [1000, 0], [1000, 1000], [0, 1000], [500, 500]],
            [[0, 0], [30, 0], [30, 30], [0, 30], [15, 15]],
        )
    rows = series("a", "one", 100) + series("b", "other", 101)
    _, tracks, engine = run(rows, project=p)
    assert len(tracks) == 1
    cropped = replace(rows[0], local_track_id="crop", bbox_xyxy=(90, 800, 110, 1000))
    assert engine.floor.projection(cropped).point is None
    assert engine.floor.projection(cropped).reason == "cropped_feet"


def test_timeout_and_explicit_exit_still_dominate_geometry():
    rows = series("a", "one", 100) + series("a", "back", 100, start=61)
    assert len(run(rows)[1]) == 2
    cams, p = setup()
    engine = AssociationEngine(cams, AssociationConfig(decision_mode="geometry"), p)
    assert (
        len(
            engine.run(
                series("a", "one", 100) + series("a", "back", 100, start=1),
                [ExitEvent("a", "one", 0.5)],
            )[1]
        )
        == 2
    )


def test_same_camera_short_fragment_recovery_survives_missing_floor():
    _cams, p = setup()
    p["cameras"][0]["calibration"] = None
    rows = series("a", "one", 100) + series("a", "next", 103, start=0.8)
    _, tracks, engine = run(rows, project=p)
    assert len(tracks) == 1
    assert (
        engine.fragment_report["edges"][0]["boundary_geometry"]["coordinate_units"]
        == "normalized_image"
    )


def test_overlap_can_be_inferred_from_measured_patches_without_manual_camera_links():
    cams, p = setup()
    cams = [replace(c, overlap_with=frozenset()) for c in cams]
    for c in p["cameras"]:
        c["calibration"] = calibrate_mesh(
            [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
            [[0, 0], [30, 0], [30, 30], [0, 30]],
        )
    engine = AssociationEngine(
        cams, AssociationConfig(decision_mode="geometry", tracklet_threshold=0.62), p
    )
    assignments, tracks = engine.run(
        series("a", "one", 100) + series("b", "other", 101)
    )
    assert len(tracks) == 1 and len(assignments) == 6
    assert len(engine.fragment_report["calibrated_overlap_patches"]) == 1


def test_cropped_silhouette_can_keep_id_by_short_image_bearing_without_fake_floor_position():
    rows = series("a", "one", 100)
    rows += [
        replace(o, bbox_xyxy=(90, 900, 110, 1000))
        for o in series("a", "partial", 100, start=0.8)
    ]
    assignments, tracks, engine = run(rows)
    assert len(tracks) == 1 and len(assignments) == 6
    assert all(engine.floor.point(o) is None for o in rows[3:])
    assert engine.fragment_report["edges"][0]["reason"] == "partial_image_bearing"


@pytest.mark.parametrize("gap,expected", [(59.999, 1), (60, 1), (60.001, 2)])
def test_minute_retention_uses_reachability_and_appearance_without_long_velocity_prediction(
    gap, expected
):
    rows = series("a", "first", 100, (1.0, 0.0)) + series(
        "a", "back", 110, (1.0, 0.0), start=0.4 + gap
    )
    _, tracks, engine = run(rows)
    assert len(tracks) == expected
    if expected == 1:
        edge = engine.fragment_report["edges"][0]
        assert edge["reason"] == "blind_gap_reachable" and edge["appearance_used"]


def test_sustained_local_id_teleport_is_split_geometrically_without_appearance():
    rows = series("a", "one", 100) + series("a", "one", 900, start=0.6)
    _, tracks, engine = run(rows)
    assert len(tracks) == 2
    assert any(
        n["split_reason"] == "sustained_geometry_jump"
        for n in engine.fragment_report["fragments"]
    )


def test_minute_reachable_region_is_not_identity_evidence_for_weak_clothing_match():
    rows = series("a", "first", 100, (1.0, 0.0)) + series(
        "a", "other", 110, (0.65, 0.7599342077), start=30
    )
    _, tracks, engine = run(rows)
    assert len(tracks) == 2
    assert any(
        e["reason"] == "weak_appearance_for_uncertain_geometry"
        for e in engine.fragment_report["edges"]
    )


def test_geometry_assignment_is_reproducible_when_input_order_changes():
    rows = series("b", "other", 101) + series("a", "one", 100)
    first = run(rows)
    second = run(list(reversed(rows)))
    assert first[:2] == second[:2]
    assert first[2].fragment_report == second[2].fragment_report
