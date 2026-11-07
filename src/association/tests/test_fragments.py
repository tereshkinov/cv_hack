from dataclasses import replace

import pytest

from src.association import AssociationConfig, AssociationEngine, ExitEvent
from src.association.confirmation import confirmation_time
from src.association.fragments import fragments_from_observations
from src.association.tests.test_geolocation import obs, setup


def group(rows, local):
    return {r["global_track_id"] for r in rows if r["local_track_id"] == local}


def test_confirmation_needs_distinct_coherent_times_and_preserves_short_real_appearances():
    cfg = AssociationConfig(confirm_tracks=True)
    assert confirmation_time([obs("a", "one", 0, 100)], cfg)[0] is None
    assert (
        confirmation_time([obs(c, "one", 0, 100) for c in ["a", "b", "c"]], cfg)[0]
        is None
    )
    assert (
        confirmation_time([obs("a", "one", t, 100) for t in [0, 0.2, 0.4]], cfg)[0]
        == 0.4
    )
    assert (
        confirmation_time([obs("a", "one", t, 100) for t in [0, 1, 2]], cfg)[0] is None
    )
    assert (
        confirmation_time(
            [obs("a", "one", t, x) for t, x in [(0, 100), (0.2, 900), (0.4, 100)]], cfg
        )[0]
        is None
    )
    cams, project = setup()
    rows = [obs("a", "brief", 0, 100)]
    assignments, tracks = AssociationEngine(cams, cfg, project).run(rows)
    assert len(assignments) == 1 and len(tracks) == 1
    assert tracks[0]["confirmation_state"] == "candidate"
    assert assignments[0]["observed_state"] == "candidate"


def test_short_fragment_recovers_confirmed_episode_before_filtering():
    cams, project = setup()
    rows = [obs("a", "old", t, 100) for t in [0, 0.2, 0.4]] + [
        obs("a", "one_frame", 1, 105)
    ]
    cfg = AssociationConfig(fragment_mode="rebuild", confirm_tracks=True, timeout_s=15)
    assignments, tracks = AssociationEngine(cams, cfg, project).run(rows)
    assert len(tracks) == 1 and len(assignments) == 4
    assert tracks[0]["confirmation_state"] == "confirmed"
    assert tracks[0]["confirmed_at_s"] == 0.4 and tracks[0]["start_s"] == 0
    assert assignments[0]["observed_state"] == "candidate"
    assert assignments[-1]["observed_state"] == "confirmed"
    assert {r["local_track_id"] for r in assignments} == {"old", "one_frame"}


def test_graph_can_revise_greedy_choice_using_joint_competition():
    cams, project = setup()
    rows = []
    for local, start, x in [("A", 0, 100), ("X", 0, 800), ("B", 1, 100), ("C", 1, 800)]:
        rows += [obs("a", local, start + t, x) for t in [0, 0.2, 0.4]]
    engine = AssociationEngine(
        cams,
        AssociationConfig(fragment_mode="rebuild", tracklet_threshold=0.62),
        project,
    )
    scores = {
        frozenset(("A", "B")): 0.91,
        frozenset(("A", "C")): 0.90,
        frozenset(("X", "B")): 0.89,
    }
    engine.boundary_provider = lambda a, b: (
        scores.get(
            frozenset(
                (a.observations[0].local_track_id, b.observations[0].local_track_id)
            )
        ),
        {"reason": "technical-test"},
    )
    assignments, tracks = engine.run(rows)
    assert group(assignments, "A") == group(assignments, "C")
    assert group(assignments, "X") == group(assignments, "B")
    assert group(assignments, "A") != group(assignments, "B")
    assert len(tracks) == 2 and engine.fragment_report["optimality_certified"] is False


def test_near_equal_alternatives_are_deferred_and_transitive_conflict_is_forbidden():
    cams, project = setup()
    rows = [obs("a", "A", t, 100) for t in [0, 0.2, 0.4]]
    for local, x in [("B", 100), ("C", 800)]:
        rows += [obs("a", local, t, x) for t in [1, 1.2, 1.4]]
    engine = AssociationEngine(
        cams,
        AssociationConfig(
            fragment_mode="rebuild", tracklet_threshold=0.62, ambiguity_margin=0.03
        ),
        project,
    )
    scores = {frozenset(("A", "B")): 0.90, frozenset(("A", "C")): 0.89}
    engine.boundary_provider = lambda a, b: (
        scores.get(
            frozenset(
                (a.observations[0].local_track_id, b.observations[0].local_track_id)
            )
        ),
        {},
    )
    assignments, tracks = engine.run(rows)
    assert len(tracks) == 3
    assert group(assignments, "B") != group(assignments, "C")
    assert engine.fragment_report["near_optimal_alternatives"] >= 2


def test_rebuild_can_split_wrong_preliminary_camera_merge():
    cams, project = setup()
    # Original identical clothes merge these overlapping camera views.
    rows = [
        obs(c, local, t, x)
        for c, local, x in [("a", "A", 100), ("b", "B", 110)]
        for t in [0, 0.2, 0.4]
    ]
    baseline = AssociationEngine(cams, AssociationConfig(), project).run(rows)
    assert len(baseline[1]) == 1
    engine = AssociationEngine(
        cams, AssociationConfig(fragment_mode="rebuild"), project
    )
    engine.boundary_provider = lambda a, b: (
        0.3,
        {"reason": "independent-contradiction"},
    )
    assignments, tracks = engine.run(rows)
    assert len(tracks) == 2
    assert all("provisional_global_track_id" in a for a in assignments)


def test_persistent_jump_splits_virtual_fragment_but_turn_or_single_bad_frame_does_not():
    cfg = AssociationConfig()
    prefix = [obs("a", "same", t, 100) for t in [0, 0.2, 0.4]]
    suffix = [obs("a", "same", t, 900, (0.0, 1.0)) for t in [0.6, 0.8, 1.0]]
    fragments = fragments_from_observations(prefix + suffix, cfg, [])
    assert len(fragments) == 2
    assert fragments[1].split_reason == "sustained_motion_and_appearance_jump"
    turned = [replace(o, bbox_xyxy=prefix[0].bbox_xyxy) for o in suffix]
    assert len(fragments_from_observations(prefix + turned, cfg, [])) == 1
    noisy = [suffix[0]] + turned[1:]
    assert len(fragments_from_observations(prefix + noisy, cfg, [])) == 1
    assert sum(len(f.references) for f in fragments) == 6


def test_exit_timeout_and_simultaneous_views_preserve_lifecycle():
    cams, project = setup()
    cfg = AssociationConfig(fragment_mode="rebuild", confirm_tracks=True, timeout_s=15)
    rows = [obs("a", "old", t, 100) for t in [0, 0.2, 0.4]] + [
        obs("b", "new", t, 100) for t in [16, 16.2, 16.4]
    ]
    assert len(AssociationEngine(cams, cfg, project).run(rows)[1]) == 2
    rows = [obs("a", "old", t, 100) for t in [0, 0.2, 0.4]] + [
        obs("b", "new", t, 100) for t in [0.6, 0.8, 1.0]
    ]
    assignments, tracks = AssociationEngine(cams, cfg, project).run(
        rows, [ExitEvent("a", "old", 0.4)]
    )
    assert len(tracks) == 2 and group(assignments, "old") != group(assignments, "new")
    rows = [
        obs(c, local, t, 100)
        for c, local in [("a", "A"), ("b", "B")]
        for t in [0, 0.2, 0.4]
    ]
    assignments, tracks = AssociationEngine(cams, cfg, project).run(rows)
    assert len(tracks) == 1 and tracks[0]["confirmed_at_s"] == 0.4


@pytest.mark.parametrize(
    "options",
    [
        {"fragment_mode": "unknown"},
        {"confirmation_min_observations": True},
        {"confirmation_window_s": 0.2},
        {"boundary_window_s": float("nan")},
        {"fragment_beam_width": 1},
    ],
)
def test_invalid_fragment_configuration(options):
    with pytest.raises((ValueError, TypeError)):
        AssociationConfig(**options)


def test_trusted_spatial_contradiction_cannot_be_bridged_transitively():
    from src.association import Camera

    cams = [
        Camera(c, overlap_with=frozenset({"a", "b", "c"} - {c}))
        for c in ["a", "b", "c"]
    ]
    rows = [obs(c, c, t, 100) for c in ["a", "b", "c"] for t in [0, 0.2, 0.4]]
    engine = AssociationEngine(
        cams, AssociationConfig(fragment_mode="rebuild", tracklet_threshold=0.62)
    )

    def provider(left, right):
        if {left.camera_id, right.camera_id} == {"a", "c"}:
            return 0.58, {"reason": "spatial_contradiction", "geometry_quality": 0.9}
        return 0.95, {"reason": "supported"}

    engine.boundary_provider = provider
    assignments, _ = engine.run(rows)
    assert group(assignments, "a") != group(assignments, "c")
    assert any(
        c["reason"] == "reliable_spatial_contradiction"
        for c in engine.fragment_report["constraints"]
    )


@pytest.mark.parametrize("fragment_mode", ["off", "boundary", "rebuild"])
@pytest.mark.parametrize("gap,expected", [(59.999, 1), (60, 1), (60.001, 2)])
def test_minute_retention_inclusive_without_extrapolating_motion(
    fragment_mode, gap, expected
):
    cams, project = setup()
    # After a minute a person can return elsewhere in the same camera.
    rows = [obs("a", "old", t, 100) for t in [0, 0.2, 0.4]]
    rows += [obs("a", "returned", 0.4 + gap + t, 800) for t in [0, 0.2, 0.4]]
    cfg = AssociationConfig(fragment_mode=fragment_mode, confirm_tracks=True)
    assignments, tracks = AssociationEngine(cams, cfg, project).run(rows)
    assert len(tracks) == expected
    assert len(assignments) == len(rows)
    if expected == 1:
        assert group(assignments, "old") == group(assignments, "returned")


def test_minute_retention_does_not_override_explicit_exit_or_appearance():
    cams, project = setup()
    prefix = [obs("a", "old", t, 100) for t in [0, 0.2, 0.4]]
    suffix = [obs("a", "return", 30 + t, 100) for t in [0, 0.2, 0.4]]
    cfg = AssociationConfig(fragment_mode="rebuild", confirm_tracks=True)
    assert (
        len(
            AssociationEngine(cams, cfg, project).run(
                prefix + suffix, [ExitEvent("a", "old", 0.4)]
            )[1]
        )
        == 2
    )
    suffix = [replace(o, embedding=(0.0, 1.0)) for o in suffix]
    assert len(AssociationEngine(cams, cfg, project).run(prefix + suffix)[1]) == 2


def test_simultaneous_views_do_not_hide_a_coherent_single_camera_path():
    rows = [
        obs("a", "A", 0, 100),
        obs("b", "B", 0, 100),
        obs("a", "A", 0.2, 100),
        obs("a", "A", 0.4, 100),
    ]
    assert confirmation_time(rows, AssociationConfig(confirm_tracks=True))[0] == 0.4
