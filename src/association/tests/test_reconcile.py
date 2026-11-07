"""Regression cases for delayed evidence and conservative offline correction."""

from dataclasses import replace

from src.association import AssociationConfig, AssociationEngine, Camera, ExitEvent

from .test_engine import cameras, observation


def segment(camera, local, start, vector=(1.0, 0.0), model="reid-v1"):
    return [
        observation(camera, local, start + i * 0.5, vector, model=model)
        for i in range(3)
    ]


def delayed_target(camera="cam2", start=3.0, vector=(1.0, 0.0), model="reid-v1"):
    return [observation(camera, "target", start, None)] + segment(
        camera, "target", start + 0.5, vector, model
    )


def route_project():
    return {
        "coordinate_units": "meters",
        "room": {"width_m": 20, "height_m": 10},
        "cameras": [{"id": "cam1"}, {"id": "cam3"}],
        "processing": {"fusion_distance_m": 1, "sync_uncertainty_s": 0.2},
        "motion_model": {
            "nodes": [
                {"id": "a", "point": [0, 0]},
                {"id": "b", "point": [10, 0]},
            ],
            "edges": [{"a": "a", "b": "b"}],
            "camera_gates": {
                "cam1": {"both": ["a"]},
                "cam3": {"both": ["b"]},
            },
            "walking_speed_m_s": {
                "min": 0.5,
                "typical": 1.25,
                "max": 2.5,
                "sigma": 0.25,
            },
            "timing_sigma_s": 0.5,
            "route_uncertainty_fraction": 0.05,
            "distance_reliability": 0.9,
        },
    }


def route_cameras():
    return [
        Camera(
            "cam1",
            width=1000,
            height=1000,
            transition_to=frozenset({"cam3"}),
        ),
        Camera("cam3", width=1000, height=1000),
    ]


def test_delayed_embeddings_revise_all_prior_target_assignments():
    rows = segment("cam1", "source", 0) + delayed_target()
    before, _ = AssociationEngine(
        cameras(), AssociationConfig(offline_reconcile=False)
    ).run(rows)
    assert len({r["global_track_id"] for r in before}) == 2
    engine = AssociationEngine(cameras())
    after, tracks = engine.run(rows)
    assert len(tracks) == 1
    assert {r["global_track_id"] for r in after} == {"g1"}
    assert (
        after[3]["reason"] == "offline_multiframe_reid"
    )  # first frame had no embedding
    assert len(engine.reconciliation_log) == 1
    assert tracks[0]["camera_ids"] == ["cam1", "cam2"]


def test_single_lucky_frame_cannot_overrule_most_inconsistent_views():
    rows = segment("cam1", "source", 0) + delayed_target(vector=(0.0, 1.0))
    rows[-1] = replace(rows[-1], embedding=(1.0, 0.0))
    _, tracks = AssociationEngine(cameras()).run(rows)
    assert len(tracks) == 2


def test_equal_looking_competing_people_are_not_forced_together():
    rows = segment("cam1", "a", 0) + segment("cam1", "b", 0) + delayed_target()
    _, tracks = AssociationEngine(cameras()).run(rows)
    assert len(tracks) == 3


def test_distinct_appearance_tracks_remain_separate():
    rows = segment("cam1", "a", 0, (1.0, 0.0)) + segment("cam1", "b", 0, (0.0, 1.0))
    rows += delayed_target(vector=(1.0, 0.0))
    assignments, tracks = AssociationEngine(cameras()).run(rows)
    aid = {
        r["global_track_id"]
        for r in assignments
        if r["local_track_id"] in {"a", "target"}
    }
    bid = {r["global_track_id"] for r in assignments if r["local_track_id"] == "b"}
    assert len(tracks) == 2 and len(aid) == 1 and aid.isdisjoint(bid)


def test_offline_correction_respects_timeout_and_explicit_exit():
    source = segment("cam1", "a", 0)
    _, tracks = AssociationEngine(cameras()).run(source + delayed_target(start=12))
    assert len(tracks) == 2
    _, tracks = AssociationEngine(cameras()).run(
        source + delayed_target(), [ExitEvent("cam1", "a", 1.1)]
    )
    assert len(tracks) == 2
    _, tracks = AssociationEngine(cameras()).run(source + delayed_target(start=11))
    assert len(tracks) == 1  # exactly 10s between last source and first target


def test_offline_correction_respects_direction_and_feature_space():
    _, tracks = AssociationEngine(cameras()).run(
        segment("cam3", "a", 0) + delayed_target("cam1")
    )
    assert len(tracks) == 2


def test_small_transition_overlap_tolerates_clock_and_tracker_tail_only():
    source = segment("cam1", "source", 0)
    slightly_overlapping = delayed_target("cam3", start=0.8)

    _, strict = AssociationEngine(
        cameras(), AssociationConfig(transition_overlap_tolerance_s=0)
    ).run(source + slightly_overlapping)
    _, tolerant = AssociationEngine(
        cameras(), AssociationConfig(transition_overlap_tolerance_s=0.25)
    ).run(source + slightly_overlapping)

    assert len(strict) == 2
    assert len(tolerant) == 1

    long_overlap = delayed_target("cam3", start=0.2)
    _, tracks = AssociationEngine(
        cameras(), AssociationConfig(transition_overlap_tolerance_s=0.25)
    ).run(source + long_overlap)
    assert len(tracks) == 2

    reverse = delayed_target("cam1", start=0.8)
    _, tracks = AssociationEngine(
        cameras(), AssociationConfig(transition_overlap_tolerance_s=0.25)
    ).run(segment("cam3", "source", 0) + reverse)
    assert len(tracks) == 2
    _, tracks = AssociationEngine(cameras()).run(
        segment("cam1", "a", 0) + delayed_target(model="other")
    )
    assert len(tracks) == 2
    _, tracks = AssociationEngine(cameras()).run(
        segment("cam1", "a", 0) + delayed_target(vector=(1.0, 0.0, 0.0))
    )
    assert len(tracks) == 2


def test_reconciliation_is_deterministic_and_recovers_same_camera_fragments():
    rows = segment("cam1", "a", 0) + delayed_target()
    assert AssociationEngine(cameras()).run(rows) == AssociationEngine(cameras()).run(
        reversed(rows)
    )
    rows = segment("cam1", "a", 0) + delayed_target("cam1")
    _, tracks = AssociationEngine(cameras()).run(rows)
    assert len(tracks) == 1


def test_offline_recovery_cannot_merge_simultaneous_same_camera_people():
    rows = segment("cam1", "a", 0) + segment("cam1", "b", 0.5)
    assert len(AssociationEngine(cameras()).run(rows)[1]) == 2


def test_offline_recovery_rejects_far_fragments_without_first_embedding():
    rows = segment("cam1", "a", 0) + [
        replace(o, bbox_xyxy=(2000, 20, 2100, 200)) for o in delayed_target("cam1")
    ]
    assert len(AssociationEngine(cameras()).run(rows)[1]) == 2


def test_route_time_can_rescue_a_borderline_visual_transition():
    config = AssociationConfig(
        similarity_threshold=0.95,
        tracklet_threshold=0.62,
        adjacency_bonus=0,
    )
    rows = segment("cam1", "source", 0, (1.0, 0.0)) + segment(
        "cam3", "target", 9.0, (0.6, 0.8)
    )

    _, visual_only = AssociationEngine(route_cameras(), config).run(rows)
    engine = AssociationEngine(route_cameras(), config, route_project())
    _, with_route = engine.run(rows)

    assert len(visual_only) == 2
    assert len(with_route) == 1
    evidence = engine.reconciliation_log[0]["boundary_evidence"]
    assert evidence["reason"] == "route:route_time_expected"
    assert evidence["details"]["distance_m"] == 10


def test_physically_impossible_route_can_block_a_visual_false_merge():
    config = AssociationConfig(
        similarity_threshold=0.95,
        tracklet_threshold=0.62,
        adjacency_bonus=0,
    )
    rows = segment("cam1", "source", 0, (1.0, 0.0)) + segment(
        "cam3", "target", 2.0, (0.65, 0.7599342077)
    )

    _, visual_only = AssociationEngine(route_cameras(), config).run(rows)
    engine = AssociationEngine(route_cameras(), config, route_project())
    _, with_route = engine.run(rows)

    assert len(visual_only) == 1
    assert len(with_route) == 2
