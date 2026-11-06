from __future__ import annotations

from src.association import (
    AssociationConfig,
    AssociationEngine,
    Camera,
    ExitEvent,
    Observation,
)


def cameras() -> list[Camera]:
    return [
        Camera(
            "cam1", overlap_with=frozenset({"cam2"}), transition_to=frozenset({"cam3"})
        ),
        Camera(
            "cam2", overlap_with=frozenset({"cam1"}), transition_to=frozenset({"cam3"})
        ),
        Camera("cam3"),
    ]


def observation(
    camera_id: str,
    local_id: str,
    timestamp: float,
    embedding: tuple[float, ...] | None = (1.0, 0.0),
    *,
    frame_index: int | None = None,
    confidence: float = 0.9,
    model: str = "reid-v1",
) -> Observation:
    return Observation(
        camera_id=camera_id,
        frame_index=frame_index if frame_index is not None else round(timestamp * 25),
        timestamp_s=timestamp,
        local_track_id=local_id,
        bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
        detection_confidence=confidence,
        embedding=embedding,
        embedding_model_id=model if embedding is not None else None,
    )


def ids(assignments: list[dict[str, object]]) -> list[str]:
    return [str(item["global_track_id"]) for item in assignments]


def test_overlap_allows_same_id_at_the_same_time() -> None:
    observations = [
        observation("cam1", "a", 1.0, (1.0, 0.0)),
        observation("cam2", "b", 1.0, (0.99, 0.01)),
    ]

    assignments, tracks = AssociationEngine(cameras()).run(observations)

    assert ids(assignments) == ["g1", "g1"]
    assert assignments[1]["reason"] == "overlap_reid"
    assert tracks == [
        {
            "global_track_id": "g1",
            "start_s": 1.0,
            "last_seen_s": 1.0,
            "end_s": 1.0,
            "status": "closed",
            "camera_ids": ["cam1", "cam2"],
            "close_reason": "end_of_video",
        }
    ]


def test_directed_transition_allows_sequential_match() -> None:
    observations = [
        observation("cam1", "a", 1.0, (1.0, 0.0)),
        observation("cam3", "c", 4.0, (0.98, 0.02)),
    ]

    assignments, _ = AssociationEngine(cameras()).run(observations)

    assert ids(assignments) == ["g1", "g1"]
    assert assignments[1]["reason"] == "transition_reid"


def test_transition_is_not_implicitly_reversed() -> None:
    observations = [
        observation("cam3", "c", 1.0),
        observation("cam1", "a", 2.0),
    ]

    assignments, _ = AssociationEngine(cameras()).run(observations)

    assert ids(assignments) == ["g1", "g2"]


def test_configured_circular_transition_can_return_to_a_previously_seen_camera() -> (
    None
):
    circular_cameras = [
        Camera("cam1", transition_to=frozenset({"cam2"})),
        Camera("cam2", transition_to=frozenset({"cam1"})),
    ]
    observations = [
        observation("cam1", "a", 1.0),
        observation("cam2", "b", 2.0, (0.99, 0.01)),
        observation("cam1", "a-new", 3.0, (0.98, 0.02)),
    ]

    assignments, _ = AssociationEngine(circular_cameras).run(observations)

    assert ids(assignments) == ["g1", "g1", "g1"]


def test_two_local_tracks_on_one_camera_never_merge_by_appearance() -> None:
    observations = [
        observation("cam1", "person-a", 1.0, (1.0, 0.0), frame_index=25),
        observation("cam1", "person-b", 1.0, (1.0, 0.0), frame_index=25),
    ]

    assignments, _ = AssociationEngine(cameras()).run(observations)

    assert ids(assignments) == ["g1", "g2"]


def test_recovery_at_timeout_boundary_is_inclusive() -> None:
    observations = [
        observation("cam1", "a", 0.0),
        observation("cam3", "c", 10.0, (0.99, 0.01)),
    ]

    assignments, _ = AssociationEngine(cameras()).run(observations)

    assert ids(assignments) == ["g1", "g1"]


def test_gap_after_timeout_creates_new_id_and_does_not_reuse_closed_id() -> None:
    observations = [
        observation("cam1", "a", 0.0),
        observation("cam3", "c", 10.001, (0.99, 0.01)),
    ]

    assignments, tracks = AssociationEngine(cameras()).run(observations)

    assert ids(assignments) == ["g1", "g2"]
    assert tracks[0]["end_s"] == 0.0
    assert tracks[0]["close_reason"] == "timeout"


def test_local_track_continuity_updates_global_last_seen() -> None:
    observations = [
        observation("cam1", "a", 0.0),
        observation("cam1", "a", 9.0, (0.98, 0.02)),
        observation("cam3", "c", 18.5, (0.99, 0.01)),
    ]

    assignments, tracks = AssociationEngine(cameras()).run(observations)

    assert ids(assignments) == ["g1", "g1", "g1"]
    assert assignments[1]["reason"] == "local_track_continuity"
    assert tracks[0]["last_seen_s"] == 18.5


def test_explicit_exit_closes_track_at_last_observation() -> None:
    observations = [
        observation("cam1", "a", 1.0),
        observation("cam3", "c", 3.0, (0.99, 0.01)),
        observation("cam2", "b", 5.0, (0.99, 0.01)),
    ]
    exits = [ExitEvent("cam3", "c", 4.0)]

    assignments, tracks = AssociationEngine(cameras()).run(observations, exits)

    assert ids(assignments) == ["g1", "g1", "g2"]
    assert tracks[0]["end_s"] == 3.0
    assert tracks[0]["close_reason"] == "explicit_exit"


def test_embedding_models_are_not_mixed() -> None:
    observations = [
        observation("cam1", "a", 1.0, model="model-a"),
        observation("cam2", "b", 1.0, model="model-b"),
    ]

    assignments, _ = AssociationEngine(cameras()).run(observations)

    assert ids(assignments) == ["g1", "g2"]


def test_ambiguous_similarity_is_left_unconfirmed() -> None:
    observations = [
        observation("cam1", "a", 1.0, (1.0, 0.00), frame_index=25),
        observation("cam1", "b", 1.0, (1.0, 0.02), frame_index=25),
        observation("cam2", "x", 1.1, (1.0, 0.01), frame_index=28),
    ]

    assignments, _ = AssociationEngine(cameras()).run(observations)

    assert ids(assignments) == ["g1", "g2", "g3"]


def test_without_embeddings_requires_explicit_baseline() -> None:
    observations = [
        observation("cam1", "a", 1.0, None),
        observation("cam2", "b", 1.0, None),
    ]

    strict_assignments, _ = AssociationEngine(cameras()).run(observations)
    baseline_assignments, _ = AssociationEngine(
        cameras(), AssociationConfig(allow_topology_only_baseline=True)
    ).run(observations)

    assert ids(strict_assignments) == ["g1", "g2"]
    assert ids(baseline_assignments) == ["g1", "g1"]
    assert baseline_assignments[1]["reason"] == "overlap_topology_baseline"


def test_baseline_refuses_ambiguous_topology_only_match() -> None:
    observations = [
        observation("cam1", "a", 1.0, None, frame_index=25),
        observation("cam1", "b", 1.0, None, frame_index=25),
        observation("cam2", "x", 1.0, None, frame_index=25),
    ]

    assignments, _ = AssociationEngine(
        cameras(), AssociationConfig(allow_topology_only_baseline=True)
    ).run(observations)

    assert ids(assignments) == ["g1", "g2", "g3"]


def test_output_is_deterministic_for_unsorted_input() -> None:
    ordered = [
        observation("cam1", "a", 1.0),
        observation("cam2", "b", 1.0, (0.99, 0.01)),
        observation("cam3", "c", 2.0, (0.98, 0.02)),
    ]

    first = AssociationEngine(cameras()).run(ordered)
    second = AssociationEngine(cameras()).run(reversed(ordered))

    assert first == second
