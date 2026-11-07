import json

import pytest

from scripts.compare_geometry import compare


def write_run(path, assignments, tracks, edges, constraints=()):
    path.mkdir()
    (path / "assignments.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in assignments), encoding="utf-8"
    )
    (path / "tracks.json").write_text(
        json.dumps({"tracks": tracks}), encoding="utf-8"
    )
    (path / "fragment_graph.json").write_text(
        json.dumps(
            {
                "candidate_edges": len(edges),
                "selected_edges": sum(row["selected"] for row in edges),
                "edges": edges,
                "constraints": list(constraints),
            }
        ),
        encoding="utf-8",
    )


def assignment(frame, global_id):
    return {
        "camera_id": "cam1",
        "frame_index": frame,
        "local_track_id": f"local-{frame}",
        "global_track_id": global_id,
    }


def track(global_id, cameras, start, end):
    return {
        "global_track_id": global_id,
        "camera_ids": cameras,
        "start_s": start,
        "end_s": end,
    }


def edge(score, selected, adjustment=0.0):
    return {
        "left": "f1",
        "right": "f2",
        "score": score,
        "selected": selected,
        "geometry_value": 0.8,
        "geometry_quality": 0.5,
        "appearance": 0.9,
        "appearance_quality": 0.8,
        "visual_adjustment": adjustment,
        "boundary_geometry": {
            "reason": "route:route_time_expected",
            "time_delta_s": 4.0,
        },
    }


def test_compare_writes_diagnostic_summary_and_review_csv(tmp_path):
    pure = tmp_path / "pure"
    visual = tmp_path / "visual"
    output = tmp_path / "comparison"
    write_run(
        pure,
        [assignment(1, "g1"), assignment(2, "g2")],
        [track("g1", ["cam1"], 0, 0), track("g2", ["cam1"], 2, 2)],
        [edge(0.61, False)],
        [{"reason": "simultaneous_geometry_unavailable"}],
    )
    write_run(
        visual,
        [assignment(1, "g1"), assignment(2, "g1")],
        [track("g1", ["cam1", "cam2"], 0, 2)],
        [edge(0.65, True, 0.04)],
    )

    summary = compare(pure, visual, output)

    assert summary["geometry"]["global_tracks"] == 2
    assert summary["geometry_visual"]["global_tracks"] == 1
    assert summary["observation_pairs_linked_by_visual_confirmation"] == 1
    assert summary["accuracy_measured"] is False
    assert "human_same_person" in (output / "edge_review.csv").read_text()


def test_compare_rejects_different_observation_sets(tmp_path):
    pure = tmp_path / "pure"
    visual = tmp_path / "visual"
    write_run(pure, [assignment(1, "g1")], [track("g1", ["cam1"], 0, 0)], [])
    write_run(visual, [assignment(2, "g1")], [track("g1", ["cam1"], 0, 0)], [])

    with pytest.raises(ValueError, match="same observation references"):
        compare(pure, visual, tmp_path / "output")
