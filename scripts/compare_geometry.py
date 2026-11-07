"""Build a reviewable comparison of geometry-first association runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reference(row):
    return row["camera_id"], row["frame_index"], row["local_track_id"]


def _partition(run: Path):
    return {
        _reference(row): row["global_track_id"]
        for row in _jsonl(run / "assignments.jsonl")
    }


def _run_summary(run: Path):
    tracks = _json(run / "tracks.json")["tracks"]
    graph = _json(run / "fragment_graph.json")
    constraints = Counter(row["reason"] for row in graph["constraints"])
    return {
        "observations": len(_partition(run)),
        "global_tracks": len(tracks),
        "single_camera_tracks": sum(len(row["camera_ids"]) == 1 for row in tracks),
        "multi_camera_tracks": sum(len(row["camera_ids"]) > 1 for row in tracks),
        "tracks_at_most_1s": sum(
            row["end_s"] - row["start_s"] <= 1.0 for row in tracks
        ),
        "tracks_at_most_3s": sum(
            row["end_s"] - row["start_s"] <= 3.0 for row in tracks
        ),
        "candidate_edges": graph["candidate_edges"],
        "selected_edges": graph["selected_edges"],
        "constraints": dict(sorted(constraints.items())),
    }


def _choose2(value):
    return value * (value - 1) // 2


def _partition_change(left, right):
    if left.keys() != right.keys():
        raise ValueError("Runs must contain exactly the same observation references")
    contingency = Counter((left[key], right[key]) for key in left)
    left_sizes = Counter(left.values())
    right_sizes = Counter(right.values())
    common = sum(_choose2(value) for value in contingency.values())
    return {
        "observation_pairs_unlinked_by_visual_confirmation": (
            sum(_choose2(value) for value in left_sizes.values()) - common
        ),
        "observation_pairs_linked_by_visual_confirmation": (
            sum(_choose2(value) for value in right_sizes.values()) - common
        ),
    }


def _edge_map(run: Path):
    graph = _json(run / "fragment_graph.json")
    return {(row["left"], row["right"]): row for row in graph["edges"]}


def _review_rows(geometry_run: Path, visual_run: Path):
    geometry = _edge_map(geometry_run)
    visual = _edge_map(visual_run)
    keys = {
        key
        for key in geometry.keys() | visual.keys()
        if geometry.get(key, {}).get("selected") or visual.get(key, {}).get("selected")
    }
    rows = []
    for key in sorted(keys):
        pure = geometry.get(key, {})
        confirmed = visual.get(key, {})
        source = confirmed or pure
        boundary = source.get("boundary_geometry") or {}
        rows.append(
            {
                "left_fragment": key[0],
                "right_fragment": key[1],
                "geometry_selected": bool(pure.get("selected")),
                "geometry_visual_selected": bool(confirmed.get("selected")),
                "geometry_score": pure.get("score"),
                "geometry_visual_score": confirmed.get("score"),
                "geometry_value": source.get("geometry_value"),
                "geometry_quality": source.get("geometry_quality"),
                "appearance": confirmed.get("appearance"),
                "appearance_quality": confirmed.get("appearance_quality"),
                "visual_adjustment": confirmed.get("visual_adjustment", 0.0),
                "geometry_reason": boundary.get("reason"),
                "time_delta_s": boundary.get("time_delta_s"),
                "human_same_person": "",
                "comment": "",
            }
        )
    return rows


def compare(geometry_run: Path, visual_run: Path, output: Path):
    pure_partition = _partition(geometry_run)
    visual_partition = _partition(visual_run)
    summary = {
        "geometry_run": str(geometry_run),
        "geometry_visual_run": str(visual_run),
        "geometry": _run_summary(geometry_run),
        "geometry_visual": _run_summary(visual_run),
        **_partition_change(pure_partition, visual_partition),
        "accuracy_measured": False,
        "note": (
            "Track count and partition changes are diagnostic only. Fill "
            "human_same_person in edge_review.csv before judging accuracy."
        ),
    }
    review = _review_rows(geometry_run, visual_run)
    summary["review_edges"] = len(review)

    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = [
        "left_fragment",
        "right_fragment",
        "geometry_selected",
        "geometry_visual_selected",
        "geometry_score",
        "geometry_visual_score",
        "geometry_value",
        "geometry_quality",
        "appearance",
        "appearance_quality",
        "visual_adjustment",
        "geometry_reason",
        "time_delta_s",
        "human_same_person",
        "comment",
    ]
    with (output / "edge_review.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-visual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    compare(arguments.geometry, arguments.geometry_visual, arguments.output)
