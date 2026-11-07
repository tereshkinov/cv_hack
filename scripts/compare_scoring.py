"""Compare partitions without treating a change in ID count as accuracy."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def ref(row):
    return row["camera_id"], row["frame_index"], row["local_track_id"]


def compare(baseline, experiment, output, allow_appearance_change=False):
    a = {
        ref(r): r["global_track_id"]
        for r in rows(baseline / "association/assignments.jsonl")
    }
    b = {
        ref(r): r["global_track_id"]
        for r in rows(experiment / "association/assignments.jsonl")
    }
    if a.keys() != b.keys():
        raise ValueError("Runs must contain exactly the same observation references")
    source_bytes = (baseline / "vision/observations.jsonl").read_bytes()
    experiment_bytes = (experiment / "vision/observations.jsonl").read_bytes()
    appearance_changed = source_bytes != experiment_bytes
    if appearance_changed:
        if not allow_appearance_change:
            raise ValueError("Detections and embeddings differ")
        left_rows = rows(baseline / "vision/observations.jsonl")
        right_rows = rows(experiment / "vision/observations.jsonl")
        if len(left_rows) != len(right_rows):
            raise ValueError("Observation count differs")
        for left, right in zip(left_rows, right_rows, strict=True):
            without_features = lambda row: {
                k: v
                for k, v in row.items()
                if k not in {"embedding", "embedding_model_id"}
            }
            if without_features(left) != without_features(right) or (
                left.get("embedding") is None
            ) != (right.get("embedding") is None):
                raise ValueError(
                    "Only appearance may differ in an appearance comparison"
                )
    if json.loads((baseline / "project.json").read_text()) != json.loads(
        (experiment / "project.json").read_text()
    ):
        raise ValueError("Project snapshots differ; not a controlled comparison")
    if json.loads((baseline / "cameras.json").read_text()) != json.loads(
        (experiment / "cameras.json").read_text()
    ):
        raise ValueError("Camera manifests differ")
    if (
        json.loads((baseline / "job.json").read_text())["options"]
        != json.loads((experiment / "job.json").read_text())["options"]
    ):
        raise ValueError("Processing options differ")
    observations = {ref(o): o for o in rows(baseline / "vision/observations.jsonl")}
    contingency = Counter((a[k], b[k]) for k in a)
    ac, bc = Counter(a.values()), Counter(b.values())
    choose2 = lambda n: n * (n - 1) // 2
    common = sum(choose2(n) for n in contingency.values())
    decision_path = experiment / "association/decisions.jsonl"
    decisions = rows(decision_path) if decision_path.exists() else []
    summary = {
        "appearance_changed": appearance_changed,
        "experiment_observations_sha256": hashlib.sha256(experiment_bytes).hexdigest(),
        "baseline_run": baseline.name,
        "experiment_run": experiment.name,
        "observations": len(a),
        "observations_sha256": hashlib.sha256(
            (baseline / "vision/observations.jsonl").read_bytes()
        ).hexdigest(),
        "baseline_tracks": len(ac),
        "experimental_tracks": len(bc),
        "observation_pairs_unlinked": sum(choose2(n) for n in ac.values()) - common,
        "observation_pairs_newly_linked": sum(choose2(n) for n in bc.values()) - common,
        "candidate_decision_reasons": dict(Counter(r["reason"] for r in decisions)),
        "gallery_updates_blocked": json.loads(
            (experiment / "association/reconciliation.json").read_text()
        )["gallery_updates_blocked"],
        "accuracy_measured": False,
        "note": "Partition changes are not errors or improvements without human labels. Support acceptance in decisions.jsonl does not imply final assignment; selected_online records online selection only.",
    }
    # Group distinct local episodes by both partitions; IDs are display labels only.
    groups = defaultdict(list)
    for key in a:
        groups[(key[0], key[2], a[key], b[key])].append(
            observations[key]["timestamp_s"]
        )
    segments = [
        (key, min(times), max(times), len(times)) for key, times in groups.items()
    ]
    cases = []
    for i, (x, start, end, count) in enumerate(segments):
        for y, start2, end2, count2 in segments[i + 1 :]:
            same_a, same_b = x[2] == y[2], x[3] == y[3]
            if same_a == same_b or max(0, start2 - end, start - end2) > 15:
                continue
            cases.append(
                {
                    "camera_a": x[0],
                    "local_a": x[1],
                    "start_a_s": start,
                    "end_a_s": end,
                    "camera_b": y[0],
                    "local_b": y[1],
                    "start_b_s": start2,
                    "end_b_s": end2,
                    "baseline_a": x[2],
                    "baseline_b": y[2],
                    "experimental_a": x[3],
                    "experimental_b": y[3],
                    "change": "split" if same_a else "merge",
                    "support_observations": min(count, count2),
                    "human_same_person": "",
                    "comment": "",
                }
            )
    cases.sort(
        key=lambda c: (-c["support_observations"], c["start_a_s"], c["start_b_s"])
    )
    summary["review_cases"] = len(cases)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if cases:
        with (output / "review_cases.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
            writer.writeheader()
            writer.writerows(cases)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-appearance-change",
        action="store_true",
        help="Permit new embeddings while requiring identical detections and eligible references",
    )
    args = parser.parse_args()
    compare(args.baseline, args.experiment, args.output, args.allow_appearance_change)
