"""Build isolated browser runs from fixed-detection appearance experiments."""

import argparse
import json
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.app.review_data import project_paths, rebuild_projection, save_review
from src.association.cli import main as associate
from src.vision.appearance_experiment import MODES, read_rows, reference
from src.vision.embeddings import sha256

ROOT = Path(__file__).resolve().parents[1]


def build(source, experiment, mode, decision_mode="balanced"):
    source, experiment = source.resolve(), experiment.resolve()
    summary = json.loads((experiment / "experiment.json").read_text())
    original = source / "vision/observations.jsonl"
    if summary["source_observations_sha256"] != sha256(original):
        raise ValueError("Appearance experiment uses different source observations")
    if summary["source_cameras_sha256"] != sha256(source / "cameras.json"):
        raise ValueError("Appearance experiment uses different source camera manifest")
    rows, previous = read_rows(experiment / f"{mode}.jsonl"), read_rows(original)
    if len(rows) != len(previous):
        raise ValueError("Observation count changed")
    for a, b in zip(previous, rows, strict=True):
        if reference(a) != reference(b) or {
            k: v for k, v in a.items() if k not in {"embedding", "embedding_model_id"}
        } != {
            k: v for k, v in b.items() if k not in {"embedding", "embedding_model_id"}
        }:
            raise ValueError("Only appearance embeddings may change")
        if (a.get("embedding") is None) != (b.get("embedding") is None):
            raise ValueError("Eligible embedding references changed")
    featured = [row for row in rows if row.get("embedding") is not None]
    if not featured or {row["embedding_model_id"] for row in featured} != {
        summary["models"][mode]
    }:
        raise ValueError("Variant contains incompatible model spaces")
    dimensions = {len(row["embedding"]) for row in featured}
    if len(dimensions) != 1:
        raise ValueError("Variant has inconsistent embedding dimensions")
    dimension = dimensions.pop()
    project = json.loads((source / "project.json").read_text())
    cameras = json.loads((source / "cameras.json").read_text())["cameras"]
    options = json.loads((source / "job.json").read_text())["options"]
    folder = (
        ROOT
        / "runs/product"
        / (
            datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
            + f"-appearance-{mode}-"
            + uuid.uuid4().hex[:8]
        )
    )
    folder.mkdir(parents=True, exist_ok=False)
    spec = {
        "kind": "pipeline",
        "source_run_id": source.name,
        "created_at": time.time(),
        "options": options,
        "camera_names": [c["name"] for c in project["cameras"]],
        "appearance_mode": mode,
        "decision_mode": decision_mode,
    }
    for name, value in [
        ("job.json", spec),
        ("project.json", project),
        ("cameras.json", {"schema_version": 1, "cameras": cameras}),
    ]:
        (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))

    def status(state, message):
        (folder / "status.json").write_text(
            json.dumps(
                {
                    "status": state,
                    "phase": "appearance",
                    "message": message,
                    "updated_at": time.time(),
                },
                ensure_ascii=False,
            )
        )
        print(message, flush=True)

    status("running", f"Внешность {mode}: готовим сравнение")
    started = time.perf_counter()
    try:
        data = rebuild_projection(source, folder, project, cameras, options)
        # This is a newly created copy. Source observations and metadata stay immutable.
        shutil.rmtree(folder / "vision")
        (folder / "vision").mkdir()
        shutil.copy2(experiment / f"{mode}.jsonl", folder / "vision/observations.jsonl")
        shutil.copy2(source / "vision/run.json", folder / "vision/source_run.json")
        (folder / "vision/run.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "kind": "reencode_fixed_detections",
                    "source_run": source.name,
                    "appearance_mode": mode,
                    "embedding_model_id": summary["models"][mode],
                    "embedding_dimension": dimension,
                    "embedding_normalization": "L2",
                    "experiment": summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        (folder / "association/decisions.jsonl").unlink(missing_ok=True)
        command = [
            "--cameras",
            str(folder / "cameras.json"),
            "--observations",
            str(folder / "vision/observations.jsonl"),
            "--output-dir",
            str(folder / "association"),
            "--project",
            str(folder / "project.json"),
            "--decision-mode",
            decision_mode,
        ]
        for key in [
            "timeout_s",
            "similarity_threshold",
            "tracklet_threshold",
            "ambiguity_margin",
            "adjacency_bonus",
        ]:
            command.extend(["--" + key.replace("_", "-"), str(options[key])])
        associate(command)
        assigned = {
            reference(a): a for a in read_rows(folder / "association/assignments.jsonl")
        }
        for camera in data["cameras"]:
            for frame in camera["frames"]:
                for box in frame["boxes"]:
                    a = assigned[
                        (
                            camera["camera_id"],
                            frame["frame_index"],
                            box["local_track_id"],
                        )
                    ]
                    box["global_track_id"], box["reason"] = (
                        a["global_track_id"],
                        a["reason"],
                    )
        data["tracks"] = json.loads((folder / "association/tracks.json").read_text())[
            "tracks"
        ]
        data["reconciliation"] = json.loads(
            (folder / "association/reconciliation.json").read_text()
        )
        data["appearance_mode"] = mode
        data["options"]["decision_mode"] = decision_mode
        data["paths"], data["projection_rejected"] = project_paths(
            project, data["cameras"], data["options"]
        )
        save_review(folder, data)
        (folder / "performance.json").write_text(
            json.dumps(
                {
                    "preview_and_association_s": time.perf_counter() - started,
                    "reencoding_experiment": summary,
                    "mode": mode,
                }
            )
        )
        (folder / "report.md").write_text(
            f"# Appearance experiment: {mode}\nDecision mode: {decision_mode}. Tracks: {len(data['tracks'])}.\nSource: {source.name}. Fixed detections; new embeddings. Color evidence is not used for ID.\nNo labeled ground truth: track count is not accuracy.\n"
        )
        status(
            "complete",
            f"Готово: {len(data['tracks'])} треков; внешность {mode}; {decision_mode}",
        )
        return {
            "mode": mode,
            "decision_mode": decision_mode,
            "run_id": folder.name,
            "tracks": len(data["tracks"]),
        }
    except Exception as error:
        status("failed", str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument(
        "--decision-mode", choices=["baseline", "balanced"], default="balanced"
    )
    args = parser.parse_args()
    results = [
        build(args.source, args.experiment, mode, args.decision_mode) for mode in MODES
    ]
    (args.experiment / f"preview-{args.decision_mode}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    print(json.dumps(results, ensure_ascii=False), flush=True)
