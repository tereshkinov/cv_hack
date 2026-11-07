"""Portable browser preview: sampled JPEG frames, exact box references, floor paths."""

import csv
import json
import math
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import cv2

from .fusion import fuse_positions, smooth_paths
from .geometry import project

OPTIONAL_ASSIGNMENT_FIELDS = (
    "confirmation_state",
    "confirmed_at_s",
    "observed_state",
    "fragment_id",
    "provisional_global_track_id",
)


def update_box_assignment(box, assignment):
    box["global_track_id"] = assignment["global_track_id"]
    box["reason"] = assignment["reason"]
    for key in OPTIONAL_ASSIGNMENT_FIELDS:
        box.pop(key, None)
        if key in assignment:
            box[key] = assignment[key]


def build_review(run_dir, project_snapshot, cameras, options, progress=lambda *a: None):
    run_dir = Path(run_dir)
    observations = [
        json.loads(s)
        for s in (run_dir / "vision/observations.jsonl").read_text().splitlines()
    ]
    assignments = [
        json.loads(s)
        for s in (run_dir / "association/assignments.jsonl").read_text().splitlines()
    ]
    tracks = json.loads((run_dir / "association/tracks.json").read_text())["tracks"]
    mapping = {
        (a["camera_id"], a["frame_index"], a["local_track_id"]): a for a in assignments
    }
    by_frame = defaultdict(list)
    for o in observations:
        key = (o["camera_id"], o["frame_index"], o["local_track_id"])
        assigned = mapping[key]
        by_frame[(o["camera_id"], o["frame_index"])].append(
            {
                "global_track_id": assigned["global_track_id"],
                "local_track_id": o["local_track_id"],
                "bbox_xyxy": o["bbox_xyxy"],
                "confidence": o["detection_confidence"],
                "reason": assigned["reason"],
                **{k: assigned[k] for k in OPTIONAL_ASSIGNMENT_FIELDS if k in assigned},
            }
        )
    frame_dir = run_dir / "frames"
    frame_dir.mkdir(exist_ok=True)
    project_cameras = {c["id"]: c for c in project_snapshot["cameras"]}
    manifest = []
    fps = options.get("preview_fps", 10)
    total_duration = 0
    for i, cam in enumerate(cameras):
        from .server import ROOT

        cap = cv2.VideoCapture(str(ROOT / cam["video_path"]))
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть {cam['camera_id']} для просмотра")
        try:
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            limit = options.get("max_duration_s", 0)
            if limit > 0:
                count = min(count, math.ceil(limit * source_fps - 1e-9))
            duration = count / source_fps
            total_duration = max(total_duration, cam["time_offset_s"] + duration)
            meta = {
                "camera_id": cam["camera_id"],
                "name": project_cameras[cam["camera_id"]]["name"],
                "video_path": cam["video_path"],
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": source_fps,
                "offset_s": cam["time_offset_s"],
                "duration_s": duration,
                "frames": [],
                "calibrated": bool(
                    project_cameras[cam["camera_id"]].get("calibration")
                ),
            }
            next_tick = 0
            frame_index = 0
            stride = options.get("frame_stride", 1)
            wanted = {
                round(t * source_fps / fps / stride) * stride
                for t in range(math.ceil(duration * fps))
            }
            while frame_index < count:
                ok, image = cap.read()
                if not ok:
                    break
                if frame_index in wanted:
                    height, width = image.shape[:2]
                    scale = min(1, 640 / max(width, height))
                    resized = cv2.resize(
                        image,
                        (max(1, round(width * scale)), max(1, round(height * scale))),
                    )
                    filename = f"c{i}_{next_tick:08d}.jpg"
                    ok = cv2.imwrite(
                        str(frame_dir / filename),
                        resized,
                        [cv2.IMWRITE_JPEG_QUALITY, 82],
                    )
                    if not ok:
                        raise ValueError("Не удалось сохранить кадр предпросмотра")
                    meta["frames"].append(
                        {
                            "timestamp_s": cam["time_offset_s"]
                            + frame_index / source_fps,
                            "frame_index": frame_index,
                            "file": filename,
                            "boxes": by_frame.get((cam["camera_id"], frame_index), []),
                        }
                    )
                    next_tick += 1
                frame_index += 1
            if not meta["frames"]:
                raise ValueError("Не удалось подготовить ни одного кадра")
            manifest.append(meta)
        finally:
            cap.release()
        progress("preview", f"Подготовлен просмотр: {i + 1}/{len(cameras)} камер")
    paths, rejected = project_paths(project_snapshot, manifest, options)
    output = {
        "schema_version": 1,
        "coordinate_units": project_snapshot.get("coordinate_units", "meters"),
        "duration_s": total_duration,
        "preview_fps": fps,
        "project": project_snapshot,
        "cameras": manifest,
        "tracks": tracks,
        "paths": dict(paths),
        "projection_rejected": rejected,
        "observation_count": len(observations),
        "options": options,
        "reconciliation": json.loads(
            (run_dir / "association/reconciliation.json").read_text()
        ),
    }
    save_review(run_dir, output)
    return output


def project_paths(project_snapshot, manifest, options):
    project_cameras = {c["id"]: c for c in project_snapshot["cameras"]}
    fps = options.get("preview_fps", 10)
    total_duration = max(c["offset_s"] + c["duration_s"] for c in manifest)
    # Projection is evaluated on actual sampled frame timestamps; no interpolation through absence.
    frame_times = {
        c["camera_id"]: [fr["timestamp_s"] for fr in c["frames"]] for c in manifest
    }
    paths = defaultdict(list)
    rejected = 0
    for tick in range(math.ceil(total_duration * fps)):
        t = tick / fps
        samples = defaultdict(list)
        for camera in manifest:
            if not camera["offset_s"] <= t < camera["offset_s"] + camera["duration_s"]:
                continue
            times = frame_times[camera["camera_id"]]
            pos = bisect_left(times, t)
            indices = [k for k in [pos - 1, pos] if 0 <= k < len(times)]
            idx = min(indices, key=lambda k: abs(times[k] - t))
            fr = camera["frames"][idx]
            if abs(fr["timestamp_s"] - t) > 1.1 / fps:
                continue
            calibration = project_cameras[camera["camera_id"]].get("calibration")
            if not calibration:
                continue
            for box in fr["boxes"]:
                x1, _y1, x2, y2 = box["bbox_xyxy"]
                try:
                    x, y = project(calibration["matrix"], [[(x1 + x2) / 2, y2]])[0]
                except ValueError:
                    rejected += 1
                    continue
                room = project_snapshot["room"]
                if not (0 <= x <= room["width_m"] and 0 <= y <= room["height_m"]):
                    rejected += 1
                    continue
                samples[box["global_track_id"]].append(
                    {
                        "camera_id": camera["camera_id"],
                        "x_m": float(x),
                        "y_m": float(y),
                        "confidence": box["confidence"],
                    }
                )
        for gid, positions in samples.items():
            point = fuse_positions(positions, options.get("fusion_distance_m", 2))
            paths[gid].append({"timestamp_s": t, **point})
    smooth_paths(paths, options.get("smoothing_tau_s", 0.25))
    return dict(paths), rejected


def save_review(run_dir, output):
    paths = output["paths"]
    tracks = output["tracks"]
    confirmations = {
        t["global_track_id"]: t.get("confirmation_state", "confirmed") for t in tracks
    }
    has_confirmation = any("confirmation_state" in t for t in tracks)
    (run_dir / "review.json").write_text(
        json.dumps(output, ensure_ascii=False, allow_nan=False)
    )
    with (run_dir / "tracks.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "global_track_id",
                "start_s",
                "end_s",
                "camera_ids",
                "close_reason",
                *(
                    ["confirmation_state", "confirmed_at_s", "confirmation_reason"]
                    if any("confirmation_state" in t for t in tracks)
                    else []
                ),
            ],
        )
        writer.writeheader()
        for t in tracks:
            writer.writerow(
                {
                    k: ",".join(t[k]) if k == "camera_ids" else t[k]
                    for k in writer.fieldnames
                }
            )
    with (run_dir / "positions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "global_track_id",
                "timestamp_s",
                "x",
                "y",
                "coordinate_units",
                *(["confirmation_state"] if has_confirmation else []),
                "status",
                "camera_ids",
            ],
        )
        writer.writeheader()
        for gid, points in paths.items():
            for point in points:
                writer.writerow(
                    {
                        "global_track_id": gid,
                        **(
                            {"confirmation_state": confirmations[gid]}
                            if has_confirmation
                            else {}
                        ),
                        "x": point["x_m"],
                        "y": point["y_m"],
                        "coordinate_units": output.get("coordinate_units", "meters"),
                        **{k: point[k] for k in ["timestamp_s", "status"]},
                        "camera_ids": ",".join(
                            s["camera_id"] for s in point["sources"]
                        ),
                    }
                )


def validate_projection_source(source_dir, cameras):
    data = json.loads((Path(source_dir) / "review.json").read_text())
    old = {c["camera_id"]: c for c in data["cameras"]}
    if set(old) != {c["camera_id"] for c in cameras}:
        raise ValueError("Для проекции нужны те же камеры, что в выбранном результате")
    for camera in cameras:
        previous = old[camera["camera_id"]]
        for key in ["video_path", "width", "height", "fps"]:
            if camera[key] != previous[key]:
                raise ValueError("Источник камеры изменился: сначала обработайте видео")
        if abs(camera["time_offset_s"] - previous["offset_s"]) > 1e-6:
            raise ValueError("Изменилось начало видео: сначала обработайте видео")
    return data


def rebuild_projection(
    source_dir, run_dir, project_snapshot, cameras, options, progress=lambda *a: None
):
    import os
    import shutil

    source_dir, run_dir = Path(source_dir), Path(run_dir)
    data = validate_projection_source(source_dir, cameras)
    for name in ["vision", "association"]:
        shutil.copytree(source_dir / name, run_dir / name)

    def link_or_copy(src, dst):
        try:
            return os.link(src, dst)
        except OSError:
            return shutil.copy2(src, dst)

    shutil.copytree(
        source_dir / "frames", run_dir / "frames", copy_function=link_or_copy
    )
    data["project"] = project_snapshot
    # Reuse the source sampling and duration; no neural inference or video decoding.
    data["options"] = {
        **data.get("options", {}),
        "fusion_distance_m": options["fusion_distance_m"],
        "smoothing_tau_s": options["smoothing_tau_s"],
    }
    data["coordinate_units"] = project_snapshot.get("coordinate_units", "meters")
    calibration = {
        c["id"]: bool(c.get("calibration")) for c in project_snapshot["cameras"]
    }
    for camera in data["cameras"]:
        camera["calibrated"] = calibration[camera["camera_id"]]
        camera["name"] = next(
            c["name"]
            for c in project_snapshot["cameras"]
            if c["id"] == camera["camera_id"]
        )
    data["paths"], data["projection_rejected"] = project_paths(
        project_snapshot, data["cameras"], data["options"]
    )
    data["projection_source_run"] = source_dir.name
    save_review(run_dir, data)
    progress("projection", "Траектории пересчитаны по новой привязке пола")
    return data


def reassociate_cached(
    source_dir,
    run_dir,
    project_snapshot,
    cameras,
    options,
    progress=lambda *a: None,
    decision_mode="baseline",
    fragment_mode="off",
    confirm_tracks=False,
):
    """Reuse immutable detections/embeddings and JPEGs, but rebuild global IDs."""
    from src.association.cli import main as associate

    data = rebuild_projection(
        source_dir, run_dir, project_snapshot, cameras, options, progress
    )
    # A copied source may have diagnostics from a different decision mode.
    (run_dir / "association/decisions.jsonl").unlink(missing_ok=True)
    (run_dir / "association/fragment_graph.json").unlink(missing_ok=True)
    progress("association", "Пересчитываем ID: внешность, граф камер и геометрия пола")
    associate(
        [
            "--cameras",
            str(run_dir / "cameras.json"),
            "--observations",
            str(run_dir / "vision/observations.jsonl"),
            "--output-dir",
            str(run_dir / "association"),
            "--project",
            str(run_dir / "project.json"),
            "--decision-mode",
            decision_mode,
            "--fragment-mode",
            fragment_mode,
            *(["--confirm-tracks"] if confirm_tracks else []),
            "--timeout-s",
            str(options["timeout_s"]),
            "--similarity-threshold",
            str(options["similarity_threshold"]),
            "--tracklet-threshold",
            str(options["tracklet_threshold"]),
            "--ambiguity-margin",
            str(options["ambiguity_margin"]),
            "--adjacency-bonus",
            str(options["adjacency_bonus"]),
        ]
    )
    rows = [
        json.loads(s)
        for s in (run_dir / "association/assignments.jsonl").read_text().splitlines()
        if s
    ]
    mapping = {(a["camera_id"], a["frame_index"], a["local_track_id"]): a for a in rows}
    for camera in data["cameras"]:
        for frame in camera["frames"]:
            for box in frame["boxes"]:
                assignment = mapping[
                    (camera["camera_id"], frame["frame_index"], box["local_track_id"])
                ]
                update_box_assignment(box, assignment)
    data["tracks"] = json.loads((run_dir / "association/tracks.json").read_text())[
        "tracks"
    ]
    data["reconciliation"] = json.loads(
        (run_dir / "association/reconciliation.json").read_text()
    )
    # Preserve actual source sampling even when current debug settings differ.
    for key in [
        "timeout_s",
        "similarity_threshold",
        "tracklet_threshold",
        "ambiguity_margin",
        "adjacency_bonus",
    ]:
        data["options"][key] = options[key]
    data["paths"], data["projection_rejected"] = project_paths(
        project_snapshot, data["cameras"], data["options"]
    )
    data["options"]["decision_mode"] = decision_mode
    data["options"]["fragment_mode"] = fragment_mode
    data["options"]["confirm_tracks"] = confirm_tracks
    data["reassociation_source_run"] = source_dir.name
    save_review(run_dir, data)
    return data
