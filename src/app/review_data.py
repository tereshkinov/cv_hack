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
            while True:
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
    output = {
        "schema_version": 1,
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
                "x_m",
                "y_m",
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
                        **{
                            k: point[k] for k in ["timestamp_s", "x_m", "y_m", "status"]
                        },
                        "camera_ids": ",".join(
                            s["camera_id"] for s in point["sources"]
                        ),
                    }
                )
    return output
