import json

import cv2
import numpy as np
import pytest

from src.app.jobs import Jobs, validate_options
from src.app.project_runtime import camera_manifest
from src.app.review_data import build_review


def video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (80, 80))
    assert writer.isOpened()
    for i in range(8):
        writer.write(np.full((80, 80, 3), 50 + i * 10, dtype=np.uint8))
    writer.release()


def test_end_to_end_review_keeps_ids_frame_references_and_fuses_floor(
    tmp_path, monkeypatch
):
    from src.app import server

    monkeypatch.setattr(server, "ROOT", tmp_path)
    project = {"room": {"width_m": 10, "height_m": 10}, "walls": [], "cameras": []}
    cameras = []
    rows = []
    assignments = []
    for i, cid in enumerate(["a", "b"]):
        path = tmp_path / "data/raw" / f"{cid}.avi"
        video(path)
        cameras.append(
            {
                "camera_id": cid,
                "video_path": str(path.relative_to(tmp_path)),
                "time_offset_s": 0,
            }
        )
        project["cameras"].append(
            {
                "id": cid,
                "name": cid,
                "calibration": {"matrix": [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]]},
            }
        )
        for index in range(8):
            rows.append(
                {
                    "camera_id": cid,
                    "frame_index": index,
                    "local_track_id": "l1",
                    "timestamp_s": index / 5,
                    "bbox_xyxy": [10 + 2 * i, 10, 30 + 2 * i, 50],
                    "detection_confidence": 1 if i == 0 else 0.5,
                }
            )
            assignments.append(
                {
                    "camera_id": cid,
                    "frame_index": index,
                    "local_track_id": "l1",
                    "global_track_id": "g1",
                    "reason": "local_track_continuity",
                }
            )
    folder = tmp_path / "run"
    (folder / "vision").mkdir(parents=True)
    (folder / "association").mkdir()
    (folder / "vision/observations.jsonl").write_text(
        "".join(json.dumps(o) + "\n" for o in rows)
    )
    (folder / "association/assignments.jsonl").write_text(
        "".join(json.dumps(o) + "\n" for o in assignments)
    )
    (folder / "association/tracks.json").write_text(
        json.dumps(
            {
                "tracks": [
                    {
                        "global_track_id": "g1",
                        "start_s": 0,
                        "end_s": 1.4,
                        "camera_ids": ["a", "b"],
                        "close_reason": "end_of_video",
                    }
                ]
            }
        )
    )
    (folder / "association/reconciliation.json").write_text('{"merges":[]}')
    result = build_review(
        folder, project, cameras, {"preview_fps": 5, "frame_stride": 1}
    )
    assert len(result["cameras"]) == 2 and len(result["paths"]["g1"]) == 8
    assert result["paths"]["g1"][0]["x_m"] == pytest.approx(2.04)
    assert all(p["status"] == "fused" for p in result["paths"]["g1"])
    assert result["cameras"][0]["frames"][6]["boxes"][0]["global_track_id"] == "g1"
    assert (folder / "frames" / result["cameras"][1]["frames"][6]["file"]).is_file()
    assert (folder / "tracks.csv").is_file() and (folder / "positions.csv").is_file()
    limited = build_review(
        folder,
        project,
        cameras,
        {"preview_fps": 5, "frame_stride": 1, "max_duration_s": 0.6},
    )
    assert limited["duration_s"] == pytest.approx(0.6)
    assert all(len(c["frames"]) == 3 for c in limited["cameras"])
    assert all(fr["frame_index"] < 3 for c in limited["cameras"] for fr in c["frames"])
    # Stride preview must use analyzed frame indices, not fill missing boxes by guessing.
    result = build_review(
        folder, project, cameras, {"preview_fps": 10, "frame_stride": 2}
    )
    assert all(
        fr["frame_index"] % 2 == 0 for c in result["cameras"] for fr in c["frames"]
    )

    # Reproject saved observations and JPEGs: no video/ML inference and no ID changes.
    import copy

    from src.app.worker import execute

    project_new = copy.deepcopy(project)
    project_new["coordinate_units"] = "relative"
    for c in project_new["cameras"]:
        c["calibration"]["matrix"][0][2] = 2
    for c in cameras:
        c.update(width=80, height=80, fps=5)
    source_json = (folder / "review.json").read_bytes()
    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "job.json").write_text(
        json.dumps(
            {
                "kind": "projection",
                "source_run_id": folder.name,
                "options": {"fusion_distance_m": 2, "smoothing_tau_s": 0.25},
            }
        )
    )
    (derived / "project.json").write_text(json.dumps(project_new))
    (derived / "cameras.json").write_text(json.dumps({"cameras": cameras}))
    execute(derived)
    updated = json.loads((derived / "review.json").read_text())
    assert updated["paths"]["g1"][0]["x_m"] == pytest.approx(4.04)
    assert updated["coordinate_units"] == "relative"
    assert json.loads((derived / "status.json").read_text())["status"] == "complete"
    assert (derived / "association/assignments.jsonl").read_bytes() == (
        folder / "association/assignments.jsonl"
    ).read_bytes()
    assert (folder / "review.json").read_bytes() == source_json
    assert (
        derived / "frames" / updated["cameras"][0]["frames"][0]["file"]
    ).read_bytes() == (
        folder / "frames" / updated["cameras"][0]["frames"][0]["file"]
    ).read_bytes()
    assert "coordinate_units" in (derived / "positions.csv").read_text().splitlines()[0]


def test_runtime_manifest_rejects_missing_video_and_carries_links(tmp_path):
    path = tmp_path / "data/raw/a.avi"
    video(path)
    c = {
        "id": "a",
        "name": "A",
        "video_id": "token",
        "time_offset_s": 2,
        "transition_to": [],
    }
    manifest = camera_manifest({"cameras": [c]}, {"token": path}, tmp_path)
    assert manifest[0]["time_offset_s"] == 2 and manifest[0]["width"] == 80
    with pytest.raises(ValueError):
        camera_manifest({"cameras": [c]}, {}, tmp_path)


def test_job_cancel_and_single_active_process(tmp_path, monkeypatch):
    class Process:
        returncode = None

        def __init__(self, *args, **kwargs):
            pass

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr("src.app.jobs.subprocess.Popen", Process)
    manager = Jobs(tmp_path)
    first = manager.start({"cameras": []}, [], {}, "prepare")
    assert first["status"] == "queued"
    with pytest.raises(ValueError):
        manager.start({"cameras": []}, [], {}, "prepare")
    assert manager.cancel(first["id"])["status"] == "cancelled"
    assert manager.list()[0]["status"] == "cancelled"
    with pytest.raises(ValueError):
        manager.folder("../outside")


@pytest.mark.parametrize(
    "settings",
    [
        {"timeout_s": float("nan")},
        {"preview_fps": 0.5},
        {"frame_stride": True},
        {"device": "unavailable"},
    ],
)
def test_invalid_processing_options_are_rejected(settings):
    with pytest.raises(ValueError):
        validate_options(settings)


def test_cached_reassociation_changes_ids_without_video_or_inference(
    tmp_path, monkeypatch
):
    from src.app.jobs import DEFAULT_OPTIONS
    from src.app.worker import execute

    source = tmp_path / "source"
    (source / "vision").mkdir(parents=True)
    (source / "association").mkdir()
    (source / "frames").mkdir()
    (source / "frames/c0_00000000.jpg").write_bytes(b"cached-jpeg")
    project = {
        "coordinate_units": "meters",
        "room": {"width_m": 10, "height_m": 10},
        "walls": [],
        "cameras": [],
        "processing": DEFAULT_OPTIONS,
    }
    manifest, previews, observations, assignments = [], [], [], []
    for cid in ["a", "b"]:
        project["cameras"].append(
            {
                "id": cid,
                "name": cid,
                "calibration": {"matrix": [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1]]},
            }
        )
        manifest.append(
            {
                "camera_id": cid,
                "video_path": cid + ".mp4",
                "width": 1000,
                "height": 1000,
                "fps": 5,
                "time_offset_s": 0,
                "overlap_with": ["b"] if cid == "a" else [],
            }
        )
        frames = []
        for i in range(3):
            box = [90, 100, 110, 200]
            observations.append(
                {
                    "schema_version": 1,
                    "camera_id": cid,
                    "frame_index": i,
                    "timestamp_s": i / 5,
                    "local_track_id": "one",
                    "bbox_xyxy": box,
                    "detection_confidence": 0.9,
                    "embedding": [1, 0] if cid == "a" else [0.59, 0.8074],
                    "embedding_model_id": "test",
                }
            )
            assignments.append(
                {
                    "camera_id": cid,
                    "frame_index": i,
                    "local_track_id": "one",
                    "global_track_id": cid,
                    "reason": "new_track",
                }
            )
            frames.append(
                {
                    "timestamp_s": i / 5,
                    "frame_index": i,
                    "file": "c0_00000000.jpg",
                    "boxes": [
                        {
                            "global_track_id": cid,
                            "local_track_id": "one",
                            "bbox_xyxy": box,
                            "confidence": 0.9,
                            "reason": "new_track",
                        }
                    ],
                }
            )
        previews.append(
            {**manifest[-1], "offset_s": 0, "duration_s": 0.6, "frames": frames}
        )
    tracks = [
        {
            "global_track_id": cid,
            "start_s": 0,
            "end_s": 0.4,
            "camera_ids": [cid],
            "close_reason": "end_of_video",
        }
        for cid in ["a", "b"]
    ]
    (source / "vision/observations.jsonl").write_text(
        "".join(json.dumps(o) + "\n" for o in observations)
    )
    (source / "association/assignments.jsonl").write_text(
        "".join(json.dumps(o) + "\n" for o in assignments)
    )
    (source / "association/tracks.json").write_text(json.dumps({"tracks": tracks}))
    (source / "association/reconciliation.json").write_text('{"merges":[]}')
    (source / "review.json").write_text(
        json.dumps(
            {
                "cameras": previews,
                "tracks": tracks,
                "options": DEFAULT_OPTIONS,
                "duration_s": 0.6,
                "preview_fps": 5,
                "observation_count": 6,
            }
        )
    )
    before = (source / "association/assignments.jsonl").read_bytes()
    derived = tmp_path / "derived"
    derived.mkdir()
    (derived / "job.json").write_text(
        json.dumps(
            {
                "kind": "reassociate",
                "source_run_id": "source",
                "options": DEFAULT_OPTIONS,
            }
        )
    )
    (derived / "project.json").write_text(json.dumps(project))
    (derived / "cameras.json").write_text(
        json.dumps({"schema_version": 1, "cameras": manifest})
    )

    def no_video(*args, **kwargs):
        raise AssertionError("Cached reassociation must not decode videos")

    monkeypatch.setattr(cv2, "VideoCapture", no_video)
    execute(derived)
    data = json.loads((derived / "review.json").read_text())
    assert len(data["tracks"]) == 1
    assert (
        len(
            {
                b["global_track_id"]
                for c in data["cameras"]
                for f in c["frames"]
                for b in f["boxes"]
            }
        )
        == 1
    )
    assert data["reconciliation"]["geolocation"]["enabled"]
    assert (source / "association/assignments.jsonl").read_bytes() == before
    assert (derived / "frames/c0_00000000.jpg").read_bytes() == b"cached-jpeg"
    assert json.loads((derived / "status.json").read_text())["status"] == "complete"
