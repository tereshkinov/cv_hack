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
    # Stride preview must use analyzed frame indices, not fill missing boxes by guessing.
    result = build_review(
        folder, project, cameras, {"preview_fps": 10, "frame_stride": 2}
    )
    assert all(
        fr["frame_index"] % 2 == 0 for c in result["cameras"] for fr in c["frames"]
    )


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
