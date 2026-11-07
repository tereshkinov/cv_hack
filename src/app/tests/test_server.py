import json
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from src.app.server import AppServer, default_project, validate_project


@pytest.fixture
def app(tmp_path):
    server = AppServer(
        ("127.0.0.1", 0), tmp_path / "project.json", run_root=tmp_path / "runs"
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", tmp_path / "project.json"
    server.shutdown()
    server.server_close()
    thread.join()


def call(url, body=None):
    req = Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(req) as response:
        return response.status, response.read()


def test_save_reload_and_existing_video_catalog(app):
    base, path = app
    status, body = call(base + "/")
    assert status == 200 and b'<canvas id="plan"' in body
    state = validate_project(default_project())
    state["walls"] = [{"id": "wall1", "a": [1, 1], "b": [10, 1]}]
    assert call(base + "/api/project", state)[0] == 200
    assert json.loads(call(base + "/api/project")[1]) == state
    assert json.loads(path.read_text()) == state
    assert isinstance(json.loads(call(base + "/api/videos")[1]), list)


def test_bad_calibration_is_400_without_changing_project(app):
    base, path = app
    with pytest.raises(HTTPError) as error:
        call(
            base + "/api/calibrate",
            {"video_points": [[0, 0]] * 4, "plan_points": [[0, 0]] * 4},
        )
    assert error.value.code == 400
    assert not path.exists()
    with pytest.raises(HTTPError) as error:
        call(base + "/api/project", [1, 2])
    assert error.value.code == 400


def test_local_plan_catalog_and_raster_upload(app, monkeypatch):
    import cv2
    import numpy as np

    from src.app import server

    base, project_path = app
    monkeypatch.setattr(server, "ROOT", project_path.parent)
    directory = project_path.parent / "data/raw"
    directory.mkdir(parents=True)
    cv2.imwrite(
        str(directory / "office.png"), np.full((20, 30, 3), 180, dtype=np.uint8)
    )
    _, body = call(base + "/api/plans")
    assets = json.loads(body)
    assert len(assets) == 1 and assets[0]["width"] == 30 and assets[0]["height"] == 20
    assert call(base + "/api/plan-image?id=" + assets[0]["id"])[1].startswith(
        b"\x89PNG"
    )
    with pytest.raises(HTTPError) as error:
        call(base + "/api/plan-image?id=../../outside")
    assert error.value.code == 404
    raw = (directory / "office.png").read_bytes()
    request = Request(
        base + "/api/plan-upload",
        data=raw,
        headers={"Content-Type": "application/octet-stream"},
    )
    with urlopen(request) as response:
        uploaded = json.load(response)
    assert uploaded["id"] != assets[0]["id"] and uploaded["width"] == 30
    request = Request(
        base + "/api/plan-upload",
        data=b"not an image",
        headers={"Content-Type": "application/octet-stream"},
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 400


def test_camera_editor_save_preserves_latest_retention_setting(app):
    base, path = app
    state = validate_project(default_project())
    state["processing"]["timeout_s"] = 60
    call(base + "/api/project", state)
    stale = validate_project(default_project())
    stale["processing"]["timeout_s"] = 15
    stale["walls"] = [{"id": "wall", "a": [1, 1], "b": [2, 1]}]
    reply = json.loads(call(base + "/api/project?preserve_timeout=1", stale)[1])
    saved = json.loads(path.read_text())
    assert saved["processing"]["timeout_s"] == 60 and reply["timeout_s"] == 60
    assert saved["walls"] == stale["walls"]
    # Already open editor versions may still POST the old full project.
    request = Request(
        base + "/api/project",
        data=json.dumps(stale).encode(),
        headers={"Content-Type": "application/json", "Referer": base + "/"},
    )
    assert urlopen(request).status == 200
    assert json.loads(path.read_text())["processing"]["timeout_s"] == 60
    # Explicit changes from processing settings remain possible for controlled experiments.
    call(base + "/api/project", stale)
    assert json.loads(path.read_text())["processing"]["timeout_s"] == 15
