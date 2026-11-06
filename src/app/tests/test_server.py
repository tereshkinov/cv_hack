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
