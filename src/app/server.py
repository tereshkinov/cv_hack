"""Loopback-only UI server; no external services or web dependencies."""

import argparse
import hashlib
import json
import math
import mimetypes
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlparse

import cv2

from .geometry import calibrate
from .jobs import DEFAULT_OPTIONS, Jobs, validate_options
from .plan_assets import describe, plans
from .project_runtime import camera_manifest

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def default_project():
    return {
        "schema_version": 1,
        "coordinate_units": "meters",
        "room": {"width_m": 20, "height_m": 14},
        "walls": [],
        "cameras": [],
    }


def validate_project(project):
    if not isinstance(project, dict) or project.get("schema_version") != 1:
        raise ValueError("Ожидается проект версии 1")

    def number(x):
        if (
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
        ):
            raise ValueError("Координаты и размеры должны быть конечными числами")
        return x

    units = project.get("coordinate_units", "meters")
    if units not in {"meters", "relative"}:
        raise ValueError("Единицы схемы: meters или relative")
    background = project.get("plan_background")
    if background is not None:
        if not isinstance(background, dict) or not re.fullmatch(
            r"[a-f0-9]{20}", str(background.get("image_id", ""))
        ):
            raise ValueError("Некорректное изображение плана")
        if not 0 <= number(background.get("opacity", 0.8)) <= 1:
            raise ValueError("Прозрачность плана: от 0 до 1")
    room = project.get("room", {})
    width, height = number(room.get("width_m")), number(room.get("height_m"))
    if not (1 <= width <= 1000 and 1 <= height <= 1000):
        raise ValueError("Размер схемы: от 1 до 1000 метров по каждой оси")
    walls, cameras = project.get("walls"), project.get("cameras")
    if (
        not isinstance(walls, list)
        or not isinstance(cameras, list)
        or len(walls) > 1000
        or len(cameras) > 100
    ):
        raise ValueError("Некорректный список стен или камер")
    ids = set()

    def identifier(obj):
        cid = obj.get("id", "")
        if (
            not isinstance(cid, str)
            or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", cid)
            or cid in ids
        ):
            raise ValueError("Идентификаторы должны быть уникальными")
        ids.add(cid)

    def point(p):
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError("Точка должна содержать x и y")
        return [number(p[0]), number(p[1])]

    for wall in walls:
        if not isinstance(wall, dict):
            raise TypeError("Некорректное описание стены")
        identifier(wall)
        a, b = point(wall.get("a")), point(wall.get("b"))
        if a == b:
            raise ValueError("Стена должна иметь ненулевую длину")
        if any(not (0 <= p[0] <= width and 0 <= p[1] <= height) for p in [a, b]):
            raise ValueError("Стена выходит за границы схемы")
    for camera in cameras:
        if not isinstance(camera, dict):
            raise TypeError("Некорректное описание камеры")
        identifier(camera)
        p = point(camera.get("position"))
        if not (0 <= p[0] <= width and 0 <= p[1] <= height):
            raise ValueError("Камера выходит за границы схемы")
        if not isinstance(camera.get("name"), str) or len(camera["name"]) > 100:
            raise ValueError("Некорректное название камеры")
        for field, low, high in [
            ("height_m", 0.1, 100),
            ("yaw_deg", -360, 360),
            ("fov_deg", 1, 179),
            ("range_m", 0.1, 1000),
        ]:
            if not low <= number(camera.get(field)) <= high:
                raise ValueError(f"Некорректное значение {field}")
        for key in ["video_points", "plan_points"]:
            values = camera.get(key, [])
            if not isinstance(values, list) or len(values) > 4:
                raise ValueError("Можно задать не более четырёх точек каждой стороны")
            for value in values:
                point(value)
        plans = camera.get("plan_points", [])
        if any(not (0 <= p[0] <= width and 0 <= p[1] <= height) for p in plans):
            raise ValueError("Опорная точка выходит за границы схемы")
        # Recompute, never trust an imported arbitrary transformation matrix.
        if camera.get("calibration") is not None:
            camera["calibration"] = calibrate(camera.get("video_points"), plans)
    camera_ids = {c["id"] for c in cameras}
    for camera in cameras:
        offset = number(camera.get("time_offset_s", 0))
        if not 0 <= offset <= 86400:
            raise ValueError("Смещение камеры должно быть от 0 до 86400 секунд")
        for key in ["overlap_with", "transition_to"]:
            links = camera.get(key, [])
            if not isinstance(links, list) or any(
                not isinstance(v, str) or v not in camera_ids or v == camera["id"]
                for v in links
            ):
                raise ValueError("Некорректная связь камер")
    project["processing"] = validate_options(project.get("processing", {}))
    return project


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, project_path, run_root=None):
        self.project_path = project_path
        self.save_lock = Lock()
        self.jobs = Jobs(ROOT, run_root)
        self.upload_dir = ROOT / "runs/app/uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(address, Handler)

    def videos(self):
        files = []
        for directory in [ROOT / "data/raw", self.upload_dir]:
            if directory.exists():
                files.extend(
                    p
                    for p in directory.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
                )
        return {
            hashlib.sha256(str(p.relative_to(ROOT)).encode()).hexdigest()[:20]: p
            for p in sorted(files)
        }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if (
            args
            and ("GET /api/frame" in str(args[0]) or "GET /api/runs" in str(args[0]))
            and len(args) > 1
            and str(args[1]) == "200"
        ):
            return
        super().log_message(fmt, *args)

    def reply(
        self,
        status,
        data,
        content_type="application/json; charset=utf-8",
        download_name=None,
    ):
        body = (
            json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            if isinstance(data, (dict, list))
            else data
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if download_name:
            self.send_header(
                "Content-Disposition", f'attachment; filename="{download_name}"'
            )
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def json_body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length < 2_000_000:
            raise ValueError("Слишком большой или пустой JSON")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise TypeError("Ожидается JSON-объект")
        return value

    def do_GET(self):
        route = urlparse(self.path)
        try:
            if route.path == "/api/system":
                return self.reply(
                    200,
                    {
                        "models_ready": all(
                            (ROOT / "models" / name).is_file()
                            for name in ["yolo11n.pt", "osnet_x0_25_msmt17.pth"]
                        ),
                        "defaults": DEFAULT_OPTIONS,
                    },
                )
            if route.path == "/api/runs":
                return self.reply(200, self.server.jobs.list())
            parts = route.path.strip("/").split("/")
            if len(parts) >= 3 and parts[:2] == ["api", "runs"]:
                folder = self.server.jobs.folder(parts[2])
                if len(parts) == 3:
                    result = self.server.jobs.status(folder)
                    log = folder / "worker.log"
                    if log.exists():
                        with log.open("rb") as handle:
                            handle.seek(0, 2)
                            size = handle.tell()
                            handle.seek(max(0, size - 16000))
                            result["log"] = handle.read().decode(errors="replace")
                    return self.reply(200, result)
                if len(parts) == 4 and parts[3] == "review":
                    if not (folder / "review.json").is_file():
                        return self.reply(409, {"error": "Результат ещё не готов"})
                    return self.reply(200, (folder / "review.json").read_bytes())
                if (
                    len(parts) == 5
                    and parts[3] == "frame"
                    and re.fullmatch(r"c[0-9]+_[0-9]{8}\.jpg", parts[4])
                ):
                    return self.reply(
                        200, (folder / "frames" / parts[4]).read_bytes(), "image/jpeg"
                    )
                exports = {
                    "tracks.csv": "tracks.csv",
                    "positions.csv": "positions.csv",
                    "tracks.json": "association/tracks.json",
                    "assignments.jsonl": "association/assignments.jsonl",
                    "observations.jsonl": "vision/observations.jsonl",
                    "project.json": "project.json",
                    "report.md": "report.md",
                    "performance.json": "performance.json",
                    "reconciliation.json": "association/reconciliation.json",
                    "decisions.jsonl": "association/decisions.jsonl",
                }
                if len(parts) == 5 and parts[3] == "export" and parts[4] in exports:
                    return self.reply(
                        200,
                        (folder / exports[parts[4]]).read_bytes(),
                        "application/octet-stream",
                        parts[4],
                    )
                return self.reply(404, {"error": "Не найдено"})
            if route.path == "/api/project":
                path = self.server.project_path
                data = (
                    json.loads(path.read_text()) if path.exists() else default_project()
                )
                return self.reply(200, data)
            if route.path == "/api/plans":
                return self.reply(
                    200,
                    [describe(ROOT, key, path) for key, path in plans(ROOT).items()],
                )
            if route.path == "/api/plan-image":
                key = parse_qs(route.query).get("id", [""])[0]
                path = plans(ROOT).get(key)
                if path is None:
                    return self.reply(404, {"error": "Изображение плана не найдено"})
                return self.reply(
                    200,
                    path.read_bytes(),
                    mimetypes.guess_type(path.name)[0] or "image/png",
                )
            if route.path == "/api/videos":
                return self.reply(
                    200,
                    [
                        {"id": key, "name": str(p.relative_to(ROOT))}
                        for key, p in self.server.videos().items()
                    ],
                )
            if route.path in {"/api/video", "/api/frame"}:
                query = parse_qs(route.query)
                path = self.server.videos().get(query.get("id", [""])[0])
                if path is None:
                    return self.reply(404, {"error": "Видео не найдено"})
                cap = cv2.VideoCapture(str(path))
                try:
                    if not cap.isOpened():
                        raise ValueError("Не удалось открыть видео")
                    metadata = {
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    }
                    if route.path == "/api/video":
                        return self.reply(200, metadata)
                    index = int(query.get("frame", ["0"])[0])
                    if not 0 <= index < metadata["frame_count"]:
                        raise ValueError("Номер кадра вне записи")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError("Не удалось прочитать кадр")
                    ok, image = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
                    )
                    if not ok:
                        raise ValueError("Не удалось подготовить кадр")
                    return self.reply(200, image.tobytes(), "image/jpeg")
                finally:
                    cap.release()
            allowed = {
                "/": "index.html",
                "/app.js": "app.js",
                "/style.css": "style.css",
                "/review": "review.html",
                "/review.js": "review.js",
                "/review.css": "review.css",
            }
            filename = allowed.get(route.path)
            if filename:
                path = STATIC / filename
                return self.reply(
                    200,
                    path.read_bytes(),
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                )
            self.reply(404, {"error": "Не найдено"})
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.reply(400, {"error": str(error)})

    def do_POST(self):
        route = urlparse(self.path)
        try:
            if route.path == "/api/plan-upload":
                import uuid

                import numpy as np

                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 20 * 1024 * 1024:
                    raise ValueError("Размер изображения: до 20 МБ")
                raw = self.rfile.read(size)
                image = cv2.imdecode(
                    np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is None or image.shape[0] * image.shape[1] > 40_000_000:
                    raise ValueError(
                        "Нужен PNG, JPEG или WebP размером до 40 мегапикселей"
                    )
                directory = ROOT / "runs/app/plan-uploads"
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / (uuid.uuid4().hex + ".png")
                if not cv2.imwrite(str(path), image):
                    raise ValueError("Не удалось сохранить изображение")
                key = hashlib.sha256(str(path.relative_to(ROOT)).encode()).hexdigest()[
                    :20
                ]
                return self.reply(200, describe(ROOT, key, path))
            parts = route.path.strip("/").split("/")
            if (
                len(parts) == 4
                and parts[:2] == ["api", "runs"]
                and parts[3] in {"project", "reassociate", "balanced"}
            ):
                project = validate_project(self.json_body()["project"])
                cameras = camera_manifest(project, self.server.videos(), ROOT)
                if not any(c.get("calibration") for c in project["cameras"]):
                    raise ValueError("Сначала привяжите к полу хотя бы одну камеру")
                result = self.server.jobs.start(
                    project,
                    cameras,
                    project["processing"],
                    "projection" if parts[3] == "project" else parts[3],
                    source_run_id=parts[2],
                )
                return self.reply(202, result)
            if route.path in {"/api/runs", "/api/prepare"}:
                body = self.json_body()
                project = validate_project(body.get("project", default_project()))
                options = validate_options(
                    body.get("options", project.get("processing", {}))
                )
                cameras = (
                    camera_manifest(project, self.server.videos(), ROOT)
                    if route.path == "/api/runs"
                    else []
                )
                if route.path == "/api/runs" and not all(
                    (ROOT / "models" / name).is_file()
                    for name in ["yolo11n.pt", "osnet_x0_25_msmt17.pth"]
                ):
                    raise ValueError("Сначала нажмите «Подготовить модели»")
                result = self.server.jobs.start(
                    project,
                    cameras,
                    options,
                    "pipeline" if route.path == "/api/runs" else "prepare",
                )
                return self.reply(202, result)
            parts = route.path.strip("/").split("/")
            if (
                len(parts) == 4
                and parts[:2] == ["api", "runs"]
                and parts[3] == "cancel"
            ):
                return self.reply(200, self.server.jobs.cancel(parts[2]))
            if route.path == "/api/calibrate":
                body = self.json_body()
                return self.reply(
                    200, calibrate(body.get("video_points"), body.get("plan_points"))
                )
            if route.path == "/api/project":
                project = validate_project(self.json_body())
                path = self.server.project_path
                path.parent.mkdir(parents=True, exist_ok=True)
                with self.server.save_lock:
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps(project, ensure_ascii=False, indent=2) + "\n"
                    )
                    temporary.replace(path)
                return self.reply(200, {"saved": True})
            if route.path == "/api/upload":
                query = parse_qs(route.query)
                name = Path(query.get("name", ["video"])[0]).name
                if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
                    raise ValueError("Выберите MOV, MP4, AVI, MKV, WEBM или M4V")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 2_000_000_000:
                    raise ValueError("Допустимый размер видео: до 2 GB")
                target = self.server.upload_dir / (uuid.uuid4().hex[:10] + "_" + name)
                remaining = length
                try:
                    with target.open("xb") as handle:
                        while remaining:
                            block = self.rfile.read(min(1024 * 1024, remaining))
                            if not block:
                                raise ValueError("Передача видео прервалась")
                            handle.write(block)
                            remaining -= len(block)
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
                token = hashlib.sha256(
                    str(target.relative_to(ROOT)).encode()
                ).hexdigest()[:20]
                return self.reply(200, {"id": token})
            self.reply(404, {"error": "Не найдено"})
        except (ValueError, TypeError, KeyError, OSError) as error:
            self.reply(400, {"error": str(error)})


def main():
    parser = argparse.ArgumentParser(
        description="Local office editor and four-point camera calibration"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--project", type=Path, default=ROOT / "configs/office.local.json"
    )
    parser.add_argument("--open", action="store_true")
    parser.add_argument(
        "--runs", type=Path, help="Optional isolated run directory for testing"
    )
    args = parser.parse_args()
    server = AppServer(("127.0.0.1", args.port), args.project, args.runs)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Office editor: {url}\nProject: {args.project}", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for run_id in list(server.jobs.processes):
            server.jobs.cancel(run_id)
        server.server_close()
