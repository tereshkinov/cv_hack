"""Manage persisted local processing jobs; no shell command interpolation."""

import json
import math
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from threading import Lock

DEFAULT_OPTIONS = {
    "device": "cpu",
    "confidence": 0.1,
    "frame_stride": 1,
    "timeout_s": 15.0,
    "max_duration_s": 180.0,
    "local_buffer_s": 5.0,
    "adjacency_bonus": 0.04,
    "transition_overlap_tolerance_s": 0.75,
    "ambiguity_margin": 0.03,
    "similarity_threshold": 0.72,
    "tracklet_threshold": 0.62,
    "preview_fps": 10,
    "fusion_distance_m": 2.0,
    "smoothing_tau_s": 0.25,
}


def validate_options(options):
    if not isinstance(options, dict):
        raise TypeError("Настройки должны быть объектом")
    out = {
        **DEFAULT_OPTIONS,
        **{k: v for k, v in options.items() if k in DEFAULT_OPTIONS},
    }
    for key, low, high in [
        ("confidence", 0.01, 1),
        ("timeout_s", 0, 300),
        ("max_duration_s", 0, 86400),
        ("local_buffer_s", 0, 30),
        ("adjacency_bonus", 0, 0.15),
        ("transition_overlap_tolerance_s", 0, 5),
        ("ambiguity_margin", 0, 0.2),
        ("similarity_threshold", 0, 1),
        ("tracklet_threshold", 0, 1),
        ("fusion_distance_m", 0.1, 50),
        ("smoothing_tau_s", 0, 10),
    ]:
        v = out[key]
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or not low <= v <= high
        ):
            raise ValueError(f"Некорректная настройка {key}")
    for key, low, high in [("frame_stride", 1, 10), ("preview_fps", 1, 15)]:
        if (
            isinstance(out[key], bool)
            or not isinstance(out[key], int)
            or not low <= out[key] <= high
        ):
            raise ValueError(f"Некорректная настройка {key}")
    if out["device"] not in {"cpu", "mps", "0"}:
        raise ValueError("Устройство: cpu, mps или 0 (CUDA)")
    return out


class Jobs:
    def __init__(self, root, run_root=None):
        self.root = root
        self.run_root = run_root or root / "runs/product"
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.processes = {}

    def folder(self, run_id):
        if not run_id or any(
            c not in "0123456789abcdefghijklmnopqrstuvwxyz-_" for c in run_id
        ):
            raise ValueError("Некорректный идентификатор запуска")
        folder = self.run_root / run_id
        if not (folder / "job.json").is_file():
            raise FileNotFoundError("Запуск не найден")
        return folder

    def status(self, folder):
        spec = json.loads((folder / "job.json").read_text())
        path = folder / "status.json"
        status = (
            json.loads(path.read_text())
            if path.exists()
            else {"status": "queued", "phase": "queued", "message": "В очереди"}
        )
        process = self.processes.get(folder.name)
        if status["status"] in {"queued", "running"} and (
            (process is not None and process.poll() is not None) or process is None
        ):
            status = {
                "status": "failed",
                "phase": "failed",
                "message": "Процесс прерван. Запустите обработку заново.",
            }
        return {**spec, **status, "id": folder.name}

    def list(self):
        results = [self.status(p.parent) for p in self.run_root.glob("*/job.json")]
        return sorted(results, key=lambda job: job["created_at"], reverse=True)

    def start(self, project, cameras, options, kind="pipeline", source_run_id=None):
        with self.lock:
            if any(p.poll() is None for p in self.processes.values()):
                raise ValueError(
                    "Уже выполняется обработка. Дождитесь её или остановите запуск."
                )
            options = validate_options(options)
            if kind in {"projection", "reassociate"}:
                from .review_data import validate_projection_source

                source = self.folder(source_run_id)
                if self.status(source)["status"] != "complete":
                    raise ValueError("Выберите завершённый результат для проекции")
                validate_projection_source(source, cameras)
            folder = self.run_root / (
                datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
            )
            folder.mkdir()
            spec = {
                "id": folder.name,
                "kind": kind,
                "source_run_id": source_run_id,
                "created_at": time.time(),
                "options": options,
                "camera_names": [c["name"] for c in project["cameras"]],
            }
            for name, body in [
                ("job.json", spec),
                ("project.json", project),
                ("cameras.json", {"schema_version": 1, "cameras": cameras}),
            ]:
                (folder / name).write_text(
                    json.dumps(body, ensure_ascii=False, indent=2)
                )
            with (folder / "worker.log").open("w") as log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "src.app.worker",
                        "--run-dir",
                        str(folder),
                    ],
                    cwd=self.root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            self.processes[folder.name] = process
            return self.status(folder)

    def cancel(self, run_id):
        with self.lock:
            folder = self.folder(run_id)
            process = self.processes.get(run_id)
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                (folder / "cancel.tmp").write_text(
                    json.dumps(
                        {
                            "status": "cancelled",
                            "phase": "cancelled",
                            "message": "Обработка остановлена пользователем",
                        }
                    )
                )
            if (folder / "cancel.tmp").exists():
                (folder / "cancel.tmp").replace(folder / "status.json")
            return self.status(folder)
