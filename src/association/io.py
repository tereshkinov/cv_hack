from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from .models import Camera, ExitEvent, Observation

T = TypeVar("T")


def load_cameras(path: Path) -> list[Camera]:
    data = _load_json(path)
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("camera configuration must be an object with schema_version=1")
    raw_cameras = data.get("cameras")
    if not isinstance(raw_cameras, list):
        raise TypeError("camera configuration must contain a cameras list")
    if not all(isinstance(item, dict) for item in raw_cameras):
        raise TypeError("each camera must be a JSON object")
    return [Camera.from_dict(item) for item in raw_cameras]


def load_observations(path: Path) -> list[Observation]:
    return _load_jsonl(path, Observation.from_dict)


def load_exit_events(path: Path) -> list[ExitEvent]:
    return _load_jsonl(path, ExitEvent.from_dict)


def write_results(
    output_dir: Path,
    assignments: list[dict[str, object]],
    tracks: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_text(
        output_dir / "assignments.jsonl",
        "".join(_json_dumps(row) + "\n" for row in assignments),
    )
    _atomic_text(
        output_dir / "tracks.json",
        _json_dumps({"schema_version": 1, "tracks": tracks}) + "\n",
    )


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def _load_jsonl(path: Path, factory: Callable[[dict[str, Any]], T]) -> list[T]:
    result: list[T] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise TypeError("row must be a JSON object")
            result.append(factory(data))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            raise ValueError(f"invalid row {line_number} in {path}: {error}") from error
    return result


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
