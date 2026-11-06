from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, sqrt
from typing import Any, Literal

SCHEMA_VERSION = 1
TrackStatus = Literal["active", "pending", "closed"]


class ContractError(ValueError):
    """Input data does not conform to the agreed schema_version=1 contract."""


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def _string_id(value: Any, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ContractError(f"{name} must be a string or integer")
    result = str(value).strip()
    if not result:
        raise ContractError(f"{name} must not be empty")
    return result


def _require_schema(data: dict[str, Any]) -> None:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(f"schema_version must be {SCHEMA_VERSION}")


def _id_list(data: dict[str, Any], name: str) -> frozenset[str]:
    values = data.get(name, [])
    if not isinstance(values, list):
        raise ContractError(f"{name} must be a list")
    return frozenset(_string_id(value, f"{name} item") for value in values)


@dataclass(frozen=True, slots=True)
class Camera:
    camera_id: str
    video_path: str | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    time_offset_s: float = 0.0
    overlap_with: frozenset[str] = frozenset()
    transition_to: frozenset[str] = frozenset()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Camera:
        camera_id = _string_id(data.get("camera_id"), "camera_id")
        width = data.get("width")
        height = data.get("height")
        if width is not None and (
            isinstance(width, bool) or not isinstance(width, int) or width <= 0
        ):
            raise ContractError("width must be a positive integer")
        if height is not None and (
            isinstance(height, bool) or not isinstance(height, int) or height <= 0
        ):
            raise ContractError("height must be a positive integer")
        fps = data.get("fps")
        if fps is not None:
            fps = _finite_number(fps, "fps")
            if fps <= 0:
                raise ContractError("fps must be positive")
        overlap = _id_list(data, "overlap_with")
        transitions = _id_list(data, "transition_to")
        video_path = data.get("video_path")
        if video_path is not None and not isinstance(video_path, str):
            raise ContractError("video_path must be a string or null")
        return cls(
            camera_id=camera_id,
            video_path=video_path,
            width=width,
            height=height,
            fps=fps,
            time_offset_s=_finite_number(
                data.get("time_offset_s", 0.0), "time_offset_s"
            ),
            overlap_with=overlap,
            transition_to=transitions,
        )


@dataclass(frozen=True, slots=True)
class Observation:
    camera_id: str
    frame_index: int
    timestamp_s: float
    local_track_id: str
    bbox_xyxy: tuple[float, float, float, float]
    detection_confidence: float
    embedding: tuple[float, ...] | None = None
    embedding_model_id: str | None = None

    @property
    def local_key(self) -> tuple[str, str]:
        return (self.camera_id, self.local_track_id)

    @property
    def reference(self) -> tuple[str, int, str]:
        return (self.camera_id, self.frame_index, self.local_track_id)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        _require_schema(data)
        frame_index = data.get("frame_index")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise ContractError("frame_index must be a non-negative integer")
        timestamp = _finite_number(data.get("timestamp_s"), "timestamp_s")
        if timestamp < 0:
            raise ContractError("timestamp_s must be non-negative")
        bbox = data.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ContractError("bbox_xyxy must be a list of four numbers")
        parsed_bbox = tuple(_finite_number(value, "bbox_xyxy item") for value in bbox)
        if parsed_bbox[2] <= parsed_bbox[0] or parsed_bbox[3] <= parsed_bbox[1]:
            raise ContractError("bbox_xyxy must satisfy x2 > x1 and y2 > y1")
        confidence = _finite_number(
            data.get("detection_confidence"), "detection_confidence"
        )
        if not 0.0 <= confidence <= 1.0:
            raise ContractError("detection_confidence must be in [0, 1]")

        raw_embedding = data.get("embedding")
        model_id = data.get("embedding_model_id")
        embedding = None
        if raw_embedding is not None:
            if not isinstance(raw_embedding, list) or not raw_embedding:
                raise ContractError("embedding must be a non-empty list")
            embedding = tuple(
                _finite_number(value, "embedding item") for value in raw_embedding
            )
            if sqrt(sum(value * value for value in embedding)) == 0:
                raise ContractError("embedding must have non-zero norm")
            model_id = _string_id(model_id, "embedding_model_id")
        elif model_id is not None:
            raise ContractError("embedding_model_id requires embedding")

        return cls(
            camera_id=_string_id(data.get("camera_id"), "camera_id"),
            frame_index=frame_index,
            timestamp_s=timestamp,
            local_track_id=_string_id(data.get("local_track_id"), "local_track_id"),
            bbox_xyxy=parsed_bbox,
            detection_confidence=confidence,
            embedding=embedding,
            embedding_model_id=model_id,
        )


@dataclass(frozen=True, slots=True)
class ExitEvent:
    camera_id: str
    local_track_id: str
    timestamp_s: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExitEvent:
        _require_schema(data)
        timestamp = _finite_number(data.get("timestamp_s"), "timestamp_s")
        if timestamp < 0:
            raise ContractError("timestamp_s must be non-negative")
        return cls(
            camera_id=_string_id(data.get("camera_id"), "camera_id"),
            local_track_id=_string_id(data.get("local_track_id"), "local_track_id"),
            timestamp_s=timestamp,
        )


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    timeout_s: float = 10.0
    similarity_threshold: float = 0.75
    ambiguity_margin: float = 0.05
    min_embedding_confidence: float = 0.50
    gallery_size: int = 5
    allow_topology_only_baseline: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.timeout_s) or self.timeout_s < 0:
            raise ValueError("timeout_s must be finite and non-negative")
        if not -1.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in [-1, 1]")
        if not 0.0 <= self.ambiguity_margin <= 2.0:
            raise ValueError("ambiguity_margin must be in [0, 2]")
        if not 0.0 <= self.min_embedding_confidence <= 1.0:
            raise ValueError("min_embedding_confidence must be in [0, 1]")
        if (
            isinstance(self.gallery_size, bool)
            or not isinstance(self.gallery_size, int)
            or self.gallery_size < 1
        ):
            raise ValueError("gallery_size must be a positive integer")


@dataclass(slots=True)
class TrackState:
    global_track_id: str
    start_s: float
    last_seen_s: float
    status: TrackStatus = "active"
    end_s: float | None = None
    close_reason: str | None = None
    camera_ids: set[str] = field(default_factory=set)
    camera_last_seen: dict[str, float] = field(default_factory=dict)
    embedding_model_id: str | None = None
    embedding_gallery: list[tuple[float, tuple[float, ...]]] = field(
        default_factory=list
    )
