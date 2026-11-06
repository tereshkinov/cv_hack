"""Translate UI configuration into the shared camera contract."""

from math import isfinite

import cv2


def camera_manifest(project, videos, root):
    if not project["cameras"]:
        raise ValueError("Добавьте хотя бы одну камеру и выберите для неё видео")
    result = []
    for camera in project["cameras"]:
        path = videos.get(camera.get("video_id"))
        if path is None:
            raise ValueError(f"{camera['name']}: выберите доступное видео")
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видео {camera['name']}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not isfinite(fps) or fps <= 0 or count < 1 or width < 1 or height < 1:
                raise ValueError("Некорректные параметры видео")
        finally:
            cap.release()
        for x, y in camera.get("video_points", []):
            if not (0 <= x <= width and 0 <= y <= height):
                raise ValueError(f"{camera['name']}: точки привязки вне кадра")
        result.append(
            {
                "camera_id": camera["id"],
                "video_path": str(path.relative_to(root)),
                "width": width,
                "height": height,
                "fps": fps,
                "time_offset_s": camera.get("time_offset_s", 0),
                "overlap_with": camera.get("overlap_with", []),
                "transition_to": camera.get("transition_to", []),
            }
        )
    return result
