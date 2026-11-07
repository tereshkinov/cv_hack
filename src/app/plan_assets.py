"""Local raster floor-plan assets; no remote URLs or arbitrary file paths."""

import hashlib
from pathlib import Path

import cv2

SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def plans(root):
    root = Path(root).resolve()
    result = {}
    for directory in [root / "data/raw", root / "runs/app/plan-uploads"]:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() not in SUFFIXES or not path.is_file():
                continue
            path = path.resolve()
            if not path.is_relative_to(root):
                continue
            key = hashlib.sha256(str(path.relative_to(root)).encode()).hexdigest()[:20]
            result[key] = path
    return result


def describe(root, key, path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Не удалось прочитать изображение плана")
    height, width = image.shape[:2]
    return {
        "id": key,
        "name": str(path.relative_to(root)),
        "width": width,
        "height": height,
    }
