"""Guard artifact imports against mismatched experiments before writing any run."""

import json

import pytest

from scripts.appearance_preview import build
from src.vision.embeddings import sha256


@pytest.mark.parametrize("mutation", ["box", "missing_feature", "model_space"])
def test_appearance_preview_rejects_changed_detections_or_mixed_spaces(
    tmp_path, monkeypatch, mutation
):
    from scripts import appearance_preview

    monkeypatch.setattr(appearance_preview, "ROOT", tmp_path)
    source, experiment = tmp_path / "source", tmp_path / "experiment"
    (source / "vision").mkdir(parents=True)
    experiment.mkdir()
    row = {
        "schema_version": 1,
        "camera_id": "a",
        "frame_index": 0,
        "timestamp_s": 0,
        "local_track_id": "l1",
        "bbox_xyxy": [10, 10, 30, 50],
        "detection_confidence": 0.9,
        "embedding": [1.0, 0.0],
        "embedding_model_id": "old",
    }
    observations = source / "vision/observations.jsonl"
    observations.write_text(json.dumps(row) + "\n")
    (source / "cameras.json").write_text('{"cameras":[]}')
    changed = {**row, "embedding": [0.0, 1.0], "embedding_model_id": "gamma"}
    if mutation == "box":
        changed["bbox_xyxy"] = [10, 10, 31, 50]
    elif mutation == "missing_feature":
        changed["embedding"] = None
        changed.pop("embedding_model_id")
    else:
        changed["embedding_model_id"] = "another-model"
    (experiment / "gamma.jsonl").write_text(json.dumps(changed) + "\n")
    (experiment / "experiment.json").write_text(
        json.dumps(
            {
                "source_observations_sha256": sha256(observations),
                "source_cameras_sha256": sha256(source / "cameras.json"),
                "models": {"gamma": "gamma"},
            }
        )
    )
    with pytest.raises(ValueError):
        build(source, experiment, "gamma")
    assert not (tmp_path / "runs").exists()
    assert json.loads(observations.read_text()) == row
