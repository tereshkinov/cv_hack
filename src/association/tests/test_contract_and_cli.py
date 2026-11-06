from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.association.cli import main
from src.association.models import ContractError, Observation


def test_observation_rejects_embedding_without_model_id() -> None:
    with pytest.raises(ContractError, match="embedding_model_id"):
        Observation.from_dict(
            {
                "schema_version": 1,
                "camera_id": "cam1",
                "frame_index": 0,
                "timestamp_s": 0,
                "local_track_id": "l1",
                "bbox_xyxy": [0, 0, 10, 20],
                "detection_confidence": 0.9,
                "embedding": [1.0, 0.0],
            }
        )


def test_cli_processes_repository_examples_in_explicit_baseline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = Path(__file__).resolve().parents[3]
    output_dir = tmp_path / "result"

    result = main(
        [
            "--cameras",
            str(root / "examples" / "cameras.json"),
            "--observations",
            str(root / "examples" / "observations.jsonl"),
            "--output-dir",
            str(output_dir),
            "--allow-topology-only-baseline",
        ]
    )

    assert result == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["assignments"] == 2
    assert summary["global_tracks"] == 1
    assignment_lines = (output_dir / "assignments.jsonl").read_text().splitlines()
    assignments = [json.loads(line) for line in assignment_lines]
    tracks = json.loads((output_dir / "tracks.json").read_text())
    assert [row["global_track_id"] for row in assignments] == ["g1", "g1"]
    assert tracks["tracks"][0]["close_reason"] == "end_of_video"


def test_cli_export_is_byte_reproducible(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    first = tmp_path / "first"
    second = tmp_path / "second"
    common = [
        "--cameras",
        str(root / "examples" / "cameras.json"),
        "--observations",
        str(root / "examples" / "observations.jsonl"),
        "--allow-topology-only-baseline",
    ]

    main([*common, "--output-dir", str(first)])
    main([*common, "--output-dir", str(second)])

    assert (first / "assignments.jsonl").read_bytes() == (
        second / "assignments.jsonl"
    ).read_bytes()
    assert (first / "tracks.json").read_bytes() == (second / "tracks.json").read_bytes()
