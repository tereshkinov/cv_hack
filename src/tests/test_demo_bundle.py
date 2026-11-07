import hashlib
import io
import json
import tarfile

import pytest

from src.demo_bundle import restore_bundle


def bundle(root, entries=None):
    entries = entries or {
        "configs/office.local.json": b'{"saved":true}',
        "data/raw/demo.txt": b"demo",
    }
    directory = root / "demo"
    directory.mkdir()
    stream = io.BytesIO()
    files = {}
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data in entries.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
            files[name] = {
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
    data = stream.getvalue()
    parts = []
    for index, chunk in enumerate([data[:23], data[23:]]):
        name = f"assets.part-{index}"
        (directory / name).write_bytes(chunk)
        parts.append({"file": name, "sha256": hashlib.sha256(chunk).hexdigest()})
    manifest = {"schema_version": 1, "files": files, "parts": parts}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_fresh_clone_restores_split_archive_and_is_idempotent(tmp_path):
    bundle(tmp_path)
    assert restore_bundle(tmp_path, verify=True)["restored"] == 2
    assert (tmp_path / "configs/office.local.json").read_bytes() == b'{"saved":true}'
    assert restore_bundle(tmp_path)["restored"] == 0


def test_restore_keeps_local_edits_when_other_assets_are_missing(tmp_path):
    bundle(tmp_path)
    (tmp_path / "configs").mkdir()
    project = tmp_path / "configs/office.local.json"
    project.write_text("local edit")
    assert restore_bundle(tmp_path)["restored"] == 1
    assert project.read_text() == "local edit"
    with pytest.raises(ValueError, match="differ"):
        restore_bundle(tmp_path, verify=True)


def test_corrupt_part_is_rejected_before_any_asset_is_written(tmp_path):
    bundle(tmp_path)
    (tmp_path / "demo/assets.part-0").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        restore_bundle(tmp_path)
    assert not (tmp_path / "configs").exists()


def test_traversal_and_symlink_destinations_are_rejected(tmp_path):
    bundle(tmp_path, {"../outside.txt": b"bad"})
    with pytest.raises(ValueError, match="Unsafe"):
        restore_bundle(tmp_path)


def test_symlink_destination_is_not_followed(tmp_path):
    bundle(tmp_path)
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "configs").symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        restore_bundle(tmp_path)
    assert not list(real.iterdir())
