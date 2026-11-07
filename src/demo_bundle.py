"""Restore the committed offline demo without replacing any local files."""

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(p in {"..", "."} for p in relative.parts)
        or "\\" in name
    ):
        raise ValueError(f"Unsafe bundle path: {name}")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Bundle path escapes root: {name}")
    # Never write through a pre-existing symlink, even if its target is in root.
    if any(p.is_symlink() for p in [target, *target.parents] if p != root.parent):
        raise ValueError(f"Symlink in bundle destination: {name}")
    return target


class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None
        super().__init__()

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open("rb")
            n = self.current.readinto(buffer)
            if n:
                return n
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def restore_bundle(root, *, verify=False):
    root = Path(root).resolve()
    bundle = root / "demo"
    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        return {"restored": 0, "available": False}
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported demo bundle version")
    files = manifest["files"]
    destinations = {name: safe_path(root, name) for name in files}
    missing = {name for name, path in destinations.items() if not path.exists()}
    if not missing and not verify:
        return {"restored": 0, "available": True}
    parts = []
    for part in manifest["parts"]:
        path = safe_path(bundle, part["file"])
        if not path.is_file() or digest(path) != part["sha256"]:
            raise ValueError(f"Demo archive checksum mismatch: {path.name}")
        parts.append(path)
    seen, restored = set(), 0
    with (
        PartsReader(parts) as raw,
        io.BufferedReader(raw) as stream,
        tarfile.open(fileobj=stream, mode="r|gz") as archive,
    ):
        for member in archive:
            name = member.name
            if name not in files or name in seen or not member.isfile():
                raise ValueError(f"Unexpected demo member: {name}")
            seen.add(name)
            info = files[name]
            if member.size != info["size"]:
                raise ValueError(f"Demo member size mismatch: {name}")
            target = destinations[name]
            if name not in missing:
                continue  # Local edits and original longer recordings win.
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent, delete=False
                ) as out:
                    temporary = Path(out.name)
                    h = hashlib.sha256()
                    source = archive.extractfile(member)
                    while block := source.read(1024 * 1024):
                        h.update(block)
                        out.write(block)
                if h.hexdigest() != info["sha256"]:
                    raise ValueError(f"Demo member checksum mismatch: {name}")
                os.chmod(temporary, 0o644)
                # Create only if still absent; never replace a concurrent edit.
                try:
                    os.link(temporary, target)
                    restored += 1
                except FileExistsError:
                    pass
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    if seen != set(files):
        raise ValueError("Incomplete demo archive")
    if verify:
        mismatches = [
            name
            for name, path in destinations.items()
            if digest(path) != files[name]["sha256"]
        ]
        if mismatches:
            raise ValueError(f"Local files differ from demo: {mismatches[:5]}")
    return {"restored": restored, "available": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Also compare local files with the committed snapshot",
    )
    args = parser.parse_args()
    print(json.dumps(restore_bundle(args.root, verify=args.verify), ensure_ascii=False))


if __name__ == "__main__":
    main()
