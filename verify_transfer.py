"""Create or verify SHA-256 inventory for the portable reviewer-revision folder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "transfer_manifest.json"
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def file_info(name: str, path: Path) -> dict:
    # Git may change text line endings across platforms. Verify canonical LF
    # bytes for source/docs, but preserve exact bytes for all dataset files.
    if not name.startswith("data/") and (path.suffix.lower() in {".py", ".md", ".json", ".txt"}
                                          or path.name in {".gitignore", ".gitattributes"}):
        canonical = path.read_bytes().replace(b"\r\n", b"\n")
        return {"bytes": len(canonical), "sha256": hashlib.sha256(canonical).hexdigest()}
    return {"bytes": path.stat().st_size, "sha256": digest(path)}


def files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == MANIFEST:
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDE_DIRS or part.startswith("outputs") for part in relative.parts):
            continue
        yield relative.as_posix(), path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true", help="Write the inventory before transfer")
    args = parser.parse_args()
    current = {name: file_info(name, path) for name, path in files()}
    if args.create:
        MANIFEST.write_text(json.dumps({"files": current}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Recorded {len(current)} files in {MANIFEST}")
        return
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    missing = sorted(set(expected) - set(current))
    changed = sorted(name for name in set(expected) & set(current) if expected[name] != current[name])
    if missing or changed:
        print(json.dumps({"missing": missing, "changed": changed}, indent=2))
        raise SystemExit(1)
    print(f"Verified {len(expected)} files; hashes and sizes match. Extra files are ignored.")


if __name__ == "__main__":
    main()
