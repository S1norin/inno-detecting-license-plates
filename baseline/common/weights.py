"""Explicit weight fetching and download-free cache verification."""

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name("models.json")
CACHE = ROOT / "baseline/weights"
DEFAULT_MODEL = "yolo11s-license-plate"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_entry(name, manifest=MANIFEST):
    entries = json.loads(Path(manifest).read_text(encoding="utf-8"))
    matches = [entry for entry in entries if entry["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest entry for {name!r}, found {len(matches)}")
    entry = matches[0]
    filename = entry["filename"]
    if not filename or "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise ValueError("Manifest filename must be a plain filename")
    if not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"]):
        raise ValueError("Manifest sha256 must be a lowercase 64-character hex digest")
    if not entry["url"].startswith("https://"):
        raise ValueError("Weight URL must use HTTPS")
    return entry


def verify(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Weight checksum mismatch for {path}: expected {expected}, actual {actual}")
    return Path(path)


def verified_path(name=DEFAULT_MODEL, cache=CACHE, manifest=MANIFEST):
    entry = model_entry(name, manifest)
    path = Path(cache) / entry["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached weights: {path}. Run python -m baseline.common.weights fetch {name}")
    return verify(path, entry["sha256"])


def fetch(name=DEFAULT_MODEL, cache=CACHE, manifest=MANIFEST):
    entry = model_entry(name, manifest)
    cache = Path(cache)
    destination = cache / entry["filename"]
    if destination.exists():
        return verified_path(name, cache, manifest)
    cache.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            with urlopen(entry["url"], timeout=60) as source:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
        verify(temporary, entry["sha256"])
        os.replace(temporary, destination)
        return verified_path(name, cache, manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fetch", "verify"])
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print((fetch if args.action == "fetch" else verified_path)(args.model))


if __name__ == "__main__":
    main()
