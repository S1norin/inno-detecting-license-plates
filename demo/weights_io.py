"""Standalone SHA-256 weight verification (no downloads)."""

import hashlib
from pathlib import Path


def sha256_of(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_weight(path, expected_sha256: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing weight file: {path}. Fetch it with: "
            "python -m baseline.common.weights fetch <name>"
        )
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise ValueError(f"Weight digest mismatch for {path}: expected {expected_sha256}, got {actual}")
    return path
