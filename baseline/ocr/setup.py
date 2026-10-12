"""Prepare OCR without changing the pinned detection stack; installation is explicit."""

import argparse
import tempfile
from pathlib import Path

from baseline.common.env import ROOT, ensure_environment, installed_versions, interpreter, read_pins


def combined_pins():
    detection = read_pins(ROOT / "baseline/requirements-detection.txt")
    ocr = read_pins(ROOT / "baseline/requirements-ocr.txt")
    conflicts = {
        name: (version, detection[name])
        for name, version in ocr.items()
        if name in detection and version != detection[name]
    }
    if conflicts:
        raise ValueError(f"OCR and detection pins conflict: {conflicts}")
    return {**detection, **ocr}


def prepare(*, install=False):
    pins = combined_pins()
    if install:
        with tempfile.TemporaryDirectory() as folder:
            requirements = Path(folder) / "requirements.txt"
            requirements.write_text("".join(f"{name}=={version}\n" for name, version in pins.items()), encoding="utf-8")
            return ensure_environment(requirements)

    python = interpreter()
    versions = installed_versions(python) if python.is_file() else {}
    return {
        "python": str(python),
        "missing_or_mismatched": {
            name: {"expected": version, "installed": versions.get(name)}
            for name, version in pins.items()
            if versions.get(name) != version
        },
    }


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Explicitly allow creating/installing the baseline env")
    args = parser.parse_args()
    print(json.dumps(prepare(install=args.install), indent=2))


if __name__ == "__main__":
    main()
