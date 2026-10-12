"""Install stage pins into a dedicated uv-managed environment."""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VENV = ROOT / ".venv-baseline"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def interpreter(venv=VENV):
    return Path(venv) / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def read_pins(requirements):
    pins = {}
    for line in Path(requirements).read_text(encoding="utf-8").splitlines():
        line = line.partition("#")[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if not match:
            raise ValueError(f"Expected an exact name==version pin: {line}")
        name, version = match.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in pins:
            raise ValueError(f"Duplicate pin: {name}")
        pins[name] = version
    if not pins:
        raise ValueError("Requirements file has no pins")
    return pins


def uv_command():
    executable = shutil.which("uv")
    if executable:
        return [executable]
    if subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True).returncode == 0:
        return [sys.executable, "-m", "uv"]
    raise RuntimeError("uv is required; install it explicitly before running environment setup")


def installed_versions(python):
    code = ("import importlib.metadata as m,json,re; "
            "print(json.dumps({re.sub(r'[-_.]+','-',d.metadata['Name']).lower():d.version "
            "for d in m.distributions()}))")
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))


def ensure_environment(requirements, venv=VENV):
    pins = read_pins(requirements)
    uv = uv_command()
    python = interpreter(venv)
    if not python.exists():
        subprocess.run([*uv, "venv", "--python", "3.12", str(venv)], check=True)
    version = subprocess.check_output(
        [str(python), "-c", "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"], text=True
    ).strip()
    if version != "3.12":
        raise RuntimeError(f"{venv} uses Python {version}; expected 3.12. Recreate it explicitly.")
    versions = installed_versions(python)
    changed = [f"{name}=={version}" for name, version in pins.items() if versions.get(name) != version]
    if changed:
        subprocess.run(
            [*uv, "pip", "install", "--python", str(python), "--constraint", str(Path(requirements).resolve()),
             "--index", CPU_INDEX, "--index-strategy", "unsafe-best-match", *changed], check=True,
        )
        actual = installed_versions(python)
        mismatches = {name: (version, actual.get(name)) for name, version in pins.items()
                      if actual.get(name) != version}
        if mismatches:
            raise RuntimeError(f"Installed versions do not match pins: {mismatches}")
    return {"python": str(python), "installed": changed, "noop": not changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=ROOT / "baseline/requirements-detection.txt")
    args = parser.parse_args()
    print(json.dumps(ensure_environment(args.requirements), indent=2))


if __name__ == "__main__":
    main()
