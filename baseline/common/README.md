# Shared baseline infrastructure

From the repository root, with uv installed and Python 3.12 available:

```text
python -m baseline.common.env
python -m baseline.common.env
python -m baseline.common.weights fetch
python -m baseline.common.weights verify
python -m unittest baseline.tests.test_common -v
```

The environment command creates `.venv-baseline` and installs only missing or
mismatched exact pins from `baseline/requirements-detection.txt`. The second
command is a no-op. Use `--requirements /path/to/stage-requirements.txt` for
another stage. Keep shared package versions compatible across stage files.
Existing environments with a different Python minor version fail explicitly.
The pinned CPU stack supports Windows and Linux on x86-64.

`models.json` is an array of `{name, url, filename, sha256}` entries. Use immutable
download URLs and verify model receipts independently when adding models.
Filenames must be plain filenames within `baseline/weights/`. EasyOCR entries
add `cache_subdir: easyocr`, `archive_member` and `archive_sha256`; both the ZIP
and extracted weight have separate pinned SHA-256 values.

Consumers call `weights.verified_path(model_id)` to obtain a verified local
`Path`; this operation never downloads. Only `weights.fetch(model_id)` downloads,
and only when the cached file is absent. It writes a temporary file, verifies
its SHA-256, then atomically promotes it. A corrupt existing file is a hard
failure showing expected and actual digests; callers cannot use it through the
verified-path API. Explicitly remove corrupt files before refetching.

For an archive entry, fetching verifies the archive, extracts only the exact
named member, verifies the extracted weight and promotes it atomically.
Unrelated archive paths are never extracted. OCR model fetches are explicit:

```text
python -m baseline.common.weights fetch easyocr-craft
python -m baseline.common.weights fetch easyocr-zh-sim-g2
```

OCR can reuse this API with its own manifest entries and stage requirements.
Do not pass a model name to a library that may auto-download its checkpoint.
