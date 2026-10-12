import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from baseline.common import env, weights


class EnvironmentTests(unittest.TestCase):
    def test_exact_pins_only(self):
        with tempfile.TemporaryDirectory() as folder:
            requirements = Path(folder) / "requirements.txt"
            requirements.write_text("Torch==2.6.0+cpu\nPyYAML==6.0.2\n")
            self.assertEqual(env.read_pins(requirements), {"torch": "2.6.0+cpu", "pyyaml": "6.0.2"})
            requirements.write_text("torch>=2\n")
            with self.assertRaisesRegex(ValueError, "exact"):
                env.read_pins(requirements)

    def test_installs_only_changed_pins_then_noop(self):
        with tempfile.TemporaryDirectory() as folder:
            requirements = Path(folder) / "requirements.txt"
            requirements.write_text("torch==2.6.0+cpu\nnumpy==2.1.3\n")
            python = Path(folder) / "python"
            python.touch()
            with patch.object(env, "interpreter", return_value=python), \
                 patch.object(env, "uv_command", return_value=["uv"]), \
                 patch.object(env.subprocess, "check_output", return_value="3.12\n"), \
                 patch.object(env.subprocess, "run") as run, \
                 patch.object(env, "installed_versions", side_effect=[
                     {"torch": "2.5.0", "numpy": "2.1.3"},
                     {"torch": "2.6.0+cpu", "numpy": "2.1.3"},
                     {"torch": "2.6.0+cpu", "numpy": "2.1.3"},
                 ]):
                self.assertEqual(env.ensure_environment(requirements)["installed"], ["torch==2.6.0+cpu"])
                self.assertTrue(env.ensure_environment(requirements)["noop"])
                run.assert_called_once()
                self.assertNotIn("numpy==2.1.3", run.call_args.args[0])

    def test_creates_missing_venv(self):
        with tempfile.TemporaryDirectory() as folder:
            requirements = Path(folder) / "requirements.txt"
            requirements.write_text("numpy==2.1.3\n")
            with patch.object(env, "uv_command", return_value=["uv"]), \
                 patch.object(env.subprocess, "check_output", return_value="3.12\n"), \
                 patch.object(env.subprocess, "run") as run, \
                 patch.object(env, "installed_versions", return_value={"numpy": "2.1.3"}):
                env.ensure_environment(requirements, Path(folder) / "missing")
                self.assertEqual(run.call_args.args[0][:4], ["uv", "venv", "--python", "3.12"])


class WeightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.content = b"test checkpoint"
        self.digest = hashlib.sha256(self.content).hexdigest()
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps([{"name": "test", "url": "https://example.test/model.pt",
                                             "filename": "model.pt", "sha256": self.digest}]))
        self.cache = self.root / "cache"

    def test_download_verified_and_cache_reused_without_network(self):
        with patch.object(weights, "urlopen", return_value=io.BytesIO(self.content)) as request:
            path = weights.fetch("test", self.cache, self.manifest)
            self.assertEqual(path.read_bytes(), self.content)
            self.assertEqual(weights.fetch("test", self.cache, self.manifest), path)
            request.assert_called_once()
        self.assertEqual(list(self.cache.glob("*.part")), [])

    def test_corrupt_cache_fails_without_repair(self):
        self.cache.mkdir()
        path = self.cache / "model.pt"
        path.write_bytes(b"corrupt")
        with patch.object(weights, "urlopen") as request:
            with self.assertRaisesRegex(ValueError, f"expected {self.digest}, actual"):
                weights.fetch("test", self.cache, self.manifest)
            with self.assertRaises(ValueError):
                weights.verified_path("test", self.cache, self.manifest)
            request.assert_not_called()
        self.assertEqual(path.read_bytes(), b"corrupt")

    def test_bad_download_never_becomes_cache(self):
        with patch.object(weights, "urlopen", return_value=io.BytesIO(b"wrong")), self.assertRaises(ValueError):
            weights.fetch("test", self.cache, self.manifest)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_interrupted_download_cleans_partial_file(self):
        with patch.object(weights, "urlopen", side_effect=TimeoutError), self.assertRaises(TimeoutError):
            weights.fetch("test", self.cache, self.manifest)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_missing_cache_does_not_download(self):
        with patch.object(weights, "urlopen") as request:
            with self.assertRaises(FileNotFoundError):
                weights.verified_path("test", self.cache, self.manifest)
            request.assert_not_called()

    def archive_manifest(self, *, expected_content=None):
        content = io.BytesIO()
        with ZipFile(content, "w") as archive:
            archive.writestr("model.pt", self.content)
            archive.writestr("../unwanted.txt", "must never be extracted")
        payload = content.getvalue()
        entry = json.loads(self.manifest.read_text())[0]
        entry.update(archive_member="model.pt", archive_sha256=hashlib.sha256(payload).hexdigest(),
                     cache_subdir="easyocr")
        if expected_content is not None:
            entry["sha256"] = hashlib.sha256(expected_content).hexdigest()
        self.manifest.write_text(json.dumps([entry]))
        return payload

    def test_archive_and_extracted_weight_are_verified_in_stage_cache(self):
        payload = self.archive_manifest()
        with patch.object(weights, "urlopen", return_value=io.BytesIO(payload)) as request:
            path = weights.fetch("test", self.cache, self.manifest)
            self.assertEqual(path, self.cache / "easyocr/model.pt")
            self.assertEqual(path.read_bytes(), self.content)
            self.assertEqual(weights.fetch("test", self.cache, self.manifest), path)
            request.assert_called_once()
        self.assertEqual(list(self.cache.rglob("*.part")), [])
        self.assertFalse((self.cache / "unwanted.txt").exists())

    def test_wrong_extracted_digest_does_not_promote_archive(self):
        payload = self.archive_manifest(expected_content=b"different weight")
        with patch.object(weights, "urlopen", return_value=io.BytesIO(payload)), self.assertRaises(ValueError):
            weights.fetch("test", self.cache, self.manifest)
        self.assertEqual(list((self.cache / "easyocr").iterdir()), [])

    def test_wrong_archive_digest_fails_before_extraction(self):
        self.archive_manifest()
        with (
            patch.object(weights, "urlopen", return_value=io.BytesIO(b"corrupt archive")),
            self.assertRaises(ValueError),
        ):
            weights.fetch("test", self.cache, self.manifest)
        self.assertEqual(list((self.cache / "easyocr").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
