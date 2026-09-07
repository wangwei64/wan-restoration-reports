"""Offline tests for download integrity, fallback and cache behavior; no GPU."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from restore_fast.weights import ensure_weights


class WeightDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.data = {
            name + ".safetensors": (name + "-payload").encode()
            for name in ["features", "restorer", "subject"]
        }
        files = [
            {
                "file": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in self.data.items()
        ]
        (self.directory / "manifest.json").write_text(json.dumps({"files": files}))
        sources = {
            name: {
                "browser_url": "https://example.invalid/browser/" + name,
                "api_url": "https://example.invalid/api/" + name,
            }
            for name in self.data
        }
        (self.directory / "sources.json").write_text(json.dumps({"assets": sources}))
        self.addCleanup(self.temporary.cleanup)

    def reply(self, request, **kwargs):
        name = request.full_url.rsplit("/", 1)[-1]
        self.assertNotIn("Authorization", dict(request.header_items()))
        return io.BytesIO(self.data[name])

    def test_download_then_offline_reuse(self):
        with patch("urllib.request.urlopen", side_effect=self.reply) as request:
            first = ensure_weights(self.directory, offline=False)
            self.assertEqual(request.call_count, 3)
        self.assertTrue(all(r["status"] == "downloaded_verified" for r in first))
        with patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("must not access network"),
        ):
            second = ensure_weights(self.directory, offline=True)
        self.assertTrue(all(r["status"] == "verified_local" for r in second))
        self.assertFalse(list(self.directory.glob("*.part")))

    def test_api_fallback(self):
        def reply(request, **kwargs):
            if "/browser/" in request.full_url:
                raise urllib.error.HTTPError(
                    request.full_url, 503, "Unavailable", {}, None
                )
            return self.reply(request, **kwargs)

        with patch("urllib.request.urlopen", side_effect=reply) as request:
            ensure_weights(self.directory, offline=False)
        self.assertEqual(request.call_count, 6)

    def test_corrupt_download_never_replaces_existing_file(self):
        target = self.directory / "features.safetensors"
        target.write_bytes(b"previous-local-file")
        with patch(
            "urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(b"bad")
        ), patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "preserved"):
                ensure_weights(self.directory, offline=False)
        self.assertEqual(target.read_bytes(), b"previous-local-file")
        self.assertFalse(list(self.directory.glob("*.part")))

    def test_offline_missing_file_does_not_access_network(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=AssertionError("must not access network"),
        ):
            with self.assertRaisesRegex(RuntimeError, "Offline mode"):
                ensure_weights(self.directory, offline=True)


if __name__ == "__main__":
    unittest.main()
