"""Acquire this release's owned checkpoints without authentication or Torch.

Each checkpoint is installed atomically after its size and SHA256 match the
committed manifest. Existing verified files work offline and are never fetched.
"""

import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import urllib.error
import urllib.request


def _valid(path, record):
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == record["sha256"]


def _download(path, record, source):
    candidates = [(source["browser_url"], "application/octet-stream")]
    if source.get("api_url"):
        # Anonymous public asset endpoint: useful when github.com is unavailable.
        candidates.append((source["api_url"], "application/octet-stream"))
    last_failure = "no download source"
    for attempt in range(2):
        for url, accept in candidates:
            if url.startswith("https://api.github.com/"):
                url += (
                    ("&" if "?" in url else "?")
                    + "download=1&cache="
                    + str(time.time_ns())
                )
            descriptor, filename = tempfile.mkstemp(
                prefix="." + path.name + ".", suffix=".part", dir=path.parent
            )
            temporary = Path(filename)
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "Wan-Restore-Fast", "Accept": accept}
                )
                digest, count, next_notice = hashlib.sha256(), 0, 32 * 1024 * 1024
                with os.fdopen(descriptor, "wb") as output:
                    with urllib.request.urlopen(request, timeout=45) as response:
                        for chunk in iter(lambda: response.read(1024 * 1024), b""):
                            count += len(chunk)
                            if count > record["bytes"]:
                                raise ValueError("download exceeds expected size")
                            output.write(chunk)
                            digest.update(chunk)
                            if count >= next_notice:
                                print(
                                    f"[weights] {path.name}: {count / 2**20:.0f}/{record['bytes'] / 2**20:.0f} MiB",
                                    file=sys.stderr,
                                    flush=True,
                                )
                                next_notice += 32 * 1024 * 1024
                    output.flush()
                    os.fsync(output.fileno())
                if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                    raise ValueError("download size/SHA256 mismatch")
                os.replace(temporary, path)
                return
            except urllib.error.HTTPError as error:
                last_failure = f"HTTP {error.code}"
            except urllib.error.URLError as error:
                last_failure = (
                    f"network/proxy unavailable ({type(error.reason).__name__}); "
                    "check internet access and proxy settings"
                )
            except (OSError, ValueError, http.client.HTTPException) as error:
                last_failure = type(error).__name__ + (
                    ": " + str(error) if isinstance(error, ValueError) else ""
                )
            finally:
                temporary.unlink(missing_ok=True)
        if attempt == 0:
            time.sleep(2)
    raise RuntimeError(
        f"Could not download {path.name} ({last_failure}). Existing weights were preserved. Retry with internet access or copy the matching file from the public GitHub Release shown in weights/sources.json."
    )


def ensure_weights(directory, *, offline=None):
    """Verify local checkpoints; fetch only missing or corrupt files by default."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    records = [row for row in manifest["files"] if row["file"].endswith(".safetensors")]
    if len(records) != 3:
        raise ValueError("Expected three checkpoint entries in weights/manifest.json")
    offline = (
        os.environ.get("WAN_RESTORE_OFFLINE", "0") == "1"
        if offline is None
        else offline
    )
    results = []
    sources = None
    for record in records:
        name = record["file"]
        if Path(name).name != name:
            raise ValueError(
                "Checkpoint filename must stay inside the weights directory"
            )
        target = directory / name
        if _valid(target, record):
            results.append({"file": name, "status": "verified_local"})
            continue
        if offline:
            raise RuntimeError(
                f"Offline mode: {name} is missing or failed SHA256 verification. Run python run.py --download-weights while connected, then retry."
            )
        if sources is None:
            sources = json.loads(
                (directory / "sources.json").read_text(encoding="utf-8")
            )["assets"]
        print(
            f"[weights] Downloading {name} ({record['bytes'] / 2**20:.1f} MiB); SHA256 will be verified.",
            file=sys.stderr,
            flush=True,
        )
        _download(target, record, sources[name])
        results.append({"file": name, "status": "downloaded_verified"})
    return results
