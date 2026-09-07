"""Optional exact fingerprint check of external Wan weights (reads large files)."""

import hashlib, json, sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from restore_fast.config import external_paths


def main():
    _, directory = external_paths()
    records = json.loads((PACKAGE / "tests/external_weights.json").read_text())["files"]
    for row in records:
        p = directory / row["file"]
        assert p.stat().st_size == row["bytes"], str(p)
        digest = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        assert digest.hexdigest() == row["sha256"], str(p)
        print("Verified " + row["file"])


if __name__ == "__main__":
    main()
