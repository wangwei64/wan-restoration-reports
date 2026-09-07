"""Check the packaged inference graph and fixed configuration without loading Wan."""

import hashlib, json, sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from restore_fast.config import GenerationConfig
from restore_fast.models.restorer import load_models


def main():
    config = GenerationConfig()
    assert config.to_dict() == json.loads(
        (PACKAGE / "configs/fastest.json").read_text(encoding="utf-8")
    )
    assert (
        config.steps,
        config.warmup_steps,
        config.refinement_budget,
        config.quality_tolerance,
    ) == (50, 10, 2.0, 0.010)
    manifest = json.loads((PACKAGE / "weights/manifest.json").read_text())
    for row in manifest["files"]:
        assert (
            hashlib.sha256((PACKAGE / "weights" / row["file"]).read_bytes()).hexdigest()
            == row["sha256"]
        ), row["file"]
    models = load_models(PACKAGE / "weights", device="cpu")
    counts = [sum(p.numel() for p in model.parameters()) for model in models]
    assert counts == [72911811, 95567, 88481], counts
    assert not any(
        name.startswith("backbone.quality_") for name, _ in models[0].named_parameters()
    )
    print(
        json.dumps(
            {
                "passed": True,
                "parameters": counts,
                "total_parameters": sum(counts),
                "external_wan_loaded": False,
                "weights_verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
