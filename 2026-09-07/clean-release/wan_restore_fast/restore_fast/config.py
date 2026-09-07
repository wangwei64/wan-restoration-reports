"""Reproducible defaults and external Wan discovery; no experiment-directory paths."""

from dataclasses import dataclass, asdict
from pathlib import Path
import os, sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


@dataclass(frozen=True)
class GenerationConfig:
    height: int = 480
    width: int = 832
    num_frames: int = 41
    fps: int = 16
    steps: int = 50
    warmup_steps: int = 10
    shift: float = 2.0
    guidance_scale: float = 5.0
    refinement_budget: float = 2.0
    quality_tolerance: float = 0.010
    default_seed: int = 20260828
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT

    def to_dict(self):
        return asdict(self)


def external_paths(repo=None, checkpoint=None):
    def resolve(explicit, env, candidates, marker):
        supplied = explicit or os.environ.get(env)
        paths = [Path(supplied).expanduser()] if supplied else candidates
        for p in paths:
            if (p / marker).is_file():
                return p.resolve()
        raise FileNotFoundError(
            f"{env}: cannot find {marker}. Put external Wan next to this package as described in docs/WAN_INTERFACE.md, or set {env}. Searched: "
            + ", ".join(map(str, paths))
        )

    base = PACKAGE_ROOT.parent
    repo = resolve(
        repo,
        "WAN_REPO",
        [base / "Wan2.1", PACKAGE_ROOT / "external/Wan2.1", Path.cwd() / "Wan2.1"],
        "wan/modules/model.py",
    )
    checkpoint = resolve(
        checkpoint,
        "WAN_CHECKPOINT",
        [
            base / "models/Wan2.1-T2V-1.3B",
            PACKAGE_ROOT / "external/Wan2.1-T2V-1.3B",
            repo / "Wan2.1-T2V-1.3B",
        ],
        "config.json",
    )
    return repo, checkpoint
