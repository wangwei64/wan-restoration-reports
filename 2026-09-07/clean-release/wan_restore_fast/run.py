"""Run the fixed fastest release with one prompt; no subject/mask/budget input required."""

import argparse, json, sys, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prompt", nargs="?", help="Text describing the video")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional reproducibility override; default 20260828",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional new output directory; otherwise generated automatically",
    )
    p.add_argument(
        "--wan-repo",
        type=Path,
        help="Optional external Wan source; auto-discovered by default",
    )
    p.add_argument(
        "--wan-checkpoint",
        type=Path,
        help="Optional external official Wan weights; auto-discovered by default",
    )
    p.add_argument(
        "--doctor",
        action="store_true",
        help="Check dependencies, GPU, external Wan and packaged weights",
    )
    a = p.parse_args()
    if a.doctor:
        from restore_fast.doctor import check

        result = check(a.wan_repo, a.wan_checkpoint)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ready"] else 1
    if not a.prompt:
        p.error(
            'Provide a prompt, for example: python run.py "A cute Corgi playing in a park"'
        )
    from restore_fast.pipeline import RestorationPipeline

    start = time.perf_counter()
    pipe = RestorationPipeline(repo=a.wan_repo, checkpoint=a.wan_checkpoint)
    try:
        result = pipe.generate(a.prompt, a.seed, a.out)
        result["report"]["cli_wall_seconds_including_load_and_artifacts"] = (
            time.perf_counter() - start
        )
        Path(result["directory"], "run.json").write_text(
            json.dumps(result["report"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "video": result["video"],
                    "directory": result["directory"],
                    "online_seconds": result["report"]["online_seconds"],
                    "model_load_seconds": pipe.load_seconds,
                    "wall_seconds": result["report"][
                        "cli_wall_seconds_including_load_and_artifacts"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        pipe.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
