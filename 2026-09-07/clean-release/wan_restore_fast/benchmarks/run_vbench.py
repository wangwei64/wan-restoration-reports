"""Fixed VBench prompt sample; fresh same-seed native GT50, resume completed runs."""

import argparse, hashlib, json, sys, time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from restore_fast.pipeline import RestorationPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PACKAGE / "benchmarks/results")
    args = parser.parse_args()
    source = PACKAGE / "benchmarks/vbench_source.json"
    protocol = json.loads((PACKAGE / "benchmarks/protocol.json").read_text())
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == protocol["vbench_source_sha256"]
    )
    full = json.loads(source.read_text())
    cases = [
        {"index": i, "seed": protocol["seed"], **full[i]} for i in protocol["indices"]
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases.json").write_text(json.dumps(cases, indent=2))
    pipe = RestorationPipeline()
    completed = []
    try:
        for case in cases:
            folder = args.output / f'{case["index"]:04d}_{case["seed"]}'
            for mode in ["fastest", "gt50"]:
                target = folder / mode
                status = {
                    "stage": mode,
                    "index": case["index"],
                    "prompt": case["prompt_en"],
                    "completed_cases": len(completed),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                (PACKAGE / "benchmarks/status.json").write_text(
                    json.dumps(status, indent=2)
                )
                print(json.dumps(status), flush=True)
                if not (target / "run.json").exists():
                    if mode == "fastest":
                        pipe.generate(case["prompt_en"], case["seed"], target)
                    else:
                        pipe.native_reference(case["prompt_en"], case["seed"], target)
                r = json.loads((target / "run.json").read_text())
                if r["prompt"] != case["prompt_en"] or r["seed"] != case["seed"]:
                    raise ValueError(f"Output does not match fixed case: {target}")
                print(
                    json.dumps({"mode": mode, "seconds": r["online_seconds"]}),
                    flush=True,
                )
            fast = json.loads((folder / "fastest/run.json").read_text())
            gt = json.loads((folder / "gt50/run.json").read_text())
            completed.append(
                {
                    **case,
                    "fast_seconds": fast["online_seconds"],
                    "gt50_seconds": gt["online_seconds"],
                    "speedup": gt["online_seconds"] / fast["online_seconds"],
                    "active_ratio": fast["active_ratio"],
                    "initial_active_ratio": fast["initial_active_ratio"],
                }
            )
            summary = {
                "protocol": protocol,
                "results": completed,
                "model_load_seconds": pipe.load_seconds,
                "aggregate_speedup": sum(x["gt50_seconds"] for x in completed)
                / sum(x["fast_seconds"] for x in completed),
                "full_vbench_score": False,
            }
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        (PACKAGE / "benchmarks/status.json").write_text(
            json.dumps(
                {"stage": "generation_complete", "completed_cases": len(completed)},
                indent=2,
            )
        )
    finally:
        pipe.close()


if __name__ == "__main__":
    main()
