"""Module benchmark_workers auto-generated docstring."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from robot_sf.benchmark.runner import run_batch
from robot_sf.render.helper_catalog import ensure_output_dir

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _make_scenarios(repeats: int) -> list[dict]:
    """Make scenarios.

    Args:
        repeats: Auto-generated placeholder description.

    Returns:
        list[dict]: Auto-generated placeholder description.
    """
    return [
        {
            "id": "bench-workers-uni-low-open",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": repeats,
        },
    ]


def bench(workers: int, repeats: int, out_dir: Path) -> dict:
    """Bench.

    Args:
        workers: Auto-generated placeholder description.
        repeats: Auto-generated placeholder description.
        out_dir: Auto-generated placeholder description.

    Returns:
        dict: Auto-generated placeholder description.
    """
    ensure_output_dir(out_dir)
    out_file = out_dir / f"episodes_w{workers}.jsonl"
    scenarios = _make_scenarios(repeats)
    start = time.perf_counter()
    summary = run_batch(
        scenarios,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=123,
        horizon=8,
        dt=0.1,
        record_forces=False,
        append=False,
        workers=workers,
        resume=False,
    )
    elapsed = time.perf_counter() - start
    return {
        "workers": workers,
        "repeats": repeats,
        "elapsed_sec": elapsed,
        "written": summary.get("written", 0),
        "total_jobs": summary.get("total_jobs", 0),
    }


def main() -> int:
    """Main.

    Returns:
        int: Auto-generated placeholder description.
    """
    parser = argparse.ArgumentParser(description="Benchmark run_batch with varying workers")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("results/bench_workers"))
    args = parser.parse_args()

    results = []
    for w in range(1, max(1, args.max_workers) + 1):
        res = bench(w, args.repeats, args.out)
        results.append(res)
        print(json.dumps(res))

    summary_path = args.out / "summary.json"
    args.out.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
