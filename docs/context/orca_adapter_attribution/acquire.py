"""Run every (configuration, cell, seed block) job of one stage with resume.

A job is complete when its receipt exists; rerunning skips complete jobs, so a
partial acquisition can be resumed or extended without discarding records.

    python acquire.py --contract contract.yaml --stage main --out-dir <run_dir> --workers 12
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


def main() -> int:
    """Dispatch pending jobs and report failures.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--stage", required=True, help="seed block key in contract.seeds")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--block", type=int, default=10, help="seeds per job")
    parser.add_argument("--configs", nargs="*")
    parser.add_argument("--cells", nargs="*")
    args = parser.parse_args()

    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    seeds = [int(s) for s in contract["seeds"][args.stage]]
    configs = args.configs or [c["key"] for c in contract["configurations"]]
    cells = args.cells or list(contract["cells"])
    harness = Path(__file__).resolve().parent / "run_xorca.py"
    # Frozen planner configs include siblings by repository-relative path.
    repo_root = harness.parents[3]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logs = args.out_dir / "logs"
    logs.mkdir(exist_ok=True)

    jobs = []
    for config in configs:
        for cell in cells:
            for i in range(0, len(seeds), args.block):
                block = seeds[i : i + args.block]
                stem = f"{args.stage}__{config}__{cell}__{block[0]}-{block[-1]}"
                if (args.out_dir / f"{stem}.receipt.json").exists():
                    continue
                jobs.append((stem, config, cell, block))
    print(f"{len(jobs)} jobs pending", flush=True)

    env = dict(os.environ, LOGURU_LEVEL="WARNING", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(job: tuple) -> tuple[str, int, float]:
        stem, config, cell, block = job
        cmd = [
            sys.executable,
            str(harness),
            "--contract",
            str(args.contract.resolve()),
            "--config",
            config,
            "--cell",
            cell,
            "--tag",
            args.stage,
            "--out-dir",
            str(args.out_dir.resolve()),
            "--seeds",
            *map(str, block),
        ]
        t0 = time.time()
        with open(logs / f"{stem}.log", "w", encoding="utf-8") as handle:
            rc = subprocess.call(
                cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=repo_root
            )
        return stem, rc, time.time() - t0

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for stem, rc, dt in pool.map(run, jobs):
            failed += rc != 0
            print(f"{'ok ' if rc == 0 else 'ERR'} {stem} {dt:.0f}s", flush=True)
    print(f"done; failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
