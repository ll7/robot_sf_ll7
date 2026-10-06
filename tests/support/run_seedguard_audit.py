"""Run worker safety proofs, then audit the full suite from an exact checkout.

Use on Slurm: uv run python tests/support/run_seedguard_audit.py --output DIR.
The optional exclusion is for independently owned merge-train migrations only;
default execution includes every test file under the configured test roots.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Preserve proof logs, suite results, and one merged runtime audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    proof = out / "workers"
    proof.mkdir()  # Refuse stale proofs from a previous run.
    os.environ["SEEDGUARD_WORKER_PROOF"] = str(proof)
    os.environ["ROBOT_SF_PYTEST_SEED_AUDIT"] = str(out / "audit-{worker}.jsonl")
    command = [sys.executable, "-m", "pytest", "-n", str(args.workers), "-q"]
    with (out / "proof.log").open("w", encoding="utf-8") as log:
        probe = subprocess.run(
            [
                *command,
                "--dist=each",
                "tests/test_heldout_seed_guard.py",
                f"--junitxml={out}/proof.xml",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    workers = {p.stem for p in proof.glob("gw*.json") if json.loads(p.read_text())["active"]}
    if probe.returncode != 0 or workers != {f"gw{i}" for i in range(args.workers)}:
        raise RuntimeError(
            f"Worker safety proofs failed: exit={probe.returncode}, workers={workers}"
        )
    paths = sorted(
        str(path)
        for base in ("tests", "fast-pysf/tests")
        for path in Path(base).rglob("*.py")
        if (path.name.startswith("test_") or path.name.endswith("_test.py"))
        and str(path) not in args.exclude
    )
    with (out / "suite.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [
                *command,
                "--continue-on-collection-errors",
                "--tb=short",
                f"--junitxml={out}/suite.xml",
                *paths,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    events = [
        json.loads(row)
        for file in sorted(out.glob("audit-*.jsonl"))
        for row in file.read_text().splitlines()
    ]
    (out / "audit.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    (out / "result.json").write_text(
        json.dumps(
            {
                "exitcode": result.returncode,
                "heldout_attempts": len(events),
                "workers_proved": sorted(workers),
                "excluded_paths": args.exclude,
            }
        )
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
