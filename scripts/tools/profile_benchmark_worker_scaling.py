"""Profile policy-search benchmark runtime across worker counts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json

REPO_ROOT = Path(__file__).resolve().parents[2]


def _worker_label(worker_count: int) -> str:
    """Return the output subdirectory label for a worker count."""
    return f"workers_{worker_count}"


def _run_candidate_stage(
    *,
    candidate: str,
    stage: str,
    worker_count: int,
    output_root: Path,
    horizon: int | None,
    extra_args: list[str],
) -> dict[str, Any]:
    """Run one candidate stage and return a compact timing row."""
    run_dir = output_root / _worker_label(worker_count)
    display_run_dir = (
        run_dir.relative_to(REPO_ROOT) if run_dir.is_relative_to(REPO_ROOT) else run_dir
    )
    command = [
        sys.executable,
        "scripts/validation/run_policy_search_candidate.py",
        "--candidate",
        candidate,
        "--stage",
        stage,
        "--workers",
        str(worker_count),
        "--output-dir",
        str(display_run_dir),
    ]
    if horizon is not None:
        command.extend(["--horizon", str(horizon)])
    command.extend(extra_args)

    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    summary_path = run_dir / "summary.json"
    row: dict[str, Any] = {
        "workers": int(worker_count),
        "command": ["python", *command[1:]],
        "exit_code": int(completed.returncode),
        "summary_path": str(summary_path.relative_to(REPO_ROOT)),
    }
    if completed.returncode != 0 or not summary_path.exists():
        row["status"] = "failed"
        row["message"] = "candidate runner failed or did not write summary.json"
        return row

    summary_doc = _load_json(summary_path)
    summary = summary_doc.get("summary")
    batch_summary = summary_doc.get("batch_summary")
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(batch_summary, dict):
        batch_summary = {}

    runtime_sec = summary.get("runtime_sec")
    jobs = batch_summary.get("total_jobs")
    row.update(
        {
            "status": "ok",
            "decision": summary_doc.get("decision"),
            "episodes": summary.get("episodes"),
            "runtime_sec": runtime_sec,
            "jobs": jobs,
            "batch_runtime_sec": batch_summary.get("batch_runtime_sec"),
            "batch_workers": batch_summary.get("workers"),
            "parallel_execution": batch_summary.get("parallel_execution"),
            "seconds_per_job": (
                float(runtime_sec) / float(jobs)
                if isinstance(runtime_sec, int | float) and isinstance(jobs, int | float) and jobs
                else None
            ),
        }
    )
    return row


def _add_speedups(rows: list[dict[str, Any]]) -> None:
    """Add speedup versus the lowest-worker successful row."""
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return
    baseline = min(ok_rows, key=lambda row: int(row["workers"]))
    baseline_runtime = baseline.get("runtime_sec")
    if not isinstance(baseline_runtime, int | float) or baseline_runtime <= 0:
        return
    for row in ok_rows:
        runtime = row.get("runtime_sec")
        if isinstance(runtime, int | float) and runtime > 0:
            row["speedup_vs_min_workers"] = float(baseline_runtime) / float(runtime)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--stage", default="nominal_sanity")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--horizon", type=int)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output") / "benchmark_worker_scaling" / "latest",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Path for compact comparison JSON. Defaults to <output-root>/worker_scaling_summary.json.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed after -- to run_policy_search_candidate.py.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the worker-scaling profile."""
    args = parse_args(argv)
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_out = args.summary_out or output_root / "worker_scaling_summary.json"
    summary_out = (
        (REPO_ROOT / summary_out).resolve() if not summary_out.is_absolute() else summary_out
    )
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    rows = [
        _run_candidate_stage(
            candidate=str(args.candidate),
            stage=str(args.stage),
            worker_count=int(worker_count),
            output_root=output_root,
            horizon=args.horizon,
            extra_args=extra_args,
        )
        for worker_count in args.workers
    ]
    _add_speedups(rows)
    payload = {
        "schema_version": "robot-sf-benchmark-worker-scaling-profile.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate": args.candidate,
        "stage": args.stage,
        "horizon": args.horizon,
        "output_root": str(output_root.relative_to(REPO_ROOT)),
        "rows": rows,
        "claim_boundary": "diagnostic-only",
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if all(row.get("status") == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
