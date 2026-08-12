"""Measure the opt-in analysis-trace overhead for issue #6972.

This command produces local wall-clock diagnostic evidence for the synthetic
runner fixture. It is not benchmark, paper, or real-world safety evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from robot_sf.benchmark.analysis_trace import canonical_json
from robot_sf.benchmark.runner import run_episode

ISSUE = 6972
SCENARIO = {
    "id": "issue-6972-overhead",
    "density": "low",
    "flow": "uni",
    "obstacle": "open",
    "groups": 0.0,
    "speed_var": "low",
    "goal_topology": "point",
    "robot_context": "embedded",
    "repeats": 1,
}
TRACE_TELEMETRY = {"analysis_trace": "all", "planner_debug_trace": "none"}


def _git_hash() -> str:
    """Return the exact repository commit used by the measurement."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _record_size_bytes(record: dict[str, Any]) -> int:
    """Return deterministic gzip size for the serialized episode record."""

    payload = canonical_json(record).encode("utf-8")
    return len(gzip.compress(payload, mtime=0))


def _control_sequence(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the requested/applied controls from an analysis trace."""

    trace = record["algorithm_metadata"]["analysis_trace"]
    return [
        {
            "requested": step.get("controls", {}).get("requested"),
            "applied": step.get("controls", {}).get("applied"),
        }
        for step in trace["steps"][1:]
    ]


def _digest(value: Any) -> str:
    """Return a digest for a canonical JSON-compatible value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _run_arm(*, trace_enabled: bool, samples: int, warmups: int) -> dict[str, Any]:
    """Run one arm and return timings plus compact deterministic checks."""

    telemetry = TRACE_TELEMETRY if trace_enabled else None
    for _ in range(warmups):
        run_episode(
            SCENARIO,
            seed=123,
            horizon=5,
            dt=0.1,
            record_forces=False,
            telemetry=telemetry,
        )

    timings: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for _ in range(samples):
        started = time.perf_counter()
        record = run_episode(
            SCENARIO,
            seed=123,
            horizon=5,
            dt=0.1,
            record_forces=False,
            telemetry=telemetry,
        )
        timings.append(
            {
                "elapsed_sec": time.perf_counter() - started,
                "compressed_bytes": _record_size_bytes(record),
            }
        )
        records.append(record)

    return {
        "trace_enabled": trace_enabled,
        "warmups": warmups,
        "samples": timings,
        "elapsed_sec_median": statistics.median(sample["elapsed_sec"] for sample in timings),
        "elapsed_sec_max": max(sample["elapsed_sec"] for sample in timings),
        "compressed_bytes_median": statistics.median(
            sample["compressed_bytes"] for sample in timings
        ),
        "control_sequence_digests": (
            [_digest(_control_sequence(record)) for record in records] if trace_enabled else []
        ),
        "outcome_digests": [_digest(record["outcome"]) for record in records],
        "metric_digests": [_digest(record["metrics"]) for record in records],
        "trace_git_hashes": (
            [record["algorithm_metadata"]["analysis_trace"]["git_hash"] for record in records]
            if trace_enabled
            else []
        ),
        "trace_artifact_matches_provenance": (
            [
                record["provenance"]["artifact_sha256"]
                == record["algorithm_metadata"]["analysis_trace"]["artifact_sha256"]
                for record in records
            ]
            if trace_enabled
            else []
        ),
    }


def measure(*, samples: int, warmups: int) -> dict[str, Any]:
    """Run both arms and build a versioned local diagnostic receipt."""

    repository_commit = _git_hash()
    off = _run_arm(trace_enabled=False, samples=samples, warmups=warmups)
    on = _run_arm(trace_enabled=True, samples=samples, warmups=warmups)
    overhead_ratio = on["elapsed_sec_median"] / off["elapsed_sec_median"] - 1.0
    paired_equal = all(
        off_outcome == on_outcome and off_metrics == on_metrics
        for off_outcome, on_outcome, off_metrics, on_metrics in zip(
            off["outcome_digests"],
            on["outcome_digests"],
            off["metric_digests"],
            on["metric_digests"],
            strict=True,
        )
    )
    return {
        "schema_version": "issue_6972_analysis_trace_overhead_receipt.v1",
        "issue": ISSUE,
        "status": "diagnostic_only",
        "claim_boundary": (
            "Local synthetic-runner wall-clock and serialized-size diagnostic; "
            "not benchmark, paper-facing, real-world, or safety evidence."
        ),
        "repository_commit": repository_commit,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "fixture": {
            "scenario": SCENARIO,
            "seed": 123,
            "horizon": 5,
            "dt": 0.1,
            "record_forces": False,
            "telemetry_on": TRACE_TELEMETRY,
        },
        "method": {
            "warmups": warmups,
            "samples_per_arm": samples,
            "timer": "time.perf_counter",
            "serialization": "analysis_trace.canonical_json",
            "compression": "gzip.compress(mtime=0)",
            "upper_tail_summary": "maximum measured sample; not a statistical percentile",
        },
        "arms": {"analysis_trace_off": off, "analysis_trace_on": on},
        "checks": {
            "paired_outcomes_and_metrics_equal": paired_equal,
            "control_sequence_digest_stable": len(set(on["control_sequence_digests"])) == 1,
            "trace_git_hash_matches_commit": all(
                git_hash == repository_commit for git_hash in on["trace_git_hashes"]
            ),
            "trace_artifact_matches_provenance": all(on["trace_artifact_matches_provenance"]),
        },
        "derived": {
            "median_overhead_fraction": overhead_ratio,
            "median_overhead_percent": overhead_ratio * 100.0,
            "median_compressed_bytes_delta": (
                on["compressed_bytes_median"] - off["compressed_bytes_median"]
            ),
            "target_percent": 10.0,
            "target_met": overhead_ratio <= 0.10,
        },
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6, help="Measured samples per arm.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs per arm.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON receipt path; stdout is used when omitted.",
    )
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 0:
        parser.error("--samples must be positive and --warmups must be non-negative")
    return args


def main() -> int:
    """Run the measurement and write its receipt."""

    args = _parse_args()
    receipt = measure(samples=args.samples, warmups=args.warmups)
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
