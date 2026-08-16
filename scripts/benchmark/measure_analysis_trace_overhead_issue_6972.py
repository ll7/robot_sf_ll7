"""Measure the opt-in analysis-trace overhead for issue #6972.

This command produces local wall-clock diagnostic evidence for the synthetic
runner fixture. It is not benchmark, paper, or real-world safety evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
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
MEASUREMENT_ISSUE = 6987
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
DEFAULT_STABILITY_TOLERANCE_FRACTION = 0.25
RECEIPT_SCHEMA_VERSION = "analysis_trace_overhead_measurement_receipt.v2"
THREAD_SETTING_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def _execution_context(*, warmups: int) -> dict[str, Any]:
    """Record timing context needed for a later host/order reconciliation."""

    return {
        "execution_order": "alternating off/on, then on/off by batch",
        "warmup_state": {
            "warmups_per_arm_per_batch": warmups,
            "excluded_from_timing": True,
        },
        "cache_state": {
            "process_warmup": "per_arm_per_batch",
            "external_cache": "uncontrolled",
        },
        "numerical_thread_settings": {key: os.environ.get(key) for key in THREAD_SETTING_KEYS},
    }


def _git_hash() -> str:
    """Return the exact repository commit used by the measurement."""

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _compressed_size_bytes(value: Any) -> int:
    """Return deterministic gzip size for a canonical JSON-compatible value."""

    payload = canonical_json(value).encode("utf-8")
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


def _run_episode(*, trace_enabled: bool) -> dict[str, Any]:
    """Run one deterministic fixture episode for the requested telemetry arm."""

    return run_episode(
        SCENARIO,
        seed=123,
        horizon=5,
        dt=0.1,
        record_forces=False,
        telemetry=TRACE_TELEMETRY if trace_enabled else None,
    )


def _timed_episode(
    *, trace_enabled: bool, sample_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one measured episode and keep timing separate from receipt serialization."""

    started = time.perf_counter()
    record = _run_episode(trace_enabled=trace_enabled)
    timing = {
        "sample": sample_index,
        "elapsed_sec": time.perf_counter() - started,
        "trace_compressed_bytes": (
            _compressed_size_bytes(record["algorithm_metadata"]["analysis_trace"])
            if trace_enabled
            else 0
        ),
        "record_compressed_bytes": _compressed_size_bytes(record),
    }
    return record, timing


def _summarize_arm(
    *,
    trace_enabled: bool,
    warmups: int,
    timings: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize one arm while retaining compact raw timing/digest samples."""

    return {
        "trace_enabled": trace_enabled,
        "warmups": warmups,
        "samples": timings,
        "elapsed_sec_median": statistics.median(sample["elapsed_sec"] for sample in timings),
        "elapsed_sec_max": max(sample["elapsed_sec"] for sample in timings),
        "trace_compressed_bytes_median": statistics.median(
            sample["trace_compressed_bytes"] for sample in timings
        ),
        "record_compressed_bytes_median": statistics.median(
            sample["record_compressed_bytes"] for sample in timings
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


def _batch_summary(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return paired runtime summaries for one alternating measurement batch."""

    off = arms["analysis_trace_off"]
    on = arms["analysis_trace_on"]
    off_median = off["elapsed_sec_median"]
    on_median = on["elapsed_sec_median"]
    overhead_fraction = on_median / off_median - 1.0 if off_median > 0 else None
    paired_deltas = [
        on_sample["elapsed_sec"] - off_sample["elapsed_sec"]
        for off_sample, on_sample in zip(off["samples"], on["samples"], strict=True)
    ]
    return {
        "off_median_sec": off_median,
        "on_median_sec": on_median,
        "overhead_fraction": overhead_fraction,
        "overhead_percent": overhead_fraction * 100.0 if overhead_fraction is not None else None,
        "paired_runtime_delta_median_sec": statistics.median(paired_deltas),
        "paired_runtime_delta_median_ms": statistics.median(paired_deltas) * 1000.0,
    }


def _run_batch(
    *,
    batch_index: int,
    samples: int,
    warmups: int,
    arm_order: tuple[bool, bool],
) -> dict[str, Any]:
    """Run an alternating paired batch with a reversed order on the next batch."""

    records: dict[str, list[dict[str, Any]]] = {"analysis_trace_off": [], "analysis_trace_on": []}
    timings: dict[str, list[dict[str, Any]]] = {"analysis_trace_off": [], "analysis_trace_on": []}
    for trace_enabled in arm_order:
        for _ in range(warmups):
            _run_episode(trace_enabled=trace_enabled)
    for sample_index in range(samples):
        for trace_enabled in arm_order:
            record, timing = _timed_episode(
                trace_enabled=trace_enabled,
                sample_index=sample_index,
            )
            arm_name = "analysis_trace_on" if trace_enabled else "analysis_trace_off"
            records[arm_name].append(record)
            timing["batch"] = batch_index
            timings[arm_name].append(timing)
    arms = {
        arm_name: _summarize_arm(
            trace_enabled=arm_name == "analysis_trace_on",
            warmups=warmups,
            timings=timings[arm_name],
            records=records[arm_name],
        )
        for arm_name in ("analysis_trace_off", "analysis_trace_on")
    }
    return {
        "batch_index": batch_index,
        "arm_order": ["on" if value else "off" for value in arm_order],
        "arms": arms,
        "derived": _batch_summary(arms),
    }


def _aggregate_arm_summaries(batches: list[dict[str, Any]], arm_name: str) -> dict[str, Any]:
    """Combine batch summaries without discarding raw timing/digest samples."""

    summaries = [batch["arms"][arm_name] for batch in batches]
    trace_enabled = arm_name == "analysis_trace_on"
    timings = [sample for summary in summaries for sample in summary["samples"]]
    return {
        "trace_enabled": trace_enabled,
        "warmups": sum(summary["warmups"] for summary in summaries),
        "samples": timings,
        "elapsed_sec_median": statistics.median(sample["elapsed_sec"] for sample in timings),
        "elapsed_sec_max": max(sample["elapsed_sec"] for sample in timings),
        "trace_compressed_bytes_median": statistics.median(
            sample["trace_compressed_bytes"] for sample in timings
        ),
        "record_compressed_bytes_median": statistics.median(
            sample["record_compressed_bytes"] for sample in timings
        ),
        "control_sequence_digests": [
            digest for summary in summaries for digest in summary["control_sequence_digests"]
        ],
        "outcome_digests": [
            digest for summary in summaries for digest in summary["outcome_digests"]
        ],
        "metric_digests": [digest for summary in summaries for digest in summary["metric_digests"]],
        "trace_git_hashes": [
            value for summary in summaries for value in summary["trace_git_hashes"]
        ],
        "trace_artifact_matches_provenance": [
            value for summary in summaries for value in summary["trace_artifact_matches_provenance"]
        ],
    }


def _stability_summary(
    batch_summaries: list[dict[str, Any]],
    *,
    tolerance_fraction: float,
) -> dict[str, Any]:
    """Classify repeated batch overhead before allowing a target decision."""

    overheads = [
        summary["overhead_fraction"]
        for summary in batch_summaries
        if summary["overhead_fraction"] is not None
    ]
    if len(overheads) < 2:
        return {
            "status": "insufficient_batches",
            "stable": False,
            "relative_spread_fraction": None,
            "tolerance_fraction": tolerance_fraction,
        }
    spread = max(overheads) - min(overheads)
    return {
        "status": "stable" if spread <= tolerance_fraction else "unstable",
        "stable": spread <= tolerance_fraction,
        "relative_spread_fraction": spread,
        "tolerance_fraction": tolerance_fraction,
    }


def measure(
    *,
    samples: int,
    warmups: int,
    batches: int = 2,
    stability_tolerance_fraction: float = DEFAULT_STABILITY_TOLERANCE_FRACTION,
) -> dict[str, Any]:
    """Run repeated paired batches and build a fail-closed diagnostic receipt."""

    if samples < 1 or warmups < 0 or batches < 1:
        raise ValueError("samples and batches must be positive; warmups must be non-negative")
    if stability_tolerance_fraction < 0:
        raise ValueError("stability_tolerance_fraction must be non-negative")
    repository_commit = _git_hash()
    batch_results = [
        _run_batch(
            batch_index=batch_index,
            samples=samples,
            warmups=warmups,
            arm_order=(False, True) if batch_index % 2 == 0 else (True, False),
        )
        for batch_index in range(batches)
    ]
    off = _aggregate_arm_summaries(batch_results, "analysis_trace_off")
    on = _aggregate_arm_summaries(batch_results, "analysis_trace_on")
    batch_summaries = [batch["derived"] for batch in batch_results]
    stability = _stability_summary(
        batch_summaries,
        tolerance_fraction=stability_tolerance_fraction,
    )
    paired_equal = all(
        all(
            off_outcome == on_outcome and off_metrics == on_metrics
            for off_outcome, on_outcome, off_metrics, on_metrics in zip(
                batch["arms"]["analysis_trace_off"]["outcome_digests"],
                batch["arms"]["analysis_trace_on"]["outcome_digests"],
                batch["arms"]["analysis_trace_off"]["metric_digests"],
                batch["arms"]["analysis_trace_on"]["metric_digests"],
                strict=True,
            )
        )
        for batch in batch_results
    )
    checks = {
        "paired_outcomes_and_metrics_equal": paired_equal,
        "control_sequence_digest_stable": len(set(on["control_sequence_digests"])) == 1,
        "trace_git_hash_matches_commit": all(
            git_hash == repository_commit for git_hash in on["trace_git_hashes"]
        ),
        "trace_artifact_matches_provenance": all(on["trace_artifact_matches_provenance"]),
        "same_commit_repeated_batch_stable": stability["stable"],
        "repeated_batch_overheads_defined": len(batch_summaries)
        == sum(summary["overhead_fraction"] is not None for summary in batch_summaries),
    }
    integrity_ok = all(
        checks[key]
        for key in (
            "paired_outcomes_and_metrics_equal",
            "control_sequence_digest_stable",
            "trace_git_hash_matches_commit",
            "trace_artifact_matches_provenance",
        )
    )
    off_median = off["elapsed_sec_median"]
    on_median = on["elapsed_sec_median"]
    overhead_ratio = on_median / off_median - 1.0 if off_median > 0 else None
    batch_overheads_defined = checks["repeated_batch_overheads_defined"]
    batch_target_met = (
        all(
            summary["overhead_fraction"] <= 0.10
            for summary in batch_summaries
            if summary["overhead_fraction"] is not None
        )
        if batch_overheads_defined
        else None
    )
    target_met = (
        None
        if (
            overhead_ratio is None
            or not integrity_ok
            or not stability["stable"]
            or not batch_overheads_defined
        )
        else batch_target_met
    )
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "issue": MEASUREMENT_ISSUE,
        "source_issue": ISSUE,
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
            "warmups_per_arm_per_batch": warmups,
            "samples_per_arm_per_batch": samples,
            "batch_count": batches,
            "arm_order": "alternating off/on, then on/off by batch",
            "stability_tolerance_fraction": stability_tolerance_fraction,
            "timer": "time.perf_counter",
            "serialization": "analysis_trace.canonical_json",
            "compression": "gzip.compress(mtime=0)",
            "upper_tail_summary": "maximum measured sample; not a statistical percentile",
            "target_rule": (
                "target_met is null unless repeated batch medians are stable, every batch "
                "overhead is within 10%, and integrity checks pass; an aggregate median "
                "cannot hide a failed batch"
            ),
        },
        "execution_context": _execution_context(warmups=warmups),
        "batches": batch_results,
        "arms": {"analysis_trace_off": off, "analysis_trace_on": on},
        "checks": checks,
        "derived": {
            "median_overhead_fraction": overhead_ratio,
            "median_overhead_percent": overhead_ratio * 100.0
            if overhead_ratio is not None
            else None,
            "median_runtime_delta_seconds": on_median - off_median,
            "median_runtime_delta_ms": (on_median - off_median) * 1000.0,
            "median_trace_compressed_bytes": on["trace_compressed_bytes_median"],
            "median_record_compressed_bytes_delta": (
                on["record_compressed_bytes_median"] - off["record_compressed_bytes_median"]
            ),
            "batch_overhead_fractions": [
                summary["overhead_fraction"] for summary in batch_summaries
            ],
            "batch_relative_spread_fraction": stability["relative_spread_fraction"],
            "stability_status": stability["status"],
            "stability_passed": stability["stable"],
            "batch_target_met": batch_target_met,
            "target_percent": 10.0,
            "target_met": target_met,
            "target_decision": (
                "inconclusive" if target_met is None else "met" if target_met else "not_met"
            ),
        },
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=int, default=6, help="Measured samples per arm per batch."
    )
    parser.add_argument("--warmups", type=int, default=1, help="Warmup runs per arm.")
    parser.add_argument(
        "--batches",
        type=int,
        default=2,
        help="Repeated paired batches; target decisions need stable medians across batches.",
    )
    parser.add_argument(
        "--stability-tolerance",
        type=float,
        default=DEFAULT_STABILITY_TOLERANCE_FRACTION,
        help="Maximum absolute spread between batch overhead fractions.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON receipt path; stdout is used when omitted.",
    )
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 0 or args.batches < 1:
        parser.error("--samples and --batches must be positive; --warmups must be non-negative")
    if args.stability_tolerance < 0:
        parser.error("--stability-tolerance must be non-negative")
    return args


def main() -> int:
    """Run the measurement and write its receipt."""

    args = _parse_args()
    receipt = measure(
        samples=args.samples,
        warmups=args.warmups,
        batches=args.batches,
        stability_tolerance_fraction=args.stability_tolerance,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
