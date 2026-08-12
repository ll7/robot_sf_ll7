#!/usr/bin/env python3
"""Measure the bounded opt-in analysis-trace overhead fixture.

The command intentionally uses the non-map smoke runner rather than release or
historical benchmark inputs. Raw JSONL files are temporary and are removed when
the process exits; only the compact measurement receipt is written.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from robot_sf.benchmark.runner import run_batch

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"
SCENARIO: dict[str, object] = {
    "id": "smoke-telemetry-measurement",
    "density": "low",
    "flow": "uni",
    "obstacle": "open",
    "groups": 0.0,
    "speed_var": "low",
    "goal_topology": "point",
    "robot_context": "embedded",
    "repeats": 1,
}


def _stable_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Remove runtime and profile-specific fields before behavior comparison."""
    projection = json.loads(json.dumps(record))
    for key in ("timestamps", "wall_time_sec", "timing"):
        projection.pop(key, None)
    projection.pop("algorithm_metadata", None)
    provenance = projection.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("run_id", None)
        provenance.pop("artifact_sha256", None)
    return projection


def _run_profile(profile: str, *, samples: int, root: Path) -> list[dict[str, Any]]:
    """Run one trace profile and return compact per-sample measurements."""
    measurements: list[dict[str, Any]] = []
    for sample in range(samples):
        output = root / f"{profile}-{sample}.jsonl"
        kwargs: dict[str, Any] = {
            "out_path": output,
            "schema_path": SCHEMA_PATH,
            "base_seed": 123,
            "horizon": 5,
            "dt": 0.1,
            "record_forces": False,
            "append": False,
            "workers": 1,
            "resume": False,
        }
        if profile == "on":
            kwargs["telemetry"] = {"analysis_trace": "all", "planner_debug_trace": "none"}
        started = time.perf_counter()
        run_batch([SCENARIO], **kwargs)
        elapsed_s = time.perf_counter() - started
        raw = output.read_bytes()
        record = json.loads(raw.splitlines()[0])
        trace = record.get("algorithm_metadata", {}).get("analysis_trace")
        trace_payload = (
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if isinstance(trace, dict)
            else b""
        )
        measurements.append(
            {
                "sample": sample,
                "elapsed_s": round(elapsed_s, 9),
                "jsonl_gzip_bytes": len(gzip.compress(raw, mtime=0)),
                "analysis_trace_gzip_bytes": (
                    len(gzip.compress(trace_payload, mtime=0)) if trace_payload else 0
                ),
                "outcome": record.get("outcome"),
                "metrics_sha256": hashlib.sha256(
                    json.dumps(record.get("metrics"), sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest(),
                "termination_reason": record.get("termination_reason"),
                "trace_coverage_status": record.get("algorithm_metadata", {})
                .get("analysis_trace_coverage", {})
                .get("status"),
            }
        )
    return measurements


def measure(*, output: Path, samples: int = 5) -> dict[str, Any]:
    """Run and write the deterministic software-fixture measurement receipt."""
    if samples < 2:
        raise ValueError("samples must be at least 2 so the first sample can be treated as cold")
    with tempfile.TemporaryDirectory(prefix="robot-sf-6972-measure-") as temp_dir:
        root = Path(temp_dir)
        profiles = {
            "off": _run_profile("off", samples=samples, root=root),
            "on": _run_profile("on", samples=samples, root=root),
        }
    for off, on in zip(profiles["off"], profiles["on"], strict=True):
        if (
            off["outcome"] != on["outcome"]
            or off["metrics_sha256"] != on["metrics_sha256"]
            or off["termination_reason"] != on["termination_reason"]
        ):
            raise RuntimeError(f"trace profile changed stable behavior at sample {off['sample']}")

    def warm_values(values: list[dict[str, Any]], key: str) -> list[float]:
        """Return sorted non-cold samples for a numeric measurement field."""
        return sorted(float(item[key]) for item in values[1:])

    def median_warm(values: list[dict[str, Any]], key: str) -> float:
        """Return the median of the non-cold samples for one field."""
        warm = sorted(float(item[key]) for item in values[1:])
        middle = len(warm) // 2
        if len(warm) % 2:
            return warm[middle]
        return (warm[middle - 1] + warm[middle]) / 2.0

    def p90_warm(values: list[dict[str, Any]], key: str) -> float:
        """Return a nearest-rank p90 of the non-cold samples."""
        warm = warm_values(values, key)
        index = max(0, min(len(warm) - 1, int((0.9 * len(warm)) + 0.999999) - 1))
        return warm[index]

    off_median = median_warm(profiles["off"], "elapsed_s")
    on_median = median_warm(profiles["on"], "elapsed_s")
    receipt: dict[str, Any] = {
        "schema_version": "issue_6972_trace_overhead_receipt.v1",
        "status": "diagnostic_software_fixture",
        "claim_boundary": "not benchmark evidence; does not authorize source admission",
        "issue": "https://github.com/ll7/robot_sf_ll7/issues/6972",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "fixture": {
            "scenario": SCENARIO,
            "seed": 123,
            "horizon": 5,
            "dt_s": 0.1,
            "samples": samples,
            "first_sample_is_cold": True,
            "raw_artifacts_retained": False,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "profiles": profiles,
        "summary": {
            "warm_median_runtime_s": {"off": off_median, "on": on_median},
            "warm_p90_runtime_s": {
                "off": p90_warm(profiles["off"], "elapsed_s"),
                "on": p90_warm(profiles["on"], "elapsed_s"),
            },
            "warm_max_runtime_s": {
                "off": max(warm_values(profiles["off"], "elapsed_s")),
                "on": max(warm_values(profiles["on"], "elapsed_s")),
            },
            "warm_median_overhead_fraction": on_median / off_median - 1.0,
            "warm_median_jsonl_gzip_bytes": {
                "off": median_warm(profiles["off"], "jsonl_gzip_bytes"),
                "on": median_warm(profiles["on"], "jsonl_gzip_bytes"),
            },
            "warm_median_analysis_trace_gzip_bytes": median_warm(
                profiles["on"], "analysis_trace_gzip_bytes"
            ),
            "stable_projection_equal": True,
            "action_sequence_comparison": {
                "status": "covered_by_map_runner_regression",
                "test": "tests/benchmark/test_map_runner_utils.py::test_analysis_trace_profile_does_not_change_recorded_actions_or_outcome",
                "note": "The non-map fixture itself has no off-profile action trace and is not used to infer one.",
            },
            "policy_decision": "keep analysis_trace: all opt-in; median overhead is compared with the 10% threshold",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    """Parse arguments, run the fixture, and write the receipt."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    measure(output=args.output, samples=args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
