#!/usr/bin/env python3
"""Measure receipt loading on a synthetic dev-only scale fixture, never a sweep receipt.

Clone one archived controller trace over seeds 1001-1030 to expose repeated-file
parsing. No episode is executed and no execution metadata or receipt is emitted.
The reference is a caller-selected Git revision of the producer. Compare exact
serialized outputs and preserve compressed inputs/results next to the timings.
"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.ci import make_behaviour_receipt as current  # noqa: E402


def _portable(value: Any) -> Any:
    """Strip inert private provenance strings from the synthetic trace copies."""
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    if isinstance(value, str) and value.startswith("/home/"):
        return "synthetic-source/" + Path(value).name
    return value


def _measure(loader: Any, sweep: Path, repeats: int) -> tuple[list[float], bytes]:
    elapsed = []
    result = b""
    for _ in range(repeats):
        start = time.perf_counter()
        rows = loader(sweep)
        elapsed.append(time.perf_counter() - start)
        encoded = json.dumps([rows[key] for key in sorted(rows)], sort_keys=True).encode()
        if result and encoded != result:
            raise ValueError("loader output changed between repeats")
        result = encoded
    return elapsed, result


def main() -> None:
    """Write reproducible tooling-only timings and exact-output comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, help="Pre-optimization Git revision")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.reference}^{{commit}}"], cwd=ROOT, text=True
    ).strip()
    source = subprocess.check_output(
        ["git", "show", f"{reference}:scripts/ci/make_behaviour_receipt.py"], cwd=ROOT
    )
    fixture = (
        ROOT
        / "tests/analysis/fixtures/behaviour_receipt_baseline_31302/goal__differential_drive.jsonl"
    )
    template = _portable(json.loads(fixture.read_text()))
    current._check_seed_range([template["seed"]])
    sweep = output / "synthetic"
    raw_path = sweep / "campaigns/empty_world_main/runs/goal__differential_drive/episodes.jsonl"
    raw_path.parent.mkdir(parents=True)
    raw_rows, summaries = [], []
    for seed in range(1001, 1031):
        raw = copy.deepcopy(template)
        raw["seed"] = seed
        raw_rows.append(raw)
        summaries.append(
            {
                "arm": "goal",
                "scenario": raw["scenario_id"],
                "seed": seed,
                "success": current._row_success(raw),
                "collisions": current._row_collisions(raw),
                "execution_status": "written",
                "source_file": "runs/goal__differential_drive/episodes.jsonl",
            }
        )
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows))
    (sweep / "episodes_main.jsonl").write_text("".join(json.dumps(row) + "\n" for row in summaries))
    # Keep the reference module under the checkout so its ROOT resolves correctly.
    with tempfile.TemporaryDirectory(prefix="receipt-profile-", dir=ROOT / "scripts") as directory:
        ref_path = Path(directory) / "reference.py"
        ref_path.write_bytes(source)
        spec = importlib.util.spec_from_file_location("receipt_profile_reference", ref_path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        before, before_bytes = _measure(module._load_sweep_rows, sweep, args.repeats)
        after, after_bytes = _measure(current._load_sweep_rows, sweep, args.repeats)
    if before_bytes != after_bytes:
        raise ValueError("reference and current loader outputs differ")
    artifacts = {}
    for name, payload in (
        ("raw.jsonl", raw_path.read_bytes()),
        ("summaries.jsonl", (sweep / "episodes_main.jsonl").read_bytes()),
        ("loaded.json", after_bytes),
    ):
        path = output / (name + ".gz")
        path.write_bytes(gzip.compress(payload, mtime=0))
        artifacts[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "uncompressed_bytes": len(payload),
        }
    report = {
        "kind": "synthetic_tooling_scale_only",
        "reference_revision": reference,
        "seeds": list(range(1001, 1031)),
        "rows": 30,
        "trace_steps": len(template["algorithm_metadata"]["simulation_step_trace"]["steps"]),
        "before_s": before,
        "after_s": after,
        "before_median_s": statistics.median(before),
        "after_median_s": statistics.median(after),
        "outputs_byte_identical": True,
        "artifacts": artifacts,
    }
    (output / "timings.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("before_median_s", "after_median_s", "outputs_byte_identical")
            }
        )
    )


if __name__ == "__main__":
    main()
