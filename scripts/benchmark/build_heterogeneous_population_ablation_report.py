#!/usr/bin/env python3
"""Build the issue #3574 heterogeneous population ablation report.

Tabulates results, computes bootstrap rank sensitivity, and generates
summary.json, analysis.md, and ablation_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.heterogeneous_population_ablation import (
    build_per_archetype_ablation_report,
)
from robot_sf.benchmark.heterogeneous_rank_sensitivity import (
    compute_bootstrap_rank_sensitivity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default="output/issue_3574_mean_matched_harness/episode_records.jsonl",
        help="Path to the simulation output records JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/issue_3574_mean_matched_harness",
        help="Directory to write output files.",
    )
    parser.add_argument(
        "--durable-dir",
        default="output/issue_3574_mean_matched_harness/durable_evidence",
        help=(
            "Directory to write the (smoke-grade) report copy. Defaults to a git-ignored "
            "output/ path (issue #4618 CI-C). Promoting these files into "
            "docs/context/evidence/ requires catalog.yaml registration and honest, "
            "smoke-evidence claim language; an 8-row CPU sweep does not prove rank stability."
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file records."""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> int:  # noqa: C901,PLR0912,PLR0915
    """Run report compilation pipeline."""
    args = parse_args()
    records_path = REPO_ROOT / args.records
    output_dir = REPO_ROOT / args.output_dir
    durable_dir = REPO_ROOT / args.durable_dir

    if not records_path.exists():
        print(f"Error: Episode records not found at {records_path}. Run the simulations first.")
        return 1

    records = load_jsonl(records_path)
    print(f"Loaded {len(records)} episode records.")

    # Validate/skip incomplete rows (issue #4618 R5): downstream tabulation reads
    # ``planner``/``seed``/``scenario_id``/``population_arm`` directly, so drop any record
    # missing (or null in) those keys instead of raising a raw KeyError mid-report.
    required_keys = ("planner", "seed", "scenario_id", "population_arm")
    complete_records = [
        rec for rec in records if all(rec.get(key) is not None for key in required_keys)
    ]
    skipped = len(records) - len(complete_records)
    if skipped:
        print(
            f"Skipped {skipped} incomplete episode record(s) missing required keys {required_keys}."
        )
    records = complete_records
    if not records:
        print("Error: no complete episode records with required keys; nothing to report.")
        return 1

    # Find planners and seeds
    planners = sorted({rec["planner"] for rec in records})
    seeds = sorted({rec["seed"] for rec in records})
    print(f"Planners found: {planners}")
    print(f"Seeds found: {seeds}")

    # 1. Compute bootstrap rank sensitivity
    # We analyze "mean_clearance" (higher is safer)
    rank_sensitivity = compute_bootstrap_rank_sensitivity(
        records,
        metric_key="mean_clearance",
        planners=planners,
        higher_is_safer=True,
        num_bootstrap=1000,
        seed=3574,
    )

    # 2. Build per-archetype reports for each (scenario, seed, planner) triplet
    # Group records by (scenario_id, seed, planner)
    triplets: dict[tuple[str, int, str], dict[str, dict[str, Any]]] = {}
    for rec in records:
        sc_id = rec["scenario_id"]
        seed = int(rec["seed"])
        planner = rec["planner"]
        arm = rec["population_arm"]
        key = (sc_id, seed, planner)

        control_trace = rec.get("algorithm_metadata", {}).get("pedestrian_control_trace")
        if control_trace:
            triplets.setdefault(key, {})[arm] = control_trace

    ablation_reports = {}
    for key, traces_by_arm in triplets.items():
        sc_id, seed, planner = key
        # Check if we have both arms
        if "heterogeneous" in traces_by_arm and "mean_matched_homogeneous" in traces_by_arm:
            report = build_per_archetype_ablation_report(
                control_traces_by_arm=traces_by_arm,
                metric_key="clearance_m",
                higher_is_safer=True,
                cvar_alpha=0.2,
                reducer="mean",
            )
            key_str = f"{sc_id}/seed_{seed}/{planner}"
            ablation_reports[key_str] = report

    # 3. Create tabulated results for CSV
    # Row format: scenario_id, seed, planner, arm, mean_clearance, cvar_clearance
    csv_rows = []
    for rec in records:
        sc_id = rec["scenario_id"]
        seed = int(rec["seed"])
        planner = rec["planner"]
        arm = rec["population_arm"]
        trace = rec.get("algorithm_metadata", {}).get("pedestrian_control_trace")

        mean_val = ""
        cvar_val = ""
        if trace:
            # We can extract pedestrian clearance metrics
            from robot_sf.benchmark.heterogeneous_population_metrics import (
                cvar,
                pedestrian_metric_observations_from_control_trace,
            )

            try:
                obs = pedestrian_metric_observations_from_control_trace(
                    trace, "clearance_m", reducer="mean"
                )
                vals = [o.value for o in obs]
                if vals:
                    mean_val = f"{sum(vals) / len(vals):.4f}"
                    cvar_val = f"{cvar(vals, 0.2, higher_is_safer=True):.4f}"
            except (KeyError, ValueError, TypeError, ZeroDivisionError):
                # Narrowed from a blind ``except`` (issue #4618 CI-B): a malformed or empty
                # control trace leaves the CSV clearance cells blank rather than aborting.
                pass

        csv_rows.append(
            {
                "scenario_id": sc_id,
                "seed": seed,
                "planner": planner,
                "arm": arm,
                "mean_clearance_m": mean_val,
                "cvar_clearance_m": cvar_val,
            }
        )

    # Ensure output dirs exist
    output_dir.mkdir(parents=True, exist_ok=True)
    durable_dir.mkdir(parents=True, exist_ok=True)

    # Write summary.json
    summary_data = {
        "schema_version": "heterogeneous_population_ablation.v1",
        "rank_sensitivity": rank_sensitivity,
        "ablation_reports": ablation_reports,
    }

    for dest in (output_dir, durable_dir):
        # JSON
        with (dest / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, sort_keys=True)
        with (dest / "rank_sensitivity.json").open("w", encoding="utf-8") as f:
            json.dump(rank_sensitivity, f, indent=2, sort_keys=True)

        # CSV
        with (dest / "ablation_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "scenario_id",
                    "seed",
                    "planner",
                    "arm",
                    "mean_clearance_m",
                    "cvar_clearance_m",
                ],
            )
            writer.writeheader()
            writer.writerows(csv_rows)

    # 4. Generate beautiful Markdown Report (analysis.md)
    markdown_content = """# Issue #3574 Heterogeneous Population Ablation Report

This report evaluates whether moving from a homogeneous mean-matched population to a heterogeneous mixture population impacts planner safety rankings and metrics.

## Executive Summary
We compared planners across two paired population arms:
1. **Heterogeneous**: A mixture of cautious (25%), standard (50%), and hurried (25%) pedestrians.
2. **Mean-Matched Homogeneous**: A homogeneous population with speed/radius set to the weighted means of the mixture.

We evaluated the safety metric **clearance_m** (distance between robot and nearest pedestrian) using **mean** and **CVaR (alpha=0.2)** (tail safety of the 20% closest encounters).

## Rank-Order Sensitivity Analysis
We ran paired bootstrap resampling (1000 iterations) over seeds to compute the probability of one planner beating another under both arms.

### Rankings (metric: clearance_m, higher is safer)
"""

    if rank_sensitivity["status"] == "ready":
        for arm_name in sorted(rank_sensitivity["arms"].keys()):
            arm_data = rank_sensitivity["arms"][arm_name]
            markdown_content += f"\n### Arm: {arm_name}\n"
            markdown_content += f"- **Rank Order**: {', '.join(arm_data['ranking'])}\n"
            markdown_content += "- **Observed Means**:\n"
            for p, m in arm_data["observed_means"].items():
                markdown_content += f"  - {p}: {m:.4f} m\n"
            markdown_content += "- **Pairwise Bootstrap Probabilities**:\n"
            for pair_name, prob in arm_data["pairwise_probabilities"].items():
                markdown_content += (
                    f"  - P({pair_name.replace('_beats_', ' beats ')}) = {prob:.2%}\n"
                )

        markdown_content += "\n### Rank Reversals\n"
        reversals = rank_sensitivity.get("reversals", [])
        if reversals:
            for rev in reversals:
                markdown_content += f"- **WARNING**: Reversal detected! {rev['description']}\n"
        else:
            markdown_content += "- **No rank reversals detected.** The planner ranking was stable across both arms.\n"
    else:
        markdown_content += (
            f"\nRank sensitivity calculation was blocked: {rank_sensitivity.get('blockers')}\n"
        )

    markdown_content += """
## Detailed Ablation Results
Below is the table of the clearance metrics per seed and arm:

| Scenario | Seed | Planner | Arm | Mean Clearance (m) | CVaR Clearance (m) |
|---|---|---|---|---|---|
"""
    for r in csv_rows:
        markdown_content += f"| {r['scenario_id']} | {r['seed']} | {r['planner']} | {r['arm']} | {r['mean_clearance_m']} | {r['cvar_clearance_m']} |\n"

    markdown_content += """
## Non-Reactive Mixture Sweeps Caveats
- Since this is a CPU-level smoke validation run on a small slice, rank sensitivity estimates carry higher uncertainty.
- In full runs, a larger sample of seeds and scenarios is required to establish statistical significance.
"""

    for dest in (output_dir, durable_dir):
        with (dest / "analysis.md").open("w", encoding="utf-8") as f:
            f.write(markdown_content)

    print(f"Report generation complete. Wrote files to {output_dir} and {durable_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
