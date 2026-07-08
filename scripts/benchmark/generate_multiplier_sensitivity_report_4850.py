#!/usr/bin/env python3
"""Generate rank-sensitivity report for the non-reactive response multiplier sweep (issue #4850)."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.heterogeneous_rank_sensitivity import compute_bootstrap_rank_sensitivity

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    """Generate and print the rank-sensitivity report."""
    output_dir = REPO_ROOT / "output/issue_4850_multiplier_sweep"

    # Load all episode records
    all_records: list[dict] = []
    for multiplier in [0.0, 0.1, 0.3]:
        jsonl_path = output_dir / f"multiplier_{multiplier}_episode_records.jsonl"
        if not jsonl_path.exists():
            print(f"Error: Episode records not found at {jsonl_path}")
            return 1

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                all_records.append(rec)

    print(f"Loaded {len(all_records)} episode records from 3 multiplier values")

    # Run rank-sensitivity analysis
    report = compute_bootstrap_rank_sensitivity(
        records=all_records,
        metric_key="min_clearance",
        planners=["goal", "social_force"],
        higher_is_safer=True,
        num_bootstrap=1000,
        seed=4850,
    )

    # Write the report
    report_path = output_dir / "rank_sensitivity_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nRank-sensitivity report written to {report_path.relative_to(REPO_ROOT)}")

    # Print summary table
    if report.get("status") == "ready":
        print("\n" + "=" * 80)
        print("RANK-SENSITIVITY SUMMARY (issue #4850)")
        print("=" * 80)
        print(f"\nMetric: {report.get('metric_key', 'N/A')}")
        print(f"Bootstrap iterations: {report.get('num_bootstrap', 'N/A')}")
        print(f"Arms: {report.get('arms', [])}")

        if "pairwise_probabilities" in report:
            print("\nPairwise win probabilities (P(row beats column)):")
            print("-" * 80)
            pairwise = report["pairwise_probabilities"]
            for comparison in pairwise:
                planner_a = comparison.get("planner_a", "N/A")
                planner_b = comparison.get("planner_b", "N/A")
                prob_a_beats_b = comparison.get("prob_a_beats_b", "N/A")
                print(f"  {planner_a} vs {planner_b}: P({planner_a} wins) = {prob_a_beats_b:.3f}")

        if "rankings_by_arm" in report:
            print("\nRankings by arm:")
            print("-" * 80)
            rankings = report["rankings_by_arm"]
            for arm_ranking in rankings:
                arm = arm_ranking.get("arm", "N/A")
                ranking = arm_ranking.get("ranking", [])
                print(f"  {arm}: {' > '.join(ranking)}")

        if "reversals" in report:
            print("\nRank reversals:")
            print("-" * 80)
            reversals = report["reversals"]
            if reversals:
                for reversal in reversals:
                    pair = reversal.get("pair", [])
                    arm_a = reversal.get("arm_a", "N/A")
                    arm_b = reversal.get("arm_b", "N/A")
                    ranking_a = reversal.get("ranking_a", [])
                    ranking_b = reversal.get("ranking_b", [])
                    print(f"  {pair}: {arm_a} ({' > '.join(ranking_a)}) vs {arm_b} ({' > '.join(ranking_b)})")
            else:
                print("  No rank reversals detected across arms.")
    else:
        print(f"\nReport status: {report.get('status', 'unknown')}")
        if "blockers" in report:
            print(f"Blockers: {report['blockers']}")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
