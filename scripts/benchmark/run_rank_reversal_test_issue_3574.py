#!/usr/bin/env python3
"""Run the issue #3574 pre-specified (preregistered) rank-reversal test.

This is the *pre-specified rank-reversal test* analysis entrypoint for issue #3574. It is
distinct from the descriptive bootstrap rank-sensitivity report emitted by
``build_heterogeneous_population_ablation_report.py``: that report records any ranking-list
disagreement (including differences within sampling noise), whereas this test only declares a
reversal when a planner pair is bootstrap-determined in *both* arms with *opposite* signs.

The decision specification (significance level, percentile-CI method, and decision rule) is
declared before the results are inspected and is echoed verbatim in the output so the
comparison is auditable once real paired campaign episode records are available.

Claim boundary: this is analysis tooling only. It establishes no benchmark, rank-stability,
realism, or sim-to-real claim on its own; a campaign conclusion requires real paired episode
records that pass the mean-matched integration-readiness check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.heterogeneous_population_ablation import (
    RANK_METRIC_KEY,
    assess_mean_matched_episode_records,
)
from robot_sf.benchmark.heterogeneous_rank_sensitivity import pre_specified_rank_reversal_test

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="output/issue_3574_mean_matched_harness/manifest.json",
        help="Paired pre-run manifest used to fail closed on missing or extra episode rows.",
    )
    parser.add_argument(
        "--records",
        default="output/issue_3574_mean_matched_harness/episode_records.jsonl",
        help="Path to the simulation episode records JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/issue_3574_mean_matched_harness",
        help="Directory to write output files.",
    )
    parser.add_argument(
        "--metric-key",
        default=RANK_METRIC_KEY,
        help="Metric under record `metrics` to test (default matches the readiness contract).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Two-sided significance level in (0, 1)."
    )
    parser.add_argument(
        "--num-bootstrap", type=int, default=1000, help="Number of bootstrap resamples."
    )
    parser.add_argument("--seed", type=int, default=3574, help="RNG seed for bootstrap.")
    parser.add_argument(
        "--response-law-fraction",
        type=float,
        help="Analyze one response-law sweep fraction; omitted for legacy unswept manifests.",
    )
    parser.add_argument(
        "--lower-is-safer",
        action="store_true",
        help="Set if smaller metric values rank a planner higher (e.g. collisions).",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file records, skipping blank lines."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _markdown_report(
    *,
    metric_key: str,
    readiness: dict[str, Any],
    test_result: dict[str, Any],
    planners: list[str],
) -> str:
    """Render a compact, claim-bounded markdown summary of the test result."""
    lines = [
        "# Issue #3574 Pre-Specified Rank-Reversal Test",
        "",
        (
            "Preregistered hypothesis test on whether planner rank order reverses between the "
            "heterogeneous and mean-matched homogeneous population arms. A reversal is declared "
            "only when a planner pair is bootstrap-determined in both arms with opposite signs."
        ),
        "",
        "## Claim boundary",
        "",
        "- Analysis tooling only. No benchmark, rank-stability, realism, or sim-to-real claim.",
        "- A campaign conclusion requires real paired episode records that pass the readiness check.",
        "",
        f"- Metric: `{metric_key}`.",
        f"- Significance level alpha: `{test_result['pre_registration']['significance_level_alpha']}`.",
        f"- CI method: `{test_result['pre_registration']['ci_method']}` at level "
        f"`{test_result['pre_registration']['ci_level']}`.",
        "",
        "## Integration readiness",
        "",
        f"- Status: `{readiness['status']}`.",
        f"- Expected rows: `{readiness['expected_row_count']}`; observed rows: "
        f"`{readiness['observed_row_count']}`.",
    ]
    if readiness.get("blockers"):
        lines.append("")
        lines.append("Blockers:")
        for blocker in readiness["blockers"]:
            lines.append(f"- {blocker}")

    lines.extend(["", "## Decision", ""])
    if test_result["status"] != "ready":
        lines.append(f"Test status: `{test_result['status']}` (no decision rendered).")
        for blocker in test_result.get("blockers", []):
            lines.append(f"- {blocker}")
    else:
        lines.append(
            f"- Decision: `{test_result['decision']}` "
            f"(reversal count: `{test_result['reversal_count']}`)."
        )
        lines.append("- Planners: " + ", ".join(f"`{p}`" for p in planners) + ".")
        lines.append("")
        lines.append("| Pair | Heterogeneous leader | Mean-matched leader | Verdict |")
        lines.append("|---|---|---|---|")
        for pair in test_result["pairwise"]:
            het_leader = pair["heterogeneous"]["determined_sign"]
            hom_leader = pair["mean_matched_homogeneous"]["determined_sign"]
            lines.append(
                f"| ({pair['planners'][0]}, {pair['planners'][1]}) "
                f"| {het_leader} | {hom_leader} | `{pair['verdict']}` |"
            )
        if test_result["reversals"]:
            lines.append("")
            lines.append("Reversals:")
            for reversal in test_result["reversals"]:
                lines.append(f"- {reversal['description']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Run the pre-specified rank-reversal test and write the report artifacts."""
    args = parse_args()
    manifest_path = REPO_ROOT / args.manifest
    records_path = REPO_ROOT / args.records
    output_dir = REPO_ROOT / args.output_dir

    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}. Build it first.")
        return 1
    if not records_path.exists():
        print(f"Error: Episode records not found at {records_path}. Run the simulations first.")
        return 1

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    records = load_jsonl(records_path)
    print(f"Loaded {len(records)} episode records.")

    # Fail closed on integration readiness before any analysis: a reversal test on rows that do
    # not satisfy the mean-matched manifest cannot be attributed to a paired campaign.
    readiness = assess_mean_matched_episode_records(manifest, records)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rank_reversal_test_readiness.json").write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not readiness["ready"]:
        print(
            "Blocked: episode records do not satisfy the mean-matched manifest; "
            f"see {output_dir / 'rank_reversal_test_readiness.json'}"
        )
        return 2

    planners = sorted({str(rec["planner"]) for rec in records})
    arms = None
    if args.response_law_fraction is not None:
        if not 0.0 <= args.response_law_fraction <= 1.0:
            print("Error: --response-law-fraction must be in [0, 1].")
            return 1
        suffix = f"/response_law_fraction_{args.response_law_fraction:g}"
        arms = (f"heterogeneous{suffix}", f"mean_matched_homogeneous{suffix}")

    test_result = pre_specified_rank_reversal_test(
        records,
        metric_key=args.metric_key,
        planners=planners,
        higher_is_safer=not args.lower_is_safer,
        alpha=args.alpha,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
        arms=arms,
    )

    (output_dir / "rank_reversal_test.json").write_text(
        json.dumps(test_result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "rank_reversal_test.md").write_text(
        _markdown_report(
            metric_key=args.metric_key,
            readiness=readiness,
            test_result=test_result,
            planners=planners,
        ),
        encoding="utf-8",
    )

    print(f"Decision: {test_result.get('decision', test_result.get('status'))}")
    print(f"Wrote rank-reversal test artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
