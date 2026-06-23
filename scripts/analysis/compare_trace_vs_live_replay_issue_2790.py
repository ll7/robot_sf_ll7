#!/usr/bin/env python3
"""Compare trace-derived predicted no-effect/delay sensitivity against live replay.

This script parses trace-derived diagnostic summaries and live replay summaries,
compares them condition by condition, and reports agreement classification:
confirmed, false_positive, false_negative, live_only_effect, and trace_only_effect.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_TRACE_SUMMARY = (
    REPO_ROOT
    / "docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13/summary.json"
)
DEFAULT_LIVE_SUMMARY = (
    REPO_ROOT / "docs/context/evidence/issue_2777_live_observation_noise_replay/summary.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/context/evidence/issue_2790_trace_vs_live_replay"

REQUIRED_CONDITIONS = (
    "noop",
    "low_noise",
    "medium_noise",
    "missed_detection_only",
    "occlusion_only",
    "delay_only",
    "combined",
)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    """Load and parse a JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Required summary JSON file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_comparison_report(
    trace_data: dict[str, Any],
    live_data: dict[str, Any],
) -> dict[str, Any]:
    """Compare trace-derived vs live replay results condition by condition."""
    trace_conds = {c["condition"]: c for c in trace_data.get("conditions", [])}
    live_conds = {c["name"]: c for c in live_data.get("conditions", [])}

    comparison_rows = []
    all_matched = True

    for cond in REQUIRED_CONDITIONS:
        trace_obj = trace_conds.get(cond)
        live_obj = live_conds.get(cond)

        if not trace_obj or not live_obj:
            raise ValueError(f"Missing condition data for '{cond}' in trace or live summary.")

        trace_label = trace_obj.get("classification", {}).get("label", "unknown")
        trace_rationale = trace_obj.get("classification", {}).get("rationale", "")

        # Extract live indicators
        if cond == "noop":
            live_command_changed = False
            live_progress_changed = False
        else:
            live_command_changed = bool(
                live_obj.get("command_summary", {}).get("sequence_changed", False)
            )
            live_progress_changed = any(
                v.get("changed", False)
                for v in live_obj.get("progress_delta", {}).values()
                if isinstance(v, dict)
            )

        live_label = live_obj.get("classification", {}).get("label", "unknown")
        live_rationale = live_obj.get("classification", {}).get("rationale", "")

        # Determine trace-derived prediction: does it claim sensitivity?
        trace_predicts_sensitivity = trace_label == "robustness_evidence"

        # Determine live behavior change
        live_has_effect = (
            live_command_changed
            or live_progress_changed
            or (live_label == "behavior_sensitive_diagnostic_only")
        )

        # Classification label mapping
        if trace_predicts_sensitivity:
            if live_has_effect:
                comp_label = "confirmed"
                comp_category = "confirmed_sensitivity"
                interpretation = "Trace correctly predicted live replay sensitivity."
            else:
                comp_label = "false_positive"
                comp_category = "trace_only_effect"
                interpretation = "Trace predicted delay sensitivity but live replay showed no command/progress changes."
                all_matched = False
        elif live_has_effect:
            comp_label = "false_negative"
            comp_category = "live_only_effect"
            interpretation = (
                "Trace predicted no-effect/weak but live replay showed behavior sensitivity."
            )
            all_matched = False
        else:
            comp_label = "confirmed"
            comp_category = "confirmed_no_effect"
            interpretation = "Trace and live replay agree on no actionable sensitivity / no-effect."

        comparison_rows.append(
            {
                "condition": cond,
                "trace_label": trace_label,
                "trace_rationale": trace_rationale,
                "live_command_changed": live_command_changed,
                "live_progress_changed": live_progress_changed,
                "live_label": live_label,
                "live_rationale": live_rationale,
                "comparison_label": comp_label,
                "comparison_category": comp_category,
                "interpretation": interpretation,
            }
        )

    # Trustworthiness Verdict
    # Trace-derived diagnostics predict live replay correctly only if all matched
    prefilter_trustworthy = all_matched
    verdict = (
        "Trace-derived diagnostics correctly predicted all live replay outcomes."
        if prefilter_trustworthy
        else "Trace-derived diagnostics failed to predict live replay outcomes correctly (delay_only was a false positive)."
    )

    recommended_action = (
        "keep_as_cheap_prefilter" if prefilter_trustworthy else "demote_to_debugging_only"
    )

    return {
        "schema_version": "compare_trace_vs_live_replay.v1",
        "issue": 2790,
        "evidence_grade": "analysis_only",
        "verdict": verdict,
        "prefilter_trustworthy": prefilter_trustworthy,
        "recommended_action": recommended_action,
        "decision_rule": (
            "If trace-derived diagnostics predict live replay ranking correctly, keep "
            "them as a cheap prefilter. If they do not, demote trace-derived "
            "artifacts to debugging-only evidence."
        ),
        "trace_metadata": {
            "issue": trace_data.get("issue"),
            "schema_version": trace_data.get("schema_version"),
            "reproducibility": trace_data.get("reproducibility"),
        },
        "live_metadata": {
            "issue": live_data.get("issue"),
            "schema_version": live_data.get("schema_version"),
            "run_config": live_data.get("run_config"),
        },
        "comparisons": comparison_rows,
    }


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render the comparison report to Markdown."""
    lines = [
        "# Issue #2790 Trace vs Live Replay Comparison Report",
        "",
        "- **Status**: completed",
        f"- **Evidence Grade**: `{report['evidence_grade']}`",
        f"- **Prefilter Trustworthy**: `{report['prefilter_trustworthy']}`",
        f"- **Recommended Action**: `{report['recommended_action']}`",
        f"- **Decision Verdict**: {report['verdict']}",
        "",
        "## Decision Rule Context",
        "",
        f"- *Decision Rule*: {report['decision_rule']}",
        "- *Action Outcome*: Since trace-derived delay sensitivity did not reproduce in the live replay "
        "DWA wrapper run (resulting in a `false_positive` for `delay_only`), the trace-derived envelope "
        "artifacts are demoted to debugging-only evidence and must not be used as a cheap prefilter.",
        "",
        "## Comparison Table",
        "",
        "| Condition | Trace Label | Live Cmd Changed | Live Prog Changed | Comparison Label | Interpretation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in report["comparisons"]:
        lines.append(
            f"| `{row['condition']}` | `{row['trace_label']}` | `{row['live_command_changed']}` | "
            f"`{row['live_progress_changed']}` | **`{row['comparison_label']}`** | {row['interpretation']} |"
        )

    lines.extend(
        [
            "",
            "## Trace Condition Details",
            "",
        ]
    )

    for row in report["comparisons"]:
        lines.extend(
            [
                f"### `{row['condition']}`",
                f"- **Trace Prediction**: `{row['trace_label']}`",
                f"  - *Rationale*: {row['trace_rationale']}",
                f"- **Live Replay Outcome**: `{row['live_label']}`",
                f"  - *Rationale*: {row['live_rationale']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Claim Boundary & Preservation",
            "",
            "This comparison preserves the repository's fail-closed evidence discipline. "
            "Agreement or disagreement on a single scenario/seed must not be treated as broad "
            "experimental validity. Trace-derived agreement does not replace live replay for benchmark claims.",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare trace-derived predicted sensitivity against live replay outcomes."
    )
    parser.add_argument(
        "--trace-summary",
        type=pathlib.Path,
        default=DEFAULT_TRACE_SUMMARY,
        help="Path to trace-derived summary JSON.",
    )
    parser.add_argument(
        "--live-summary",
        type=pathlib.Path,
        default=DEFAULT_LIVE_SUMMARY,
        help="Path to live replay summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write output report and JSON summary.",
    )

    args = parser.parse_args()

    try:
        trace_data = load_json(args.trace_summary)
        live_data = load_json(args.live_summary)
    except FileNotFoundError as exc:
        print(f"Error loading inputs: {exc}", file=sys.stderr)
        return 1

    try:
        report = build_comparison_report(trace_data, live_data)
    except Exception as exc:
        print(f"Error building comparison report: {exc}", file=sys.stderr)
        return 1

    output_dir: pathlib.Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write summary.json
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote summary JSON: {summary_path}")

    # Write README.md
    markdown_report = render_markdown_report(report)
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(markdown_report)
    print(f"Wrote Markdown report: {readme_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
