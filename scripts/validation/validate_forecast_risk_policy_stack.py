"""Diagnostic same-seed comparison for forecast risk scoring in policy_stack_v1.

Produces a bounded diagnostic report comparing baseline (forecast_risk_weight=0)
versus diagnostic (forecast_risk_weight>0) on identical deterministic observations.
This is NOT a live benchmark claim. The output carries claim_boundary
``diagnostic_only_not_benchmark_evidence``.

Usage::

    uv run python scripts/validation/validate_forecast_risk_policy_stack.py
    uv run python scripts/validation/validate_forecast_risk_policy_stack.py \
        --out-dir docs/context/evidence/issue_2759_forecast_risk_policy_stack_2026-06-15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.planner.policy_stack_v1 import (
    PolicyStackV1Adapter,
    PolicyStackV1Config,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

CLAIM_BOUNDARY = "diagnostic_only_not_benchmark_evidence"
SCHEMA_VERSION = "forecast_risk_policy_stack.diagnostic_comparison.v1"
_DIAGNOSTIC_WEIGHT = 5.0


class _FixedRiskDWA:
    """Deterministic risk_dwa test double returning a fixed command."""

    def __init__(self, command: tuple[float, float] = (0.2, 0.0)) -> None:
        self.command = command
        self.calls = 0

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        _ = observation
        self.calls += 1
        return self.command

    def reset(self) -> None:
        self.calls = 0


def _base_observation() -> dict[str, Any]:
    """Return the canonical deterministic observation fixture."""
    return {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [1.0, 0.0]},
        "pedestrians": {"positions": [[2.0, 0.5]]},
    }


def _case_high_risk() -> dict[str, Any]:
    """High-risk scenario: diagnostic scoring should slow/stop the goal proposal."""
    obs = _base_observation()
    obs["forecast_risk_channel"] = {
        "status": "available",
        "risk": 1.0,
        "occupancy_risk": 0.9,
        "collision_relevance": 0.8,
    }
    return obs


def _case_false_positive_suppress() -> dict[str, Any]:
    """False-positive scenario: penalty suppressed, unnecessary stop avoided."""
    obs = _base_observation()
    obs["forecast_risk_channel"] = {
        "status": "available",
        "risk": 1.0,
        "false_positive_risk": 1.0,
        "unnecessary_stop_risk": 1.0,
    }
    return obs


_CASES: list[dict[str, Any]] = [
    {"name": "high_risk_diagnostic_slows_goal", "observation": _case_high_risk()},
    {"name": "false_positive_suppresses_penalty", "observation": _case_false_positive_suppress()},
]


def _run_single(
    config: PolicyStackV1Config,
    observation: dict[str, Any],
    *,
    risk_dwa: _FixedRiskDWA,
) -> dict[str, Any]:
    """Run one adapter step and return the last-step diagnostic packet."""
    stack = PolicyStackV1Adapter(config=config, risk_dwa=risk_dwa)
    stack.plan(observation)
    last = stack.diagnostics()["last_step"]
    if last is None:
        raise AssertionError("diagnostics last_step must exist after plan()")
    return last


def _extract_metrics(step: dict[str, Any]) -> dict[str, Any]:
    """Pull scalar metrics from a diagnostics last_step packet."""
    ranking = step.get("candidate_ranking", [])
    goal_components = step.get("risk_score_components", {}).get("goal", {})
    linear_speed = abs(float(step.get("selected_command", [0.0, 0.0])[0]))
    forecast_penalty = float(goal_components.get("forecast_risk_penalty", 0.0))
    high_risk_speed_reduction = 1.0 if linear_speed < 0.5 else 0.0
    shield_stop_count = 1 if step.get("selected_proposal_key") == "shield_stop" else 0
    return {
        "selected_proposal": step.get("selected_proposal_key", ""),
        "speed": round(linear_speed, 6),
        "progress_proxy": round(linear_speed, 6),
        "forecast_penalty": round(forecast_penalty, 6),
        "high_risk_speed_reduction": high_risk_speed_reduction,
        "shield_stop_count": shield_stop_count,
        "ranking_top": ranking[0]["proposal_key"] if ranking else "",
        "shield_intervened": bool(step.get("shield_intervened", False)),
    }


def build_report() -> dict[str, Any]:
    """Build the full diagnostic comparison report.

    Returns:
        dict with claim_boundary, cases, and recommendation fields.
    """
    baseline_config = PolicyStackV1Config(
        proposal_sources=("goal", "risk_dwa"),
        forecast_risk_weight=0.0,
    )
    diagnostic_config = PolicyStackV1Config(
        proposal_sources=("goal", "risk_dwa"),
        forecast_risk_weight=_DIAGNOSTIC_WEIGHT,
    )

    case_results: list[dict[str, Any]] = []
    for case in _CASES:
        baseline_dwa = _FixedRiskDWA()
        diagnostic_dwa = _FixedRiskDWA()
        baseline_step = _run_single(baseline_config, case["observation"], risk_dwa=baseline_dwa)
        diagnostic_step = _run_single(
            diagnostic_config, case["observation"], risk_dwa=diagnostic_dwa
        )
        baseline_metrics = _extract_metrics(baseline_step)
        diagnostic_metrics = _extract_metrics(diagnostic_step)

        delta_progress = round(
            diagnostic_metrics["progress_proxy"] - baseline_metrics["progress_proxy"], 6
        )
        delta_penalty = round(
            diagnostic_metrics["forecast_penalty"] - baseline_metrics["forecast_penalty"], 6
        )
        false_positive_unnecessary_slowdown = int(
            "false_positive" in str(case["name"])
            and diagnostic_metrics["speed"] < baseline_metrics["speed"]
        )

        case_results.append(
            {
                "case_name": case["name"],
                "baseline": baseline_metrics,
                "diagnostic": diagnostic_metrics,
                "delta_progress_proxy": delta_progress,
                "delta_forecast_penalty": delta_penalty,
                "false_positive_unnecessary_slowdown_count": false_positive_unnecessary_slowdown,
            }
        )

    recommendation = _decide_recommendation(case_results)
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "diagnostic_weight": _DIAGNOSTIC_WEIGHT,
        "cases": case_results,
        "recommendation": recommendation,
    }


def _decide_recommendation(cases: list[dict[str, Any]]) -> str:
    """Classify the diagnostic outcome into a short recommendation string.

    Returns:
        One of the bounded diagnostic recommendation labels.
    """
    high_risk = next(
        (c for c in cases if c["case_name"] == "high_risk_diagnostic_slows_goal"), None
    )
    fp_case = next(
        (c for c in cases if c["case_name"] == "false_positive_suppresses_penalty"), None
    )
    if high_risk is None or fp_case is None:
        return "incomplete_cases"

    hr_diag = high_risk["diagnostic"]
    hr_base = high_risk["baseline"]
    fp_diag = fp_case["diagnostic"]

    high_risk_slows = hr_diag["forecast_penalty"] > 0.0 and hr_diag["speed"] < hr_base["speed"]
    fp_suppressed = fp_diag["forecast_penalty"] == 0.0 and fp_diag["selected_proposal"] == "goal"

    if high_risk_slows and fp_suppressed:
        return "forecast_risk_scoring_diagnostic_consistent"
    if high_risk_slows and not fp_suppressed:
        return "false_positive_suppression_inconsistent"
    if not high_risk_slows and fp_suppressed:
        return "high_risk_penalization_inconsistent"
    return "no_diagnostic_signal"


def _human_summary(report: dict[str, Any]) -> str:
    """Render the report as a human-readable Markdown string."""
    lines = [
        "# Forecast Risk Policy Stack Diagnostic Comparison",
        "",
        f"- **claim_boundary**: `{report['claim_boundary']}`",
        f"- **diagnostic_weight**: {report['diagnostic_weight']}",
        f"- **recommendation**: `{report['recommendation']}`",
        "",
        "## Cases",
        "",
    ]
    for case in report["cases"]:
        lines.append(f"### {case['case_name']}")
        lines.append("")
        lines.append("| Field | Baseline | Diagnostic | Delta |")
        lines.append("|---|---|---|---|")
        for key in (
            "selected_proposal",
            "speed",
            "progress_proxy",
            "forecast_penalty",
            "high_risk_speed_reduction",
            "shield_stop_count",
        ):
            b = case["baseline"][key]
            d = case["diagnostic"][key]
            if key == "delta_progress_proxy":
                continue
            if isinstance(b, float):
                delta = round(d - b, 6)
            elif isinstance(b, int):
                delta = d - b
            else:
                delta = ""
            lines.append(f"| {key} | {b} | {d} | {delta} |")
        lines.append(
            f"| false_positive_unnecessary_slowdown_count |  | "
            f"{case['false_positive_unnecessary_slowdown_count']} |  |"
        )
        lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"```{report['recommendation']}```")
    lines.append("")
    lines.append(
        "> This is a diagnostic-only comparison. "
        "No safety claim is made. "
        f"claim_boundary={CLAIM_BOUNDARY}"
    )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for report.json, report.md, summary.json (default: stdout only).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the diagnostic comparison and optionally write artifacts.

    Returns:
        Exit code 0 on success.
    """
    args = build_arg_parser().parse_args(argv)
    report = build_report()
    summary = {
        "claim_boundary": CLAIM_BOUNDARY,
        "recommendation": report["recommendation"],
        "case_count": len(report["cases"]),
        "diagnostic_weight": report["diagnostic_weight"],
        "false_positive_unnecessary_slowdown_count": sum(
            case["false_positive_unnecessary_slowdown_count"] for case in report["cases"]
        ),
    }
    print(json.dumps({"summary": summary, "report": report}, indent=2, sort_keys=True))

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        (args.out_dir / "report.md").write_text(_human_summary(report), encoding="utf-8")
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
