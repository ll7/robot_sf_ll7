"""Diagnostic comparison of forecast-risk scoring modes before planner coupling.

Compares five risk-channel modes on identical deterministic observations:

- ``no_risk``: baseline with ``forecast_risk_weight=0``
- ``raw_risk``: unfiltered forecast-risk penalty
- ``calibration_filtered``: gated by Issue #2865 calibration eligibility
- ``actor_class_aware``: gated by per-actor-class calibration rows
- ``observation_tier_aware``: gated by cross-observation-tier calibration rows

The tool reads the Issue #2865 calibration report and marks unavailable/blocked
modes explicitly. It does NOT run a live benchmark and makes no safety,
navigation, paper, or dissertation claim.

Usage::

    uv run python scripts/validation/validate_forecast_risk_calibration_filter.py
    uv run python scripts/validation/validate_forecast_risk_calibration_filter.py \
        --out-dir docs/context/evidence/issue_2869_forecast_risk_calibration_filter_2026-06-15
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

SCHEMA_VERSION = "forecast_risk_calibration_filter.diagnostic_comparison.v1"
CLAIM_BOUNDARY = "diagnostic_only_not_benchmark_evidence"
_DIAGNOSTIC_WEIGHT = 5.0
_DEFAULT_CALIBRATION_PATH = (
    "docs/context/evidence/issue_2865_forecast_calibration_report_2026-06-15/"
    "calibration_report.json"
)
_RISK_MODES = (
    "no_risk",
    "raw_risk",
    "calibration_filtered",
    "actor_class_aware",
    "observation_tier_aware",
)


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
    """High-risk scenario: raw risk should slow/stop the goal proposal."""
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


def _load_calibration_report(path: Path) -> dict[str, Any]:
    """Load and basic-validate the Issue #2865 calibration report."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("reliability_rows"), list):
        raise ValueError("calibration report missing reliability_rows")
    return data


def _any_eligible_for_risk_scoring(rows: list[dict[str, Any]]) -> bool:
    """Return True if at least one row is eligible for forecast-risk scoring.

    Recognizes the eligibility vocabulary emitted by
    :func:`robot_sf.benchmark.forecast_calibration_report._risk_scoring_eligibility`
    (canonical token ``eligible_analysis_only``) as well as the legacy ``eligible`` /
    ``calibrated`` tokens kept for backward compatibility.
    """
    eligible = {"eligible", "calibrated", "eligible_analysis_only"}
    return any(str(row.get("risk_scoring_eligibility", "")) in eligible for row in rows)


def _actor_class_available(rows: list[dict[str, Any]]) -> bool:
    """Return True if any row carries a usable actor class."""
    return any(str(row.get("actor_class", "unavailable")).lower() != "unavailable" for row in rows)


def _observation_tier_variation(rows: list[dict[str, Any]]) -> set[str]:
    """Return the set of observation tiers present in calibration rows."""
    return {str(row.get("observation_tier", "unknown")) for row in rows}


def _mode_availability(mode: str, rows: list[dict[str, Any]]) -> tuple[str, str]:
    """Resolve (status, reason) for a risk mode from calibration rows.

    Returns:
        tuple[str, str]: ``("available", "")`` or ``("blocked", "reason")``.
    """
    if mode == "no_risk":
        return "available", "baseline forecast_risk_weight=0"
    if mode == "raw_risk":
        return "available", "diagnostic unfiltered forecast risk"
    if mode == "calibration_filtered":
        if _any_eligible_for_risk_scoring(rows):
            return "available", "calibration report contains risk-scoring-eligible rows"
        return "blocked", "no_rows_risk_scoring_eligible"
    if mode == "actor_class_aware":
        if _actor_class_available(rows):
            return "available", "per-actor-class calibration rows available"
        return "blocked", "actor_class_unavailable_in_all_rows"
    if mode == "observation_tier_aware":
        tiers = _observation_tier_variation(rows)
        if len(tiers) > 1:
            return "available", f"cross-tier calibration rows available: {sorted(tiers)}"
        return "blocked", f"single_observation_tier: {sorted(tiers)[0] if tiers else 'unknown'}"
    return "blocked", f"unknown_mode: {mode}"


def _run_single(
    *,
    mode: str,
    case: dict[str, Any],
    calibration_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Run one mode/case combination and return metrics, or None if blocked."""
    status, _reason = _mode_availability(mode, calibration_rows)
    if status != "available":
        return None

    if mode == "raw_risk":
        config = PolicyStackV1Config(
            proposal_sources=("goal", "risk_dwa"),
            forecast_risk_weight=_DIAGNOSTIC_WEIGHT,
        )
    else:
        config = PolicyStackV1Config(
            proposal_sources=("goal", "risk_dwa"),
            forecast_risk_weight=0.0,
        )

    stack = PolicyStackV1Adapter(config=config, risk_dwa=_FixedRiskDWA())
    stack.plan(case["observation"])
    last = stack.diagnostics()["last_step"]
    if last is None:
        raise AssertionError("diagnostics last_step must exist after plan()")

    goal_components = last.get("risk_score_components", {}).get("goal", {})
    command = last.get("selected_command", [0.0, 0.0])
    speed = abs(float(command[0]))
    return {
        "selected_proposal": str(last.get("selected_proposal_key", "")),
        "speed": round(speed, 6),
        "progress_proxy": round(speed, 6),
        "forecast_penalty": round(float(goal_components.get("forecast_risk_penalty", 0.0)), 6),
        "shield_stop_count": 1 if last.get("selected_proposal_key") == "shield_stop" else 0,
    }


def _build_mode_rows(calibration_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build one table row per risk mode with per-case metrics."""
    case_metrics: dict[str, dict[str, dict[str, Any] | None]] = {mode: {} for mode in _RISK_MODES}
    for mode in _RISK_MODES:
        for case in _CASES:
            metrics = _run_single(
                mode=mode,
                case=case,
                calibration_rows=calibration_rows,
            )
            case_metrics[mode][case["name"]] = metrics

    rows: list[dict[str, Any]] = []
    for mode in _RISK_MODES:
        status, reason = _mode_availability(mode, calibration_rows)
        high = case_metrics[mode].get("high_risk_diagnostic_slows_goal")
        fp = case_metrics[mode].get("false_positive_suppresses_penalty")
        row: dict[str, Any] = {
            "mode": mode,
            "status": status,
            "reason": reason,
            "forecast_risk_weight": 0.0 if mode == "no_risk" else _DIAGNOSTIC_WEIGHT,
            "uses_calibration_filter": mode
            in {"calibration_filtered", "actor_class_aware", "observation_tier_aware"},
            "high_risk": high,
            "false_positive": fp,
        }
        rows.append(row)
    return rows


def _compute_tradeoffs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute cross-mode tradeoffs for progress and false-positive stopping."""
    by_mode = {row["mode"]: row for row in rows}
    no_risk_high = by_mode.get("no_risk", {}).get("high_risk")
    raw_risk_high = by_mode.get("raw_risk", {}).get("high_risk")
    no_risk_fp = by_mode.get("no_risk", {}).get("false_positive")
    raw_risk_fp = by_mode.get("raw_risk", {}).get("false_positive")

    route_progress_tradeoff: float | None = None
    if (
        no_risk_high is not None
        and raw_risk_high is not None
        and isinstance(no_risk_high, dict)
        and isinstance(raw_risk_high, dict)
    ):
        route_progress_tradeoff = round(
            float(no_risk_high["progress_proxy"]) - float(raw_risk_high["progress_proxy"]),
            6,
        )

    false_positive_stopping: bool | None = None
    false_positive_unnecessary_slowdown_count: int | None = None
    if (
        no_risk_fp is not None
        and raw_risk_fp is not None
        and isinstance(no_risk_fp, dict)
        and isinstance(raw_risk_fp, dict)
    ):
        false_positive_stopping = (
            raw_risk_fp["selected_proposal"] == "goal" and raw_risk_fp["forecast_penalty"] == 0.0
        )
        false_positive_unnecessary_slowdown_count = int(raw_risk_fp["speed"] < no_risk_fp["speed"])

    return {
        "route_progress_tradeoff": route_progress_tradeoff,
        "false_positive_stopping_avoided": false_positive_stopping,
        "false_positive_unnecessary_slowdown_count": false_positive_unnecessary_slowdown_count,
    }


def _decide_recommendation(rows: list[dict[str, Any]], tradeoffs: dict[str, Any]) -> str:
    """Return a bounded recommendation for calibration-filter gating."""
    by_mode = {row["mode"]: row for row in rows}
    calibration_filtered = by_mode.get("calibration_filtered", {})
    if calibration_filtered.get("status") == "blocked":
        return "wait"
    # If calibration filtering were ever available, require raw_risk consistency too.
    raw_risk = by_mode.get("raw_risk", {})
    raw_high = raw_risk.get("high_risk") if isinstance(raw_risk.get("high_risk"), dict) else None
    if raw_high is None or raw_high.get("forecast_penalty", 0.0) <= 0.0:
        return "wait"
    if tradeoffs.get("false_positive_stopping_avoided") is False:
        return "wait"
    return "diagnostic_only"


def _recommendation_rationale(recommendation: str) -> str:
    """Return a rationale matching the current recommendation."""
    if recommendation == "wait":
        return (
            "Calibration filtering cannot gate forecast-risk scoring: the Issue #2865 report "
            "contains no risk-scoring-eligible rows. Keep forecast-risk scoring opt-in and "
            "diagnostic-only until eligible calibration evidence exists."
        )
    return (
        "Calibration-filtered rows are available in the input report, but this tool remains a "
        "deterministic diagnostic comparison rather than benchmark evidence. Use it only to "
        "decide whether a same-seed benchmark comparison is now warranted."
    )


def build_report(calibration_path: Path | None = None) -> dict[str, Any]:
    """Build the full diagnostic comparison report.

    Args:
        calibration_path: Path to Issue #2865 calibration report JSON.
            Defaults to the tracked evidence location.

    Returns:
        dict with claim_boundary, mode table, tradeoffs, and recommendation.
    """
    path = calibration_path or Path(_DEFAULT_CALIBRATION_PATH)
    report_data = _load_calibration_report(path)
    rows = report_data.get("reliability_rows", [])

    mode_rows = _build_mode_rows(rows)
    tradeoffs = _compute_tradeoffs(mode_rows)
    recommendation = _decide_recommendation(mode_rows, tradeoffs)

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 2869,
        "claim_boundary": CLAIM_BOUNDARY,
        "calibration_report": str(path),
        "diagnostic_weight": _DIAGNOSTIC_WEIGHT,
        "modes": mode_rows,
        "tradeoffs": tradeoffs,
        "recommendation": recommendation,
        "recommendation_rationale": _recommendation_rationale(recommendation),
    }


def _format_optional_float(value: Any) -> str:
    """Format optional metric values for Markdown tables."""
    if value is None:
        return "NA"
    return f"{float(value):.6f}"


def _format_optional_int(value: Any) -> str:
    """Format optional integer values for Markdown tables."""
    if value is None:
        return "NA"
    return str(int(value))


def _render_markdown(report: dict[str, Any]) -> str:
    """Render the report as a human-readable Markdown string."""
    lines = [
        "# Issue #2869 Forecast Risk Calibration Filter Diagnostic",
        "",
        "Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2869>",
        "",
        "## Boundary",
        "",
        f"- **schema_version**: `{report['schema_version']}`",
        f"- **claim_boundary**: `{report['claim_boundary']}`",
        f"- **calibration_report**: `{report['calibration_report']}`",
        f"- **diagnostic_weight**: {report['diagnostic_weight']}",
        f"- **recommendation**: `{report['recommendation']}`",
        "",
        "## Risk Mode Comparison",
        "",
        "| mode | status | forecast_risk_weight | high_risk selected | high_risk speed | "
        "high_risk penalty | false_positive selected | false_positive speed | "
        "false_positive penalty |",
        "|---|---:|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in report["modes"]:
        high = row["high_risk"] if isinstance(row["high_risk"], dict) else {}
        fp = row["false_positive"] if isinstance(row["false_positive"], dict) else {}
        lines.append(
            f"| {row['mode']} | {row['status']} | "
            f"{row['forecast_risk_weight']:.1f} | "
            f"{high.get('selected_proposal', 'NA')} | "
            f"{_format_optional_float(high.get('speed'))} | "
            f"{_format_optional_float(high.get('forecast_penalty'))} | "
            f"{fp.get('selected_proposal', 'NA')} | "
            f"{_format_optional_float(fp.get('speed'))} | "
            f"{_format_optional_float(fp.get('forecast_penalty'))} |"
        )
    lines.append("")
    lines.append("### Blocked mode reasons")
    lines.append("")
    for row in report["modes"]:
        if row["status"] == "blocked":
            lines.append(f"- **{row['mode']}**: {row['reason']}")
    lines.append("")
    lines.append("## Tradeoffs")
    lines.append("")
    lines.append("| tradeoff | value |")
    lines.append("|---|---|")
    tradeoffs = report["tradeoffs"]
    lines.append(
        f"| route_progress_tradeoff (no_risk - raw_risk) | "
        f"{_format_optional_float(tradeoffs.get('route_progress_tradeoff'))} |"
    )
    lines.append(
        f"| false_positive_stopping_avoided | "
        f"{tradeoffs.get('false_positive_stopping_avoided', 'NA')} |"
    )
    lines.append(
        f"| false_positive_unnecessary_slowdown_count | "
        f"{_format_optional_int(tradeoffs.get('false_positive_unnecessary_slowdown_count'))} |"
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"**{report['recommendation'].upper()}**")
    lines.append("")
    lines.append(report["recommendation_rationale"])
    lines.append("")
    lines.append(
        "> This report is diagnostic-only. It does not establish safety, navigation benefit, "
        "human realism, benchmark-strength predictor quality, or paper/dissertation claims."
    )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-report",
        type=Path,
        default=Path(_DEFAULT_CALIBRATION_PATH),
        help="Path to Issue #2865 calibration report JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for report.json, report.md, summary.json, README.md (default: stdout only).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the diagnostic comparison and optionally write artifacts.

    Returns:
        Exit code 0 on success.
    """
    args = build_arg_parser().parse_args(argv)
    if not args.calibration_report.exists():
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"calibration report not found: {args.calibration_report}",
                }
            )
        )
        return 1

    try:
        report = build_report(args.calibration_report)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1

    summary = {
        "issue": report["issue"],
        "schema_version": report["schema_version"],
        "claim_boundary": report["claim_boundary"],
        "recommendation": report["recommendation"],
        "mode_count": len(report["modes"]),
        "blocked_mode_count": sum(1 for row in report["modes"] if row["status"] == "blocked"),
        "route_progress_tradeoff": report["tradeoffs"].get("route_progress_tradeoff"),
        "false_positive_stopping_avoided": report["tradeoffs"].get(
            "false_positive_stopping_avoided"
        ),
        "false_positive_unnecessary_slowdown_count": report["tradeoffs"].get(
            "false_positive_unnecessary_slowdown_count"
        ),
    }
    print(json.dumps({"summary": summary, "report": report}, indent=2, sort_keys=True))

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        (args.out_dir / "report.md").write_text(_render_markdown(report), encoding="utf-8")
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        readme = _render_readme(report)
        (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    return 0


def _render_readme(report: dict[str, Any]) -> str:
    """Render the evidence-bundle README."""
    calibration_report = str(report["calibration_report"])
    calibration_link = (
        "../issue_2865_forecast_calibration_report_2026-06-15/calibration_report.json"
    )
    return (
        "# Issue #2869 Forecast Risk Calibration Filter Diagnostic\n\n"
        "## Scope\n\n"
        "This bundle compares five forecast-risk scoring modes before trusting planner coupling:\n"
        "`no_risk`, `raw_risk`, `calibration_filtered`, `actor_class_aware`, and "
        "`observation_tier_aware`.\n\n"
        "## Evidence status\n\n"
        f"- `schema`: `{report['schema_version']}`\n"
        f"- `calibration_report`: [{calibration_report}]({calibration_link})\n"
        "- `report`: [report.json](report.json) and [report.md](report.md)\n"
        f"- `claim_boundary`: {report['claim_boundary']}\n"
        f"- `recommendation`: `{report['recommendation']}`\n\n"
        "## Reproducible command\n\n"
        "```\n"
        "uv run python scripts/validation/validate_forecast_risk_calibration_filter.py \\\n"
        "  --out-dir docs/context/evidence/"
        "issue_2869_forecast_risk_calibration_filter_2026-06-15\n"
        "```\n\n"
        "## Validation\n\n"
        "```\n"
        "uv run pytest tests/validation/test_forecast_risk_calibration_filter.py\n"
        "```\n\n"
        "## Claim boundary\n\n"
        "This report is diagnostic-only. It does not establish safety, navigation benefit, "
        "human realism, benchmark-strength predictor quality, or paper/dissertation claims. "
        "Forecast-risk scoring remains opt-in with `forecast_risk_weight=0.0` by default.\n"
    )


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
