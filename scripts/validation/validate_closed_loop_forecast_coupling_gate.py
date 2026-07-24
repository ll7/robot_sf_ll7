#!/usr/bin/env python3
"""Closed-loop forecast coupling gate report.

Synthesizes forecast comparison evidence (issue #2781) with closed-loop gate
metrics (issue #1897) into a deterministic revise/continue/stop recommendation.
Diagnostic-only: no learned training, no expensive campaign run.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any

_DEFAULT_FORECAST_PATH = (
    "docs/context/evidence/issue_2781_interaction_aware_forecast_2026-06-15/comparison_report.json"
)
_DEFAULT_GATE_README = (
    "docs/context/evidence/issue_1897_predictive_coupling_gate_2026-05-31/README.md"
)


def _load_forecast_comparison(path: Path) -> dict[str, Any]:
    """Load and basic-validate the forecast comparison JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if "interaction_effect" not in data:
        raise ValueError("forecast comparison missing interaction_effect block")
    if "comparison_rows" not in data:
        raise ValueError("forecast comparison missing comparison_rows")
    return data


def _parse_gate_metrics_from_readme(path: Path) -> dict[str, Any]:
    """Extract closed-loop gate metrics from the issue #1897 README.

    Uses regex patterns tolerant of surrounding backticks and extra whitespace.
    Normalizes status/reason to lowercase.  Fail-closed: raises ValueError if
    any required field is missing or malformed.
    """
    text = path.read_text(encoding="utf-8")
    metrics: dict[str, Any] = {}

    patterns: list[tuple[str, re.Pattern[str], str]] = [
        ("status", re.compile(r"-\s*status:\s*`?([A-Za-z_]+)`?\s*$"), "str"),
        ("reason", re.compile(r"-\s*reason:\s*`?(.+?)`?\s*$"), "str"),
        (
            "global_success_delta",
            re.compile(r"-\s*global success delta:\s*`?([-+]?\d+\.?\d*)`?\s*$"),
            "float",
        ),
        (
            "hard_success_delta",
            re.compile(r"-\s*hard success delta:\s*`?([-+]?\d+\.?\d*)`?\s*$"),
            "float",
        ),
        (
            "global_min_distance_delta",
            re.compile(r"-\s*global mean-min-distance delta:\s*`?([-+]?\d+\.?\d*)`?\s*$"),
            "float",
        ),
    ]

    for line in text.splitlines():
        stripped = line.strip()
        for key, pat, kind in patterns:
            if key in metrics:
                continue
            m = pat.search(stripped)
            if m:
                raw = m.group(1)
                if kind == "str":
                    metrics[key] = raw.strip().lower()
                else:
                    metrics[key] = float(raw)

    required = {
        "status",
        "reason",
        "global_success_delta",
        "hard_success_delta",
        "global_min_distance_delta",
    }
    missing = required - set(metrics)
    if missing:
        raise ValueError(f"gate README missing fields: {sorted(missing)}")
    return metrics


def _summarize_interaction_effect(forecast: dict[str, Any]) -> dict[str, Any]:
    """Compact summary of the forecast interaction effect."""
    ie = forecast.get("interaction_effect")
    if not isinstance(ie, dict):
        raise ValueError("forecast comparison interaction_effect must be an object")
    rows_value = forecast.get("comparison_rows")
    rows = rows_value if isinstance(rows_value, list) else []
    evaluable = [r for r in rows if r.get("status") == "evaluated"]
    return {
        "matched_rows": ie.get("matched_rows", 0),
        "mean_ade_1s_delta_vs_cv": ie.get("mean_ade_1s_delta_vs_cv"),
        "mean_nll_1s_delta_vs_cv": ie.get("mean_nll_1s_delta_vs_cv"),
        "evaluable_row_count": len(evaluable),
        "conclusion": ie.get("conclusion", ""),
    }


def _decide_recommendation(
    forecast_effect: dict[str, Any],
    gate: dict[str, Any],
) -> str:
    """Deterministic recommendation: revise | continue | stop.

    Rules:
    - closed-loop gate failed → revise (forecast quality insufficient for coupling).
    - closed-loop gate passed + positive forecast signal → continue.
    - both negative or contradictory → stop.
    """
    gate_passed = gate.get("status") == "passed"
    ade_delta = forecast_effect.get("mean_ade_1s_delta_vs_cv")
    nll_delta = forecast_effect.get("mean_nll_1s_delta_vs_cv")

    forecast_positive = False
    if ade_delta is not None and nll_delta is not None:
        # Positive means better: lower ADE (negative delta) and lower NLL (negative delta).
        forecast_positive = (ade_delta < 0) and (nll_delta < 0)

    if not gate_passed:
        return "revise"
    if gate_passed and forecast_positive:
        return "continue"
    return "stop"


def build_gate_report(
    forecast_path: Path,
    gate_readme_path: Path,
) -> dict[str, Any]:
    """Build the full gate report dict."""
    forecast = _load_forecast_comparison(forecast_path)
    gate = _parse_gate_metrics_from_readme(gate_readme_path)
    effect = _summarize_interaction_effect(forecast)
    recommendation = _decide_recommendation(effect, gate)

    return {
        "issue": 2843,
        "claim_boundary": (
            "Diagnostic-only closed-loop forecast coupling gate. "
            "Not paper-facing evidence. No learned training involved."
        ),
        "inputs": {
            "forecast_comparison": str(forecast_path),
            "gate_readme": str(gate_readme_path),
        },
        "forecast_interaction_effect": effect,
        "closed_loop_gate": gate,
        "recommendation": recommendation,
        "diagnostic_notes": _build_diagnostic_notes(effect, gate, recommendation),
    }


def _build_diagnostic_notes(
    effect: dict[str, Any],
    gate: dict[str, Any],
    recommendation: str,
) -> str:
    parts: list[str] = []
    ade = effect.get("mean_ade_1s_delta_vs_cv")
    nll = effect.get("mean_nll_1s_delta_vs_cv")
    if ade is not None and nll is not None:
        ade_dir = "worsened" if ade > 0 else "improved"
        nll_dir = "worsened" if nll > 0 else "improved"
        parts.append(
            f"Forecast interaction_aware {ade_dir} 1s ADE by {abs(ade):.4f} m vs CV "
            f"and {nll_dir} 1s NLL by {abs(nll):.4f}."
        )
    parts.append(
        f"Closed-loop gate {gate.get('status', '?')} "
        f"(reason={gate.get('reason', '?')}, "
        f"global_success_delta={gate.get('global_success_delta', '?')})."
    )
    if recommendation == "revise":
        parts.append("Recommendation: revise forecast coupling before closed-loop claims.")
    elif recommendation == "stop":
        parts.append("Recommendation: stop; contradictory signals across forecast and gate.")
    else:
        parts.append("Recommendation: continue; both forecast and gate are positive.")
    return " ".join(parts)


def _format_optional_float(value: Any) -> str:
    """Format optional metric values for Markdown tables."""
    if value is None:
        return "NA"
    return f"{float(value):.4f}"


def _render_markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown report."""
    eff = report["forecast_interaction_effect"]
    gate = report["closed_loop_gate"]
    rec = report["recommendation"]
    return textwrap.dedent(f"""\
        # Issue #2843 Closed-Loop Forecast Coupling Gate

        Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2843>

        ## Boundary

        Diagnostic-only closed-loop forecast coupling gate. Not paper-facing evidence.
        No learned training involved. Deterministic: no expensive campaign run.

        ## Inputs

        - Forecast comparison: `{report["inputs"]["forecast_comparison"]}`
        - Gate readme: `{report["inputs"]["gate_readme"]}`

        ## Forecast Interaction Effect

        | Metric | Value |
        |---|---:|
        | Matched rows | {eff["matched_rows"]} |
        | Evaluable rows | {eff["evaluable_row_count"]} |
        | Mean ADE 1s delta vs CV (m) | {_format_optional_float(eff["mean_ade_1s_delta_vs_cv"])} |
        | Mean NLL 1s delta vs CV | {_format_optional_float(eff["mean_nll_1s_delta_vs_cv"])} |

        {eff["conclusion"]}

        ## Closed-Loop Gate

        | Metric | Value |
        |---|---|
        | Status | `{gate["status"]}` |
        | Reason | `{gate["reason"]}` |
        | Global success delta | {gate["global_success_delta"]:.4f} |
        | Hard success delta | {gate["hard_success_delta"]:.4f} |
        | Global min-distance delta | {gate["global_min_distance_delta"]:.4f} |

        ## Recommendation

        **{rec.upper()}**

        {report["diagnostic_notes"]}
    """)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Closed-loop forecast coupling gate report (issue #2843)."
    )
    parser.add_argument(
        "--forecast-comparison",
        type=Path,
        default=Path(_DEFAULT_FORECAST_PATH),
        help="Path to forecast comparison JSON (issue #2781).",
    )
    parser.add_argument(
        "--gate-readme",
        type=Path,
        default=Path(_DEFAULT_GATE_README),
        help="Path to closed-loop gate README (issue #1897).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write gate_report.json and gate_report.md.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the gate check and return a shell-friendly exit code."""
    str_argv = [str(a) for a in argv] if argv is not None else None
    args = build_arg_parser().parse_args(str_argv)

    if not args.forecast_comparison.exists():
        print(
            json.dumps(
                {"status": "error", "error": f"forecast not found: {args.forecast_comparison}"}
            )
        )
        return 1
    if not args.gate_readme.exists():
        print(
            json.dumps({"status": "error", "error": f"gate readme not found: {args.gate_readme}"})
        )
        return 1

    try:
        report = build_gate_report(args.forecast_comparison, args.gate_readme)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 1

    md = _render_markdown(report)

    if args.output_dir:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "gate_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (args.output_dir / "gate_report.md").write_text(md, encoding="utf-8")
        except OSError as exc:
            print(json.dumps({"status": "error", "error": str(exc)}))
            return 1

    print(json.dumps(report, indent=2, sort_keys=True))
    if report["recommendation"] == "stop":
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
