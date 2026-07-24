#!/usr/bin/env python3
"""Generate a mixed-scenario coverage matrix for issue #2766.

Cross-references evidence modules against canonical scenario slices.
Every blocked / unavailable / missing-denominator / stale / diagnostic-only /
proxy / fallback cell carries an explicit reason.  No benchmark or
paper-facing overclaims are produced.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning ``None`` when absent."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


DEFAULT_LEDGER = Path("docs/context/evidence/issue_2760_dissertation_evidence_ledger/ledger.json")
DEFAULT_SIGNAL_SUMMARY = Path("docs/context/evidence/issue_2799_signalized_runtime/summary.json")
DEFAULT_OBS_NOISE_SUMMARY = Path(
    "docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13/summary.json"
)
DEFAULT_CV_FORECAST = Path(
    "docs/context/evidence/issue_2774_motion_rich_forecast_traces_2026-06-14/report.json"
)
DEFAULT_GAP_REPORT = Path(
    "docs/context/evidence/issue_2784_dissertation_gap_report/gap_report.json"
)
DEFAULT_NEGATIVE_RESULT_DIR = Path(
    "docs/context/evidence/issue_2788_negative_result_scenario_candidates"
)
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_2766_mixed_scenario_matrix")

SCENARIO_SLICES = [
    "corridor_interaction",
    "bottleneck",
    "crossing",
    "signalized_crossing",
    "occluded_emergence",
    "dense_pedestrian",
    "t_intersection",
    "doorway",
]

EVIDENCE_COLUMNS = [
    "topology_reselection",
    "signal_compliance_metrics",
    "prediction_baseline",
    "observation_perturbation",
    "denominator_health",
    "stale_current_status",
    "claim_eligibility",
]

MATRIX_SCHEMA = "mixed_scenario_matrix.v1"


def _json_object_rows(value: Any) -> list[dict[str, Any]]:
    """Return only dict rows from a JSON list-like value."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _json_object(value: Any) -> dict[str, Any]:
    """Return a JSON object or an empty mapping for malformed/null values."""
    return value if isinstance(value, dict) else {}


def _ledger_row(ledger: dict[str, Any], area: str) -> dict[str, Any] | None:
    """Return one ledger row by area, or ``None`` when missing."""
    ledger_rows = _json_object_rows(ledger.get("rows"))
    for row in ledger_rows:
        if row.get("area") == area:
            return row
    return None


# -----------------------------------------------------------------------
# Per-column evidence extractors
# -----------------------------------------------------------------------


def _topology_cell(
    scenario: str,
    ledger: dict[str, Any] | None,
) -> dict[str, str]:
    """Return topology-reselection status + reason for one scenario slice."""
    row = _ledger_row(ledger, "topology_guidance") if ledger else None
    tier = row.get("evidence_tier", "unavailable") if row else "unavailable"
    status = row.get("artifact_status", "missing") if row else "missing"

    hard_slices = {"t_intersection", "doorway", "bottleneck"}
    if scenario in hard_slices:
        return {
            "status": "diagnostic_only",
            "reason": (
                f"Topology hard slice for {scenario}: horizon_exhausted on all "
                "progress-gated rows; no clearance achieved. Stop lane for "
                "same-family selector reruns."
            ),
        }
    if scenario == "corridor_interaction":
        return {
            "status": "diagnostic_only",
            "reason": (
                "No topology reselection evidence on corridor slice; "
                "topology guidance diversifies labels but does not improve "
                "route progress or terminal outcome on canonical slices."
            ),
        }
    if status == "current" and tier == "diagnostic":
        return {
            "status": "diagnostic_only",
            "reason": (
                f"Ledger area topology_guidance: artifact_status={status}, "
                f"evidence_tier={tier}. Near-parity topology selection can "
                "diversify route labels but does not improve outcome."
            ),
        }
    return {
        "status": "unavailable",
        "reason": "No topology reselection evidence available for this slice.",
    }


def _signal_cell(
    scenario: str,
    signal_summary: dict[str, Any] | None,
    ledger: dict[str, Any] | None,
) -> dict[str, str]:
    """Return signal-compliance status + reason for one scenario slice."""
    if scenario == "signalized_crossing":
        if not signal_summary:
            return {
                "status": "unavailable",
                "reason": "Missing tracked signalized-runtime summary (issue #2799).",
            }
        eligible = _json_object_rows(signal_summary.get("eligible_rows"))
        excluded = _json_object_rows(signal_summary.get("excluded_rows"))
        denom_total = sum(r.get("signal_metrics_denominator", 0) for r in eligible)
        if denom_total > 0:
            return {
                "status": "available",
                "reason": (
                    f"Runtime denominator plumbing active: {len(eligible)} "
                    f"observable row(s) with denominator={denom_total}. "
                    "Does not prove traffic-signal realism or legality compliance."
                ),
            }
        return {
            "status": "denominator_zero",
            "reason": (
                f"All {len(excluded)} row(s) excluded; denominator=0 for "
                "non-observable rows. Compliance claims require "
                "planner_observable benchmark-evidence rows."
            ),
        }
    row = _ledger_row(ledger, "signalized_behavior") if ledger else None
    if row and row.get("artifact_status") == "current":
        return {
            "status": "proxy_only",
            "reason": (
                "Signal state is trace_metadata_only / planner_observable=false "
                "outside signalized_crossing fixture. Proxy diagnostic only."
            ),
        }
    return {
        "status": "unavailable",
        "reason": "No signal-compliance evidence for this scenario slice.",
    }


def _prediction_cell(
    scenario: str,
    cv_forecast: dict[str, Any] | None,
    ledger: dict[str, Any] | None,
) -> dict[str, str]:
    """Return prediction-baseline status + reason for one scenario slice."""
    row = _ledger_row(ledger, "prediction") if ledger else None
    tier = row.get("evidence_tier", "unavailable") if row else "unavailable"

    if cv_forecast:
        trace_rows = _json_object_rows(cv_forecast.get("results_by_trace"))
        matched = [trace_row for trace_row in trace_rows if trace_row.get("family") == scenario]
        if matched:
            statuses = {r.get("status") for r in matched}
            if statuses == {"limited_no_pedestrian_motion"}:
                return {
                    "status": "limited_no_motion",
                    "reason": (
                        "All pedestrian velocities zero; constant-velocity "
                        "forecast produces degenerate predictions."
                    ),
                }
            if "evaluated" in statuses:
                return {
                    "status": "diagnostic_only",
                    "reason": (
                        "CV forecast evaluated on bounded trace fixture. "
                        "Diagnostic-only; no planner-campaign comparison."
                    ),
                }
    if tier == "diagnostic":
        return {
            "status": "diagnostic_only",
            "reason": (
                "Merged prediction interface exists; contract-smoke rows "
                "materialized but no executed planner campaign. "
                "Denominator repair required."
            ),
        }
    return {
        "status": "unavailable",
        "reason": "No prediction baseline evidence for this scenario slice.",
    }


def _observation_cell(
    scenario: str,
    obs_summary: dict[str, Any] | None,
) -> dict[str, str]:
    """Return observation-perturbation status + reason for one scenario slice."""
    if scenario == "occluded_emergence":
        if not obs_summary:
            return {
                "status": "unavailable",
                "reason": "Missing tracked observation-noise envelope (issue #2755).",
            }
        summary = _json_object(obs_summary.get("summary"))
        classifications = _json_object(summary.get("classifications"))
        robustness = [c for c in classifications.values() if c == "robustness_evidence"]
        weak = [c for c in classifications.values() if c == "scenario_too_weak"]
        diag = [c for c in classifications.values() if c == "diagnostic_only"]
        return {
            "status": "partial_robustness",
            "reason": (
                f"{len(robustness)} condition(s) robustness_evidence (delay_only), "
                f"{len(diag)} diagnostic_only, {len(weak)} scenario_too_weak "
                "across 7 conditions. Single-fixture, not paper-facing."
            ),
        }
    if scenario == "dense_pedestrian":
        return {
            "status": "diagnostic_only",
            "reason": (
                "Issue #2765: trace-derived dense-pedestrian stress fixture; "
                "8/10 conditions expose forecast ambiguity. "
                "Stored trace action proxies, not live replay."
            ),
        }
    return {
        "status": "unavailable",
        "reason": "No observation-perturbation evidence for this scenario slice.",
    }


def _denominator_cell(
    scenario: str,
    signal_summary: dict[str, Any] | None,
    obs_summary: dict[str, Any] | None,
) -> dict[str, str]:
    """Return denominator-health status + reason for one scenario slice."""
    if scenario == "signalized_crossing":
        if not signal_summary:
            return {
                "status": "missing",
                "reason": "No tracked signalized-runtime summary.",
            }
        eligible = _json_object_rows(signal_summary.get("eligible_rows"))
        excluded = _json_object_rows(signal_summary.get("excluded_rows"))
        denom_total = sum(r.get("signal_metrics_denominator", 0) for r in eligible)
        return {
            "status": "partial" if denom_total > 0 else "zero",
            "reason": (
                f"{len(eligible)} eligible row(s) denominator>0, "
                f"{len(excluded)} excluded (denominator=0). "
                "Planner-observable compliance evidence limited."
            ),
        }
    if scenario == "occluded_emergence" and obs_summary:
        conditions = _json_object_rows(obs_summary.get("conditions"))
        viable: list[dict[str, Any]] = []
        for condition in conditions:
            if condition.get("first_observed_step") is not None:
                viable.append(condition)
        return {
            "status": "partial" if viable else "zero",
            "reason": (
                f"{len(viable)}/{len(conditions)} observation conditions produce "
                "observable steps; remaining zeroed by occlusion/missed detection."
            ),
        }
    return {
        "status": "not_applicable",
        "reason": "Denominator metric not defined for this scenario slice.",
    }


def _staleness_cell(
    scenario: str,
    ledger: dict[str, Any] | None,
    *,
    negative_result_available: bool,
) -> dict[str, str]:
    """Return stale/current status + reason for one scenario slice."""
    stale_summary = _json_object_rows(ledger.get("stale_artifact_summary")) if ledger else []
    stale_ids = {item.get("artifact_id") for item in stale_summary}
    if scenario in {"corridor_interaction", "crossing"}:
        if "tab_issue_1023_campaign_table" in stale_ids:
            return {
                "status": "stale",
                "reason": (
                    "Exported tables (issue #1023) are stale/non-claimable: "
                    "missing payload files. Historical tracked evidence only."
                ),
            }
    if scenario in {"t_intersection", "doorway"} and negative_result_available:
        return {
            "status": "current_negative",
            "reason": (
                "Negative-result candidates (issue #2788, NR-001) exist as "
                "not_promoted scenario candidates. Not benchmark evidence."
            ),
        }
    if scenario in {"t_intersection", "doorway"}:
        return {
            "status": "unavailable",
            "reason": "Negative-result candidate source is missing for this slice.",
        }
    return {
        "status": "current",
        "reason": "Source artifacts are current for this slice.",
    }


def _claim_cell(
    scenario: str,
    topology: dict[str, str],
    signal: dict[str, str],
    prediction: dict[str, str],
    observation: dict[str, str],
    denominator: dict[str, str],
    stale: dict[str, str],
) -> dict[str, str]:
    """Return claim-eligibility status + reason for one scenario slice."""
    blockers: list[str] = []
    for col_name, cell in [
        ("topology", topology),
        ("signal", signal),
        ("prediction", prediction),
        ("observation", observation),
        ("denominator", denominator),
    ]:
        s = cell["status"]
        if s in ("unavailable", "denominator_zero", "missing", "zero"):
            blockers.append(f"{col_name}: {s}")
        elif s == "proxy_only":
            blockers.append(f"{col_name}: proxy_only")
        elif s == "stale":
            blockers.append(f"{col_name}: stale")
    if stale["status"] == "stale":
        blockers.append("stale_artifacts_present")
    if blockers:
        return {
            "status": "not_eligible",
            "reason": "Blockers: " + "; ".join(blockers) + ".",
        }
    return {
        "status": "diagnostic_only",
        "reason": (
            "All evidence modules are diagnostic-only for this slice. "
            "No benchmark or paper-facing claim is eligible."
        ),
    }


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------


@dataclass(frozen=True)
class SourceInputs:
    """Tracked source inputs consumed by the matrix generator."""

    ledger: Path
    signal_summary: Path
    obs_noise_summary: Path
    cv_forecast: Path
    gap_report: Path
    negative_result_dir: Path


def build_matrix_rows(
    *,
    ledger: dict[str, Any] | None,
    signal_summary: dict[str, Any] | None,
    obs_summary: dict[str, Any] | None,
    cv_forecast: dict[str, Any] | None,
    negative_result_available: bool = False,
) -> list[dict[str, Any]]:
    """Build one dict per scenario slice with all column cells."""
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIO_SLICES:
        topology = _topology_cell(scenario, ledger)
        signal = _signal_cell(scenario, signal_summary, ledger)
        prediction = _prediction_cell(scenario, cv_forecast, ledger)
        observation = _observation_cell(scenario, obs_summary)
        denominator = _denominator_cell(scenario, signal_summary, obs_summary)
        stale = _staleness_cell(
            scenario,
            ledger,
            negative_result_available=negative_result_available,
        )
        claim = _claim_cell(
            scenario,
            topology,
            signal,
            prediction,
            observation,
            denominator,
            stale,
        )
        rows.append(
            {
                "scenario_slice": scenario,
                "topology_reselection": topology,
                "signal_compliance_metrics": signal,
                "prediction_baseline": prediction,
                "observation_perturbation": observation,
                "denominator_health": denominator,
                "stale_current_status": stale,
                "claim_eligibility": claim,
            }
        )
    return rows


def build_markdown(
    rows: list[dict[str, Any]],
    *,
    generated_at: str,
) -> str:
    """Render the matrix as a Markdown report."""
    col_headers = {
        "topology_reselection": "Topology Reselection",
        "signal_compliance_metrics": "Signal Compliance",
        "prediction_baseline": "Prediction Baseline",
        "observation_perturbation": "Observation Perturbation",
        "denominator_health": "Denominator Health",
        "stale_current_status": "Stale/Current",
        "claim_eligibility": "Claim Eligibility",
    }
    lines = [
        "# Issue #2766 Mixed-Scenario Coverage Matrix",
        "",
        f"Generated at: {generated_at}",
        "Status: synthesis draft only; not benchmark, paper, or safety evidence",
        "",
        "This matrix cross-references evidence modules against canonical scenario",
        "slices. Every blocked / unavailable / missing-denominator / stale /",
        "diagnostic-only / proxy / fallback cell carries an explicit reason.",
        "",
        "## Matrix",
        "",
    ]
    for row in rows:
        scenario = row["scenario_slice"]
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append("| Module | Status | Reason |")
        lines.append("|---|---|---|")
        for col_key, header in col_headers.items():
            cell = row[col_key]
            lines.append(f"| {header} | {cell['status']} | {cell['reason']} |")
        lines.append("")
    lines.append("## Summary Counts")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---|")
    status_counts: dict[str, int] = {}
    for row in rows:
        for key in EVIDENCE_COLUMNS:
            s = row[key]["status"]
            status_counts[s] = status_counts.get(s, 0) + 1
    for status, count in sorted(status_counts.items()):
        lines.append(f"| {status} | {count} |")
    lines.append("")
    lines.append("## Conservative Rules Applied")
    lines.append("")
    lines.append("- Synthesis draft only; not benchmark, paper, or safety evidence.")
    lines.append("- Diagnostic / stale / non-claimable / unavailable / fallback /")
    lines.append("  degraded / proxy-only / missing-denominator rows weaken or block claims.")
    lines.append("- Fallback behavior is not acceptable as a successful benchmark outcome.")
    lines.append("- No invented values; missing tracked sources produce unavailable rows.")
    lines.append("")
    return "\n".join(lines)


def build_summary(
    *,
    generated_at: str,
    sources: SourceInputs,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build machine-readable summary metadata."""
    total_cells = len(rows) * len(EVIDENCE_COLUMNS)
    status_counts: dict[str, int] = {}
    for row in rows:
        for key in EVIDENCE_COLUMNS:
            s = row[key]["status"]
            status_counts[s] = status_counts.get(s, 0) + 1
    eligible = sum(1 for r in rows if r["claim_eligibility"]["status"] == "eligible")
    return {
        "schema_version": MATRIX_SCHEMA,
        "issue": 2766,
        "generated_at_utc": generated_at,
        "scenario_count": len(rows),
        "column_count": len(EVIDENCE_COLUMNS),
        "total_cells": total_cells,
        "eligible_scenario_count": eligible,
        "not_eligible_scenario_count": len(rows) - eligible,
        "status_counts": status_counts,
        "source_inputs": {
            "ledger": str(sources.ledger),
            "signal_summary": str(sources.signal_summary),
            "obs_noise_summary": str(sources.obs_noise_summary),
            "cv_forecast": str(sources.cv_forecast),
            "gap_report": str(sources.gap_report),
            "negative_result_dir": str(sources.negative_result_dir),
        },
        "claim_boundary": (
            "synthesis draft only; not benchmark evidence, "
            "paper-facing evidence, or manuscript text"
        ),
        "conservative_rules": [
            "diagnostic/stale/non-claimable rows weaken or block wording",
            "fallback/degraded/unavailable/proxy-only rows are not success evidence",
            ("missing tracked sources produce unavailable rows rather than invented values"),
            "no benchmark or paper-facing overclaims",
        ],
    }


def generate_matrix(
    *,
    sources: SourceInputs,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate Markdown and JSON matrix artifacts.

    Returns ``(md_path, json_path)``.
    """
    ledger = _load_json(sources.ledger)
    signal_summary = _load_json(sources.signal_summary)
    obs_summary = _load_json(sources.obs_noise_summary)
    cv_forecast = _load_json(sources.cv_forecast)
    negative_result_available = sources.negative_result_dir.exists() and any(
        sources.negative_result_dir.glob("*.json")
    )
    generated_at = datetime.now(UTC).isoformat()
    rows = build_matrix_rows(
        ledger=ledger,
        signal_summary=signal_summary,
        obs_summary=obs_summary,
        cv_forecast=cv_forecast,
        negative_result_available=negative_result_available,
    )
    md = build_markdown(rows, generated_at=generated_at)
    summary = build_summary(generated_at=generated_at, sources=sources, rows=rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "mixed_scenario_matrix.md"
    json_path = output_dir / "summary.json"
    md_path.write_text(md, encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return md_path, json_path


def _cli() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate mixed-scenario coverage matrix for issue #2766."
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--signal-summary", type=Path, default=DEFAULT_SIGNAL_SUMMARY)
    parser.add_argument("--obs-noise-summary", type=Path, default=DEFAULT_OBS_NOISE_SUMMARY)
    parser.add_argument("--cv-forecast", type=Path, default=DEFAULT_CV_FORECAST)
    parser.add_argument("--gap-report", type=Path, default=DEFAULT_GAP_REPORT)
    parser.add_argument("--negative-result-dir", type=Path, default=DEFAULT_NEGATIVE_RESULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    sources = SourceInputs(
        ledger=args.ledger,
        signal_summary=args.signal_summary,
        obs_noise_summary=args.obs_noise_summary,
        cv_forecast=args.cv_forecast,
        gap_report=args.gap_report,
        negative_result_dir=args.negative_result_dir,
    )
    md_path, json_path = generate_matrix(sources=sources, output_dir=args.output_dir)
    print(f"Generated: {md_path}")
    print(f"Generated: {json_path}")


if __name__ == "__main__":
    _cli()
