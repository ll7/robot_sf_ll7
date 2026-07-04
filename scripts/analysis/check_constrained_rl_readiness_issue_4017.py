#!/usr/bin/env python3
"""Fail-closed readiness gate for issue #4017 constrained-RL evidence.

This checker consolidates the existing CPU-smoke diagnostic comparison into a
single machine-readable handoff. It does not train policies, run benchmark
campaigns, submit Slurm jobs, or promote a paper-facing safety claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

SCHEMA_VERSION = "issue_4017.constrained_rl_readiness.v1"
EXPECTED_REPORT_SCHEMA = "issue_4017.constrained_rl_diagnostic.v1"
READY_STATUS = "diagnostic_ready_for_empirical_campaign"
BLOCKED_STATUS = "diagnostic_blocked"
BLOCKED_EXIT_CODE = 3


def assess_readiness(report_path: str | Path) -> dict[str, object]:
    """Return fail-closed readiness assessment for a diagnostic comparison report."""
    path = Path(report_path).resolve()
    report = _read_json_object(path)
    blockers = _collect_blockers(report)
    status = BLOCKED_STATUS if blockers else READY_STATUS
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 4017,
        "source_report": str(path),
        "status": status,
        "claim_boundary": (
            "diagnostic constrained-RL smoke readiness only; not benchmark-strength, "
            "paper-grade, or dissertation safety evidence"
        ),
        "ready_for_benchmark_claim": False,
        "ready_for_empirical_campaign": not blockers,
        "blockers_remaining": blockers,
        "intentional_limits": [
            "no full benchmark campaign run",
            "no Slurm/GPU submission",
            "no metric semantics redefinition",
            "no safety-wrapper or uncertainty-buffer claim",
            "no paper/dissertation claim edit",
        ],
        "next_empirical_action": _next_empirical_action(blockers),
        "contract": {
            "requires_report_schema": EXPECTED_REPORT_SCHEMA,
            "requires_report_status": "diagnostic_ready",
            "requires_diagnostic_only_tier": True,
            "requires_non_degraded_evidence": True,
            "requires_matched_seed_and_timesteps": True,
            "requires_budget_violation_and_multiplier_update": True,
            "requires_runtime_recorded": True,
        },
    }


def _collect_blockers(report: Mapping[str, object]) -> list[str]:
    """Collect readiness blockers without treating diagnostic evidence as a claim."""
    blockers: list[str] = []
    blockers.extend(_report_contract_blockers(report))
    blockers.extend(_run_summary_blockers("baseline", _mapping(report.get("baseline"))))
    blockers.extend(_run_summary_blockers("constrained", _mapping(report.get("constrained"))))
    blockers.extend(_constraint_effect_blockers(_mapping(report.get("constraint_effect"))))

    trace = _mapping(report.get("constraint_trace"))
    if int(trace.get("record_count") or 0) <= 0:
        blockers.append("constraint trace has no completed episode records")
    return _dedupe(blockers)


def _report_contract_blockers(report: Mapping[str, object]) -> list[str]:
    """Return blockers from top-level report contract fields."""
    blockers: list[str] = []
    if report.get("schema_version") != EXPECTED_REPORT_SCHEMA:
        blockers.append("comparison report schema is missing or unsupported")
    if report.get("issue") != 4017:
        blockers.append("comparison report does not target issue 4017")
    if report.get("evidence_tier") != "diagnostic-only":
        blockers.append("comparison report must stay diagnostic-only")
    if report.get("status") != "diagnostic_ready":
        blockers.append("comparison report status is not diagnostic_ready")
    blockers.extend(_string_list(report.get("blockers")))
    if bool(report.get("fallback_or_degraded", True)):
        blockers.append("comparison report is fallback/degraded or omits non-degraded proof")
    return blockers


def _run_summary_blockers(role: str, summary: Mapping[str, object]) -> list[str]:
    """Return blockers for one training manifest summary."""
    blockers: list[str] = []
    if bool(summary.get("dry_run", True)):
        blockers.append(f"{role} manifest is dry-run only")
    if summary.get("runtime_status") != "recorded":
        blockers.append(f"{role} manifest does not record runtime_seconds")
    if bool(summary.get("fallback_or_degraded", True)):
        blockers.append(f"{role} manifest is fallback/degraded")
    if _string_list(summary.get("missing_fields")):
        blockers.append(f"{role} manifest is missing required fields")
    return blockers


def _constraint_effect_blockers(effect: Mapping[str, object]) -> list[str]:
    """Return blockers from the constrained-vs-unconstrained effect summary."""
    blockers: list[str] = []
    if effect.get("interpretation") != "diagnostic_only":
        blockers.append("constraint effect interpretation must remain diagnostic_only")
    if effect.get("matched_seed") is not True:
        blockers.append("baseline and constrained manifests do not use the same seed")
    if effect.get("matched_total_timesteps") is not True:
        blockers.append("baseline and constrained manifests do not use matched timesteps")
    if bool(effect.get("benchmark_safety_claim", True)):
        blockers.append("comparison report attempts a benchmark safety claim")
    if not _string_list(effect.get("budget_violation_constraints")):
        blockers.append("no positive budget violation was observed")
    if not _string_list(effect.get("multiplier_changed_constraints")):
        blockers.append("no Lagrange multiplier update was observed")
    return blockers


def _next_empirical_action(blockers: Sequence[str]) -> str:
    if blockers:
        return (
            "Regenerate the matched CPU-smoke manifests and comparison report until this "
            "readiness gate reports diagnostic_ready_for_empirical_campaign."
        )
    return (
        "Run the paired constrained/unconstrained empirical campaign on the configured "
        "diagnostic scenario set, then archive the generated manifests, trace, comparison "
        "report, and this readiness report before making any stronger safety claim."
    )


def _read_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"comparison report does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"comparison report must be a JSON object: {path}")
    return data


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [str(item) for item in value]


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/benchmarks/issue4017/comparison_report.json"),
        help="Issue #4017 diagnostic comparison report JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the readiness report JSON.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Exit 0 even when the report remains blocked.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the readiness CLI."""
    args = _build_parser().parse_args(argv)
    readiness = assess_readiness(args.report)
    text = json.dumps(readiness, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if readiness["status"] == BLOCKED_STATUS and not args.allow_blocked:
        return BLOCKED_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
