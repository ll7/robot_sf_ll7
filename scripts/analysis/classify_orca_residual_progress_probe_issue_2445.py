#!/usr/bin/env python3
"""Classify the ORCA-residual progress-probe decision for issue #2445.

This script applies the issue #2408 / PR #2420 stop rule deterministically to the
revised v1 bounded smoke result (SLURM job 12913) and decides whether the
ORCA-residual BC lane should ``continue``, ``revise``, or ``stop``.

It is a *decision over existing evidence*. It does not launch training, submit
SLURM jobs, promote checkpoints, or claim learned-residual success.

Decision rule (from issue #2408 / PR #2420, cited in #2445):
    "If the v1 bounded smoke also has success_rate=0.0 with timeout_low_progress,
    stop the current residual-BC lane shape, or reopen only through a named
    objective/dataset redesign; do NOT rerun unchanged v0 or submit
    nominal_sanity."

Fail-closed posture (repository top value: honest, transparent, reproducible):
    If the input artifact is missing or invalid, or required smoke-evidence
    fields are absent, the decision is ``stop`` (the smoke cannot justify
    continuation or escalation) and the missing input/fields are named
    explicitly. A failed-closed smoke is never readiness evidence and must not
    justify nominal or larger reruns.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _rel(path: pathlib.Path) -> str:
    """Render a path repo-relative when possible, for reproducible artifacts.

    Absolute machine/worktree paths must not be baked into promoted evidence, so
    paths under the repo root are recorded relative to it; others are returned as-is.
    """
    resolved = pathlib.Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


DEFAULT_SMOKE_SUMMARY = (
    REPO_ROOT
    / "docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17/summary.json"
)

PRIOR_DECISION_ISSUE = 2408
REQUIRED_SMOKE_FIELDS = (
    "residual_clipping_rate",
    "guard_veto_rate",
    "fallback_degraded_status",
    "artifact_pointer_status",
)
REOPEN_CONDITION = (
    "Reopen only through a named objective/dataset redesign (a new BC objective, "
    "dataset, scenario, or instrumentation lane); do not rerun unchanged v0 BC and "
    "do not submit nominal_sanity from this failed-closed smoke."
)


def capture_git_head() -> str:
    """Return the current git HEAD commit, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_summary(path: pathlib.Path) -> dict[str, Any] | None:
    """Load the smoke summary JSON, returning None on missing/invalid input."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _missing_required_fields(summary: dict[str, Any]) -> list[str]:
    """Return the list of required smoke-evidence fields that are absent/null."""
    block = summary.get("required_smoke_evidence")
    if not isinstance(block, dict):
        # The whole block is missing: every required field is missing.
        return list(REQUIRED_SMOKE_FIELDS)
    missing: list[str] = []
    for field in REQUIRED_SMOKE_FIELDS:
        if block.get(field) is None:
            missing.append(field)
    return missing


def classify_decision(
    summary: dict[str, Any] | None,
    summary_path: pathlib.Path,
) -> dict[str, Any]:
    """Apply the #2408 stop rule and return the decision block.

    Fail-closed: a missing/invalid artifact or absent required fields yields a
    ``stop`` decision naming the missing input/fields. A reproduced v0 failure
    pattern (success_rate 0.0 + timeout_low_progress) yields ``stop``. A clean
    passing smoke (success_rate > 0, no timeout_low_progress, all required
    fields present) yields ``continue``.
    """
    git_head = capture_git_head()

    # --- Fail closed on missing/invalid artifact ---------------------------------
    if summary is None:
        return {
            "schema_version": "orca_residual_progress_probe_decision.v1",
            "issue": 2445,
            "evidence_grade": "analysis_only",
            "git_head": git_head,
            "input_artifact": _rel(summary_path),
            "orca_residual_progress_probe_decision": {
                "prior_decision_issue": PRIOR_DECISION_ISSUE,
                "v1_smoke_success_rate": None,
                "v1_smoke_failure_mode": None,
                "artifact_pointer_status": "missing_or_invalid",
                "missing_required_fields": list(REQUIRED_SMOKE_FIELDS),
                "decision": "stop",
                "reopen_condition": REOPEN_CONDITION,
                "rationale": (
                    f"Fail-closed: the required v1 smoke summary artifact at "
                    f"'{summary_path}' is missing or invalid, so no smoke evidence "
                    "exists to justify continuation or escalation. Per the #2408 "
                    "stop rule, the lane stops until a valid smoke (or a named "
                    "objective/dataset redesign) is supplied."
                ),
            },
        }

    metrics = summary.get("metrics") or {}
    success_rate = metrics.get("success_rate")
    failure_mode_counts = metrics.get("failure_mode_counts") or {}
    timeout_low_progress = int(failure_mode_counts.get("timeout_low_progress", 0) or 0)
    failure_mode = (
        "timeout_low_progress"
        if timeout_low_progress > 0
        else (next(iter(failure_mode_counts), None) if failure_mode_counts else None)
    )

    missing_fields = _missing_required_fields(summary)
    smoke_evidence = summary.get("required_smoke_evidence") or {}
    raw_pointer = smoke_evidence.get("artifact_pointer_status")
    artifact_pointer_status = raw_pointer if raw_pointer is not None else "missing"

    status = summary.get("status")
    nominal_escalation_allowed = summary.get("nominal_escalation_allowed")

    # --- Fail closed on missing required smoke-evidence fields -------------------
    if missing_fields:
        return {
            "schema_version": "orca_residual_progress_probe_decision.v1",
            "issue": 2445,
            "evidence_grade": "analysis_only",
            "git_head": git_head,
            "input_artifact": _rel(summary_path),
            "orca_residual_progress_probe_decision": {
                "prior_decision_issue": PRIOR_DECISION_ISSUE,
                "v1_smoke_success_rate": success_rate,
                "v1_smoke_failure_mode": failure_mode,
                "artifact_pointer_status": artifact_pointer_status,
                "missing_required_fields": missing_fields,
                "decision": "stop",
                "reopen_condition": REOPEN_CONDITION,
                "rationale": (
                    "Fail-closed: the v1 smoke is missing required smoke-evidence "
                    f"fields {missing_fields} (summary status="
                    f"'{status}', nominal_escalation_allowed="
                    f"{nominal_escalation_allowed}). Missing required fields "
                    "independently forbid nominal escalation; combined with "
                    f"success_rate={success_rate} and failure_mode='{failure_mode}', "
                    "the #2408 stop rule applies. Reopen only via a named "
                    "objective/dataset redesign."
                ),
            },
        }

    # --- Reproduced v0 failure pattern => stop ----------------------------------
    reproduced_v0_failure = (success_rate == 0.0) and (timeout_low_progress > 0)
    if reproduced_v0_failure:
        decision = "stop"
        rationale = (
            f"The v1 bounded smoke reproduces the v0 failure pattern: "
            f"success_rate={success_rate} with failure_mode='timeout_low_progress' "
            f"(summary status='{status}'). Per the #2408 / PR #2420 stop rule, this "
            "stops the current residual-BC lane shape. Reopen only via a named "
            "objective/dataset redesign; do not rerun unchanged v0 or submit "
            "nominal_sanity."
        )
    else:
        # All required fields present AND not the v0 failure pattern => continue.
        decision = "continue"
        rationale = (
            f"The v1 bounded smoke did not reproduce the v0 failure pattern "
            f"(success_rate={success_rate}, failure_mode='{failure_mode}') and all "
            "required smoke-evidence fields are present. The current lane shape may "
            "continue to the next predeclared gate; this remains smoke evidence only "
            "and does not by itself justify nominal/paper-grade claims."
        )

    return {
        "schema_version": "orca_residual_progress_probe_decision.v1",
        "issue": 2445,
        "evidence_grade": "analysis_only",
        "git_head": git_head,
        "input_artifact": _rel(summary_path),
        "orca_residual_progress_probe_decision": {
            "prior_decision_issue": PRIOR_DECISION_ISSUE,
            "v1_smoke_success_rate": success_rate,
            "v1_smoke_failure_mode": failure_mode,
            "artifact_pointer_status": artifact_pointer_status,
            "missing_required_fields": missing_fields,
            "decision": decision,
            "reopen_condition": REOPEN_CONDITION,
            "rationale": rationale,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render the decision block to Markdown."""
    block = report["orca_residual_progress_probe_decision"]
    lines = [
        "# Issue #2445 ORCA-Residual Progress-Probe Decision",
        "",
        f"- **Decision**: `{block['decision']}`",
        f"- **Evidence Grade**: `{report['evidence_grade']}`",
        f"- **Prior decision issue**: #{block['prior_decision_issue']}",
        f"- **Input artifact**: `{report['input_artifact']}`",
        f"- **Git HEAD**: `{report['git_head']}`",
        "",
        "## V1 Smoke Facts (job 12913)",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| v1_smoke_success_rate | `{block['v1_smoke_success_rate']}` |",
        f"| v1_smoke_failure_mode | `{block['v1_smoke_failure_mode']}` |",
        f"| artifact_pointer_status | `{block['artifact_pointer_status']}` |",
        f"| missing_required_fields | `{block['missing_required_fields']}` |",
        "",
        "## Rationale",
        "",
        block["rationale"],
        "",
        "## Reopen Condition",
        "",
        block["reopen_condition"],
        "",
        "## Claim Boundary",
        "",
        "This is an analysis-only routing decision over existing failed-closed smoke "
        "evidence. It is not a benchmark result, not a learned-component success "
        "claim, and must not justify nominal or larger SLURM reruns.",
    ]
    return "\n".join(lines)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Classify the ORCA-residual progress-probe decision (issue #2445) by "
            "applying the #2408 stop rule to the v1 bounded smoke summary. "
            "Fails closed on a missing/invalid artifact or absent required fields."
        )
    )
    parser.add_argument(
        "--smoke-summary",
        type=pathlib.Path,
        default=DEFAULT_SMOKE_SUMMARY,
        help="Path to the #1475 v1 smoke summary JSON (default: the 12913 bundle).",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Optional directory to write decision.json and decision.md.",
    )
    args = parser.parse_args()

    summary = load_summary(args.smoke_summary)
    report = classify_decision(summary, args.smoke_summary)

    print(json.dumps(report, indent=2))

    if args.output_dir is not None:
        output_dir: pathlib.Path = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "decision.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        md_path = output_dir / "decision.md"
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(render_markdown(report))
        print(f"Wrote decision JSON: {json_path}", file=sys.stderr)
        print(f"Wrote decision Markdown: {md_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
