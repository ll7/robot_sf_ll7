#!/usr/bin/env python3
"""Build issue #1358 parent acceptance-criteria audit evidence."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.orca_residual_lane_readiness import assess_lane_readiness
from scripts.validation.build_issue_1475_acceptance_audit import build_audit as build_1475_audit

SCHEMA_VERSION = "issue-1358-acceptance-audit.v1"
DEFAULT_CLOSURE_AUDIT = Path("docs/context/issue_1358_closure_audit_2026-07-07.md")
DEFAULT_STATE_SURFACE = Path("docs/context/issue_1358_state.yaml")
DEFAULT_OUTPUT = Path("docs/context/evidence/issue_1358_acceptance_audit_2026-07-07.json")


@dataclass(frozen=True)
class CriterionAudit:
    """One issue #1358 acceptance criterion mapped to repository evidence."""

    criterion: str
    status: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        """Return JSON-friendly representation."""
        return {
            "criterion": self.criterion,
            "status": self.status,
            "evidence": self.evidence,
        }


def _repo_root_from(path: Path) -> Path:
    return path.resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"failed to read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"failed to parse YAML {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"expected YAML mapping in {path}")
    return data


def _state_surface_check(
    *,
    repo_root: Path,
    state_surface_path: Path,
    acceptance_evidence: list[CriterionAudit],
    closure_call: str,
) -> dict[str, Any]:
    """Check append-only high-churn state row matches the generated audit."""

    state_surface = _load_yaml(repo_root / state_surface_path)
    entries = state_surface.get("entries")
    if not isinstance(entries, list) or not entries:
        return {
            "path": str(state_surface_path),
            "status": "invalid",
            "errors": ["state surface has no entries"],
        }

    latest = entries[-1]
    errors: list[str] = []
    if state_surface.get("issue") != 1358:
        errors.append(f"issue must be 1358, got {state_surface.get('issue')!r}")

    closure_boundary = latest.get("closure_boundary", {})
    if closure_boundary.get("closure_call_for_this_pr") != closure_call:
        errors.append(
            "latest entry closure_call_for_this_pr "
            f"{closure_boundary.get('closure_call_for_this_pr')!r} != {closure_call!r}"
        )

    state_evidence = latest.get("acceptance_evidence")
    if state_evidence is None:
        state_evidence = closure_boundary.get("acceptance_evidence", [])
    state_by_criterion = {
        item.get("criterion"): item.get("status")
        for item in state_evidence
        if isinstance(item, dict)
    }
    for item in acceptance_evidence:
        if state_by_criterion.get(item.criterion) != item.status:
            errors.append(
                f"{item.criterion!r} status "
                f"{state_by_criterion.get(item.criterion)!r} != {item.status!r}"
            )

    return {
        "path": str(state_surface_path),
        "status": "valid" if not errors else "invalid",
        "latest_recorded_at_utc": latest.get("recorded_at_utc"),
        "entry_status": latest.get("status"),
        "errors": errors,
    }


def _stable_issue_1475_audit_summary(issue_1475_audit: dict[str, Any]) -> dict[str, Any]:
    """Return #1475 facts needed by #1358 without volatile child state-row metadata."""

    state_surface = issue_1475_audit["state_surface"]
    return {
        "status": issue_1475_audit["status"],
        "closure_call": issue_1475_audit["closure_call"],
        "remaining_criteria": issue_1475_audit["remaining_criteria"],
        "state_surface": {
            "path": state_surface["path"],
            "status": state_surface["status"],
            "errors": state_surface["errors"],
            "integration_report_status": state_surface.get("integration_report_status"),
        },
    }


def build_audit(
    *,
    repo_root: Path,
    closure_audit_path: Path = DEFAULT_CLOSURE_AUDIT,
    state_surface_path: Path = DEFAULT_STATE_SURFACE,
) -> dict[str, Any]:
    """Build a fail-closed issue #1358 acceptance audit."""

    readiness = assess_lane_readiness(repo_root, validate_packet=True)
    issue_1475_audit = build_1475_audit(repo_root=repo_root)
    integration = readiness["integration_report"]

    local_handoff_ready = (
        readiness["overall_status"] == "blocked_on_followup"
        and integration["integration_status"] == "local_handoff_ready_parent_blocked"
        and not readiness["errors"]
    )
    child_1475_blocked = (
        issue_1475_audit["issue"] == 1475
        and issue_1475_audit["status"] == "blocked"
        and issue_1475_audit["closure_call"] == "keep_open"
    )
    state_1475_valid = issue_1475_audit["state_surface"]["status"] == "valid"

    criteria = [
        CriterionAudit(
            criterion="Candidate design records exact observation additions and residual action bounds.",
            status="met" if local_handoff_ready else "not_met",
            evidence=(
                "Readiness report status="
                f"{readiness['overall_status']!r}, prerequisites "
                f"{integration['local_contract']['prerequisites_ready']}/"
                f"{integration['local_contract']['prerequisites_total']} ready; "
                "merged PRs #1409/#1875/#3770 provide the local handoff surface."
            ),
        ),
        CriterionAudit(
            criterion="Training config versioned and runnable by one canonical handoff command.",
            status="met" if local_handoff_ready else "not_met",
            evidence=(
                "Readiness report route set includes lineage validation, smoke candidate, "
                "and SLURM handoff command shapes; no commands are executed by this audit."
            ),
        ),
        CriterionAudit(
            criterion="Trained checkpoint durable lineage explicit artifact pointer.",
            status="not_met",
            evidence=(
                "Issue #1475 audit status="
                f"{issue_1475_audit['status']!r}; remaining criteria include durable "
                "dataset/checkpoint/report artifacts from the next Slurm smoke rerun."
            ),
        ),
        CriterionAudit(
            criterion="Smoke nominal-sanity policy-search stages run without fallback/degraded success.",
            status="not_met",
            evidence=(
                "Issue #1475 audit keeps fallback/degraded success fail-closed, but tracked smoke "
                f"gate status remains {issue_1475_audit['smoke_gate']['status']!r}; "
                "no nominal result exists."
            ),
        ),
        CriterionAudit(
            criterion="Report compares ORCA, current PPO leader, failed guarded-PPO variants, and new residual policy.",
            status="not_met",
            evidence=(
                "No trained residual checkpoint and no scenario-stratified nominal report exist; "
                "comparison remains blocked on child #1475 durable evidence."
            ),
        ),
        CriterionAudit(
            criterion="Result classified promote, revise, or reject using scenario-stratified evidence.",
            status="not_met",
            evidence=(
                "Issue #2445 closed an earlier progress-probe decision, but parent #1358 thread "
                "still requires #1475 durable evidence before continue/revise/stop classification."
            ),
        ),
        CriterionAudit(
            criterion="Parent stays open until Issue #1475 classifies lane continue/revise/stop durable evidence.",
            status="met" if child_1475_blocked and state_1475_valid else "not_met",
            evidence=(
                "Issue #1475 executable audit closure_call="
                f"{issue_1475_audit['closure_call']!r}; state surface status="
                f"{issue_1475_audit['state_surface']['status']!r}."
            ),
        ),
        CriterionAudit(
            criterion="No new residual-policy training children added before Issue #1475 reports durable evidence or fail-closed blocker.",
            status="met",
            evidence=(
                "This audit is CPU-only evidence generation; it does not add children, submit "
                "Slurm/GPU work, run training, or mutate planner behavior."
            ),
        ),
    ]

    status = "complete" if all(item.status == "met" for item in criteria) else "blocked"
    closure_call = "close" if status == "complete" else "keep_open"
    state_surface = _state_surface_check(
        repo_root=repo_root,
        state_surface_path=state_surface_path,
        acceptance_evidence=criteria,
        closure_call=closure_call,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "issue": 1358,
        "status": status,
        "closure_call": closure_call,
        "claim_boundary": (
            "CPU-only parent acceptance audit over tracked merged evidence; no Slurm/GPU "
            "submission, benchmark campaign, training run, release, or paper-facing claim."
        ),
        "checked_paths": [
            str(closure_audit_path),
            "robot_sf/benchmark/orca_residual_lane_readiness.py",
            "docs/context/evidence/issue_1475_acceptance_audit_2026-07-06.json",
            "docs/context/issue_1475_state.yaml",
            str(state_surface_path),
        ],
        "readiness": {
            "overall_status": readiness["overall_status"],
            "integration_status": integration["integration_status"],
            "remaining_blocker_keys": integration["remaining_blocker_keys"],
            "errors": readiness["errors"],
        },
        "issue_1475_audit": {
            **_stable_issue_1475_audit_summary(issue_1475_audit),
        },
        "acceptance_evidence": [item.to_dict() for item in criteria],
        "remaining_criteria": [item.to_dict() for item in criteria if item.status != "met"],
        "state_surface": state_surface,
        "next_empirical_action": (
            "On a Slurm-capable lane, run one bounded ORCA-residual BC smoke rerun against "
            "current post-gate code. Only if the smoke gate passes, escalate a bounded nominal "
            "run and record the #1358 continue/revise/stop classification."
        ),
        "forbidden_actions_confirmed": {
            "full_benchmark_campaign_run": False,
            "slurm_or_gpu_submission": False,
            "paper_or_dissertation_claim_edit": False,
            "issue_comment": False,
            "release": False,
            "merge": False,
            "deletion": False,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root resolving relative paths (default: current directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON artifact path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the audit JSON artifact instead of printing to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run CLI."""

    args = build_arg_parser().parse_args(argv)
    repo_root = _repo_root_from(args.repo_root)
    report = build_audit(repo_root=repo_root)

    if report["state_surface"]["status"] != "valid":
        print(
            f"issue #1358 state surface mismatch: {report['state_surface']['errors']}",
            file=sys.stderr,
        )
        return 2

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.write:
        output = args.output if args.output.is_absolute() else repo_root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
