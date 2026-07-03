#!/usr/bin/env python3
"""Evaluate named retained h600 roots against the #3556 seed-sufficiency contract.

Plain-language summary: issue #4328 proposes three retained h600 campaign report
roots as inputs to the issue #3556 ScenarioBelief seed-sufficiency *closure*
resolver (PR #4310). This runner evaluates each named candidate root against the
resolver's input contract — existence on the current analysis host, the two
analyzer-required report files, and a #3556 ScenarioBelief provenance/lineage
check — and either runs the analyzer on the best fully compatible candidate or
fails closed with an explicit, per-candidate blocker.

Why a provenance check: the #3556 closure resolves seed-sufficiency for the
ScenarioBelief drop-vs-retain contrast specifically. The named candidates are
foreign h600 campaign lineages (long-horizon confirm, extended roster, hybrid
roster). Even when their report files are structurally loadable by the
planner-agnostic analyzer, their seed-sufficiency verdict answers a different
question, so promoting it as #3556 closure evidence would be a provenance
overclaim. Compatibility is therefore gated on lineage as well as report files.

It never launches a campaign, never submits Slurm/GPU work, never changes
ScenarioBelief runtime behavior, and never edits paper/dissertation claims.

Example (fail-closed candidate closure packet on a host without the roots):

    uv run python \\
        scripts/validation/evaluate_issue_4328_h600_seed_sufficiency_candidates.py \\
        --evidence-date 2026-07-03
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Ensure repository-root imports resolve when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from robot_sf.benchmark.scenario_belief_screening import (  # noqa: E402
    REQUIRED_SEED_SUFFICIENCY_REPORTS,
    build_h600_candidate_closure_packet,
    evaluate_seed_sufficiency_candidate,
)
from scripts.tools.analyze_seed_sufficiency import analyze_seed_sufficiency  # noqa: E402

BLOCKED_EXIT_CODE = 3
DEFAULT_ANALYZER_OUTPUT_DIR = "output/issue_4328_h600_seed_sufficiency"

# The named retained h600 report roots proposed in issue #4328. Each entry pins the
# repository-relative campaign root (the parent of ``reports/``) and its declared
# campaign lineage. These are foreign h600 campaigns, not #3556 ScenarioBelief
# drop-vs-retain campaigns; the provenance check below records that explicitly.
# Host-routing state (which machine currently retains these roots) is intentionally
# NOT encoded here: this evaluation reports only observable, repo-relative facts.
DEFAULT_CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "name": "issue3810-h600-longhorizon-confirm-run",
        "root": "output/issue3810-h600-longhorizon-confirm-run/13268",
        "lineage": "issue #3810 h600 long-horizon confirmation run",
    },
    {
        "name": "issue3810-h600-extroster-run",
        "root": "output/issue3810-h600-extroster-run/13273",
        "lineage": "issue #3810 h600 extended-roster run",
    },
    {
        "name": "issue4230-h600-hybrid-roster-run",
        "root": "output/issue4230-h600-hybrid-roster-run/13282",
        "lineage": "issue #4230 h600 hybrid-roster run",
    },
)

# Recorded when no candidate qualifies: the #3556-specific campaign that would
# actually satisfy the contract. This is a queue-row request (provenance for the
# next empirical step), not an execution instruction.
QUEUE_ROW_REQUEST: dict[str, Any] = {
    "kind": "scenario_belief_seed_sufficiency_campaign",
    "issue": 3556,
    "why": (
        "A #3556-specific ScenarioBelief drop-vs-retain campaign is required to close "
        "seed-sufficiency; foreign h600 roster campaigns answer a different question."
    ),
    "runner": "scripts/benchmark/run_belief_mode_safety_campaign_issue_3556.py",
    "required_reports": [f"reports/{name}" for name in REQUIRED_SEED_SUFFICIENCY_REPORTS],
    "min_seed_budget_for_headline": 20,
    "execution": "deferred (no run performed by this evaluation)",
}


def _rel(path: Path) -> str:
    """Return a repository-root-relative path string when possible."""
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_root(root: str) -> Path:
    """Resolve a repo-relative or absolute candidate root string to a Path."""
    candidate = Path(root)
    return candidate if candidate.is_absolute() else _REPO_ROOT / candidate


def _missing_report_files(campaign_root: Path) -> list[str]:
    """Return required report files absent under ``campaign_root/reports``.

    When the root itself is absent it cannot be probed, so every required report
    is reported missing; this keeps the blocker reproducible on any host.
    """
    if not campaign_root.exists():
        return list(REQUIRED_SEED_SUFFICIENCY_REPORTS)
    reports = campaign_root / "reports"
    return [name for name in REQUIRED_SEED_SUFFICIENCY_REPORTS if not (reports / name).is_file()]


def _analyzer_command(resolved_root: str | None, output_dir: Path) -> list[str]:
    """Build the exact analyzer argv (resolved root, or best candidate when blocked)."""
    base = ["uv", "run", "python", "scripts/tools/analyze_seed_sufficiency.py"]
    if resolved_root is not None:
        base += ["--campaign-root", resolved_root]
    base += ["--output-dir", _rel(output_dir)]
    return base


def _compact_analyzer_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a small, durable summary from the analyzer payload."""
    headline = payload.get("headline_rank_stability_contract", {})
    return {
        "campaign_count": payload.get("summary", {}).get("campaign_count"),
        "seed_counts": headline.get("seed_counts"),
        "max_seed_count": headline.get("max_seed_count"),
        "headline_label": headline.get("label"),
        "headline_claim_status": headline.get("claim_status"),
        "promotion_allowed": headline.get("promotion_allowed"),
    }


def _render_readme(packet: dict[str, Any]) -> str:
    """Render a reviewable Markdown closure note from the candidate packet."""
    resolved = packet["evidence_status"] == "promoted"
    lines = [
        "# Issue #4328 h600 Candidate Seed-Sufficiency Closure Packet",
        "",
        "Plain-language summary: this packet evaluates the named retained h600 report roots "
        "proposed in issue #4328 against the issue #3556 ScenarioBelief seed-sufficiency closure "
        "contract (existence on host + the two analyzer-required reports + ScenarioBelief "
        "provenance), then either runs the analyzer on the best fully compatible candidate or "
        "fails closed with an explicit per-candidate blocker.",
        "",
        "## Claim Boundary",
        "",
        f"- Closure target issue: `#{packet['closure_target_issue']}` "
        f"(attempt filed under `#{packet['closure_attempt_issue']}`).",
        f"- Evidence status: `{packet['evidence_status']}`.",
        f"- Decision label: `{packet['decision_label']}`.",
        f"- {packet['claim_boundary']}",
        "",
        "## Required Report Files (per candidate)",
        "",
    ]
    lines += [f"- `reports/{name}`" for name in packet["required_report_files"]]
    lines += [
        "",
        "## Candidate Compatibility",
        "",
        "| candidate | root | exists on host | reports present | provenance compatible | "
        "blockers |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for candidate in packet["candidates"]:
        blockers = ", ".join(candidate["blockers"]) or "—"
        lines.append(
            f"| `{candidate['name']}` | `{candidate['root']}` | {candidate['exists_on_host']} | "
            f"{candidate['required_reports_present']} | "
            f"{candidate['provenance']['provenance_compatible']} | {blockers} |"
        )
    lines += [
        "",
        "## Analyzer Command",
        "",
        "```bash",
        " ".join(packet["analyzer_command"]),
        "```",
        "",
    ]
    if resolved:
        summary = packet.get("analyzer_summary") or {}
        lines += [
            "## Analyzer Result (compact)",
            "",
            f"- Resolved candidate: `{packet['resolved_candidate']['name']}`",
            f"- Analyzer output dir: `{packet['analyzer_output_dir']}`",
            f"- Headline claim status: `{summary.get('headline_claim_status')}`",
            f"- Headline label: `{summary.get('headline_label')}`",
            f"- Promotion allowed: `{summary.get('promotion_allowed')}`",
            f"- Seed counts: `{summary.get('seed_counts')}`",
            "",
        ]
    else:
        request = packet.get("queue_row_request") or {}
        lines += [
            "## Queue-Row Request (deferred; no execution)",
            "",
            f"- Kind: `{request.get('kind')}`",
            f"- Runner: `{request.get('runner')}`",
            f"- Why: {request.get('why')}",
            f"- Required reports: {', '.join(request.get('required_reports', []))}",
            "",
        ]
    lines += [
        "## Decision",
        "",
        f"`{packet['decision_label']}`: {packet['next_empirical_action']}",
        "",
        "## Out of Scope (confirmed)",
        "",
        "- No full benchmark campaign run.",
        "- No Slurm/GPU submission.",
        "- No ScenarioBelief belief-mode semantic change.",
        "- No paper/dissertation claim edit.",
        "",
        "Complements the durable-root closure packet in "
        "`docs/context/evidence/issue_3556_seed_sufficiency_closure_2026-07-03/` by evaluating the "
        "specific named h600 candidate roots (which fall outside the resolver's default search "
        "roots) and adding an explicit ScenarioBelief provenance gate.",
    ]
    return "\n".join(lines) + "\n"


def run_candidate_closure(
    *,
    candidates: list[dict[str, str]],
    analyzer_output_dir: Path,
    evidence_dir: Path,
    queue_row_request: dict[str, Any] | None = QUEUE_ROW_REQUEST,
) -> dict[str, Any]:
    """Probe named candidate roots, run the analyzer when one qualifies, write the packet."""
    records: list[dict[str, Any]] = []
    for spec in candidates:
        root_path = _resolve_root(spec["root"])
        record = evaluate_seed_sufficiency_candidate(
            name=spec["name"],
            root=_rel(root_path),
            exists_on_host=root_path.exists(),
            missing_report_files=_missing_report_files(root_path),
            lineage=spec.get("lineage", ""),
        )
        records.append(record)

    resolved_candidate = next((record for record in records if record["compatible"]), None)

    analyzer_summary: dict[str, Any] | None = None
    output_dir_str: str | None = None
    resolved_root_str: str | None = None
    if resolved_candidate is not None:
        resolved_root_str = resolved_candidate["root"]
        payload = analyze_seed_sufficiency(
            [_resolve_root(resolved_root_str)],
            analyzer_output_dir,
        )
        analyzer_summary = _compact_analyzer_summary(payload)
        output_dir_str = _rel(analyzer_output_dir)

    command = _analyzer_command(resolved_root_str, analyzer_output_dir)
    packet = build_h600_candidate_closure_packet(
        candidates=records,
        analyzer_command=command,
        resolved_candidate=resolved_candidate,
        analyzer_output_dir=output_dir_str,
        analyzer_summary=analyzer_summary,
        queue_row_request=None if resolved_candidate is not None else queue_row_request,
    )

    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "summary.json").write_text(
        json.dumps(packet, indent=2) + "\n", encoding="utf-8"
    )
    (evidence_dir / "README.md").write_text(_render_readme(packet), encoding="utf-8")
    return packet


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analyzer-output-dir",
        default=DEFAULT_ANALYZER_OUTPUT_DIR,
        help="Where analyzer artifacts are written when a compatible candidate resolves.",
    )
    parser.add_argument(
        "--evidence-dir",
        default=None,
        help="Closure packet output dir. Default: "
        "docs/context/evidence/issue_4328_h600_seed_sufficiency_candidates_<evidence-date>/",
    )
    parser.add_argument(
        "--evidence-date",
        default=None,
        help="YYYY-MM-DD stamp for the default evidence dir (default: today, UTC).",
    )
    parser.add_argument(
        "--exit-zero-on-blocked",
        action="store_true",
        help="Exit 0 even when the closure fails closed (packet is always written).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Resolve CLI paths, evaluate the named candidates, and return the exit code."""
    args = _parse_args(argv)
    analyzer_output_dir = (
        _REPO_ROOT / args.analyzer_output_dir
        if not Path(args.analyzer_output_dir).is_absolute()
        else Path(args.analyzer_output_dir)
    )
    if args.evidence_dir is not None:
        evidence_dir = Path(args.evidence_dir)
        if not evidence_dir.is_absolute():
            evidence_dir = _REPO_ROOT / evidence_dir
    else:
        stamp = args.evidence_date or datetime.now(UTC).strftime("%Y-%m-%d")
        evidence_dir = (
            _REPO_ROOT
            / "docs/context/evidence"
            / f"issue_4328_h600_seed_sufficiency_candidates_{stamp}"
        )

    packet = run_candidate_closure(
        candidates=[dict(candidate) for candidate in DEFAULT_CANDIDATES],
        analyzer_output_dir=analyzer_output_dir,
        evidence_dir=evidence_dir,
    )

    print(f"evidence_status={packet['evidence_status']} decision={packet['decision_label']}")
    print(f"compatible_candidates={packet['compatible_candidates']}")
    print(f"closure_packet={_rel(evidence_dir)}")
    if packet["evidence_status"] == "promoted" or args.exit_zero_on_blocked:
        return 0
    return BLOCKED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
