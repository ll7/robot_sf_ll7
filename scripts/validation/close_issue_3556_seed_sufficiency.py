#!/usr/bin/env python3
"""Resolve retained issue #3556 ScenarioBelief campaign reports and close seed sufficiency.

Plain-language summary: this is the seed-sufficiency *closure* step for issue
#3556. Earlier slices built the analyzer handoff command (PR #4261) and recorded
a single-path blocker when the ephemeral runner output root was missing
(PR #4273). This resolver goes further: it searches an ordered set of **durable**
roots (``docs/context/evidence`` first, then the runner output root) for a
retained ScenarioBelief campaign root that exposes the analyzer-required report
files. When one is found it runs ``analyze_seed_sufficiency.py`` and promotes a
compact closure summary; when none is found it fails closed with an explicit,
reproducible missing-artifact blocker that names every searched root and every
missing report file.

It never launches a campaign, never submits Slurm/GPU work, never changes
ScenarioBelief runtime behavior, and never edits paper/dissertation claims.

Example (fail-closed closure packet when retained reports are absent):

    uv run python scripts/validation/close_issue_3556_seed_sufficiency.py \\
        --evidence-date 2026-07-03

Example (point at a restored retained campaign root):

    uv run python scripts/validation/close_issue_3556_seed_sufficiency.py \\
        --search-root docs/context/evidence \\
        --search-root output/issue_3556_belief_mode_campaign
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
    build_seed_sufficiency_closure_packet,
)
from scripts.tools.analyze_seed_sufficiency import (  # noqa: E402
    analyze_seed_sufficiency,
    resolve_campaign_roots,
)

# Durable roots are searched before the ephemeral runner output root so a promoted
# retained campaign takes precedence over transient local scratch output.
DEFAULT_SEARCH_ROOTS = (
    "docs/context/evidence",
    "output/issue_3556_belief_mode_campaign",
)
DEFAULT_CAMPAIGN_ID = "issue_3556"
DEFAULT_ANALYZER_OUTPUT_DIR = "output/issue_3556_seed_sufficiency"
BLOCKED_EXIT_CODE = 3


def _rel(path: Path) -> str:
    """Return a repository-root-relative path string when possible."""
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _missing_report_files(campaign_root: Path) -> list[str]:
    """Return required report files absent under ``campaign_root/reports``."""
    reports = campaign_root / "reports"
    return [name for name in REQUIRED_SEED_SUFFICIENCY_REPORTS if not (reports / name).is_file()]


def _probe_search_root(search_root: Path, campaign_id: str) -> dict[str, Any]:
    """Probe one durable root for retained #3556 campaign roots.

    Reuses the analyzer's own discovery (``resolve_campaign_roots``) so the
    resolver and the analyzer agree on what counts as a campaign root. Discovery
    keys off ``seed_variability_by_scenario.json``; this probe additionally checks
    that ``seed_episode_rows.csv`` is present, because the analyzer needs both.
    """
    try:
        found = resolve_campaign_roots(
            campaign_output_roots=[search_root],
            campaign_ids=[campaign_id],
        )
    except FileNotFoundError:
        found = []
    campaign_roots_found = [_rel(root) for root in found]
    usable_campaign_roots: list[str] = []
    missing_report_files: dict[str, list[str]] = {}
    for root in found:
        missing = _missing_report_files(root)
        if missing:
            missing_report_files[_rel(root)] = missing
        else:
            usable_campaign_roots.append(_rel(root))
    return {
        "search_root": _rel(search_root),
        "exists": search_root.exists(),
        "campaign_roots_found": campaign_roots_found,
        "usable_campaign_roots": usable_campaign_roots,
        "missing_report_files": missing_report_files,
    }


def _analyzer_command(
    campaign_root: str | None, search_roots: list[Path], output_dir: Path, campaign_id: str
) -> list[str]:
    """Build the exact analyzer argv (resolved or would-run-when-restored)."""
    base = ["uv", "run", "python", "scripts/tools/analyze_seed_sufficiency.py"]
    if campaign_root is not None:
        base += ["--campaign-root", campaign_root]
    else:
        for root in search_roots:
            base += ["--campaign-output-root", _rel(root)]
        base += ["--campaign-id", campaign_id]
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
    """Render a reviewable Markdown closure note from the packet."""
    resolved = packet["evidence_status"] == "promoted"
    lines = [
        "# Issue #3556 Seed-Sufficiency Closure Packet",
        "",
        "Plain-language summary: this packet resolves whether a retained ScenarioBelief "
        "campaign root with the analyzer-required seed-sufficiency reports exists under the "
        "searched durable roots, then either runs the analyzer or fails closed with an explicit "
        "missing-artifact blocker.",
        "",
        "## Claim Boundary",
        "",
        f"- Evidence status: `{packet['evidence_status']}`.",
        f"- Decision label: `{packet['decision_label']}`.",
        f"- {packet['claim_boundary']}",
        "",
        "## Required Retained Report Files",
        "",
    ]
    lines += [f"- `reports/{name}`" for name in packet["required_report_files"]]
    lines += [
        "",
        "## Searched Durable Roots",
        "",
        "| search root | exists | campaign roots found | usable (all reports) | missing reports |",
        "| --- | --- | --- | --- | --- |",
    ]
    for probe in packet["searched_roots"]:
        found = ", ".join(probe["campaign_roots_found"]) or "—"
        usable = ", ".join(probe["usable_campaign_roots"]) or "—"
        missing = (
            "; ".join(
                f"{root}: {', '.join(files)}"
                for root, files in probe["missing_report_files"].items()
            )
            or "—"
        )
        lines.append(
            f"| `{probe['search_root']}` | {probe['exists']} | {found} | {usable} | {missing} |"
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
            f"- Resolved campaign root: `{packet['resolved_campaign_root']}`",
            f"- Analyzer output dir: `{packet['analyzer_output_dir']}`",
            f"- Headline claim status: `{summary.get('headline_claim_status')}`",
            f"- Headline label: `{summary.get('headline_label')}`",
            f"- Promotion allowed: `{summary.get('promotion_allowed')}`",
            f"- Seed counts: `{summary.get('seed_counts')}`",
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
        "Supersedes the single-path handoff probe in "
        "`docs/context/evidence/issue_3556_seed_sufficiency_handoff_2026-07-03/` by searching "
        "durable roots and recording a reproducible per-root manifest.",
    ]
    return "\n".join(lines) + "\n"


def run_closure(
    *,
    search_roots: list[Path],
    campaign_id: str,
    analyzer_output_dir: Path,
    evidence_dir: Path,
) -> dict[str, Any]:
    """Probe durable roots, run the analyzer when resolved, and write the packet."""
    probes = [_probe_search_root(root, campaign_id) for root in search_roots]

    resolved_campaign_root: str | None = None
    for probe in probes:
        if probe["usable_campaign_roots"]:
            resolved_campaign_root = probe["usable_campaign_roots"][0]
            break

    analyzer_summary: dict[str, Any] | None = None
    output_dir_str: str | None = None
    if resolved_campaign_root is not None:
        payload = analyze_seed_sufficiency(
            [_REPO_ROOT / resolved_campaign_root],
            analyzer_output_dir,
        )
        analyzer_summary = _compact_analyzer_summary(payload)
        output_dir_str = _rel(analyzer_output_dir)

    command = _analyzer_command(
        resolved_campaign_root, search_roots, analyzer_output_dir, campaign_id
    )
    packet = build_seed_sufficiency_closure_packet(
        searched_roots=probes,
        resolved_campaign_root=resolved_campaign_root,
        analyzer_command=command,
        analyzer_output_dir=output_dir_str,
        analyzer_summary=analyzer_summary,
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
        "--search-root",
        action="append",
        default=None,
        help="Durable root to search for retained #3556 campaign reports (repeatable). "
        f"Default: {', '.join(DEFAULT_SEARCH_ROOTS)}",
    )
    parser.add_argument(
        "--campaign-id",
        default=DEFAULT_CAMPAIGN_ID,
        help="Substring filter for retained campaign root names.",
    )
    parser.add_argument(
        "--analyzer-output-dir",
        default=DEFAULT_ANALYZER_OUTPUT_DIR,
        help="Where analyzer artifacts are written when a retained root is resolved.",
    )
    parser.add_argument(
        "--evidence-dir",
        default=None,
        help="Closure packet output dir. Default: "
        "docs/context/evidence/issue_3556_seed_sufficiency_closure_<evidence-date>/",
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
    """Resolve CLI paths, run the closure, and return the resolved/blocked exit code."""
    args = _parse_args(argv)
    search_root_strs = args.search_root if args.search_root else list(DEFAULT_SEARCH_ROOTS)
    search_roots = [
        (_REPO_ROOT / s if not Path(s).is_absolute() else Path(s)) for s in search_root_strs
    ]
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
            _REPO_ROOT / "docs/context/evidence" / f"issue_3556_seed_sufficiency_closure_{stamp}"
        )

    packet = run_closure(
        search_roots=search_roots,
        campaign_id=args.campaign_id,
        analyzer_output_dir=analyzer_output_dir,
        evidence_dir=evidence_dir,
    )

    print(f"evidence_status={packet['evidence_status']} decision={packet['decision_label']}")
    print(f"resolved_campaign_root={packet['resolved_campaign_root']}")
    print(f"closure_packet={_rel(evidence_dir)}")
    if packet["evidence_status"] == "promoted" or args.exit_zero_on_blocked:
        return 0
    return BLOCKED_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
