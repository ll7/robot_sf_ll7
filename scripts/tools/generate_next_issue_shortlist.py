"""Generate deterministic next-issue shortlist from repo-local evidence sources.

Read-only planning artifact. Consumes the negative-result register, dissertation
evidence ledger, dissertation gap report, and algorithm-readiness metadata. Emits
JSON and Markdown shortlist with reasons, blockers, data-source status, and
degradation notes. No GitHub mutation, no Project #5 writes, no issue creation.

Route-efficiency data is optional; if missing a degradation note is recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON file; return None if missing (degradation, not error)."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


SCHEMA_VERSION = "next_issue_shortlist.v1"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# -- Source paths -----------------------------------------------------------

LEDGER_PATH = (
    REPO_ROOT / "docs/context/evidence/issue_2760_dissertation_evidence_ledger/ledger.json"
)
REGISTER_PATH = (
    REPO_ROOT / "docs/context/evidence/issue_2762_negative_result_register/register.json"
)
GAP_REPORT_PATH = (
    REPO_ROOT / "docs/context/evidence/issue_2784_dissertation_gap_report/gap_report.json"
)
ALGORITHM_READINESS_PATH = REPO_ROOT / "robot_sf/benchmark/algorithm_readiness.py"
ROUTE_EFFICIENCY_PATH = REPO_ROOT / "output"  # directory probe for route reports

# -- Ranking weights --------------------------------------------------------

_BUCKET_SCORES: dict[str, int] = {
    "negative_revise_only": 80,
    "blocked": 60,
    "remove_weaken": 40,
    "supported": 10,
}
_TIER_SCORES: dict[str, int] = {
    "diagnostic": 15,
    "non-claimable": 10,
    "release-backed": 5,
}

CLAIM_BOUNDARIES = [
    "This shortlist is a synthesis/planning aid. It does not produce new benchmark "
    "evidence, paper-facing results, or safety claims.",
    "Ranking is deterministic: same inputs always produce the same order.",
    "Missing sources are recorded as degradation notes; the shortlist still emits "
    "candidates from available sources.",
    "No GitHub mutation, no Project #5 writes, and no issue creation occurred.",
]


# -- Source loaders ---------------------------------------------------------


def _load_optional_json(path: Path | None) -> dict[str, Any] | list[Any] | None:
    """Load optional JSON snapshot; return None if no path or missing."""
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot_items(snapshot: dict[str, Any] | list[Any] | None, key: str) -> list[dict[str, Any]]:
    """Return a list of dict items from common snapshot shapes."""
    if snapshot is None:
        return []
    if isinstance(snapshot, list):
        return [item for item in snapshot if isinstance(item, dict)]
    values = snapshot.get(key, [])
    if isinstance(values, list):
        return [item for item in values if isinstance(item, dict)]
    return []


def _label_names(issue: dict[str, Any]) -> list[str]:
    """Extract label names from gh JSON and local snapshot shapes."""
    labels = issue.get("labels") or []
    names: list[str] = []
    for label in labels:
        if isinstance(label, str):
            names.append(label)
        elif isinstance(label, dict) and label.get("name"):
            names.append(str(label["name"]))
    return names


def _probe_route_efficiency() -> dict[str, Any]:
    """Probe for route-efficiency data in output/.

    Returns a dict with available flag and optional manifest paths.
    """
    if not ROUTE_EFFICIENCY_PATH.is_dir():
        return {"available": False, "reason": "output/ directory not found"}
    manifests = sorted(ROUTE_EFFICIENCY_PATH.glob("**/routing_manifest.json"))
    if not manifests:
        return {"available": False, "reason": "no routing_manifest.json found in output/"}
    return {
        "available": True,
        "manifest_count": len(manifests),
        "manifests": [str(m.relative_to(REPO_ROOT)) for m in manifests[:5]],
    }


def _load_algorithm_readiness() -> list[dict[str, Any]]:
    """Load canonical algorithm readiness tuples from the in-repo registry."""
    try:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from robot_sf.benchmark.algorithm_readiness import _ALGORITHMS
    except Exception:
        return []
    return [
        {
            "canonical_name": algo.canonical_name,
            "tier": algo.tier,
            "note": algo.note,
        }
        for algo in _ALGORITHMS
    ]


def _summarize_recent_prs(snapshot: dict[str, Any] | list[Any] | None) -> dict[str, Any]:
    """Summarize recent PR snapshot metadata for source-status reporting."""
    prs = _snapshot_items(snapshot, "prs")
    return {
        "available": snapshot is not None,
        "count": len(prs),
        "latest": [
            {"number": pr.get("number"), "title": pr.get("title")}
            for pr in prs[:5]
            if pr.get("number") is not None
        ],
    }


# -- Candidate builders -----------------------------------------------------


def _candidates_from_register(register: dict[str, Any]) -> list[dict[str, Any]]:
    """Build candidate issues from negative-result register entries."""
    candidates: list[dict[str, Any]] = []
    for entry in register.get("entries", []):
        next_action = entry.get("recommended_next_action")
        if not next_action:
            continue
        source_issue = entry["linked_issues"][0] if entry.get("linked_issues") else None
        candidates.append(
            {
                "id": entry["id"],
                "title": next_action,
                "source": "negative_result_register",
                "source_issue": source_issue,
                "bucket": "negative_revise_only",
                "evidence_tier": None,
                "result_classification": entry.get("result_classification"),
                "reason": entry.get("why_failed_or_inconclusive", ""),
                "blockers": [entry.get("claim_boundary", "")],
                "data_source_status": "available",
                "degradation_notes": [],
            }
        )
    return candidates


def _candidates_from_gap_report(gap_report: dict[str, Any]) -> list[dict[str, Any]]:
    """Build candidate issues from dissertation gap report."""
    candidates: list[dict[str, Any]] = []
    for gap in gap_report.get("gaps", []):
        promotion = gap.get("promotion_step_or_reason")
        if not promotion:
            continue
        label = gap.get("area") or gap.get("entry_id", "unknown")
        candidates.append(
            {
                "id": f"gap-{label}",
                "title": promotion,
                "source": "dissertation_gap_report",
                "source_issue": gap.get("source_issue"),
                "bucket": gap.get("bucket", "blocked"),
                "evidence_tier": gap.get("evidence_tier"),
                "result_classification": gap.get("result_classification"),
                "reason": gap.get("claim_gap_or_reason", ""),
                "blockers": [gap.get("caveat", "")],
                "data_source_status": "available",
                "degradation_notes": [],
            }
        )
    return candidates


def _candidates_from_ledger(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """Build candidate issues from dissertation evidence ledger rows."""
    candidates: list[dict[str, Any]] = []
    for row in ledger.get("rows", []):
        gap = row.get("claim_gap")
        promotion = row.get("evidence_promotion_path")
        if not gap and not promotion:
            continue
        area = row.get("area", "unknown")
        candidates.append(
            {
                "id": f"ledger-{area}",
                "title": promotion or f"Address gap in {area}",
                "source": "dissertation_evidence_ledger",
                "source_issue": row.get("source_issues", [None])[0]
                if row.get("source_issues")
                else None,
                "bucket": "blocked"
                if row.get("evidence_promotion_path")
                else "negative_revise_only",
                "evidence_tier": row.get("evidence_tier"),
                "result_classification": None,
                "reason": gap or "",
                "blockers": [row.get("caveat", "")],
                "data_source_status": "available",
                "degradation_notes": [],
            }
        )
    return candidates


def _candidates_from_algorithm_readiness(
    algorithms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build candidate issues from experimental algorithms needing validation."""
    candidates: list[dict[str, Any]] = []
    for algo in algorithms:
        if algo.get("tier") != "experimental":
            continue
        name = algo["canonical_name"]
        candidates.append(
            {
                "id": f"algo-{name}",
                "title": f"Validate experimental algorithm: {name}",
                "source": "algorithm_readiness",
                "source_issue": None,
                "bucket": "blocked",
                "evidence_tier": None,
                "result_classification": None,
                "reason": algo.get("note", ""),
                "blockers": ["Requires benchmark validation before baseline promotion"],
                "data_source_status": "available",
                "degradation_notes": [],
            }
        )
    return candidates


def _candidates_from_open_issues(
    snapshot: dict[str, Any] | list[Any] | None,
) -> list[dict[str, Any]]:
    """Build candidate rows from an optional open-issue snapshot."""
    candidates: list[dict[str, Any]] = []
    for issue in _snapshot_items(snapshot, "issues"):
        if issue.get("state", "OPEN") not in {"OPEN", "open"}:
            continue
        number = issue.get("number")
        if number is None:
            continue
        labels = _label_names(issue)
        bucket = "blocked"
        evidence_label = next((label for label in labels if label.startswith("evidence:")), None)
        evidence_tier = evidence_label.split(":", 1)[1] if evidence_label else None
        candidates.append(
            {
                "id": f"open-issue-{number}",
                "title": issue.get("title", f"Advance issue #{number}"),
                "source": "open_issues_snapshot",
                "source_issue": number,
                "bucket": bucket,
                "evidence_tier": evidence_tier,
                "result_classification": None,
                "reason": issue.get("body_excerpt", "Open issue snapshot candidate."),
                "blockers": [],
                "data_source_status": "available",
                "degradation_notes": [],
            }
        )
    return candidates


# -- Ranking ----------------------------------------------------------------


def _rank_score(candidate: dict[str, Any]) -> int:
    """Compute deterministic ranking score (higher = more actionable)."""
    bucket = candidate.get("bucket", "blocked")
    tier = candidate.get("evidence_tier")
    score = _BUCKET_SCORES.get(bucket, 30)
    score += _TIER_SCORES.get(tier or "", 0)
    return score


def _deduplicate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by source issue when present, otherwise by source-local id."""
    seen: set[tuple[str, Any]] = set()
    deduped: list[dict[str, Any]] = []
    for c in candidates:
        source_issue = c.get("source_issue")
        key = (c["source"], source_issue if source_issue is not None else c["id"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


# -- Generation pipeline ----------------------------------------------------


def _safe_relative(path: Path, base: Path) -> str:
    """Return relative path string, or absolute path string if not under base."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def generate_shortlist(
    *,
    open_issues_snapshot_path: Path | None = None,
    recent_prs_snapshot_path: Path | None = None,
) -> tuple[dict[str, Any], str]:
    """Load sources, build candidates, rank, and return (json_report, markdown)."""
    sources_status: dict[str, Any] = {}
    degradation_notes: list[str] = []

    # Load sources (None = missing, recorded as degradation)
    register = _load_json(REGISTER_PATH)
    ledger = _load_json(LEDGER_PATH)
    gap_report = _load_json(GAP_REPORT_PATH)
    route_eff = _probe_route_efficiency()
    algorithms = _load_algorithm_readiness()
    open_issues_snapshot = _load_optional_json(open_issues_snapshot_path)
    recent_prs_snapshot = _load_optional_json(recent_prs_snapshot_path)

    sources_status["negative_result_register"] = {
        "path": _safe_relative(REGISTER_PATH, REPO_ROOT),
        "available": register is not None,
    }
    sources_status["dissertation_evidence_ledger"] = {
        "path": _safe_relative(LEDGER_PATH, REPO_ROOT),
        "available": ledger is not None,
    }
    sources_status["dissertation_gap_report"] = {
        "path": _safe_relative(GAP_REPORT_PATH, REPO_ROOT),
        "available": gap_report is not None,
    }
    sources_status["route_efficiency"] = {
        "available": route_eff["available"],
        "reason": route_eff.get("reason"),
    }
    sources_status["algorithm_readiness"] = {
        "path": _safe_relative(ALGORITHM_READINESS_PATH, REPO_ROOT),
        "available": len(algorithms) > 0,
    }
    sources_status["open_issues_snapshot"] = {
        "path": _safe_relative(open_issues_snapshot_path, REPO_ROOT)
        if open_issues_snapshot_path
        else None,
        "available": open_issues_snapshot is not None,
        "count": len(_snapshot_items(open_issues_snapshot, "issues")),
    }
    recent_pr_summary = _summarize_recent_prs(recent_prs_snapshot)
    sources_status["recent_prs_snapshot"] = {
        "path": _safe_relative(recent_prs_snapshot_path, REPO_ROOT)
        if recent_prs_snapshot_path
        else None,
        **recent_pr_summary,
    }

    if not route_eff["available"]:
        degradation_notes.append(
            f"route_efficiency: {route_eff.get('reason', 'data not available')} "
            "-- signal skipped, no route-efficiency ranking component applied"
        )
    if open_issues_snapshot is None:
        degradation_notes.append(
            "open_issues_snapshot: no local JSON snapshot supplied -- open-issue ranking "
            "signal skipped"
        )
    if recent_prs_snapshot is None:
        degradation_notes.append(
            "recent_prs_snapshot: no local JSON snapshot supplied -- recent-PR ranking "
            "context recorded as missing"
        )

    # Build candidates from each available source
    all_candidates: list[dict[str, Any]] = []

    if register is not None:
        all_candidates.extend(_candidates_from_register(register))
    else:
        degradation_notes.append(
            f"negative_result_register: source not found at "
            f"{_safe_relative(REGISTER_PATH, REPO_ROOT)}"
        )

    if gap_report is not None:
        all_candidates.extend(_candidates_from_gap_report(gap_report))
    else:
        degradation_notes.append(
            f"dissertation_gap_report: source not found at "
            f"{_safe_relative(GAP_REPORT_PATH, REPO_ROOT)}"
        )

    if ledger is not None:
        all_candidates.extend(_candidates_from_ledger(ledger))
    else:
        degradation_notes.append(
            f"dissertation_evidence_ledger: source not found at "
            f"{_safe_relative(LEDGER_PATH, REPO_ROOT)}"
        )

    if algorithms:
        all_candidates.extend(_candidates_from_algorithm_readiness(algorithms))

    if open_issues_snapshot is not None:
        all_candidates.extend(_candidates_from_open_issues(open_issues_snapshot))

    # Deduplicate and rank
    candidates = _deduplicate(all_candidates)
    candidates.sort(key=lambda c: (-_rank_score(c), c["id"]))

    # Add rank numbers
    for i, c in enumerate(candidates, 1):
        c["rank"] = i
        c["ranking_score"] = _rank_score(c)

    json_report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": (
            "synthesis/planning aid; not new benchmark, paper, dissertation, or safety evidence"
        ),
        "sources_status": sources_status,
        "degradation_notes": degradation_notes,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "claim_boundaries": CLAIM_BOUNDARIES,
    }

    md_text = _render_markdown(json_report)
    return json_report, md_text


def _render_source_status_lines(report: dict[str, Any]) -> list[str]:
    """Render data-source status section."""
    lines: list[str] = ["## Data Sources", ""]
    for name, status in report["sources_status"].items():
        avail = "available" if status["available"] else "missing"
        extra = f" ({status.get('reason')})" if status.get("reason") else ""
        extra += f" [{status.get('path', '')}]" if status.get("path") else ""
        lines.append(f"- **{name}**: {avail}{extra}")
    lines.append("")
    return lines


def _render_candidate_lines(c: dict[str, Any]) -> list[str]:
    """Render a single candidate entry."""
    lines: list[str] = []
    lines.append(f"### #{c['rank']} {c['id']} (score: {c['ranking_score']})")
    lines.append("")
    lines.append(f"- **Title**: {c['title']}")
    lines.append(f"- **Source**: {c['source']}")
    if c.get("source_issue"):
        lines.append(f"- **Source issue**: #{c['source_issue']}")
    lines.append(f"- **Bucket**: {c['bucket']}")
    if c.get("evidence_tier"):
        lines.append(f"- **Evidence tier**: {c['evidence_tier']}")
    if c.get("result_classification"):
        lines.append(f"- **Result classification**: {c['result_classification']}")
    lines.append(f"- **Reason**: {c['reason']}")
    raw_blockers = c.get("blockers")
    blockers = [
        str(blocker) for blocker in (raw_blockers if raw_blockers is not None else []) if blocker
    ]
    if blockers:
        lines.append(f"- **Blockers**: {'; '.join(blockers)}")
    lines.append(f"- **Data-source status**: {c['data_source_status']}")
    if c.get("degradation_notes"):
        for dn in c["degradation_notes"]:
            lines.append(f"  - {dn}")
    lines.append("")
    return lines


def _render_markdown(report: dict[str, Any]) -> str:
    """Render shortlist as Markdown."""
    lines: list[str] = []
    lines.append("# Next-Issue Shortlist")
    lines.append("")
    lines.append(
        "**Purpose**: synthesis/planning aid; not new benchmark, paper, "
        "dissertation, or safety evidence."
    )
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}  | Schema: {report['schema_version']}")
    lines.append("")

    lines.extend(_render_source_status_lines(report))

    # Degradation notes
    if report["degradation_notes"]:
        lines.append("## Degradation Notes")
        lines.append("")
        for note in report["degradation_notes"]:
            lines.append(f"- {note}")
        lines.append("")

    # Ranked candidates
    lines.append(f"## Candidates ({report['candidate_count']} total)")
    lines.append("")
    if not report["candidates"]:
        lines.append("No candidates generated from available sources.")
        lines.append("")
    else:
        for c in report["candidates"]:
            lines.extend(_render_candidate_lines(c))

    # Claim boundaries
    lines.append("## Claim Boundaries")
    lines.append("")
    for b in report["claim_boundaries"]:
        lines.append(f"- {b}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic next-issue shortlist from repo-local evidence."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write JSON and Markdown outputs.",
    )
    parser.add_argument(
        "--open-issues-snapshot",
        type=Path,
        default=None,
        help="Optional JSON snapshot of open issues, for deterministic ranking input.",
    )
    parser.add_argument(
        "--recent-prs-snapshot",
        type=Path,
        default=None,
        help="Optional JSON snapshot of recent PRs, for deterministic source-status input.",
    )
    args = parser.parse_args()

    json_report, md_text = generate_shortlist(
        open_issues_snapshot_path=args.open_issues_snapshot,
        recent_prs_snapshot_path=args.recent_prs_snapshot,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = out_dir / "summary.json"
    json_out.write_text(
        json.dumps(json_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    md_out = out_dir / "README.md"
    md_out.write_text(md_text, encoding="utf-8")

    print(f"JSON shortlist written to {json_out}")
    print(f"Markdown shortlist written to {md_out}")
    print(f"Candidates: {json_report['candidate_count']}")
    if json_report["degradation_notes"]:
        print(f"Degradation notes: {len(json_report['degradation_notes'])}")


if __name__ == "__main__":
    main()
