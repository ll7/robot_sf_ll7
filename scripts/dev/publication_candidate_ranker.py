#!/usr/bin/env python3
"""Read-only ranker for publication-scout candidate issues.

Consumes publication scout/linter candidate output and pre-computed dimension
scores, then emits a deterministic Markdown report (or JSON) with ranked
candidates.  Each ranked item includes reasons, evidence caveats, and
duplication/blocker notes.

The script is intentionally read-only and deterministic; all inputs are loaded
from fixture or preloaded JSON payloads so tests can cover edge cases without
contacting GitHub or any remote service.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "publication_candidate_ranker.v2"

DIMENSION_WEIGHTS: dict[str, float] = {
    "dissertation_relevance": 0.20,
    "evidence_readiness": 0.25,
    "implementation_boundedness": 0.15,
    "artifact_readiness": 0.10,
    "expected_research_value": 0.20,
    "duplication_safety": 0.10,
}

LEDGER_BOOST = 0.08
NEGATIVE_RESULT_PENALTY = 0.10


class RankerInputError(ValueError):
    """Raised when an optional ranking input has the wrong schema or shape."""


@dataclass(frozen=True)
class CandidateInput:
    """One candidate issue with linter output and ranking signals."""

    issue_number: int | None
    issue_url: str | None
    title: str
    linter_ok: bool
    linter_findings: tuple[dict[str, str], ...]
    scores: dict[str, float]
    reasons: tuple[str, ...]
    evidence_caveats: tuple[str, ...]
    duplication_notes: tuple[str, ...]
    blocked_by: tuple[int, ...]


@dataclass(frozen=True)
class RankedCandidate:
    """A ranked candidate with computed aggregate score and metadata."""

    rank: int
    issue_number: int | None
    issue_url: str | None
    title: str
    total_score: float
    scores: dict[str, float]
    reasons: tuple[str, ...]
    evidence_caveats: tuple[str, ...]
    duplication_notes: tuple[str, ...]
    blocked_by: tuple[int, ...]
    linter_ok: bool
    linter_findings: tuple[dict[str, str], ...]
    ledger_relevance: float
    negative_result_awareness: float
    ledger_notes: tuple[str, ...]
    negative_result_caveats: tuple[str, ...]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value into [lo, hi]."""
    return max(lo, min(hi, value))


def _parse_candidate(raw: dict[str, Any]) -> CandidateInput:
    """Parse one candidate object from the input JSON."""
    scores_raw = raw.get("scores", {})
    if not isinstance(scores_raw, dict):
        scores_raw = {}

    scores: dict[str, float] = {}
    all_keys = set(DIMENSION_WEIGHTS) | {"artifact_dependency", "duplication_risk"}
    for key in all_keys:
        raw_value = scores_raw.get(key)
        if isinstance(raw_value, (int, float)):
            scores[key] = _clamp(float(raw_value))
        else:
            scores[key] = 0.0

    findings_raw = raw.get("linter_findings", [])
    if not isinstance(findings_raw, list):
        findings_raw = []
    findings = tuple(
        {"code": str(f.get("code", "")), "detail": str(f.get("detail", ""))}
        for f in findings_raw
        if isinstance(f, dict)
    )

    reasons_raw = raw.get("reasons", [])
    reasons = tuple(str(r) for r in reasons_raw if isinstance(r, str))

    caveats_raw = raw.get("evidence_caveats", [])
    evidence_caveats = tuple(str(c) for c in caveats_raw if isinstance(c, str))

    dup_raw = raw.get("duplication_notes", [])
    duplication_notes = tuple(str(d) for d in dup_raw if isinstance(d, str))

    blocked_raw = raw.get("blocked_by", [])
    blocked_by = tuple(int(b) for b in blocked_raw if isinstance(b, (int, float)))

    issue_number = raw.get("issue_number")
    if isinstance(issue_number, (int, float)):
        issue_number = int(issue_number)
    else:
        issue_number = None

    issue_url = raw.get("issue_url")
    if not isinstance(issue_url, str):
        issue_url = None

    title = raw.get("title", "")
    if not isinstance(title, str):
        title = str(title)

    linter_ok = raw.get("linter_ok", False)
    if not isinstance(linter_ok, bool):
        linter_ok = bool(linter_ok)

    return CandidateInput(
        issue_number=issue_number,
        issue_url=issue_url,
        title=title,
        linter_ok=linter_ok,
        linter_findings=findings,
        scores=scores,
        reasons=reasons,
        evidence_caveats=evidence_caveats,
        duplication_notes=duplication_notes,
        blocked_by=blocked_by,
    )


def _parse_ledger(raw: Any) -> dict[int, list[dict[str, Any]]]:
    """Parse a dissertation evidence ledger into {issue_number: rows} lookup.

    Expected schema: dissertation_evidence_ledger.v2 with top-level ``rows``.
    """
    lookup: dict[int, list[dict[str, Any]]] = {}
    if not isinstance(raw, dict):
        raise RankerInputError("ledger-json payload must be an object")
    if raw.get("schema_version") != "dissertation_evidence_ledger.v2":
        raise RankerInputError("ledger-json schema_version must be dissertation_evidence_ledger.v2")
    rows = raw.get("rows", [])
    if not isinstance(rows, list):
        raise RankerInputError("ledger-json rows must be a list")
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_issues = row.get("source_issues", [])
        if not isinstance(source_issues, list):
            continue
        for issue_num in source_issues:
            issue = _safe_int(issue_num)
            if issue is not None:
                lookup.setdefault(issue, []).append(row)
    return lookup


def _parse_register(raw: Any) -> dict[int, list[dict[str, Any]]]:
    """Parse a negative result register into {issue_number: entries} lookup.

    Expected schema: negative_result_register.v1 with a top-level
    ``entries`` list where each entry has ``linked_issues`` (list of ints)
    and ``result_classification`` such as revise or diagnostic_only.
    """
    lookup: dict[int, list[dict[str, Any]]] = {}
    if not isinstance(raw, dict):
        raise RankerInputError("register-json payload must be an object")
    if raw.get("schema_version") != "negative_result_register.v1":
        raise RankerInputError("register-json schema_version must be negative_result_register.v1")
    entries = raw.get("entries", [])
    if not isinstance(entries, list):
        raise RankerInputError("register-json entries must be a list")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        linked_issues = entry.get("linked_issues", [])
        if not isinstance(linked_issues, list):
            continue
        for issue_num in linked_issues:
            issue = _safe_int(issue_num)
            if issue is not None:
                lookup.setdefault(issue, []).append(entry)
    return lookup


def _safe_int(value: object) -> int | None:
    """Return a strict issue number int, skipping bools and float coercions."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _short_text(value: object) -> str:
    """Return a stripped string when value is meaningful, otherwise empty."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _ledger_notes_for_rows(rows: list[dict[str, Any]]) -> tuple[float, tuple[str, ...]]:
    """Return ledger relevance and human-readable notes for matched ledger rows."""
    if not rows:
        return 0.0, ()

    notes: list[str] = []
    actionable = False
    for row in rows:
        area = _short_text(row.get("area"))
        prefix = f"ledger {area}:" if area else "ledger:"
        claim_gap = _short_text(row.get("claim_gap"))
        if claim_gap:
            actionable = True
            notes.append(f"{prefix} open claim gap - {claim_gap}")
        promotion = _short_text(row.get("evidence_promotion_path"))
        if promotion:
            actionable = True
            notes.append(f"{prefix} promotion path - {promotion}")
        tier = _short_text(row.get("evidence_tier"))
        if tier:
            notes.append(f"{prefix} evidence tier {tier}")

    if not notes:
        notes.append("ledger: issue tracked in evidence ledger")
    return (1.0 if actionable else 0.0), tuple(notes)


def _register_caveats_for_entries(entries: list[dict[str, Any]]) -> tuple[float, tuple[str, ...]]:
    """Return negative-result awareness and conservative caveats for register entries."""
    negative_statuses = {"revise", "diagnostic_only", "failed", "inconclusive"}
    caveats: list[str] = []
    awareness = 0.0

    for entry in entries:
        status = _short_text(entry.get("result_classification")).lower()
        if status not in negative_statuses:
            continue
        awareness = 1.0
        message = f"negative-result register: status={status}"
        if status == "diagnostic_only":
            message += (
                "; diagnostic-only repetition requires an explicit why-this-is-different rationale"
            )
        caveats.append(message)
        for key, label in (
            ("why_failed_or_inconclusive", "reason"),
            ("recommended_next_action", "next action"),
            ("claim_boundary", "boundary"),
        ):
            value = _short_text(entry.get(key))
            if value:
                caveats.append(f"negative-result {label}: {value}")

    return awareness, tuple(caveats)


def _compute_total_score(scores: dict[str, float]) -> float:
    """Compute a weighted aggregate score from dimension scores.

    All dimension scores are expected in [0, 1].  ``artifact_readiness`` is
    the inverse of artifact dependency (lower dependency is better).
    ``duplication_safety`` is the inverse of duplication risk (lower risk is
    better).
    """
    artifact_dependency = scores.get("artifact_dependency", 0.0)
    duplication_risk = scores.get("duplication_risk", 0.0)

    derived = {
        "dissertation_relevance": scores.get("dissertation_relevance", 0.0),
        "evidence_readiness": scores.get("evidence_readiness", 0.0),
        "implementation_boundedness": scores.get("implementation_boundedness", 0.0),
        "artifact_readiness": _clamp(1.0 - artifact_dependency),
        "expected_research_value": scores.get("expected_research_value", 0.0),
        "duplication_safety": _clamp(1.0 - duplication_risk),
    }

    total = 0.0
    for dimension, weight in DIMENSION_WEIGHTS.items():
        total += derived.get(dimension, 0.0) * weight
    return round(total, 6)


def _apply_ledger_register_modifiers(
    candidate: CandidateInput,
    ledger: dict[int, list[dict[str, Any]]],
    register: dict[int, list[dict[str, Any]]],
) -> tuple[float, float, tuple[str, ...], tuple[str, ...]]:
    """Compute ledger_relevance, negative_result_awareness, and notes.

    Returns (ledger_relevance, negative_result_awareness, ledger_notes,
    negative_result_caveats) where both relevance scores are in [0, 1] and
    notes/caveats are human-readable strings.
    """
    issue_num = candidate.issue_number
    ledger_relevance, ledger_notes = _ledger_notes_for_rows(
        ledger.get(issue_num, []) if issue_num is not None else []
    )
    negative_result_awareness, negative_caveats = _register_caveats_for_entries(
        register.get(issue_num, []) if issue_num is not None else []
    )
    return ledger_relevance, negative_result_awareness, ledger_notes, negative_caveats


def rank_candidates(
    candidates: list[CandidateInput],
    *,
    ledger: dict[int, list[dict[str, Any]]] | None = None,
    register: dict[int, list[dict[str, Any]]] | None = None,
) -> list[RankedCandidate]:
    """Rank candidates by aggregate score (descending), breaking ties by issue number.

    Optional ``ledger`` and ``register`` lookups add bounded additive modifiers
    and human-readable notes/caveats.
    """
    if ledger is None:
        ledger = {}
    if register is None:
        register = {}

    ranked: list[RankedCandidate] = []
    for idx, candidate in enumerate(candidates):
        total_score = _compute_total_score(candidate.scores)
        ledger_rel, neg_aware, ledger_notes, neg_caveats = _apply_ledger_register_modifiers(
            candidate, ledger, register
        )

        modifier = 0.0
        if ledger_rel > 0:
            modifier += LEDGER_BOOST * ledger_rel
        if neg_aware > 0:
            modifier -= NEGATIVE_RESULT_PENALTY * neg_aware
        total_score = _clamp(round(total_score + modifier, 6))

        new_reasons = list(candidate.reasons)
        new_caveats = list(candidate.evidence_caveats)
        new_reasons.extend(ledger_notes)
        new_caveats.extend(neg_caveats)

        ranked.append(
            RankedCandidate(
                rank=0,
                issue_number=candidate.issue_number,
                issue_url=candidate.issue_url,
                title=candidate.title,
                total_score=total_score,
                scores=candidate.scores,
                reasons=tuple(new_reasons),
                evidence_caveats=tuple(new_caveats),
                duplication_notes=candidate.duplication_notes,
                blocked_by=candidate.blocked_by,
                linter_ok=candidate.linter_ok,
                linter_findings=candidate.linter_findings,
                ledger_relevance=ledger_rel,
                negative_result_awareness=neg_aware,
                ledger_notes=ledger_notes,
                negative_result_caveats=neg_caveats,
            )
        )

    ranked.sort(
        key=lambda r: (-r.total_score, r.issue_number or 0, r.title),
    )

    for position, item in enumerate(ranked, start=1):
        ranked[position - 1] = RankedCandidate(
            rank=position,
            issue_number=item.issue_number,
            issue_url=item.issue_url,
            title=item.title,
            total_score=item.total_score,
            scores=item.scores,
            reasons=item.reasons,
            evidence_caveats=item.evidence_caveats,
            duplication_notes=item.duplication_notes,
            blocked_by=item.blocked_by,
            linter_ok=item.linter_ok,
            linter_findings=item.linter_findings,
            ledger_relevance=item.ledger_relevance,
            negative_result_awareness=item.negative_result_awareness,
            ledger_notes=item.ledger_notes,
            negative_result_caveats=item.negative_result_caveats,
        )

    return ranked


def _format_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> list[str]:
    """Format compact Markdown-style table rows."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def _format_issue_link(issue_url: str | None, issue_number: int | None) -> str:
    """Format an issue reference for Markdown output."""
    if issue_url and issue_number:
        return f"[#{issue_number}]({issue_url})"
    if issue_number:
        return f"#{issue_number}"
    return "N/A"


def _format_candidate_detail(item: RankedCandidate) -> list[str]:
    """Format the detail section for a single ranked candidate."""
    lines: list[str] = []
    issue_ref = _format_issue_link(item.issue_url, item.issue_number)
    lines.append(f"### {issue_ref} - {item.title}")
    lines.append("")
    lines.append(f"**Total score:** {item.total_score:.4f}")
    lines.append("")

    reasons = "; ".join(item.reasons) if item.reasons else "none"
    lines.append(f"**Reasons:** {reasons}")
    evidence_caveats = "; ".join(item.evidence_caveats) if item.evidence_caveats else "none"
    lines.append(f"**Evidence caveats:** {evidence_caveats}")
    duplication_notes = "; ".join(item.duplication_notes) if item.duplication_notes else "none"
    lines.append(f"**Duplication notes:** {duplication_notes}")
    blocked_by = ", ".join(f"#{b}" for b in item.blocked_by) if item.blocked_by else "none"
    lines.append(f"**Blocked by:** {blocked_by}")
    ledger_notes = "; ".join(item.ledger_notes) if item.ledger_notes else "none"
    lines.append(f"**Ledger notes:** {ledger_notes}")
    if item.ledger_notes:
        lines.append(f"**Ledger relevance:** {item.ledger_relevance:.2f}")
    negative_result_caveats = (
        "; ".join(item.negative_result_caveats) if item.negative_result_caveats else "none"
    )
    lines.append(f"**Negative-result caveats:** {negative_result_caveats}")
    if item.negative_result_caveats:
        lines.append(f"**Negative-result awareness:** {item.negative_result_awareness:.2f}")

    linter_status = "pass" if item.linter_ok else "fail"
    lines.append(f"**Linter:** {linter_status}")
    for finding in item.linter_findings:
        code = finding.get("code", "")
        detail = finding.get("detail", "")
        lines.append(f"- `{code}`: {detail}")

    lines.append("")
    return lines


def format_markdown_report(ranked: list[RankedCandidate], *, total_count: int) -> str:
    """Format a deterministic Markdown report from ranked candidates."""
    lines: list[str] = ["# Publication Candidate Ranking", ""]

    if not ranked:
        lines.append("No candidates to rank.")
        return "\n".join(lines)

    lines.append(f"Ranked {len(ranked)} candidate(s) by publication readiness.")
    lines.append("")

    table_headers = (
        "Rank",
        "Issue",
        "Title",
        "Score",
        "Relevance",
        "Evidence",
        "Bounded",
        "Blocked",
    )
    table_rows: list[tuple[object, ...]] = []
    for item in ranked:
        issue_ref = _format_issue_link(item.issue_url, item.issue_number)
        blocked_str = ", ".join(f"#{b}" for b in item.blocked_by) if item.blocked_by else "-"
        table_rows.append(
            (
                item.rank,
                issue_ref,
                item.title[:60],
                f"{item.total_score:.4f}",
                f"{item.scores.get('dissertation_relevance', 0.0):.2f}",
                f"{item.scores.get('evidence_readiness', 0.0):.2f}",
                f"{item.scores.get('implementation_boundedness', 0.0):.2f}",
                blocked_str,
            )
        )
    lines.extend(_format_table(table_headers, table_rows))

    lines.extend(["", "## Candidate Details", ""])

    for item in ranked:
        lines.extend(_format_candidate_detail(item))

    return "\n".join(lines)


def _build_report(
    ranked: list[RankedCandidate],
    *,
    total_count: int,
) -> dict[str, Any]:
    """Build the machine-readable CLI report."""
    return {
        "schema": SCHEMA,
        "ok": True,
        "read_only": True,
        "project_writes": False,
        "total_count": total_count,
        "ranked_count": len(ranked),
        "ranked_candidates": [
            {
                "rank": item.rank,
                "issue_number": item.issue_number,
                "issue_url": item.issue_url,
                "title": item.title,
                "total_score": item.total_score,
                "scores": item.scores,
                "reasons": list(item.reasons),
                "evidence_caveats": list(item.evidence_caveats),
                "duplication_notes": list(item.duplication_notes),
                "blocked_by": list(item.blocked_by),
                "linter_ok": item.linter_ok,
                "linter_findings": list(item.linter_findings),
                "ledger_relevance": item.ledger_relevance,
                "negative_result_awareness": item.negative_result_awareness,
                "ledger_notes": list(item.ledger_notes),
                "negative_result_caveats": list(item.negative_result_caveats),
            }
            for item in ranked
        ],
        "failure_summary": None,
    }


def _load_json(payload_path: Path) -> Any:
    """Load one JSON payload from disk."""
    data = payload_path.read_text(encoding="utf-8")
    return json.loads(data)


def _dump_json(payload: dict[str, Any]) -> None:
    """Write sorted JSON for stable, machine-readable output."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    """Build a deterministic CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-json",
        type=Path,
        required=True,
        help="JSON file containing a list of candidate objects.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format: markdown (default) or json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write output to file instead of stdout.",
    )
    parser.add_argument(
        "--ledger-json",
        type=Path,
        default=None,
        help="Optional dissertation evidence ledger JSON (dissertation_evidence_ledger.v2).",
    )
    parser.add_argument(
        "--register-json",
        type=Path,
        default=None,
        help="Optional negative result register JSON (negative_result_register.v1).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _build_parser().parse_args(argv)

    try:
        payload = _load_json(args.candidates_json)
    except (OSError, json.JSONDecodeError) as exc:
        _dump_json(
            {
                "schema": SCHEMA,
                "ok": False,
                "read_only": True,
                "project_writes": False,
                "error": str(exc),
            }
        )
        return 2

    if not isinstance(payload, list):
        _dump_json(
            {
                "schema": SCHEMA,
                "ok": False,
                "read_only": True,
                "project_writes": False,
                "error": "candidates-json payload must be a list of candidate objects",
            }
        )
        return 2

    ledger: dict[int, list[dict[str, Any]]] = {}
    if args.ledger_json is not None:
        try:
            ledger_raw = _load_json(args.ledger_json)
            ledger = _parse_ledger(ledger_raw)
        except (OSError, json.JSONDecodeError, RankerInputError) as exc:
            _dump_json(
                {
                    "schema": SCHEMA,
                    "ok": False,
                    "read_only": True,
                    "project_writes": False,
                    "error": f"ledger-json load failed: {exc}",
                }
            )
            return 2

    register: dict[int, list[dict[str, Any]]] = {}
    if args.register_json is not None:
        try:
            register_raw = _load_json(args.register_json)
            register = _parse_register(register_raw)
        except (OSError, json.JSONDecodeError, RankerInputError) as exc:
            _dump_json(
                {
                    "schema": SCHEMA,
                    "ok": False,
                    "read_only": True,
                    "project_writes": False,
                    "error": f"register-json load failed: {exc}",
                }
            )
            return 2

    candidates = [_parse_candidate(item) for item in payload if isinstance(item, dict)]
    ranked = rank_candidates(candidates, ledger=ledger, register=register)

    if args.format == "json":
        report = _build_report(ranked, total_count=len(candidates))
        output_text = json.dumps(report, indent=2, sort_keys=True)
    else:
        output_text = format_markdown_report(ranked, total_count=len(candidates))

    if args.output is not None:
        args.output.write_text(output_text + "\n", encoding="utf-8")
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
