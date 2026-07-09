"""Generate dissertation gap report from evidence ledger and negative-result register.

This is a synthesis/planning aid. It does not produce new benchmark evidence,
paper-facing results, or safety claims.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.identity.hash_utils import load_json as _load_json

LEDGER_PATH = Path("docs/context/evidence/issue_2760_dissertation_evidence_ledger/ledger.json")
REGISTER_PATH = Path("docs/context/evidence/issue_2762_negative_result_register/register.json")

SCHEMA_VERSION = "dissertation_gap_report.v1"

CLAIM_BOUNDARIES = [
    "This gap report is a synthesis/planning aid. It does not produce new benchmark "
    "evidence, paper-facing results, or safety claims.",
    "All allowed_wording and caveat fields are copied verbatim from source rows; no new "
    "wording is introduced.",
    "A non-null promotion_step_or_reason does not upgrade a row to stronger evidence. "
    "Promotion requires completing the path and reclassifying the evidence tier.",
    "Fallback behavior is not acceptable as a successful benchmark outcome unless the "
    "task explicitly measures fallback mode.",
    "Every gap classification preserves the source evidence tier and classification "
    "without upgrade.",
]


def _classify_ledger_row(row: dict[str, Any]) -> str:
    """Classify a single ledger row into one of four buckets.

    Decision table (order matters for ambiguity):
    1. evidence_tier == release-backed AND artifact_status == current -> supported
    2. evidence_tier == non-claimable OR artifact_status == stale -> remove_weaken
    3. evidence_promotion_path is not None AND evidence_tier != release-backed -> blocked
    4. evidence_promotion_path is None AND evidence_tier != release-backed
       AND artifact_status == current -> negative_revise_only
    5. default -> blocked
    """
    tier = row["evidence_tier"]
    status = row["artifact_status"]
    path = row["evidence_promotion_path"]

    if tier == "release-backed" and status == "current":
        return "supported"
    if tier == "non-claimable" or status == "stale":
        return "remove_weaken"
    if path is not None and tier != "release-backed":
        return "blocked"
    if path is None and tier != "release-backed" and status == "current":
        return "negative_revise_only"
    return "blocked"


def _classify_register_entry(entry: dict[str, Any]) -> str:
    """Classify a single register entry into one of four buckets."""
    classification = entry["result_classification"]
    if classification == "revise":
        return "negative_revise_only"
    if classification == "diagnostic_only":
        return "negative_revise_only"
    if classification == "failed":
        return "remove_weaken"
    if classification == "inconclusive":
        return "negative_revise_only"
    return "blocked"


def _build_gap_from_ledger(row: dict[str, Any]) -> dict[str, Any]:
    """Build a gap record from a ledger row."""
    bucket = _classify_ledger_row(row)
    return {
        "area": row["area"],
        "entry_id": None,
        "bucket": bucket,
        "promotion_step_or_reason": row["evidence_promotion_path"],
        "source": "ledger",
        "source_issue": row["source_issues"][0] if row["source_issues"] else None,
        "evidence_tier": row["evidence_tier"],
        "result_classification": None,
        "allowed_wording_or_boundary": row["allowed_wording"],
        "caveat": row["caveat"],
        "claim_gap_or_reason": row["claim_gap"],
    }


def _build_gap_from_register(entry: dict[str, Any]) -> dict[str, Any]:
    """Build a gap record from a register entry."""
    bucket = _classify_register_entry(entry)
    return {
        "area": None,
        "entry_id": entry["id"],
        "bucket": bucket,
        "promotion_step_or_reason": entry.get("recommended_next_action"),
        "source": "register",
        "source_issue": entry["linked_issues"][0] if entry["linked_issues"] else None,
        "evidence_tier": None,
        "result_classification": entry["result_classification"],
        "allowed_wording_or_boundary": entry["claim_boundary"],
        "caveat": entry["claim_boundary"],
        "claim_gap_or_reason": entry["why_failed_or_inconclusive"],
    }


def generate_gap_report() -> tuple[dict[str, Any], str]:
    """Load sources, classify, and return (json_report, markdown_text)."""
    ledger = _load_json(LEDGER_PATH)
    register = _load_json(REGISTER_PATH)

    gaps: list[dict[str, Any]] = []
    for row in ledger["rows"]:
        gaps.append(_build_gap_from_ledger(row))
    for entry in register["entries"]:
        gaps.append(_build_gap_from_register(entry))

    json_report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).date().isoformat(),
        "purpose": "synthesis/planning aid; not new benchmark, paper, dissertation, or safety evidence",
        "source_ledger": {
            "path": str(LEDGER_PATH),
            "issue": ledger.get("issue"),
            "row_count": len(ledger["rows"]),
        },
        "source_register": {
            "path": str(REGISTER_PATH),
            "issue": register.get("issue"),
            "entry_count": len(register["entries"]),
        },
        "gaps": gaps,
        "claim_boundaries": CLAIM_BOUNDARIES,
    }

    md_text = _render_markdown(json_report)
    return json_report, md_text


def _render_markdown(report: dict[str, Any]) -> str:
    """Render the gap report as Markdown."""
    lines: list[str] = []
    lines.append("# Dissertation Gap Report")
    lines.append("")
    lines.append(
        "**Purpose**: synthesis/planning aid; not new benchmark, paper, dissertation, "
        "or safety evidence."
    )
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}  | Schema: {report['schema_version']}")
    lines.append("")
    lines.append(
        f"Sources: ledger #{report['source_ledger']['issue']} "
        f"({report['source_ledger']['row_count']} rows), "
        f"register #{report['source_register']['issue']} "
        f"({report['source_register']['entry_count']} entries)"
    )
    lines.append("")

    bucket_order = ["supported", "blocked", "negative_revise_only", "remove_weaken"]
    bucket_titles = {
        "supported": "Supported (release-backed, current)",
        "blocked": "Blocked (promotion path exists but not yet completed)",
        "negative_revise_only": "Negative / Revise-Only",
        "remove_weaken": "Remove / Weaken",
    }

    for bucket in bucket_order:
        bucket_gaps = [g for g in report["gaps"] if g["bucket"] == bucket]
        lines.append(f"## {bucket_titles[bucket]}")
        lines.append("")
        if not bucket_gaps:
            lines.append("No gaps in this bucket.")
            lines.append("")
            continue

        for gap in bucket_gaps:
            label = gap["area"] or gap["entry_id"]
            source_tag = f"[{gap['source']}]"
            lines.append(f"### {label} {source_tag}")
            lines.append("")
            tier_info = gap["evidence_tier"] or gap["result_classification"]
            lines.append(f"- **Tier/Classification**: {tier_info}")
            lines.append(
                f"- **Promotion step or reason**: "
                f"{gap['promotion_step_or_reason'] or 'None (limitation remains)'}"
            )
            lines.append(f"- **Allowed wording / boundary**: {gap['allowed_wording_or_boundary']}")
            lines.append(f"- **Caveat**: {gap['caveat']}")
            lines.append(f"- **Claim gap / reason**: {gap['claim_gap_or_reason']}")
            lines.append("")

    lines.append("## Claim Boundaries")
    lines.append("")
    for b in report["claim_boundaries"]:
        lines.append(f"- {b}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate dissertation gap report from ledger and register."
    )
    parser.add_argument(
        "--json-output",
        required=True,
        help="Path to write the JSON gap report.",
    )
    parser.add_argument(
        "--markdown-output",
        required=True,
        help="Path to write the Markdown gap report.",
    )
    args = parser.parse_args()

    try:
        json_report, md_text = generate_gap_report()
    except FileNotFoundError as exc:
        parser.error(str(exc))

    json_out = Path(args.json_output)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(json_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    md_out = Path(args.markdown_output)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(md_text, encoding="utf-8")

    print(f"JSON report written to {json_out}")
    print(f"Markdown report written to {md_out}")


if __name__ == "__main__":
    main()
