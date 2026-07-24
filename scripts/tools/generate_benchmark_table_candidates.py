#!/usr/bin/env python3
"""Generate conservative benchmark table candidates for issue #2767."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON mapping, returning ``None`` when the source is absent."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


DEFAULT_LEDGER = Path("docs/context/evidence/issue_2760_dissertation_evidence_ledger/ledger.json")
DEFAULT_CLAIM_MAP = Path("docs/context/issue_1542_manuscript_claim_evidence_map.md")
DEFAULT_SIGNAL_SUMMARY = Path("docs/context/evidence/issue_2799_signalized_runtime/summary.json")
DEFAULT_OBS_NOISE_SUMMARY = Path(
    "docs/context/evidence/issue_2755_observation_noise_envelope_2026-06-13/summary.json"
)
DEFAULT_OUTPUT_DIR = Path("docs/context/evidence/issue_2767_benchmark_table_candidates")

DRAFT_STATUS = "draft_only_not_for_manuscript_use_without_verification"
TABLE_IDS = [
    "amv_metric_summary",
    "topology_diagnostic_summary",
    "signalized_crossing_metric_summary",
    "prediction_baseline_summary",
    "observation_noise_diagnostic_summary",
    "negative_result_summary",
]
UNAVAILABLE_SOURCE_SECTION_TABLE = (
    "| Status | Reason |\n|---|---|\n| unavailable | source section not found |"
)
UNAVAILABLE_SECTION_TABLE = (
    "| Status | Reason |\n|---|---|\n| unavailable | table not found under section |"
)


@dataclass(frozen=True)
class SourceInputs:
    """Tracked source inputs used to build the table candidates."""

    ledger: Path
    claim_map: Path
    signal_summary: Path
    observation_noise_summary: Path


def _extract_markdown_table(markdown: str, heading_pattern: str) -> str:
    """Extract the first Markdown table after a heading or marker pattern."""
    lines = markdown.splitlines()
    start_index = next(
        (index for index, line in enumerate(lines) if re.search(heading_pattern, line)),
        -1,
    )
    if start_index == -1:
        return UNAVAILABLE_SOURCE_SECTION_TABLE

    table_lines: list[str] = []
    found_table = False
    for line in lines[start_index:]:
        if line.strip().startswith("|"):
            table_lines.append(line)
            found_table = True
        elif found_table and line.strip():
            break
    if not table_lines:
        return UNAVAILABLE_SECTION_TABLE
    return "\n".join(table_lines)


def _ledger_row(ledger: dict[str, Any], area: str) -> dict[str, Any]:
    """Return one ledger row by area, failing closed if it is missing."""
    rows = ledger.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("ledger rows must be a list")
    for row in rows:
        if isinstance(row, dict) and row.get("area") == area:
            return row
    raise KeyError(f"Missing ledger row for {area}")


def _topology_table(ledger: dict[str, Any]) -> str:
    """Build the topology diagnostic table from the machine-readable ledger."""
    row = _ledger_row(ledger, "topology_guidance")
    claim = row.get("claim", "N/A")
    artifact_status = row.get("artifact_status", "N/A")
    evidence_tier = row.get("evidence_tier", "N/A")
    allowed_wording = row.get("allowed_wording", "N/A")
    caveat = row.get("caveat", "N/A")
    return "\n".join(
        [
            "| Field | Value |",
            "|---|---|",
            f"| Claim | {claim} |",
            f"| Artifact status | {artifact_status} |",
            f"| Evidence tier | {evidence_tier} |",
            f"| Allowed wording | {allowed_wording} |",
            f"| Caveat | {caveat} |",
        ]
    )


def _signalized_table(signal_summary: dict[str, Any] | None) -> str:
    """Build the signalized-crossing denominator table."""
    rows = ["| Episode ID | Row type | Observable | Denominator |", "|---|---|---|---|"]
    if not signal_summary:
        rows.append("| unavailable | missing tracked summary | false | 0 |")
        return "\n".join(rows)

    for key in ("eligible_rows", "excluded_rows"):
        values = signal_summary.get(key, [])
        if not isinstance(values, list):
            continue
        for row in values:
            if not isinstance(row, dict):
                continue
            denominator = str(row.get("signal_metrics_denominator", "unknown"))
            exclusion = row.get("exclusion_reason")
            if exclusion:
                denominator = f"{denominator} (excluded: {exclusion})"
            rows.append(
                "| {episode_id} | {row_type} | {planner_observable} | {denominator} |".format(
                    episode_id=row.get("episode_id", "unknown"),
                    row_type=row.get("row_type", "unknown"),
                    planner_observable=row.get("planner_observable", "unknown"),
                    denominator=denominator,
                )
            )
    return "\n".join(rows)


def _observation_noise_table(summary: dict[str, Any] | None) -> str:
    """Build the observation-noise diagnostic classification table."""
    rows = ["| Condition | Classification | Rationale |", "|---|---|---|"]
    if not summary:
        rows.append("| unavailable | missing tracked summary | no tracked input found |")
        return "\n".join(rows)

    summary_section = summary.get("summary")
    classifications = (
        summary_section.get("classifications", {}) if isinstance(summary_section, dict) else {}
    )
    conditions = summary.get("conditions", [])
    rationale_by_condition = {}
    if not isinstance(conditions, list):
        conditions = []
    for row in conditions:
        if not isinstance(row, dict):
            continue
        condition = row.get("condition")
        if condition is None:
            continue
        classification = row.get("classification")
        rationale = "N/A"
        if isinstance(classification, dict):
            rationale = classification.get("rationale", "N/A")
        rationale_by_condition[condition] = rationale
    for condition, classification in classifications.items():
        rows.append(
            f"| {condition} | {classification} | {rationale_by_condition.get(condition, 'N/A')} |"
        )
    return "\n".join(rows)


def _negative_result_table(ledger: dict[str, Any]) -> str:
    """Build a compact table of claim-blocking negative or stale rows."""
    stale_rows = ledger.get("stale_artifact_summary")
    if not isinstance(stale_rows, list):
        stale_rows = []
    rows = [
        "| Area | Verdict | Reason |",
        "|---|---|---|",
        "| CARLA Replay Parity | blocked | Robot actor spawn failure prevents oracle replay. |",
        (
            "| Predictive Planner v2 | negative | Obstacle-feature success worsened despite "
            "forecast improvement. |"
        ),
    ]
    for item in stale_rows:
        if not isinstance(item, dict):
            continue
        rows.append(
            "| {artifact} | {state} | {reason} |".format(
                artifact=item.get("artifact_id", "unnamed"),
                state=item.get("state", "unknown"),
                reason=item.get("reason", "unknown"),
            )
        )
    return "\n".join(rows)


def build_markdown(
    *,
    ledger: dict[str, Any],
    claim_map: str,
    signal_summary: dict[str, Any] | None,
    observation_noise_summary: dict[str, Any] | None,
    generated_at: str,
) -> str:
    """Build the Markdown table-candidate report."""
    amv_primary = _extract_markdown_table(claim_map, r"Primary protocol \(Issue #1344\):")
    prediction = _extract_markdown_table(claim_map, r"Final #1427 same-seed comparison:")
    topology_row = _ledger_row(ledger, "topology_guidance")

    return f"""# Issue #2767 Benchmark Table Candidates

Generated at: {generated_at}
Status: {DRAFT_STATUS}

This bundle contains conservative draft benchmark-results table candidates synthesized from
tracked claim/evidence inputs. It is not a manuscript draft and does not promote diagnostic,
stale, unavailable, fallback, degraded, proxy-only, or missing-denominator evidence.

## 1. Metric Summary (AMV Primary Protocol)
{amv_primary}

Allowed wording: "Baseline AMV performance remains limited; stress success remains low and caveated."

Caveat: AMV coverage remains incomplete and SNQI contract status is warn/fail in the source map.

## 2. Topology Diagnostic Summary
{_topology_table(ledger)}

Allowed wording: "{topology_row.get("allowed_wording", "N/A")}"

Caveat: {topology_row.get("caveat", "N/A")}

## 3. Signalized-Crossing Metric Summary
{_signalized_table(signal_summary)}

Allowed wording: "Simulator-backed signalized-crossing denominator plumbing exists for explicit
planner-observable rows."

Caveat: Diagnostic only; does not prove traffic-signal realism, crossing-legality compliance, or
planner-ranking performance.

## 4. Prediction Baseline Summary
{prediction}

Allowed wording: "The prediction interface and comparison surfaces exist, but current closed-loop
performance deltas are unproven or negative."

Caveat: Do not claim prediction-quality or planner-improvement evidence from contract or negative
same-seed rows.

## 5. Observation-Noise Diagnostic Summary
{_observation_noise_table(observation_noise_summary)}

Allowed wording: "Observation-noise diagnostics can identify fixture and forecast ambiguity for
planning follow-up."

Caveat: Diagnostic only; no sim-to-real, perception, or paper-facing robustness claim.

## 6. Negative-Result Summary
{_negative_result_table(ledger)}

## Conservative Rules Applied

- Draft-only unless dependencies are current and claimable.
- Diagnostic, stale, non-claimable, unavailable, fallback, degraded, proxy-only, or
  missing-denominator rows weaken or block wording.
- Fallback behavior is not acceptable as a successful benchmark outcome.
- No invented values; missing tracked sources produce unavailable rows.
"""


def build_summary(*, generated_at: str, sources: SourceInputs) -> dict[str, Any]:
    """Build machine-readable summary metadata for the generated report."""
    return {
        "schema_version": "benchmark_table_candidates.v1",
        "issue": 2767,
        "generated_at_utc": generated_at,
        "status": DRAFT_STATUS,
        "tables": TABLE_IDS,
        "source_inputs": {
            "ledger": str(sources.ledger),
            "claim_map": str(sources.claim_map),
            "signal_summary": str(sources.signal_summary),
            "observation_noise_summary": str(sources.observation_noise_summary),
        },
        "claim_boundary": (
            "synthesis draft only; not benchmark evidence, paper-facing evidence, or manuscript text"
        ),
        "conservative_rules": [
            "draft-only unless dependencies are current and claimable",
            "diagnostic/stale/non-claimable rows weaken or block wording",
            "fallback/degraded/unavailable/proxy-only rows are not success evidence",
            "missing tracked sources render a table unavailable rather than inventing values",
        ],
    }


def generate_table_candidates(
    *,
    sources: SourceInputs,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Generate Markdown and JSON table-candidate artifacts."""
    ledger = _load_json(sources.ledger)
    if ledger is None:
        raise FileNotFoundError(f"Missing ledger: {sources.ledger}")
    if not sources.claim_map.exists():
        raise FileNotFoundError(f"Missing claim map: {sources.claim_map}")
    claim_map = sources.claim_map.read_text(encoding="utf-8")
    signal_summary = _load_json(sources.signal_summary)
    observation_noise_summary = _load_json(sources.observation_noise_summary)

    generated_at = datetime.now(UTC).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "table_candidates.md"
    summary_path = output_dir / "summary.json"
    markdown_path.write_text(
        build_markdown(
            ledger=ledger,
            claim_map=claim_map,
            signal_summary=signal_summary,
            observation_noise_summary=observation_noise_summary,
            generated_at=generated_at,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(build_summary(generated_at=generated_at, sources=sources), indent=2) + "\n",
        encoding="utf-8",
    )
    return markdown_path, summary_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--claim-map", type=Path, default=DEFAULT_CLAIM_MAP)
    parser.add_argument("--signal-summary", type=Path, default=DEFAULT_SIGNAL_SUMMARY)
    parser.add_argument("--observation-noise-summary", type=Path, default=DEFAULT_OBS_NOISE_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _build_parser().parse_args(argv)
    markdown_path, summary_path = generate_table_candidates(
        sources=SourceInputs(
            ledger=args.ledger,
            claim_map=args.claim_map,
            signal_summary=args.signal_summary,
            observation_noise_summary=args.observation_noise_summary,
        ),
        output_dir=args.output_dir,
    )
    print(f"Wrote {markdown_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
