#!/usr/bin/env python3
"""Validate the integrated job-13274 rank-stability claim packet.

The preserved bundle contains historical machine-readable analysis output and a
human-authored integration report.  This checker keeps those surfaces aligned
without regenerating the campaign or changing metric semantics.  It is
intentionally specific to the completed #5247 evidence bundle because its
purpose is to catch claim-card drift before the result is reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_BUNDLE = Path("docs/context/evidence/issue_5247_job_13274_rank_stability")
EXPECTED_RESULT_SCHEMA = "issue_3216_headline_ci_rank_stability.v1"
EXPECTED_PROVENANCE_SCHEMA = "issue_5247_verified_harvest_rank_stability.v2"


def _load_json(path: Path, problems: list[str]) -> dict[str, Any] | None:
    """Load a JSON object, recording a useful fail-closed diagnostic."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        problems.append(f"{path}: cannot load JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        problems.append(f"{path}: expected a JSON object")
        return None
    return payload


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expect(problems: list[str], label: str, actual: Any, expected: Any) -> None:
    """Record a mismatch between a packet field and its contract value."""
    if actual != expected:
        problems.append(f"{label}: expected {expected!r}, got {actual!r}")


def _headline_counts(result: dict[str, Any], problems: list[str]) -> dict[str, int]:
    """Derive headline counts from the canonical result arrays."""
    adjacent = result.get("adjacent_rank_claims")
    stability = result.get("rank_stability")
    cells = result.get("cells")
    if not isinstance(adjacent, list):
        problems.append("result.json: adjacent_rank_claims must be a list")
        adjacent = []
    if not isinstance(stability, list):
        problems.append("result.json: rank_stability must be a list")
        stability = []
    if not isinstance(cells, list):
        problems.append("result.json: cells must be a list")
        cells = []

    separable = sum(
        isinstance(claim, dict) and claim.get("decision") == "ci_separable" for claim in adjacent
    )
    overlapping = sum(
        isinstance(claim, dict)
        and claim.get("decision") == "not_statistically_distinguishable_budget"
        for claim in adjacent
    )
    stable_top1 = sum(
        isinstance(entry, dict) and entry.get("top1_stable") is True for entry in stability
    )
    counted_cells = sum(isinstance(cell, dict) and cell.get("counted") is True for cell in cells)
    excluded_cells = len(cells) - counted_cells
    return {
        "adjacent_total": len(adjacent),
        "ci_separable": separable,
        "ci_overlap": overlapping,
        "scenario_total": len(stability),
        "stable_top1": stable_top1,
        "cell_total": len(cells),
        "counted_cells": counted_cells,
        "excluded_cells": excluded_cells,
    }


def _check_provenance(
    bundle: Path,
    result: dict[str, Any],
    provenance: dict[str, Any],
    problems: list[str],
) -> None:
    """Check provenance, ranking profile, metric boundary, and output hashes."""
    _expect(
        problems,
        "provenance.schema_version",
        provenance.get("schema_version"),
        EXPECTED_PROVENANCE_SCHEMA,
    )
    _expect(
        problems, "provenance.evidence_status", provenance.get("evidence_status"), "diagnostic-only"
    )
    ranking = provenance.get("ranking")
    if not isinstance(ranking, dict):
        problems.append("analysis_provenance.json: ranking must be an object")
    else:
        _expect(problems, "provenance.ranking.profile", ranking.get("profile"), "constraints_first")
        _expect(problems, "provenance.ranking.rank_metric", ranking.get("rank_metric"), "success")
        _expect(
            problems,
            "provenance.ranking.snqi_rank_status",
            ranking.get("snqi_rank_status"),
            "blocked_invalid_metric",
        )

    failure = provenance.get("snqi_contract_failure")
    if not isinstance(failure, dict):
        problems.append("analysis_provenance.json: snqi_contract_failure must be an object")
    else:
        _expect(
            problems,
            "provenance.snqi_contract_failure.contract_status",
            failure.get("contract_status"),
            "fail",
        )
        _expect(
            problems,
            "provenance.snqi_contract_failure.enforcement",
            failure.get("enforcement"),
            "warn",
        )
        failed_checks = failure.get("failed_checks")
        if not isinstance(failed_checks, list) or not any(
            isinstance(check, dict)
            and check.get("check") == "rank_alignment_spearman"
            and check.get("value") == -0.19999999999999998
            for check in failed_checks
        ):
            problems.append(
                "analysis_provenance.json: missing the recorded rank_alignment_spearman=-0.2 failure"
            )

    output_hashes = provenance.get("output_sha256")
    if not isinstance(output_hashes, dict):
        problems.append("analysis_provenance.json: output_sha256 must be an object")
        return
    for filename in ("result.json", "report.md"):
        path = bundle / filename
        expected = output_hashes.get(filename)
        if not path.is_file():
            problems.append(f"{path}: required preserved output is missing")
        elif not isinstance(expected, str) or _sha256(path) != expected:
            problems.append(f"{path}: SHA-256 does not match analysis_provenance.json")

    _expect(problems, "result.schema_version", result.get("schema_version"), EXPECTED_RESULT_SCHEMA)


def _check_claim_card(claim_card: str, counts: dict[str, int], problems: list[str]) -> None:
    """Check the human-readable claim boundary without interpreting new claims."""
    required_fragments = (
        "**Evidence classification:** `diagnostic-only`",
        "**S30 disposition:** `deferred_for_dissertation_draft`",
        "No planner-superiority, benchmark, paper, or dissertation claim is promoted.",
        "No SNQI rank or SNQI-based adjacent-order claim is promoted.",
        "No S30 result is implied: no S30 campaign rows exist in this bundle.",
    )
    for fragment in required_fragments:
        if fragment not in claim_card:
            problems.append(f"claim_decision.md: missing required claim-boundary text: {fragment}")

    normalized = " ".join(claim_card.split())
    expected_headline = (
        f"{counts['ci_separable']} of {counts['adjacent_total']} adjacent comparisons are "
        f"CI-separable, {counts['ci_overlap']} are not distinguishable at this budget, "
        f"only {counts['stable_top1']} of {counts['scenario_total']} scenario top-1 planners are stable"
    )
    if expected_headline not in normalized:
        problems.append(
            "claim_decision.md: headline counts do not match the preserved result packet"
        )


def _check_decision_packet(
    result: dict[str, Any], counts: dict[str, int], problems: list[str]
) -> None:
    """Check the machine-readable decision packet against derived counts."""
    packet = result.get("decision_packet")
    if not isinstance(packet, dict):
        problems.append("result.json: decision_packet must be an object")
        return
    _expect(
        problems, "decision_packet.rank_profile", packet.get("rank_profile"), "constraints_first"
    )
    _expect(
        problems,
        "decision_packet.s30_decision_status",
        packet.get("s30_decision_status"),
        "needs_review",
    )
    _expect(
        problems,
        "decision_packet.adjacent_overlap_count",
        packet.get("adjacent_overlap_count"),
        counts["ci_overlap"],
    )
    _expect(
        problems,
        "decision_packet.identifiable_scenario_count",
        packet.get("identifiable_scenario_count"),
        counts["scenario_total"],
    )
    _expect(
        problems,
        "decision_packet.excluded_cell_count",
        packet.get("excluded_cell_count"),
        counts["excluded_cells"],
    )
    _expect(problems, "decision_packet.min_seed_count", packet.get("min_seed_count"), 20)
    grid = packet.get("grid_completeness")
    if not isinstance(grid, dict):
        problems.append("decision_packet.grid_completeness must be an object")
    else:
        _expect(
            problems,
            "decision_packet.grid_completeness.observed_cell_count",
            grid.get("observed_cell_count"),
            counts["counted_cells"],
        )


def _check_rank_metrics(result: dict[str, Any], problems: list[str]) -> None:
    """Ensure every preserved ranking entry uses the constraints-first metric."""
    claims = result.get("adjacent_rank_claims", [])
    if not isinstance(claims, list):
        return
    for claim in claims:
        if isinstance(claim, dict):
            _expect(
                problems, "adjacent_rank_claim.rank_metric", claim.get("rank_metric"), "success"
            )
    stability = result.get("rank_stability", [])
    if not isinstance(stability, list):
        return
    for entry in stability:
        if isinstance(entry, dict):
            _expect(problems, "rank_stability.rank_metric", entry.get("rank_metric"), "success")


def check_bundle(bundle: Path) -> list[str]:
    """Return fail-closed diagnostics for the #5247 evidence bundle."""
    problems: list[str] = []
    result_path = bundle / "result.json"
    provenance_path = bundle / "analysis_provenance.json"
    claim_path = bundle / "claim_decision.md"
    for path in (result_path, provenance_path, bundle / "report.md", claim_path):
        if not path.is_file():
            problems.append(f"{path}: required evidence artifact is missing")

    result = _load_json(result_path, problems) if result_path.is_file() else None
    provenance = _load_json(provenance_path, problems) if provenance_path.is_file() else None
    if result is None or provenance is None:
        return problems

    counts = _headline_counts(result, problems)
    _check_decision_packet(result, counts, problems)
    _check_rank_metrics(result, problems)

    _check_provenance(bundle, result, provenance, problems)
    if claim_path.is_file():
        try:
            _check_claim_card(claim_path.read_text(encoding="utf-8"), counts, problems)
        except OSError as exc:
            problems.append(f"{claim_path}: cannot read claim card: {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the claim-packet validator and return a process status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--json", action="store_true", help="emit a machine-readable result")
    args = parser.parse_args(argv)

    problems = check_bundle(args.bundle_dir)
    payload = {
        "status": "failed" if problems else "passed",
        "bundle": str(args.bundle_dir),
        "problems": problems,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    elif problems:
        print("FAIL: #5247 rank-stability claim packet")
        for problem in problems:
            print(f"- {problem}")
    else:
        print(f"PASS: #5247 rank-stability claim packet ({args.bundle_dir})")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
