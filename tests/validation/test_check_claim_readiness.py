"""Tests for the claim-readiness guardrail checker."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.validation import check_claim_readiness

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> Path:
    """Write test text and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def test_ready_diagnostic_claim_reports_diagnostic_review_status(tmp_path: Path) -> None:
    """A complete diagnostic bundle should be ready only for diagnostic review."""
    claim = _write(
        tmp_path / "claim.md",
        """
        evidence_tier: diagnostic_only
        comparator: baseline planner row
        mechanism activation: activation_count changed on selected steps
        trace support: simulation_trace frames show step-level behavior
        seed/slice boundary: scenario classic, seed 111, horizon 160
        claim_boundary: diagnostic_only_not_benchmark_success
        fallback/degraded limitations: fallback rows are not benchmark evidence
        """,
    )
    evidence_dir = tmp_path / "evidence"
    _write(
        evidence_dir / "summary.json",
        """
        {
          "command": "uv run python scripts/demo/reproduce_mechanism_report.py",
          "commit": "abc123",
          "sha256": "0"
        }
        """,
    )

    result = check_claim_readiness.evaluate_claim_readiness(claim, evidence_dir)["claim_readiness"]

    assert result["schema_version"] == "claim_readiness.v1"
    assert result["status"] == "ready_for_diagnostic_review"
    assert result["claim_class"] == "diagnostic"
    assert result["missing_fields"] == []
    assert result["warnings"] == []


def test_missing_fields_report_is_not_ready(tmp_path: Path) -> None:
    """Sparse material should report missing fields instead of passing readiness."""
    claim = _write(tmp_path / "claim.md", "claim_boundary: diagnostic only")
    evidence = _write(tmp_path / "summary.json", '{"command": "uv run ..."}')

    result = check_claim_readiness.evaluate_claim_readiness(claim, evidence)["claim_readiness"]

    assert result["status"] == "not_ready_missing_fields"
    assert "comparator_or_baseline" in result["missing_fields"]
    assert "trace_support" in result["missing_fields"]
    assert "artifact_provenance" in result["present_fields"]


def test_benchmark_claim_with_diagnostic_or_fallback_evidence_warns(tmp_path: Path) -> None:
    """Benchmark/paper claims should not silently absorb diagnostic/fallback evidence."""
    claim = _write(
        tmp_path / "claim.md",
        """
        evidence_tier: benchmark-strength
        comparator: baseline
        mechanism activation: mechanism_signal present
        trace support: trace frames
        artifact provenance: command, commit, sha256
        seed/slice boundary: scenario x seed 1
        claim_boundary: benchmark_success
        fallback/degraded limitations: degraded rows and diagnostic_only evidence are caveats
        """,
    )
    evidence = _write(tmp_path / "evidence.md", "fallback rows are diagnostic_only caveats")

    result = check_claim_readiness.evaluate_claim_readiness(claim, evidence)["claim_readiness"]

    assert result["claim_class"] == "benchmark"
    assert result["status"] == "not_ready_claim_boundary_warning"
    assert result["missing_fields"] == []
    assert result["warnings"] == [
        "benchmark_or_paper_claim_mentions_fallback_degraded_or_diagnostic_only_evidence"
    ]


def test_evidence_tokens_do_not_override_claim_class(tmp_path: Path) -> None:
    """Incidental evidence text should not upgrade a diagnostic claim class."""
    claim = _write(
        tmp_path / "claim.md",
        """
        evidence_tier: diagnostic_only
        comparator: baseline
        mechanism activation: activation_count changed
        trace support: simulation_trace frames
        artifact provenance: command, commit, sha256
        seed/slice boundary: scenario x seed 1
        claim_boundary: diagnostic_only_not_benchmark_success
        fallback/degraded limitations: fallback rows are caveats
        """,
    )
    evidence = _write(
        tmp_path / "evidence.md",
        "Historical benchmark-strength manuscript note; not the claim under review.",
    )

    result = check_claim_readiness.evaluate_claim_readiness(claim, evidence)["claim_readiness"]

    assert result["claim_class"] == "diagnostic"
    assert result["status"] == "ready_for_diagnostic_review"


def test_cli_writes_json_and_sets_exit_code(tmp_path: Path, capsys) -> None:
    """CLI should print/write JSON and return readiness as the exit code."""
    claim = _write(tmp_path / "claim.md", "claim_boundary: diagnostic only")
    evidence = _write(tmp_path / "summary.json", '{"command": "uv run ..."}')
    output_json = tmp_path / "readiness.json"

    exit_code = check_claim_readiness.main(
        [
            "--claim-file",
            str(claim),
            "--evidence",
            str(evidence),
            "--output-json",
            str(output_json),
        ]
    )

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert stdout_payload == file_payload
    assert file_payload["claim_readiness"]["status"] == "not_ready_missing_fields"
