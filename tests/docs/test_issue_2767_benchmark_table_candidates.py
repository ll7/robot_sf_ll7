"""Tests for issue #2767 benchmark table candidate generation."""

from __future__ import annotations

import json

from scripts.tools.generate_benchmark_table_candidates import (
    DEFAULT_CLAIM_MAP,
    DEFAULT_LEDGER,
    DEFAULT_OBS_NOISE_SUMMARY,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SIGNAL_SUMMARY,
    DRAFT_STATUS,
    TABLE_IDS,
    SourceInputs,
    generate_table_candidates,
)


def _default_sources() -> SourceInputs:
    return SourceInputs(
        ledger=DEFAULT_LEDGER,
        claim_map=DEFAULT_CLAIM_MAP,
        signal_summary=DEFAULT_SIGNAL_SUMMARY,
        observation_noise_summary=DEFAULT_OBS_NOISE_SUMMARY,
    )


def test_generated_table_candidates_are_current() -> None:
    """Tracked artifacts should match the generator output shape."""
    markdown = (DEFAULT_OUTPUT_DIR / "table_candidates.md").read_text(encoding="utf-8")
    summary = json.loads((DEFAULT_OUTPUT_DIR / "summary.json").read_text(encoding="utf-8"))

    assert DRAFT_STATUS in markdown
    assert summary["schema_version"] == "benchmark_table_candidates.v1"
    assert summary["status"] == DRAFT_STATUS
    assert summary["tables"] == TABLE_IDS
    assert "not benchmark evidence" in summary["claim_boundary"]


def test_generator_writes_all_requested_tables(tmp_path) -> None:
    """Regeneration should produce all requested table candidates."""
    markdown_path, summary_path = generate_table_candidates(
        sources=_default_sources(),
        output_dir=tmp_path,
    )

    markdown = markdown_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    expected_headings = [
        "## 1. Metric Summary",
        "## 2. Topology Diagnostic Summary",
        "## 3. Signalized-Crossing Metric Summary",
        "## 4. Prediction Baseline Summary",
        "## 5. Observation-Noise Diagnostic Summary",
        "## 6. Negative-Result Summary",
    ]
    for heading in expected_headings:
        assert heading in markdown
    assert summary["tables"] == TABLE_IDS


def test_draft_only_boundary_and_caveats_are_explicit(tmp_path) -> None:
    """Generated text must preserve draft-only and no-overclaim language."""
    markdown_path, summary_path = generate_table_candidates(
        sources=_default_sources(),
        output_dir=tmp_path,
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    blocked_terms = [
        "not a manuscript draft",
        "does not promote diagnostic",
        "Diagnostic only",
        "Fallback behavior is not acceptable as a successful benchmark outcome",
        "No invented values",
    ]
    for term in blocked_terms:
        assert term in markdown
    assert (
        "draft-only unless dependencies are current and claimable" in summary["conservative_rules"]
    )


def test_stale_and_negative_rows_weaken_table_status(tmp_path) -> None:
    """Stale or negative rows should remain visible as blockers."""
    markdown_path, _summary_path = generate_table_candidates(
        sources=_default_sources(),
        output_dir=tmp_path,
    )
    markdown = markdown_path.read_text(encoding="utf-8")

    assert "tab_issue_1023_campaign_table" in markdown
    assert "non-claimable" in markdown
    assert "stale-needs-refresh" in markdown
    assert "Predictive Planner v2 | negative" in markdown
    assert "CARLA Replay Parity | blocked" in markdown


def test_missing_optional_signal_source_fails_closed(tmp_path) -> None:
    """Missing optional table inputs should produce unavailable rows, not invented values."""
    missing = tmp_path / "missing.json"
    markdown_path, summary_path = generate_table_candidates(
        sources=SourceInputs(
            ledger=DEFAULT_LEDGER,
            claim_map=DEFAULT_CLAIM_MAP,
            signal_summary=missing,
            observation_noise_summary=missing,
        ),
        output_dir=tmp_path / "out",
    )
    markdown = markdown_path.read_text(encoding="utf-8")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "| unavailable | missing tracked summary | false | 0 |" in markdown
    assert "| unavailable | missing tracked summary | no tracked input found |" in markdown
    assert str(missing) == summary["source_inputs"]["signal_summary"]
