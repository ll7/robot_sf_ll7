"""Tests for headline rank-stability output reconciliation packet."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "tools"
    / "reconcile_headline_rank_stability_outputs.py"
)
_spec = importlib.util.spec_from_file_location("headline_reconcile_3802", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
sys.modules["headline_reconcile_3802"] = mod
_spec.loader.exec_module(mod)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_present_outputs(
    root: Path,
    *,
    generated_at: str,
    decision_packet: dict[str, Any] | None = None,
) -> None:
    _write_json(
        root / "result.json",
        {
            "schema_version": "headline-ci-rank-stability-report.v1",
            "generated_at_utc": generated_at,
            "classification": "diagnostic",
            "classification_rationale": "fixture remains diagnostic",
            "decision_packet": decision_packet or {},
        },
    )
    (root / "report.md").write_text("# report\n", encoding="utf-8")
    _write_json(
        root / "seed_sufficiency_analysis.json",
        {
            "schema_version": "seed_sufficiency_analysis.v1",
            "generated_at_utc": generated_at,
            "rank_metric": "snqi",
            "summary": {"campaign_count": 1, "underpowered_or_unstable": True},
        },
    )
    _write_json(
        root / "headline_rank_stability_contract.json",
        {
            "schema_version": "headline-rank-stability-contract.v1",
            "generated_at_utc": generated_at,
            "claim_status": "blocked_missing_increased_seed_rows",
            "label": "blocked_pending_s20_s30",
            "promotion_allowed": False,
            "max_seed_count": 5,
        },
    )
    (root / "headline_rank_stability_pairwise.csv").write_text(
        "from_campaign,to_campaign,rank_label\n",
        encoding="utf-8",
    )


def test_packet_marks_present_outputs_available_but_not_claim_ready(tmp_path: Path) -> None:
    """Present artifacts reconcile without promoting headline ranking claims."""

    root = tmp_path / "outputs"
    _write_present_outputs(root, generated_at="2026-06-29T10:00:00+00:00")

    packet = mod.build_packet(
        [root],
        fresh_after=mod._parse_datetime("2026-06-29T09:00:00+00:00"),
        now=mod._parse_datetime("2026-06-29T11:00:00+00:00"),
    )

    assert packet["summary"]["required_missing"] == 0
    assert packet["summary"]["required_stale"] == 0
    assert packet["summary"]["readiness"] == "not_claim_ready"
    assert packet["claim_inputs"]["headline_contract_claim_status"] == (
        "blocked_missing_increased_seed_rows"
    )
    assert any(
        item["claim"] == "dissertation_ready_rank_stability_claim"
        for item in packet["cannot_claim"]
    )
    markdown = mod.render_markdown(packet)
    assert "Artifact Status" in markdown
    assert "paper/dissertation claim edits" in markdown


def test_packet_surfaces_decision_packet_overlap_blockers(tmp_path: Path) -> None:
    """Decision packet overlap state blocks strict local ranking claims."""
    root = tmp_path / "outputs"
    _write_present_outputs(
        root,
        generated_at="2026-06-29T10:00:00+00:00",
        decision_packet={
            "manuscript_table_status": "ready_for_table_review_no_claim_promotion",
            "s30_decision_status": "needs_review",
            "s30_reasons": ["adjacent_rank_ci_overlap_requires_claim_downgrade_or_more_data"],
            "adjacent_overlap_count": 2,
            "invalid_metric_claim_count": 0,
            "manuscript_blockers": [],
        },
    )

    packet = mod.build_packet([root], now=mod._parse_datetime("2026-06-29T11:00:00+00:00"))

    assert packet["claim_inputs"]["manuscript_table_status"] == (
        "ready_for_table_review_no_claim_promotion"
    )
    assert packet["claim_inputs"]["s30_decision_status"] == "needs_review"
    assert packet["claim_inputs"]["adjacent_overlap_count"] == 2
    blocked_claims = {item["claim"] for item in packet["cannot_claim"]}
    assert "s30_not_required_by_local_preflight" in blocked_claims
    assert "strict_adjacent_planner_ordering" in blocked_claims


def test_packet_surfaces_blocked_s30_decision_status(tmp_path: Path) -> None:
    """A blocked S30 decision remains explicit in read-only reconciliation."""
    root = tmp_path / "outputs"
    _write_present_outputs(
        root,
        generated_at="2026-06-29T10:00:00+00:00",
        decision_packet={
            "manuscript_table_status": "blocked",
            "s30_decision_status": "blocked",
            "s30_reasons": ["missing_expected_headline_cells"],
            "adjacent_overlap_count": 0,
            "invalid_metric_claim_count": 0,
            "manuscript_blockers": ["headline_grid_incomplete"],
        },
    )

    packet = mod.build_packet([root], now=mod._parse_datetime("2026-06-29T11:00:00+00:00"))

    assert packet["claim_inputs"]["s30_decision_status"] == "blocked"
    assert any(
        item["claim"] == "s30_not_required_by_local_preflight"
        and "missing_expected_headline_cells" in item["reason"]
        for item in packet["cannot_claim"]
    )


def test_packet_normalizes_scalar_s30_reasons(tmp_path: Path) -> None:
    """Scalar S30 reasons stay whole instead of being split into characters."""
    root = tmp_path / "outputs"
    _write_present_outputs(
        root,
        generated_at="2026-06-29T10:00:00+00:00",
        decision_packet={
            "manuscript_table_status": "ready_for_table_review_no_claim_promotion",
            "s30_decision_status": "blocked",
            "s30_reasons": "missing_expected_headline_cells",
            "adjacent_overlap_count": 0,
            "invalid_metric_claim_count": 0,
            "manuscript_blockers": [],
        },
    )

    packet = mod.build_packet([root], now=mod._parse_datetime("2026-06-29T11:00:00+00:00"))

    blocker = next(
        item
        for item in packet["cannot_claim"]
        if item["claim"] == "s30_not_required_by_local_preflight"
    )
    assert blocker["reason"] == "missing_expected_headline_cells"


def test_packet_falls_back_for_invalid_s30_reasons(tmp_path: Path) -> None:
    """Malformed S30 reasons keep the fail-closed status explicit."""
    root = tmp_path / "outputs"
    _write_present_outputs(
        root,
        generated_at="2026-06-29T10:00:00+00:00",
        decision_packet={
            "manuscript_table_status": "ready_for_table_review_no_claim_promotion",
            "s30_decision_status": "blocked",
            "s30_reasons": {"unexpected": "shape"},
            "adjacent_overlap_count": 0,
            "invalid_metric_claim_count": 0,
            "manuscript_blockers": [],
        },
    )

    packet = mod.build_packet([root], now=mod._parse_datetime("2026-06-29T11:00:00+00:00"))

    assert any(
        item["claim"] == "s30_not_required_by_local_preflight"
        and item["reason"] == "decision packet s30_decision_status is 'blocked'"
        for item in packet["cannot_claim"]
    )


def test_packet_records_clear_local_preflight_status_without_promotion(tmp_path: Path) -> None:
    """Clear S20 local decision packet is visible but remains read-only."""
    root = tmp_path / "outputs"
    _write_present_outputs(
        root,
        generated_at="2026-06-29T10:00:00+00:00",
        decision_packet={
            "manuscript_table_status": "ready_for_table_review_no_claim_promotion",
            "s30_decision_status": "not_required_by_local_preflight",
            "s30_reasons": [],
            "adjacent_overlap_count": 0,
            "invalid_metric_claim_count": 0,
            "manuscript_blockers": [],
        },
    )

    packet = mod.build_packet([root], now=mod._parse_datetime("2026-06-29T11:00:00+00:00"))

    assert packet["claim_inputs"]["s30_decision_status"] == "not_required_by_local_preflight"
    blocked_claims = {item["claim"] for item in packet["cannot_claim"]}
    assert "s30_not_required_by_local_preflight" not in blocked_claims
    assert "strict_adjacent_planner_ordering" not in blocked_claims
    assert "new_paper_or_dissertation_text" in blocked_claims


def test_packet_marks_required_outputs_missing(tmp_path: Path) -> None:
    """Missing fixture artifacts fail closed as unavailable outputs."""

    root = tmp_path / "empty"
    root.mkdir()

    packet = mod.build_packet([root], now=mod._parse_datetime("2026-06-29T11:00:00+00:00"))

    assert packet["summary"]["readiness"] == "blocked_missing_outputs"
    assert packet["summary"]["required_missing"] == 4
    missing = [
        artifact
        for root_summary in packet["roots"]
        for artifact in root_summary["artifacts"]
        if artifact["status"] == "missing"
    ]
    assert {artifact["key"] for artifact in missing} >= {
        "headline_report_json",
        "headline_report_markdown",
        "seed_sufficiency_json",
        "headline_contract_json",
    }


def test_packet_marks_old_generated_outputs_stale(tmp_path: Path) -> None:
    """Freshness floor flags stale JSON artifacts by generated timestamp."""

    root = tmp_path / "outputs"
    _write_present_outputs(root, generated_at="2026-06-28T10:00:00+00:00")

    packet = mod.build_packet(
        [root],
        fresh_after=mod._parse_datetime("2026-06-29T09:00:00+00:00"),
        now=mod._parse_datetime("2026-06-29T11:00:00+00:00"),
    )

    stale_keys = {
        artifact["key"]
        for root_summary in packet["roots"]
        for artifact in root_summary["artifacts"]
        if artifact["status"] == "stale"
    }
    assert {
        "headline_report_json",
        "seed_sufficiency_json",
        "headline_contract_json",
    } <= stale_keys
    assert packet["summary"]["readiness"] == "blocked_stale_outputs"


def test_packet_marks_non_dict_json_required_artifact_missing(tmp_path: Path) -> None:
    """A structurally wrong JSON artifact (list/primitive) fails closed without crashing."""

    root = tmp_path / "outputs"
    _write_present_outputs(root, generated_at="2026-06-29T10:00:00+00:00")
    # Overwrite a required JSON artifact with a non-object payload.
    (root / "result.json").write_text(json.dumps([1, 2, 3]) + "\n", encoding="utf-8")

    packet = mod.build_packet([root], now=mod._parse_datetime("2026-06-29T11:00:00+00:00"))

    assert packet["summary"]["readiness"] == "blocked_missing_outputs"
    report = next(
        artifact
        for root_summary in packet["roots"]
        for artifact in root_summary["artifacts"]
        if artifact["key"] == "headline_report_json"
    )
    assert report["status"] == "missing"
    assert "expected object" in report["detail"]


def test_cli_writes_json_and_markdown_packet(tmp_path: Path, capsys: Any) -> None:
    """CLI writes the reconciliation packet files without requiring campaign execution."""

    root = tmp_path / "outputs"
    out_dir = tmp_path / "packet"
    _write_present_outputs(root, generated_at="2026-06-29T10:00:00+00:00")

    exit_code = mod.main(
        [
            str(root),
            "--fresh-after",
            "2026-06-29T09:00:00+00:00",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "readiness=not_claim_ready" in captured.out
    assert (out_dir / "headline_rank_stability_reconciliation.json").exists()
    assert (out_dir / "headline_rank_stability_reconciliation.md").exists()
