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


def _write_present_outputs(root: Path, *, generated_at: str) -> None:
    _write_json(
        root / "result.json",
        {
            "schema_version": "headline-ci-rank-stability-report.v1",
            "generated_at_utc": generated_at,
            "classification": "diagnostic",
            "classification_rationale": "fixture remains diagnostic",
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
