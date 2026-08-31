"""Tests for future work bridge status cards and summary builder (issue #8048)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analysis.build_future_work_bridge_status import (
    SCHEMA,
    SUMMARY_SCHEMA,
    build_all,
    check_all,
    get_bridge_cards,
    validate_safe_sentence,
)

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analysis"
    / "build_future_work_bridge_status.py"
)


def test_four_named_bridges_present() -> None:
    """Verify all 4 required future work bridges are defined with required fields."""
    cards = get_bridge_cards()
    assert len(cards) == 4
    bridge_ids = {c.bridge_id for c in cards}
    assert bridge_ids == {
        "carla_cross_simulator_bridge",
        "route_choice_homotopy_observability",
        "incident_to_scenario_provenance",
        "amv_actuation_realism_bridge",
    }
    for card in cards:
        assert card.schema == SCHEMA
        assert card.card_digest == card.compute_digest()
        assert len(card.implemented_now) >= 1
        assert len(card.verified_now) >= 1
        assert len(card.missing_proof) >= 1
        assert len(card.forbidden_inferences) >= 1
        assert "requirements" in card.next_decisive_experiment
        assert card.admission_status in ("diagnostic_only", "not_requested", "blocked")
        validate_safe_sentence(card.safe_sentence)


def test_build_all_creates_valid_cards_and_summary(tmp_path: Path) -> None:
    """Test generating cards and markdown summary into a temporary directory."""
    cards_dir = tmp_path / "cards"
    summary_file = tmp_path / "summary.md"

    result = build_all(cards_dir, summary_file)
    assert result["schema"] == SUMMARY_SCHEMA
    assert result["card_count"] == 4
    assert summary_file.exists()
    assert len(list(cards_dir.glob("*.v1.json"))) == 4

    summary_content = summary_file.read_text(encoding="utf-8")
    assert "# Future-Work Bridge Status Summary" in summary_content
    assert "| Bridge |" in summary_content
    assert "carla_cross_simulator_bridge" in summary_content


def test_check_all_detects_drift(tmp_path: Path) -> None:
    """Test that check_all detects matching vs drifted files."""
    cards_dir = tmp_path / "cards"
    summary_file = tmp_path / "summary.md"

    build_all(cards_dir, summary_file)
    assert check_all(cards_dir, summary_file) is True

    # Mutate a card
    card_path = cards_dir / "carla_cross_simulator_bridge.v1.json"
    data = json.loads(card_path.read_text(encoding="utf-8"))
    data["title"] = "Modified Title"
    card_path.write_text(json.dumps(data), encoding="utf-8")

    assert check_all(cards_dir, summary_file) is False


def test_validate_safe_sentence_fails_on_forbidden_claims() -> None:
    """Verify that unverified overclaims in safe sentences raise ValueError."""
    with pytest.raises(ValueError, match="Unsafe claim detected"):
        validate_safe_sentence("We have achieved proven transfer to real robots.")

    with pytest.raises(ValueError, match="Unsafe claim detected"):
        validate_safe_sentence("This demonstrates a physically realistic model.")

    with pytest.raises(ValueError, match="Unsafe claim detected"):
        validate_safe_sentence("The robot was found legally liable in this scenario.")


def test_cli_check_and_json(tmp_path: Path) -> None:
    """Test CLI execution with --check and --json flags."""
    cards_dir = tmp_path / "cards"
    summary_file = tmp_path / "summary.md"

    # Build via CLI
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cards-dir",
            str(cards_dir),
            "--summary-file",
            str(summary_file),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    res = json.loads(proc.stdout)
    assert res["schema"] == SUMMARY_SCHEMA
    assert res["card_count"] == 4

    # Check via CLI
    proc_check = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--cards-dir",
            str(cards_dir),
            "--summary-file",
            str(summary_file),
            "--check",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_check.returncode == 0
    res_check = json.loads(proc_check.stdout)
    assert res_check["ok"] is True
