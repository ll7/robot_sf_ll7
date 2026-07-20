"""Regression guard for the fail-closed continue-patch retry contract (#6008)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = ROOT / "docs/context/issue_6008_state.yaml"


def _contract() -> dict:
    state = yaml.safe_load(STATE_PATH.read_text(encoding="utf-8"))
    assert state["issue"] == 6008
    assert state["schema"] == "robot_sf.issue_state.v1"
    assert len(state["entries"]) >= 1
    return state["entries"][-1]["reconciliation_contract"]


def test_continue_patch_contract_fails_closed_for_stale_or_ambiguous_retries() -> None:
    """Missing, merged, and path-only retries cannot create duplicate work."""
    contract = _contract()
    decisions = contract["decisions"]

    assert contract["applies_when"] == "retry.kind == continue_patch before worker dispatch"
    assert decisions["patch_already_merged"] == {
        "dispatch": False,
        "terminal_state": "superseded",
        "receipt": "records merged evidence and selects no duplicate continuation",
    }
    assert decisions["patch_reference_missing"]["dispatch"] is False
    assert decisions["patch_reference_missing"]["terminal_state"] == "blocked"
    assert decisions["patch_identity_indeterminate"]["dispatch"] is False
    assert decisions["patch_identity_indeterminate"]["terminal_state"] == "blocked"
    assert (
        "A path-only continue_patch retry is never enough authority to dispatch."
        in contract["invariants"]
    )


def test_continue_patch_contract_allows_only_an_identified_unmerged_patch() -> None:
    """The positive lane remains available after the stale-retry guard is added."""
    contract = _contract()
    available = contract["decisions"]["patch_available_and_unmerged"]

    assert available["dispatch"] is True
    assert "identity and changed paths" in available["receipt"]
    assert contract["required_identity"] == [
        "stable patch identity (commit, content digest, or immutable review reference)",
        "declared changed-path set or equivalent patch manifest",
    ]
