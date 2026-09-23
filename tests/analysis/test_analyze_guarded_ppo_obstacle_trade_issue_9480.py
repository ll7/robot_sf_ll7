"""Focused tests for the issue #9480 frozen-bundle analyzer."""

from __future__ import annotations

import copy
import hashlib
from typing import TYPE_CHECKING

import pytest

from scripts.analysis.analyze_guarded_ppo_obstacle_trade_issue_9480 import (
    BASE_ARM,
    EXPECTED_ARMS,
    GUARDED_ARM,
    _contact_step,
    _contact_table_row,
    _decision_is_substitution,
    _validate_runtime_row,
    verify_bundle_checksums,
)

if TYPE_CHECKING:
    from pathlib import Path


def _guarded_contact_row() -> dict:
    return {
        "episode_id": "classic_bottleneck_low--111--abc123",
        "scenario_id": "classic_bottleneck_low",
        "seed": 111,
        "metrics": {"obstacle_collision_count": 1},
        "event_ledger": {
            "collision_events": [
                {"collision_partner_type": "static_geometry", "collision_time": 4.3}
            ]
        },
        "scenario_params": {"map_file": "maps/svg_maps/classic_bottleneck_low.svg"},
        "algorithm_metadata": {
            "guard_stats": {
                "fallback_safe": 1,
                "stop_safe": 0,
                "ppo_clear": 4,
                "ppo_safe": 0,
            },
            "shield_stats": {
                "decision_count": 5,
                "intervention_count": 1,
                "override_count": 1,
                "pass_through_count": 4,
                "last_decision": {
                    "decision_label": "fallback_safe",
                    "intervened": True,
                    "override_applied": True,
                },
            },
        },
    }


def _runtime_row(*, status: str = "collision") -> dict:
    expected = EXPECTED_ARMS[GUARDED_ARM]
    return {
        "status": status,
        "algorithm_metadata": {
            "status": "ok",
            "planner_runtime": {
                "checkpoint_provenance": {
                    "model_id": expected["model_id"],
                    "load_succeeded": True,
                    "fallback_triggered": False,
                    "load_status": "loaded",
                    "load_error": None,
                }
            },
        },
        "integrity": {"effective_view": {"degraded": False}},
    }


def test_substitution_uses_intervention_or_override_flags() -> None:
    assert _decision_is_substitution({"intervened": True, "override_applied": False}) is True
    assert _decision_is_substitution({"intervened": False, "override_applied": True}) is True
    assert _decision_is_substitution({"intervened": False, "override_applied": False}) is False


def test_contact_step_uses_zero_based_runtime_step() -> None:
    assert _contact_step(0.1) == 0
    assert _contact_step(57.6) == 575
    with pytest.raises(ValueError, match="outside"):
        _contact_step(0.0)


def test_contact_table_contains_retained_fields_and_explicit_na() -> None:
    table_row = _contact_table_row(_guarded_contact_row(), arm=GUARDED_ARM)
    assert table_row["scenario_cell"] == "classic_bottleneck_low"
    assert table_row["map_file"].endswith("classic_bottleneck_low.svg")
    assert table_row["seed"] == 111
    assert table_row["contact_time_s"] == 4.3
    assert table_row["contact_step"] == 42
    assert table_row["final_guard_substitution"] is True
    assert table_row["final_guard_pass_through"] is False
    assert table_row["contact_x_m"] == "NA"
    assert table_row["contact_y_m"] == "NA"
    assert table_row["preceding_pedestrian_clearance"] == "NA"
    assert table_row["guard_active_last_k_steps"] == "NA"


def test_runtime_allows_ordinary_outcomes_but_rejects_execution_failures() -> None:
    expected = EXPECTED_ARMS[GUARDED_ARM]
    for status in ("collision", "failure"):
        _validate_runtime_row(_runtime_row(status=status), arm=GUARDED_ARM, expected=expected)
    for status in ("failed", "skipped", "fallback", "degraded", "not_available", "unknown", ""):
        with pytest.raises(ValueError, match="non-success execution status"):
            _validate_runtime_row(_runtime_row(status=status), arm=GUARDED_ARM, expected=expected)

    degraded = _runtime_row()
    degraded["integrity"]["effective_view"]["degraded"] = True
    with pytest.raises(ValueError, match="degraded flag"):
        _validate_runtime_row(degraded, arm=GUARDED_ARM, expected=expected)

    fallback = _runtime_row()
    fallback["algorithm_metadata"]["planner_runtime"]["checkpoint_provenance"][
        "fallback_triggered"
    ] = True
    with pytest.raises(ValueError, match="fallback flag"):
        _validate_runtime_row(fallback, arm=GUARDED_ARM, expected=expected)


def test_payload_checksum_verifier_fails_closed(tmp_path: Path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_bytes(b"frozen payload\n")
    checksum = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "checksums.sha256").write_text(f"{checksum}  payload.txt\n", encoding="utf-8")
    assert verify_bundle_checksums(tmp_path) == (1, payload.stat().st_size)

    corrupted = copy.copy(payload)
    corrupted.write_bytes(b"tampered payload\n")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_bundle_checksums(tmp_path)


def test_base_contact_table_keeps_guard_fields_na() -> None:
    table_row = _contact_table_row(_guarded_contact_row(), arm=BASE_ARM)
    assert table_row["arm"] == "ppo"
    assert table_row["final_guard_label"] == "NA"
    assert table_row["final_guard_intervened"] == "NA"
    assert table_row["final_guard_override_applied"] == "NA"
