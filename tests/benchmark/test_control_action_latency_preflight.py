"""Tests for the control-action-latency sweep preflight (issue #5034)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.control_action_latency_preflight import (
    AXIS_KEY,
    DECISION_BLOCKED,
    DECISION_READY,
    REQUIRED_LATENCY_STEPS,
    SCHEMA_VERSION,
    check_control_action_latency_axis,
    write_control_action_latency_preflight,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = REPO_ROOT / "configs/research/fidelity_sensitivity_v1.yaml"


def _latency_axis() -> dict[str, Any]:
    """Return a control_action_latency axis covering the required steps (PR #5026 schema)."""
    return {
        "key": AXIS_KEY,
        "rationale": "Measure how a reset-safe control-to-actuation queue changes safety outcomes.",
        "variants": [
            {
                "key": "zero_step_nominal",
                "baseline": True,
                "patch": {"sim_config": {"action_latency_steps": 0}},
            },
            {"key": "one_step_100ms", "patch": {"sim_config": {"action_latency_steps": 1}}},
            {"key": "three_step_300ms", "patch": {"sim_config": {"action_latency_steps": 3}}},
        ],
    }


def _config_with_latency_axis() -> dict[str, Any]:
    """Return a minimal config carrying the latency axis (post-#5026 shape)."""
    return {
        "schema_version": "fidelity-sensitivity.v1",
        "issue": 3207,
        "study_id": "unit_test_latency",
        "axes": [
            {
                "key": "integration_timestep",
                "variants": [
                    {"key": "dt_nominal", "baseline": True, "patch": {"dt": 0.1}},
                    {"key": "dt_alt", "patch": {"dt": 0.2}},
                ],
            },
            _latency_axis(),
        ],
    }


def test_axis_absent_blocks_with_dependency_prerequisite() -> None:
    """A config with no latency axis (current main) fails closed and names PR #5026."""
    config = {"axes": [{"key": "integration_timestep", "variants": []}]}
    packet = check_control_action_latency_axis(
        config, config_path="configs/research/fidelity_sensitivity_v1.yaml", git_head="deadbeef"
    )
    assert packet["decision"] == DECISION_BLOCKED
    assert packet["ready"] is False
    assert packet["axis_present"] is False
    assert packet["missing_latency_steps"] == list(REQUIRED_LATENCY_STEPS)
    assert any("#5026" in b for b in packet["blockers"])


def test_axis_present_with_all_steps_is_ready() -> None:
    """Once the latency axis covers 0/1/3 steps, the preflight reports ready."""
    packet = check_control_action_latency_axis(
        _config_with_latency_axis(), config_path="cfg.yaml", git_head="abc123"
    )
    assert packet["decision"] == DECISION_READY
    assert packet["ready"] is True
    assert packet["axis_present"] is True
    assert packet["observed_latency_steps"] == [0, 1, 3]
    assert packet["missing_latency_steps"] == []
    assert packet["blockers"] == []


def test_axis_missing_a_required_step_blocks() -> None:
    """Dropping the 3-step variant fails closed and lists the missing step."""
    config = _config_with_latency_axis()
    latency_axis = next(a for a in config["axes"] if a["key"] == AXIS_KEY)
    latency_axis["variants"] = [
        v for v in latency_axis["variants"] if v["patch"]["sim_config"]["action_latency_steps"] != 3
    ]
    packet = check_control_action_latency_axis(config, config_path="cfg.yaml", git_head="abc123")
    assert packet["decision"] == DECISION_BLOCKED
    assert packet["missing_latency_steps"] == [3]
    assert any("missing required" in b for b in packet["blockers"])


def test_axis_without_baseline_blocks() -> None:
    """A latency axis that marks no baseline variant fails closed."""
    config = _config_with_latency_axis()
    latency_axis = next(a for a in config["axes"] if a["key"] == AXIS_KEY)
    for variant in latency_axis["variants"]:
        variant.pop("baseline", None)
    packet = check_control_action_latency_axis(config, config_path="cfg.yaml", git_head="abc123")
    assert packet["decision"] == DECISION_BLOCKED
    assert any("exactly one baseline" in b for b in packet["blockers"])


def test_live_config_reflects_dependency_state() -> None:
    """The shipped config either blocks (pre-#5026) or is ready (post-#5026), never silent.

    This asserts the preflight tracks the real dependency state rather than a
    fixed expectation: while PR #5026 is unmerged the axis is absent and the
    packet must be blocked; once merged it must be ready with all required steps.
    """
    config = yaml.safe_load(REAL_CONFIG.read_text(encoding="utf-8")) or {}
    packet = check_control_action_latency_axis(
        config,
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="unknown",
    )
    if packet["axis_present"]:
        assert packet["decision"] == DECISION_READY
        assert packet["missing_latency_steps"] == []
    else:
        assert packet["decision"] == DECISION_BLOCKED
        assert any("#5026" in b for b in packet["blockers"])


def test_packet_is_serializable_and_written(tmp_path: Path) -> None:
    """The packet is JSON-serializable and written deterministically."""
    packet = check_control_action_latency_axis(
        _config_with_latency_axis(), config_path="cfg.yaml", git_head="abc123"
    )
    assert packet["schema_version"] == SCHEMA_VERSION
    # Round-trips through JSON without loss.
    assert json.loads(json.dumps(packet)) == packet
    path = write_control_action_latency_preflight(packet, tmp_path)
    assert path.exists()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written == copy.deepcopy(packet)
