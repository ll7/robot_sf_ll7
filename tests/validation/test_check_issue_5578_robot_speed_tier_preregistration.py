"""Tests for the issue #5578 robot speed-tier preregistration checker."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import yaml

if TYPE_CHECKING:
    from pathlib import Path

from scripts.validation.check_issue_5578_robot_speed_tier_preregistration import (
    DEFAULT_CONFIG,
    EXPECTED_SEEDS,
    _scenario_ids,
    load_preregistration,
    validate_preregistration,
)


def _payload() -> dict[str, Any]:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_checked_in_preregistration_passes() -> None:
    """The tracked packet freezes all speed, roster, pairing, and inference fields."""
    payload = load_preregistration()

    assert payload["seed_policy"]["seeds"] == EXPECTED_SEEDS
    assert payload["robot_speed_axis"]["baseline_cap_m_s"] == 2.0
    assert payload["result_contract"]["expected_cell_count"] == 2160


def test_checker_fails_closed_when_inference_population_is_removed() -> None:
    """A comparative packet without a frozen inference population is not executable evidence."""
    payload = _payload()
    del payload["inference_contract"]["inference_population"]

    with pytest.raises(ValueError, match="inference population is not frozen"):
        validate_preregistration(payload)


def test_checker_fails_closed_when_speed_tier_changes() -> None:
    """Changing a tier silently would invalidate the pre-registered speed contrast."""
    payload = _payload()
    payload["robot_speed_axis"]["tiers"][2]["cap_m_s"] = 4.0

    with pytest.raises(ValueError, match="speed tiers must be exactly"):
        validate_preregistration(payload)


def test_checker_rejects_transient_host_or_queue_state() -> None:
    """Tracked preregistration must not encode queue routing or target-host state."""
    payload = _payload()
    payload["execution_boundary"]["target_host"] = "imech036"

    with pytest.raises(ValueError, match="transient queue/host routing state"):
        validate_preregistration(payload)


def test_checker_requires_the_complete_typed_collision_output_contract() -> None:
    """Future summaries must retain typed collision rows, not only an aggregate rate."""
    payload = _payload()
    payload["result_contract"]["required_per_tier_summary"].remove("typed_collision_breakdown")

    with pytest.raises(ValueError, match="per-tier output contract incomplete"):
        validate_preregistration(payload)


def test_checker_validates_declared_scenario_sources() -> None:
    """The six selected IDs must resolve to the tracked archetype sources."""
    payload = _payload()
    payload["scenario_contract"]["selected_scenarios"][0]["source_path"] = "missing.yaml"

    with pytest.raises(ValueError, match="missing scenario source"):
        validate_preregistration(payload)


def test_scenario_ids_ignore_null_names(tmp_path: Path) -> None:
    """A null source-row name must not become the literal scenario ID ``None``."""
    source = tmp_path / "scenarios.yaml"
    source.write_text("scenarios:\n  - name:\n  - name: valid_scenario\n", encoding="utf-8")

    assert _scenario_ids(source) == {"valid_scenario"}


def test_checker_requires_non_empty_string_planner_ids() -> None:
    """Roster identity fields must not accept null or coerced planner IDs."""
    payload = _payload()
    payload["planner_roster"]["arms"][0]["planner_id"] = None

    with pytest.raises(ValueError, match="planner_id must be a non-empty string"):
        validate_preregistration(payload)


def test_checker_requires_list_typed_per_tier_summary() -> None:
    """Malformed summary types must fail before set comparison can raise a TypeError."""
    payload = _payload()
    payload["result_contract"]["required_per_tier_summary"] = "collision_rate"

    with pytest.raises(ValueError, match="required_per_tier_summary must be a list"):
        validate_preregistration(payload)
