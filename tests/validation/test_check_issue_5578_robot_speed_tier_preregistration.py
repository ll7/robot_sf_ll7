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
    assert payload["primary_claim_scope"] == "per_planner_robustness"
    assert payload["ranking_claim_scope"] == "descriptive_only"
    assert payload["inference_contract"]["resampling_unit"] == "paired_seed_block"
    assert payload["robot_speed_axis"]["tiers"][2]["cap_m_s"] == 4.0
    assert (
        payload["robot_speed_axis"]["tiers"][2]["runtime_variant_key"]
        == "bicycle_4_0_mps_micromobility"
    )
    assert payload["inference_contract"]["multiplicity"]["directional_family_alpha"] == 0.025


def test_checker_fails_closed_when_inference_population_is_removed() -> None:
    """A comparative packet without a frozen inference population is not executable evidence."""
    payload = _payload()
    del payload["inference_contract"]["inference_population"]

    with pytest.raises(ValueError, match="inference population must be fixed_declared_suite"):
        validate_preregistration(payload)


def test_checker_fails_closed_when_speed_tier_changes() -> None:
    """Changing a tier silently would invalidate the pre-registered speed contrast."""
    payload = _payload()
    payload["robot_speed_axis"]["tiers"][2]["cap_m_s"] = 4.2

    with pytest.raises(ValueError, match=r"tier\[2\]\.cap_m_s drifted"):
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


def test_checker_fails_closed_when_activation_diagnostics_missing() -> None:
    """Missing required activation diagnostics fails the preregistration check."""
    payload = _payload()
    payload["robot_speed_axis"]["activation_contract"]["required_diagnostics"].remove(
        "fraction_above_2_0_mps"
    )

    with pytest.raises(ValueError, match="required_diagnostics drifted"):
        validate_preregistration(payload)


def test_checker_fails_closed_when_actuation_envelope_inconsistent() -> None:
    """Mathematical inconsistency in stopping_distance_envelope_m fails closed."""
    payload = _payload()
    payload["robot_speed_axis"]["tiers"][2]["stopping_distance_envelope_m"] = 5.0

    with pytest.raises(ValueError, match="stopping_distance_envelope_m drifted"):
        validate_preregistration(payload)


def test_checker_fails_closed_when_ppo_estimand_drifts() -> None:
    """PPO arm must explicitly state zero-shot OOD robustness estimand."""
    payload = _payload()
    ppo_arm = next(a for a in payload["planner_roster"]["arms"] if a["planner_id"] == "ppo")
    ppo_arm["estimand_type"] = "fine_tuned"

    with pytest.raises(ValueError, match="PPO estimand_type must be zero_shot_ood_robustness"):
        validate_preregistration(payload)


def test_checker_fails_closed_when_resampling_unit_drifts() -> None:
    """Resampling unit must be paired_seed_block."""
    payload = _payload()
    payload["inference_contract"]["resampling_unit"] = "unpaired"

    with pytest.raises(ValueError, match="resampling unit must be paired_seed_block"):
        validate_preregistration(payload)


def test_checker_fails_closed_when_safety_notes_missing() -> None:
    """Missing safety interpretation contract notes fails closed."""
    payload = _payload()
    del payload["inference_contract"]["safety_interpretation_contract"]

    with pytest.raises(ValueError, match="safety_interpretation_contract must be a mapping"):
        validate_preregistration(payload)


def test_checker_binds_every_tier_to_exact_supported_runtime_variant() -> None:
    payload = _payload()
    payload["robot_speed_axis"]["tiers"][2]["runtime_variant_key"] = "bicycle_4_2_mps"
    with pytest.raises(ValueError, match=r"tier\[2\]\.runtime_variant_key drifted"):
        validate_preregistration(payload)


@pytest.mark.parametrize(
    ("index", "field", "value", "message"),
    [
        (2, "linear_speed_range_m_s", [99.0, 999.0], "linear_speed_range_m_s"),
        (1, "normalized_action_range", [-1.0, 1.0], "normalized_action_range"),
        (0, "scaling_method", "identity", "scaling_method drifted"),
    ],
)
def test_checker_freezes_exact_command_action_scaling(
    index: int, field: str, value: object, message: str
) -> None:
    payload = _payload()
    payload["robot_speed_axis"]["tiers"][index]["command_action_scaling"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_preregistration(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("min_fraction_above_2_0_mps", 0.0, "must be 0.05"),
        ("min_peak_speed_m_s", 0.0, "must be 2.2"),
    ],
)
def test_checker_freezes_activation_thresholds(field: str, value: float, message: str) -> None:
    payload = _payload()
    payload["robot_speed_axis"]["activation_contract"]["minimum_activation_rule"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_preregistration(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("success_rate_harm_threshold", -0.99, "must be -0.05"),
        ("collision_rate_harm_threshold", 0.99, "must be 0.02"),
        ("near_miss_rate_harm_threshold", 0.99, "must be 0.05"),
    ],
)
def test_checker_freezes_harm_margins(field: str, value: float, message: str) -> None:
    payload = _payload()
    payload["inference_contract"]["decision_rule"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_preregistration(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("interval_method", "central_two_sided", "interval_method"),
        ("bootstrap_replicates", 100, "bootstrap_replicates"),
        ("tail_probability_rule", "zero_effect", "tail_probability_rule"),
    ],
)
def test_checker_freezes_one_sided_inference_contract(
    field: str, value: object, message: str
) -> None:
    payload = _payload()
    payload["inference_contract"]["decision_rule"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_preregistration(payload)


def test_checker_requires_all_exposure_fields_per_cell() -> None:
    payload = _payload()
    payload["result_contract"]["required_cell_keys"].remove("total_exposure_seconds")
    with pytest.raises(ValueError, match="metrics, typed collisions, activation, and exposure"):
        validate_preregistration(payload)


def test_checker_rejects_top_hybrid_roster_role_drift() -> None:
    payload = _payload()
    payload["planner_roster"]["arms"][0]["role"] = "top_hybrid_candidate"
    with pytest.raises(ValueError, match="top_hybrid_promoted"):
        validate_preregistration(payload)
