"""Tests for the dummy learned local-policy adapter fixture."""

from __future__ import annotations

import pytest

from robot_sf.planner.learned_policy_adapter import (
    DummyLearnedLocalPolicyAdapter,
    LearnedPolicyAdapterContractError,
)
from scripts.validation.check_learned_policy_eligibility import (
    validate_learned_policy_eligibility,
)


def _lidar_observation() -> dict[str, object]:
    """Return the minimal LiDAR-style observation expected by the dummy adapter."""
    return {
        "drive_state": [0.0, 0.0, 0.0, 0.0, 0.0],
        "rays": [2.0, 1.5, 1.0, 1.5, 2.0],
    }


def test_dummy_learned_policy_metadata_declares_non_benchmark_fixture_boundary() -> None:
    """The dummy adapter should expose checklist-style metadata without overclaiming."""
    adapter = DummyLearnedLocalPolicyAdapter()

    metadata = adapter.metadata()

    assert metadata["policy_id"] == "dummy_learned_local_policy_adapter"
    assert metadata["claim_boundary"] == "adapter_fixture_only_not_benchmark_evidence"
    assert metadata["verdict"] == "eligible_for_adapter"
    assert metadata["observation_t"] == "decision step t before action selection"
    assert metadata["observation_contract"]["observation_level"] == "lidar_2d"
    assert metadata["observation_contract"]["required_inputs"] == ["drive_state", "rays"]
    assert metadata["observation_fields"]["deployment_observable"] == ["drive_state", "rays"]
    assert metadata["action_contract"]["command_space"] == "unicycle_vw"
    assert metadata["action_contract"]["output_keys"] == ["v", "omega"]
    assert metadata["candidate_registry"]["entry_planned"] is False


def test_dummy_learned_policy_metadata_passes_eligibility_helper() -> None:
    """The fixture metadata should satisfy the learned-policy checklist helper."""
    adapter = DummyLearnedLocalPolicyAdapter()

    assert validate_learned_policy_eligibility(adapter.metadata()) == []


def test_dummy_learned_policy_predicts_deterministic_action_with_logging() -> None:
    """The fixture should emit one predictable action and the required step diagnostics."""
    adapter = DummyLearnedLocalPolicyAdapter()

    result = adapter.predict(_lidar_observation())

    assert result.action == {"v": 0.25, "omega": 0.0}
    assert result.raw_model_action == {"v": 0.25, "omega": 0.0}
    assert result.adapted_action == {"v": 0.25, "omega": 0.0}
    assert result.post_guard_action == {"v": 0.25, "omega": 0.0}
    assert result.guard_applied is False
    assert result.guard_or_fallback_reason == "none"
    assert result.observation_level == "lidar_2d"
    assert result.planner_observation_mode == "sensor_fusion_state"
    assert result.action_bounds == {"v": [0.0, 0.5], "omega": [-0.5, 0.5]}
    assert result.action_projection_metadata == {
        "projected": False,
        "projection_policy": "none",
    }
    assert result.to_metadata()["post_guard_action"] == {"v": 0.25, "omega": 0.0}
    assert result.to_metadata()["action_bounds"] == {"v": [0.0, 0.5], "omega": [-0.5, 0.5]}


def test_dummy_learned_policy_plan_returns_unicycle_tuple_for_planner_smokes() -> None:
    """Planner-style smoke tests can consume the deterministic command tuple directly."""
    adapter = DummyLearnedLocalPolicyAdapter()

    assert adapter.plan(_lidar_observation()) == (0.25, 0.0)
    assert adapter.step(_lidar_observation()) == {"v": 0.25, "omega": 0.0}
    adapter.reset(seed=7)
    adapter.configure({})
    adapter.configure(None)
    adapter.close()


def test_dummy_learned_policy_fails_closed_for_none_observation() -> None:
    """None observations should fail closed before returning an action."""
    adapter = DummyLearnedLocalPolicyAdapter()

    with pytest.raises(LearnedPolicyAdapterContractError, match="observation must not be None"):
        adapter.predict(None)

    with pytest.raises(LearnedPolicyAdapterContractError, match="observation must not be None"):
        adapter.plan(None)


@pytest.mark.parametrize(
    ("observation", "match"),
    [
        ({"drive_state": [0.0]}, "missing required observation inputs: rays"),
        ({"drive_state": [0.0], "rays": [], "future_states": []}, "forbidden"),
    ],
)
def test_dummy_learned_policy_fails_closed_for_bad_observation(
    observation: dict[str, object],
    match: str,
) -> None:
    """Unsupported observation payloads should fail before returning an action."""
    adapter = DummyLearnedLocalPolicyAdapter()

    with pytest.raises(LearnedPolicyAdapterContractError, match=match):
        adapter.predict(observation)

    with pytest.raises(LearnedPolicyAdapterContractError, match=match):
        adapter.plan(observation)


def test_dummy_learned_policy_fails_closed_for_unsupported_request() -> None:
    """Unsupported observation levels or action spaces should fail closed."""
    adapter = DummyLearnedLocalPolicyAdapter()

    with pytest.raises(LearnedPolicyAdapterContractError, match="observation_level"):
        adapter.predict(_lidar_observation(), observation_level="oracle_full_state")

    with pytest.raises(LearnedPolicyAdapterContractError, match="observation_level"):
        adapter.predict(_lidar_observation(), observation_level="")

    with pytest.raises(LearnedPolicyAdapterContractError, match="action_command_space"):
        adapter.predict(_lidar_observation(), action_command_space="holonomic_vxy_world")

    with pytest.raises(LearnedPolicyAdapterContractError, match="action_command_space"):
        adapter.predict(_lidar_observation(), action_command_space="")


def test_dummy_learned_policy_fails_closed_for_runtime_configuration() -> None:
    """The fixture should not accept runtime reconfiguration that changes its contract."""
    adapter = DummyLearnedLocalPolicyAdapter()

    with pytest.raises(LearnedPolicyAdapterContractError, match="runtime configuration"):
        adapter.configure({"action": "turn"})
