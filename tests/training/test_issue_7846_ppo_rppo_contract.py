"""Focused contract tests for the #7846 PPO vs RecurrentPPO successor contract.

The successor contract freezes an executable matched comparison: both arms must
use the ``default_gym`` observation (the historical nested ``socnav_struct``
recurrent arm was rejected by Stable-Baselines3 before learning), the recurrent
arm must be the matched executable config, and the historical #4244
preregistration must remain byte-for-byte unmutated. These tests give the
freeze a deterministic gate without mutating the #4244-specific validator.
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

SUCCESSOR_CONTRACT = (
    REPO_ROOT
    / "configs"
    / "training"
    / "comparison_matrix"
    / "issue_7846_ppo_rppo_contract_v1.yaml"
)
HISTORICAL_SEVEN_ARM = (
    REPO_ROOT
    / "configs"
    / "training"
    / "comparison_matrix"
    / "issue_4244_seven_arm_preregistration.yaml"
)
RECURRENT_MATCHED = (
    REPO_ROOT / "configs" / "training" / "ppo" / "issue_4014_recurrent_ppo_lstm_smoke_matched.yaml"
)
PPO_MATCHED = REPO_ROOT / "configs" / "training" / "ppo" / "issue_4014_ppo_smoke_matched.yaml"
BROKEN_RECURRENT = (
    REPO_ROOT / "configs" / "training" / "ppo" / "issue_4014_ppo_lstm_recurrent_smoke.yaml"
)


def _load_yaml(path: Path) -> dict:
    assert path.is_file(), f"missing contract file: {path}"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"{path} must be a YAML mapping"
    return payload


def test_successor_contract_is_loaded() -> None:
    """The successor contract file exists and parses as a mapping."""
    payload = _load_yaml(SUCCESSOR_CONTRACT)
    assert payload["issue"] == 7846
    assert payload["schema_version"] == "training_comparison_matrix_preregistration.v1"


def test_successor_contract_has_exactly_two_matched_arms() -> None:
    """The freeze must contain exactly the matched PPO and RecurrentPPO arms."""
    payload = _load_yaml(SUCCESSOR_CONTRACT)
    arms = payload["arms"]
    assert [arm["id"] for arm in arms] == [
        "ppo_default_gym_control_v1",
        "recurrent_ppo_lstm_default_gym_v1",
    ]
    assert payload["comparison"]["arms_expected"] == 2


def test_successor_arms_use_default_gym_observation() -> None:
    """Both arms must resolve to the executable default_gym observation."""
    payload = _load_yaml(SUCCESSOR_CONTRACT)
    for arm in payload["arms"]:
        assert arm["observation_mode"] == "default_gym", arm["id"]


def test_recurrent_arm_points_at_matched_executable_config() -> None:
    """The recurrent arm must reference the matched default_gym config."""
    payload = _load_yaml(SUCCESSOR_CONTRACT)
    recurrent = next(arm for arm in payload["arms"] if arm["id"].startswith("recurrent_ppo_lstm"))
    assert recurrent["training_config"] == (
        "configs/training/ppo/issue_4014_recurrent_ppo_lstm_smoke_matched.yaml"
    )
    matched = _load_yaml(RECURRENT_MATCHED)
    assert matched["env_overrides"]["observation_mode"] == "default_gym"
    assert matched["recurrent_policy"] == "MultiInputLstmPolicy"


def test_feedforward_arm_points_at_matched_config() -> None:
    """The feed-forward arm must reference the matched default_gym PPO config."""
    payload = _load_yaml(SUCCESSOR_CONTRACT)
    ppo = next(arm for arm in payload["arms"] if arm["id"].startswith("ppo_default_gym"))
    assert ppo["training_config"] == "configs/training/ppo/issue_4014_ppo_smoke_matched.yaml"
    matched = _load_yaml(PPO_MATCHED)
    assert matched["env_overrides"]["observation_mode"] == "default_gym"


def test_broken_recurrent_arm_is_not_referenced_by_successor() -> None:
    """The nested socnav_struct recurrent arm stays historical and unreferenced."""
    payload = _load_yaml(SUCCESSOR_CONTRACT)
    referenced = [arm["training_config"] for arm in payload["arms"]]
    assert BROKEN_RECURRENT.as_posix() not in referenced
    broken = _load_yaml(BROKEN_RECURRENT)
    assert broken["env_overrides"]["observation_mode"] == "socnav_struct"


def test_successor_budget_is_frozen_at_full_matched_budget() -> None:
    """The successor fixes the #4244-matched 15M budget and five seeds."""
    payload = _load_yaml(SUCCESSOR_CONTRACT)
    budget = payload["shared_budget"]
    assert budget["total_timesteps"] == 15_000_000
    assert budget["seeds"] == [123, 231, 777, 992, 1337]
    assert budget["per_seed_independent_config_and_job"] is True


def test_successor_fixes_guardrails_and_metrics() -> None:
    """Collision regression must not be compensated by completion gains."""
    payload = _load_yaml(SUCCESSOR_CONTRACT)
    guardrails = payload["comparison"]["guardrails"]
    assert guardrails["success_improvement_cannot_compensate_collision_regression"] is True
    assert "collision_probability" in payload["comparison"]["metrics"]["secondary"]
    assert payload["comparison"]["safety_wrapper_enabled"] is False


def test_historical_seven_arm_preregistration_is_unmutated() -> None:
    """The frozen #4244 file must be byte-identical to what the runner expects."""
    payload = _load_yaml(HISTORICAL_SEVEN_ARM)
    assert payload["issue"] == 4244
    assert len(payload["arms"]) == 7
    historic_recurrent = next(arm for arm in payload["arms"] if arm["id"] == "recurrent_ppo_lstm")
    assert historic_recurrent["training_config"].endswith(
        "issue_4014_ppo_lstm_recurrent_smoke.yaml"
    )


def test_successor_contract_claims_no_execution_authority() -> None:
    """The freeze must not authorize training or claim evidence."""
    payload = _load_yaml(SUCCESSOR_CONTRACT)
    claim_boundary = str(payload["claim_boundary"]).lower()
    assert "not" in claim_boundary and "authorize" in claim_boundary
    assert payload["queue_plan"]["submit_in_this_pr"] is False
