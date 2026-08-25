"""Focused deterministic-smoke tests for the stateful RecurrentPPO adapter.

The tests exercise the issue #7848 local-planner contract with a stubbed
checkpoint: fail-closed observation and checkpoint validation, recurrent
hidden-state lifecycle across steps, reset at episode boundaries, action bounds
clipping, raw-command observability, and the canonical metadata checklist.
No real checkpoint or training run is involved.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from robot_sf.planner.learned_policy_adapter import LearnedPolicyAdapterContractError
from robot_sf.planner.recurrent_ppo_learned_adapter import (
    RecurrentPPOLearnedAdapterConfig,
    RecurrentPPOLearnedLocalPolicyAdapter,
    build_recurrent_ppo_adapter_config,
)

LSTM_STATE_NUMERIC = (np.zeros((2, 16), dtype=np.float32), np.zeros((2, 16), dtype=np.float32))


class _FakeRecurrentPPOModel:
    """Deterministic RecurrentPPO stand-in recording predict call state."""

    def __init__(self) -> None:
        self.observation_space = spaces.Dict(
            {
                "drive_state": spaces.Box(low=-10.0, high=10.0, shape=(7,), dtype=np.float32),
                "rays": spaces.Box(low=0.0, high=15.0, shape=(64,), dtype=np.float32),
            }
        )
        self.predict_calls: list[dict] = []

    def predict(
        self,
        observation: dict,
        *,
        state=None,
        episode_start=None,
        deterministic: bool = False,
    ):
        self.predict_calls.append(
            {
                "state": state,
                "episode_start": episode_start,
                "deterministic": deterministic,
            }
        )
        action = np.asarray([0.3, -0.2], dtype=float)
        return action, LSTM_STATE_NUMERIC


class _WrongSpaceModel:
    """Checkpoint stub whose observation space lacks drive_state/rays."""

    observation_space = spaces.Dict({"occupancy_grid": spaces.Box(0, 1, (4, 4))})

    def predict(self, *_args, **_kwargs):  # pragma: no cover - never reached
        raise AssertionError("predict must not be reached")


def _obs(drive_state: object | None = None, *, extra: dict | None = None) -> dict:
    payload = {
        "drive_state": np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        if drive_state is None
        else drive_state,
        "rays": np.zeros(64, dtype=np.float32) + 0.5,
    }
    if extra:
        payload.update(extra)
    return payload


@pytest.fixture()
def adapter(monkeypatch) -> RecurrentPPOLearnedLocalPolicyAdapter:
    """Adapter bound to the deterministic fake model."""
    inst = RecurrentPPOLearnedLocalPolicyAdapter(
        RecurrentPPOLearnedAdapterConfig(
            model_path="tests/fixtures/stub_recurrent_ppo.zip",
            deterministic=True,
        ),
        defer_model_loading=True,
    )
    inst._model = _FakeRecurrentPPOModel()
    inst._initialized = True
    return inst


def test_missing_required_observation_inputs_fail_closed(adapter) -> None:
    """A missing drive_state or rays key must raise before any prediction."""
    with pytest.raises(LearnedPolicyAdapterContractError, match="missing required"):
        adapter.predict({"drive_state": _obs()["drive_state"]})
    with pytest.raises(LearnedPolicyAdapterContractError, match="missing required"):
        adapter.predict({"rays": _obs()["rays"]})


def test_forbidden_future_inputs_fail_closed(adapter) -> None:
    """Leaked future/trajectory inputs must be rejected at evaluation time."""
    observation = _obs(extra={"future_trajectory": np.zeros((10, 2))})
    with pytest.raises(LearnedPolicyAdapterContractError, match="forbidden"):
        adapter.predict(observation)


def test_nonfinite_observation_payload_fails_closed(adapter) -> None:
    """NaN or inf ray payloads must fail closed, not reach the model."""
    with pytest.raises(LearnedPolicyAdapterContractError, match="non-finite"):
        adapter.predict(_obs(drive_state=np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])))


def test_missing_checkpoint_fails_closed_without_fallback(monkeypatch, tmp_path) -> None:
    """A corrupt checkpoint must raise; no silent fallback to goal-seeking."""
    bad_checkpoint = tmp_path / "broken_recurrent_ppo.zip"
    bad_checkpoint.write_bytes(b"not a real checkpoint")

    monkeypatch.setattr(
        "robot_sf.planner.recurrent_ppo_learned_adapter.RecurrentPPO.load",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("checkpoint corrupt")),
    )
    with pytest.raises(LearnedPolicyAdapterContractError, match="failed to load"):
        RecurrentPPOLearnedLocalPolicyAdapter(
            RecurrentPPOLearnedAdapterConfig(model_path=str(bad_checkpoint))
        )


def test_checkpoint_observation_contract_fails_closed_for_wrong_keys() -> None:
    """A checkpoint that does not declare drive_state/rays must be rejected."""
    inst = RecurrentPPOLearnedLocalPolicyAdapter(
        RecurrentPPOLearnedAdapterConfig(model_path="x.zip"),
        defer_model_loading=True,
    )
    inst._model = _WrongSpaceModel()
    inst._initialized = True
    with pytest.raises(LearnedPolicyAdapterContractError, match="does not declare"):
        inst._validate_model_observation_contract()


def test_config_bounds_validation() -> None:
    """Non-finite or implausible action bounds must fail at construction."""
    with pytest.raises(ValueError, match="v_max"):
        RecurrentPPOLearnedAdapterConfig(v_max=-1.0)
    with pytest.raises(ValueError, match="omega_max"):
        RecurrentPPOLearnedAdapterConfig(omega_max=float("nan"))
    with pytest.raises(ValueError, match="plausibility"):
        RecurrentPPOLearnedAdapterConfig(v_max=0.1, omega_max=2.0)


def test_recurrent_state_lifecycle(adapter) -> None:
    """LSTM state is carried across steps and reset at episode boundaries."""
    result1 = adapter.predict(_obs())
    fake = adapter._model
    assert fake.predict_calls[0]["episode_start"] is True
    assert fake.predict_calls[0]["state"] is None
    assert result1.action_projection_metadata["recurrent_state_carried"] is True

    adapter.predict(_obs())
    assert fake.predict_calls[1]["episode_start"] is False
    assert fake.predict_calls[1]["state"] is LSTM_STATE_NUMERIC

    adapter.reset()
    assert adapter._lstm_states is None
    assert adapter._episode_start is True

    adapter.predict(_obs())
    assert fake.predict_calls[2]["episode_start"] is True


def test_deterministic_inference_default(adapter) -> None:
    """Inference must default to deterministic mode."""
    adapter.predict(_obs())
    assert adapter._model.predict_calls[0]["deterministic"] is True


def test_action_bounds_clipping(adapter) -> None:
    """Raw actions outside the configured bounds are clipped to the command space."""
    fake = adapter._model
    original = fake.predict

    def _big_action(*_args, **_kwargs):
        return np.asarray([5.0, -5.0], dtype=float), LSTM_STATE_NUMERIC

    fake.predict = _big_action
    result = adapter.predict(_obs())
    assert result.action["v"] == 0.5
    assert result.action["omega"] == -0.5
    assert result.raw_model_action == {"v": 5.0, "omega": -5.0}
    fake.predict = original


def test_raw_command_observability(adapter) -> None:
    """The raw model command must be observable separate from the adapted one."""
    result = adapter.predict(_obs())
    assert set(result.raw_model_action) == {"v", "omega"}
    assert result.post_guard_action == result.adapted_action
    assert result.guard_applied is False
    assert "inference_latency_ms" in result.action_projection_metadata


def test_plan_and_step_protocol(adapter) -> None:
    """The planner protocol plan/step surfaces return the bounded unicycle command."""
    command = adapter.plan(_obs())
    assert isinstance(command, tuple)
    assert len(command) == 2
    step_action = adapter.step(_obs())
    assert set(step_action) == {"v", "omega"}


def test_metadata_contract_checklist(adapter) -> None:
    """Metadata mirrors the canonical learned-policy checklist with real values."""
    meta = adapter.metadata()
    assert meta["policy_id"] == "recurrent_ppo_lstm_default_gym_v1"
    assert meta["claim_boundary"] == "implementation_smoke_only_not_benchmark_evidence"
    action = meta["action_contract"]
    assert action["command_space"] == "unicycle_vw"
    assert action["output_keys"] == ["v", "omega"]
    assert action["units"] == "m/s and rad/s"
    assert meta["recurrent_contract"]["deterministic_inference"] is True
    assert meta["recurrent_contract"]["reset_at_episode_boundary"] is True
    assert "future_trajectory" in meta["observation_contract"]["rejected_evaluation_time_inputs"]


def test_diagnostics_latency_telemetry(adapter) -> None:
    """Diagnostics expose inference latency statistics after steps."""
    adapter.predict(_obs())
    adapter.predict(_obs())
    diagnostics = adapter.diagnostics()
    assert diagnostics["checkpoint_loaded"] is True
    assert diagnostics["recurrent_state_carried"] is True
    assert diagnostics["inference_latency_ms"]["samples"] == 2
    assert diagnostics["inference_latency_ms"]["mean"] >= 0.0


def test_reset_accounting_tracks_reason_and_count(adapter) -> None:
    """Reset records the boundary reason and increments the reset count."""
    adapter.predict(_obs())
    assert adapter.diagnostics()["reset_count"] == 0
    adapter.reset()
    diagnostics = adapter.diagnostics()
    assert diagnostics["reset_count"] == 1
    assert diagnostics["last_reset_reason"] == "explicit_reset"
    assert diagnostics["episode_start"] is True
    adapter.reset_state(reason="episode_terminated")
    diagnostics = adapter.diagnostics()
    assert diagnostics["reset_count"] == 2
    assert diagnostics["last_reset_reason"] == "episode_terminated"
    adapter.reset_state(reason="scenario_replaced")
    assert adapter.diagnostics()["reset_count"] == 3


def test_reset_with_seed_sets_sequence_identity(adapter) -> None:
    """A seeded reset binds the sequence identity for determinism checks."""
    assert adapter.diagnostics()["sequence_identity"] is None
    adapter.reset(seed=42)
    assert adapter.diagnostics()["sequence_identity"] == 42


def test_inference_call_count_and_state_shape(adapter) -> None:
    """Inference call count increments and state shape is reported boundedly."""
    adapter.predict(_obs())
    diagnostics = adapter.diagnostics()
    assert diagnostics["inference_call_count"] == 1
    assert diagnostics["recurrent_state_shape"] == [[2, 16], [2, 16]]
    assert diagnostics["recurrent_state_finite"] is True


def test_diagnostics_with_observation_reports_raw_command_and_validation(adapter) -> None:
    """Passing an observation reports validation status and raw (v, omega)."""
    diagnostics = adapter.diagnostics(observation=_obs())
    assert diagnostics["observation_validation_status"] == "validated"
    assert diagnostics["raw_desired_command"]["v"] == 0.3
    assert diagnostics["raw_desired_command"]["omega"] == -0.2


def test_invalid_observation_reported_in_diagnostics(adapter) -> None:
    """A contract-violating observation reports invalid, not validated."""
    diagnostics = adapter.diagnostics(observation={"drive_state": np.zeros(7)})
    assert diagnostics["observation_validation_status"] == "invalid"


def test_build_config_from_mapping() -> None:
    """The mapping builder selects only supported config keys."""
    config = build_recurrent_ppo_adapter_config({"v_max": 0.7, "omega_max": 0.9, "unrecognized": 1})
    assert config.v_max == 0.7
    assert config.omega_max == 0.9
    assert not hasattr(config, "unrecognized")
