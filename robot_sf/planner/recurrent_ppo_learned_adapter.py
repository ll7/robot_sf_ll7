"""Stateful RecurrentPPO learned local-policy adapter.

This adapter implements the issue #7848 local-planner contract on top of the
canonical learned-policy boundary in ``learned_policy_adapter.py``.  It loads a
checkpoint trained by ``sb3_contrib.RecurrentPPO`` under the #7846
``default_gym`` observation contract (``drive_state`` + ``rays``), preserves
the LSTM hidden/cell state across control steps, resets it at every episode
boundary, validates observations fail closed, and emits the desired unicycle
command ``(v, omega)`` with full raw-command observability.

It intentionally owns no safety wrapper, trains nothing, changes no planner
default, and does not duplicate the velocity-to-acceleration conversion; the
canonical ``policy_command_to_env_action`` path performs that conversion during
environment execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from robot_sf.baselines.ppo import _configure_torch_213_runtime
from robot_sf.benchmark.local_model_artifacts import validate_no_local_model_path_value
from robot_sf.common.optional_import import try_import
from robot_sf.models import resolve_model_path
from robot_sf.planner.learned_policy_adapter import (
    LearnedPolicyAdapterContractError,
    LearnedPolicyStepResult,
)

_configure_torch_213_runtime()

try:  # Lazy import; not required for type-check only
    from sb3_contrib import RecurrentPPO
except ImportError:  # pragma: no cover - envs without training deps installed
    RecurrentPPO = None  # type: ignore


@dataclass
class RecurrentPPOLearnedAdapterConfig:
    """Configuration for the stateful RecurrentPPO learned adapter."""

    model_path: str = "model/recurrent_ppo_lstm_contract_v1.zip"
    model_id: str | None = None
    device: str = "auto"
    deterministic: bool = True
    v_max: float = 0.5
    omega_max: float = 0.5
    observation_level: str = "default_gym"
    action_command_space: str = "unicycle_vw"
    required_observation_inputs: tuple[str, ...] = ("drive_state", "rays")
    forbidden_observation_inputs: tuple[str, ...] = (
        "future_states",
        "future_trajectory",
        "future_collision_label",
        "simulator_outcome",
        "termination_reason",
    )

    def __post_init__(self) -> None:
        """Reject non-finite or invalid bounds fail closed at construction."""
        if not np.isfinite(float(self.v_max)) or float(self.v_max) < 0.0:
            raise ValueError("v_max must be a finite non-negative value")
        if not np.isfinite(float(self.omega_max)) or float(self.omega_max) < 0.0:
            raise ValueError("omega_max must be a finite non-negative value")
        if float(self.v_max) > 0.0 and (float(self.omega_max) > float(self.v_max) * 4.0):
            raise ValueError(
                "omega_max exceeds a plausibility guard of 4x v_max; check action bounds"
            )


class RecurrentPPOLearnedLocalPolicyAdapter:
    """Load one registered RecurrentPPO checkpoint and own its recurrent state.

    The adapter is fail closed: a missing, ambiguous, rights-blocked, or
    incompatible checkpoint raises instead of silently falling back to another
    policy or to goal-seeking.  Recurrent state is held across control steps and
    cleared on ``reset``; deterministic inference is the default.
    """

    policy_id = "recurrent_ppo_lstm_default_gym_v1"
    claim_boundary = "implementation_smoke_only_not_benchmark_evidence"

    def __init__(
        self,
        config: RecurrentPPOLearnedAdapterConfig | dict[str, Any] | None = None,
        *,
        seed: int | None = None,
        defer_model_loading: bool = False,
    ) -> None:
        """Initialize the adapter and load the checkpoint unless deferred."""
        torch = try_import("torch")
        if torch is not None:
            torch.set_num_threads(1)
        self.config = self._parse_config(config)
        self._seed = seed
        self._model = None
        self._status = "ok"
        self._lstm_states: Any = None
        self._episode_start = True
        self._defer_model_loading = defer_model_loading
        self._initialized = False
        self._latency_samples_ms: list[float] = []
        self._inference_calls = 0
        self._reset_count = 0
        self._last_reset_reason: str | None = None
        self._state_shape_drift_count = 0
        self._sequence_identity: int | None = None
        if not defer_model_loading:
            self._ensure_model_loaded()

    def _parse_config(
        self, cfg: RecurrentPPOLearnedAdapterConfig | dict[str, Any] | None
    ) -> RecurrentPPOLearnedAdapterConfig:
        """Normalize config input into the dataclass form.

        Returns:
            RecurrentPPOLearnedAdapterConfig: Parsed adapter configuration.
        """
        if cfg is None:
            return RecurrentPPOLearnedAdapterConfig()
        if isinstance(cfg, RecurrentPPOLearnedAdapterConfig):
            return cfg
        if isinstance(cfg, dict):
            return RecurrentPPOLearnedAdapterConfig(**cfg)
        raise TypeError(f"Invalid config type: {type(cfg)}")

    # --- Model loading -------------------------------------------------
    def _load_model(self) -> None:
        """Resolve and load one RecurrentPPO checkpoint, failing closed on defects."""
        if RecurrentPPO is None:
            raise LearnedPolicyAdapterContractError(
                "sb3_contrib is not installed; the RecurrentPPO checkpoint cannot be loaded"
            )
        if self.config.model_id is None:
            validate_no_local_model_path_value(
                self.config.model_path,
                owner="RecurrentPPOLearnedAdapterConfig",
            )
        try:
            resolved = (
                resolve_model_path(self.config.model_id)
                if self.config.model_id
                else Path(self.config.model_path)
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            raise LearnedPolicyAdapterContractError(
                f"recurrent checkpoint resolution failed: {exc}"
            ) from exc
        if not resolved.exists():
            raise LearnedPolicyAdapterContractError(f"recurrent checkpoint not found: {resolved}")
        try:
            model = RecurrentPPO.load(
                str(resolved),
                device=self.config.device,
                print_system_info=False,
            )
        except (RuntimeError, ValueError, OSError) as exc:
            raise LearnedPolicyAdapterContractError(
                f"failed to load recurrent checkpoint {resolved}: {exc}"
            ) from exc
        self._model = model
        self._validate_model_observation_contract()
        self._status = "ok"

    def _validate_model_observation_contract(self) -> None:
        """Fail closed when the checkpoint does not declare the default_gym keys."""
        observation_space = getattr(self._model, "observation_space", None)
        spaces = getattr(observation_space, "spaces", None)
        if not isinstance(spaces, dict):
            raise LearnedPolicyAdapterContractError(
                "recurrent checkpoint observation_space must be a Dict with "
                f"drive_state and rays; got {type(observation_space).__name__}"
            )
        declared = {str(key) for key in spaces}
        missing = [key for key in self.config.required_observation_inputs if key not in declared]
        if missing:
            raise LearnedPolicyAdapterContractError(
                "recurrent checkpoint does not declare required observation keys: "
                + ", ".join(missing)
            )

    def _ensure_model_loaded(self) -> None:
        """Lazily load the model exactly once unless the config changed."""
        if getattr(self, "_initialized", False):
            return
        if self._model is None:
            self._load_model()
        self._initialized = True

    # --- Observation validation ---------------------------------------
    def _validate_request(
        self,
        observation: Mapping[str, Any],
        *,
        observation_level: str,
        action_command_space: str,
    ) -> None:
        """Validate the observation/action contract before any prediction."""
        if observation_level != self.config.observation_level:
            raise LearnedPolicyAdapterContractError(
                "unsupported observation_level "
                f"'{observation_level}'; expected '{self.config.observation_level}'"
            )
        if action_command_space != self.config.action_command_space:
            raise LearnedPolicyAdapterContractError(
                "unsupported action_command_space "
                f"'{action_command_space}'; expected '{self.config.action_command_space}'"
            )
        if observation is None or not isinstance(observation, Mapping):
            raise LearnedPolicyAdapterContractError("observation must be a mapping")
        missing = [key for key in self.config.required_observation_inputs if key not in observation]
        if missing:
            raise LearnedPolicyAdapterContractError(
                "missing required observation inputs: " + ", ".join(missing)
            )
        forbidden = [key for key in self.config.forbidden_observation_inputs if key in observation]
        if forbidden:
            raise LearnedPolicyAdapterContractError(
                "forbidden evaluation-time observation inputs: " + ", ".join(forbidden)
            )

    @staticmethod
    def _validate_sensor_payload(key: str, value: Any) -> None:
        """Require a finite, well-shaped numeric payload for one observation key."""
        array = np.asarray(value, dtype=float)
        if array.ndim == 0 or array.size == 0:
            raise LearnedPolicyAdapterContractError(
                f"observation '{key}' must be a non-empty array"
            )
        if not np.all(np.isfinite(array)):
            raise LearnedPolicyAdapterContractError(
                f"observation '{key}' contains non-finite values"
            )

    def _validate_observation_payload(self, observation: Mapping[str, Any]) -> None:
        """Fail closed on non-finite or empty sensor payloads."""
        for key in self.config.required_observation_inputs:
            value = observation.get(key)
            if value is None:
                raise LearnedPolicyAdapterContractError(f"observation '{key}' is missing a value")
            self._validate_sensor_payload(key, value)

    # --- Inference -----------------------------------------------------
    def _predict_raw(
        self,
        observation: Mapping[str, Any],
    ) -> np.ndarray:
        """Run deterministic RecurrentPPO inference with the owned recurrent state.

        Returns:
            np.ndarray: Raw model action vector, squeezed to 1-D.
        """
        self._ensure_model_loaded()
        payload = {
            str(key): np.asarray(observation[key], dtype=np.float32)
            for key in self.config.required_observation_inputs
        }
        try:
            action, lstm_states = self._model.predict(
                payload,
                state=self._lstm_states,
                episode_start=self._episode_start,
                deterministic=self.config.deterministic,
            )
        except (RuntimeError, ValueError, OSError, IndexError) as exc:
            raise LearnedPolicyAdapterContractError(
                f"recurrent inference failed at step {exc}"
            ) from exc
        if (
            self._lstm_states is not None
            and lstm_states is not None
            and self._state_shape_mismatch(self._lstm_states, lstm_states)
        ):
            self._state_shape_drift_count += 1
        self._lstm_states = lstm_states
        self._episode_start = False
        self._inference_calls += 1
        return np.asarray(action, dtype=float).reshape(-1)

    @staticmethod
    def _state_shape_mismatch(previous: Any, current: Any) -> bool:
        """Return whether two recurrent state payloads disagree in shape."""
        try:
            prev_shapes = tuple(
                np.asarray(part).shape
                for part in (previous if isinstance(previous, tuple) else [previous])
            )
            curr_shapes = tuple(
                np.asarray(part).shape
                for part in (current if isinstance(current, tuple) else [current])
            )
        except (TypeError, ValueError):
            return True
        return prev_shapes != curr_shapes

    def _command_from_raw(self, raw: np.ndarray) -> dict[str, float]:
        """Map a raw action vector to the bounded unicycle command.

        Returns:
            dict[str, float]: Command with ``v`` and ``omega`` keys in m/s and rad/s.
        """
        v = float(raw[0]) if raw.size >= 1 else 0.0
        omega = float(raw[1]) if raw.size >= 2 else 0.0
        v = float(np.clip(v, 0.0, self.config.v_max))
        omega = float(np.clip(omega, -self.config.omega_max, self.config.omega_max))
        return {"v": v, "omega": omega}

    # --- Public contract ----------------------------------------------
    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        observation_level: str | None = None,
        action_command_space: str | None = None,
    ) -> LearnedPolicyStepResult:
        """Return one step action and its full observability payload."""
        self._validate_request(
            observation,
            observation_level=(
                self.config.observation_level if observation_level is None else observation_level
            ),
            action_command_space=(
                self.config.action_command_space
                if action_command_space is None
                else action_command_space
            ),
        )
        self._validate_observation_payload(observation)
        started = perf_counter()
        raw = self._predict_raw(observation)
        latency_ms = (perf_counter() - started) * 1000.0
        self._latency_samples_ms.append(latency_ms)
        adapted = self._command_from_raw(raw)
        raw_command = {
            "v": float(raw[0]) if raw.size >= 1 else 0.0,
            "omega": float(raw[1]) if raw.size >= 2 else 0.0,
        }
        return LearnedPolicyStepResult(
            action=dict(adapted),
            raw_model_action=raw_command,
            adapted_action=dict(adapted),
            post_guard_action=dict(adapted),
            guard_applied=False,
            guard_or_fallback_reason="none",
            observation_level=self.config.observation_level,
            planner_observation_mode="default_gym",
            action_bounds={
                "v": [0.0, float(self.config.v_max)],
                "omega": [-float(self.config.omega_max), float(self.config.omega_max)],
            },
            action_projection_metadata={
                "projected": True,
                "projection_policy": "clip_to_configured_bounds",
                "recurrent_state_carried": self._lstm_states is not None,
                "episode_start_reset": self._episode_start,
                "inference_latency_ms": latency_ms,
            },
        )

    def plan(self, observation: Mapping[str, Any]) -> tuple[float, float]:
        """Return the desired unicycle command as a planner-style tuple."""
        result = self.predict(observation)
        return result.action["v"], result.action["omega"]

    def step(self, obs: Mapping[str, Any]) -> dict[str, float]:
        """Return the unicycle command as a planner-protocol action dictionary."""
        return dict(self.predict(obs).action)

    def reset(self, *, seed: int | None = None) -> None:
        """Clear recurrent state for a new episode and scenario boundary.

        The reset is idempotent: clearing an already-clear state is a no-op for
        the recurrent payload but still records the explicit boundary reason.
        """
        self.reset_state(seed=seed, reason="explicit_reset")

    def reset_state(self, *, seed: int | None = None, reason: str = "explicit_reset") -> None:
        """Clear recurrent state with an explicit lifecycle reason recorded."""
        if seed is not None:
            self._seed = seed
        self._lstm_states = None
        self._episode_start = True
        self._reset_count += 1
        self._last_reset_reason = reason
        if seed is not None:
            self._sequence_identity = int(seed)

    def configure(self, config: Any) -> None:
        """Apply a new configuration and re-load the checkpoint on change."""
        new_config = self._parse_config(config)
        changed = new_config != self.config
        self.config = new_config
        if changed:
            self._model = None
            self._initialized = False
            if not self._defer_model_loading:
                self._ensure_model_loaded()

    def close(self) -> None:
        """Release the loaded model and recurrent state."""
        self._model = None
        self._lstm_states = None

    def bind_env(self, env: Any) -> None:
        """Bind a runtime observation space and validate the default_gym keys."""
        del env  # observation contract is validated per step; no env binding needed

    def diagnostics(self, *, observation: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return execution diagnostics including recurrent-state presence.

        Args:
            observation: Optional current observation used to report validation
                status and raw desired command values. When omitted, the raw
                command fields are reported as ``None``.

        Returns:
            dict[str, Any]: Versioned diagnostic mapping.
        """
        latency = np.asarray(self._latency_samples_ms, dtype=float)
        state_shape = self._describe_state_shape(self._lstm_states)
        state_finite = self._state_is_finite(self._lstm_states)
        validation_status = "not_evaluated"
        if observation is not None:
            try:
                self._validate_request(
                    observation,
                    observation_level=self.config.observation_level,
                    action_command_space=self.config.action_command_space,
                )
                self._validate_observation_payload(observation)
                validation_status = "validated"
            except LearnedPolicyAdapterContractError:
                validation_status = "invalid"
        raw_velocity = None
        raw_omega = None
        saturation: list[str] = []
        if observation is not None and validation_status == "validated":
            try:
                step = self.predict(observation)
                raw_velocity = step.raw_model_action["v"]
                raw_omega = step.raw_model_action["omega"]
                saturation = self._saturation_flags(step)
            except LearnedPolicyAdapterContractError:
                validation_status = "invalid"
        return {
            "planner_type": "RecurrentPPOLearnedLocalPolicyAdapter",
            "policy_id": self.policy_id,
            "checkpoint_loaded": self._model is not None,
            "recurrent_state_carried": self._lstm_states is not None,
            "episode_start": self._episode_start,
            "recurrent_state_shape": state_shape,
            "recurrent_state_finite": state_finite,
            "inference_call_count": self._inference_calls,
            "reset_count": self._reset_count,
            "last_reset_reason": self._last_reset_reason,
            "sequence_identity": self._sequence_identity,
            "state_shape_drift_count": self._state_shape_drift_count,
            "observation_validation_status": validation_status,
            "raw_desired_command": {"v": raw_velocity, "omega": raw_omega},
            "action_saturation": saturation,
            "inference_latency_ms": {
                "mean": float(np.mean(latency)) if latency.size else None,
                "p95": float(np.percentile(latency, 95)) if latency.size else None,
                "max": float(np.max(latency)) if latency.size else None,
                "samples": int(latency.size),
            },
        }

    @staticmethod
    def _describe_state_shape(state: Any) -> list[list[int]] | None:
        """Return a bounded list of part shapes for the recurrent state."""
        if state is None:
            return None
        parts = state if isinstance(state, tuple) else [state]
        return [list(np.asarray(part).shape) for part in parts]

    @staticmethod
    def _state_is_finite(state: Any) -> bool | None:
        """Return whether every recurrent-state part is finite (None when empty)."""
        if state is None:
            return None
        parts = state if isinstance(state, tuple) else [state]
        try:
            return bool(all(bool(np.all(np.isfinite(np.asarray(part)))) for part in parts))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _saturation_flags(step: LearnedPolicyStepResult) -> list[str]:
        """Return the bound keys that are at their configured limit."""
        flags = []
        for key, value in step.adapted_action.items():
            bounds = step.action_bounds.get(key)
            if bounds is None:
                continue
            lower, upper = bounds
            if float(value) <= float(lower) + 1e-9 or float(value) >= float(upper) - 1e-9:
                flags.append(key)
        return flags

    def metadata(self) -> dict[str, Any]:
        """Return checklist-style metadata mirroring the canonical fixture contract."""
        checkpoint_source = (
            f"model_id={self.config.model_id}"
            if self.config.model_id
            else Path(self.config.model_path).name
        )
        return {
            "policy_id": self.policy_id,
            "claim_boundary": self.claim_boundary,
            "verdict": "eligible_for_adapter",
            "observation_t": "decision step t before action selection",
            "observation_contract": {
                "observation_level": self.config.observation_level,
                "planner_observation_mode": "default_gym",
                "required_inputs": list(self.config.required_observation_inputs),
                "deployment_observable": list(self.config.required_observation_inputs),
                "training_only": [],
                "rejected_evaluation_time_inputs": list(self.config.forbidden_observation_inputs),
                "normalization": "pre-normalized_base_gym_values",
            },
            "action_contract": {
                "output_family": "velocity_command",
                "command_space": self.config.action_command_space,
                "output_keys": ["v", "omega"],
                "frame": "robot",
                "units": "m/s and rad/s",
                "bounds": {
                    "v": [0.0, float(self.config.v_max)],
                    "omega": [-float(self.config.omega_max), float(self.config.omega_max)],
                },
                "kinematics_compatibility": "differential-drive unicycle_vw",
                "projection_policy": "clip_to_configured_bounds",
                "raw_to_robot_sf_action": "raw_model_action_to_v_omega_clip",
                "guard_or_projection_policy": "no guard; post_guard_action equals adapted_action",
            },
            "recurrent_contract": {
                "policy_class": "RecurrentPPO",
                "policy_kind": "MultiInputLstmPolicy",
                "state_carried_across_steps": True,
                "reset_at_episode_boundary": True,
                "deterministic_inference": bool(self.config.deterministic),
            },
            "checkpoint_provenance": {
                "source": checkpoint_source,
                "resolution": "resolve_model_path_or_explicit_path",
                "missing_or_incompatible_policy": "fail closed before action emission",
                "fallback_to_another_checkpoint": False,
                "fallback_to_goal_seeking": False,
            },
            "candidate_registry": {
                "entry_planned": False,
                "adapter_path": "robot_sf/planner/recurrent_ppo_learned_adapter.py",
                "missing_checkpoint_policy": "fail closed with structured error",
                "unsupported_observation_policy": "fail closed before action emission",
                "guard_activation_policy": "not_applicable_no_guard",
            },
        }


def build_recurrent_ppo_adapter_config(
    cfg: dict[str, Any] | None,
) -> RecurrentPPOLearnedAdapterConfig:
    """Build the adapter config from a mapping payload.

    Returns:
        RecurrentPPOLearnedAdapterConfig: Parsed adapter configuration.
    """
    if not isinstance(cfg, dict):
        return RecurrentPPOLearnedAdapterConfig()
    allowed = set(RecurrentPPOLearnedAdapterConfig.__dataclass_fields__)
    return RecurrentPPOLearnedAdapterConfig(
        **{key: value for key, value in cfg.items() if key in allowed}
    )


__all__ = [
    "RecurrentPPOLearnedAdapterConfig",
    "RecurrentPPOLearnedLocalPolicyAdapter",
    "build_recurrent_ppo_adapter_config",
]
