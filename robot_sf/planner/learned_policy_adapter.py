"""Learned local-policy adapter fixtures for contract-level tests.

The dummy adapter in this module exists to exercise the Robot SF learned-policy
boundary without loading a checkpoint or making a benchmark claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping
from robot_sf.errors import RobotSfError


class LearnedPolicyAdapterContractError(RobotSfError, ValueError):
    """Raised when a learned-policy adapter request violates its declared contract."""


@dataclass(frozen=True)
class LearnedPolicyStepResult:
    """Per-step learned-policy action and logging payload."""

    action: dict[str, float]
    raw_model_action: dict[str, float]
    adapted_action: dict[str, float]
    post_guard_action: dict[str, float]
    guard_applied: bool
    guard_or_fallback_reason: str
    observation_level: str
    planner_observation_mode: str
    action_bounds: dict[str, list[float]]
    action_projection_metadata: dict[str, Any]

    _ACTION_BOUND_LOWER = 0
    _ACTION_BOUND_UPPER = 1

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnostic payload for this decision step."""
        return {
            "action": dict(self.action),
            "raw_model_action": dict(self.raw_model_action),
            "adapted_action": dict(self.adapted_action),
            "post_guard_action": dict(self.post_guard_action),
            "guard_applied": self.guard_applied,
            "guard_or_fallback_reason": self.guard_or_fallback_reason,
            "observation_level": self.observation_level,
            "planner_observation_mode": self.planner_observation_mode,
            "action_bounds": {
                key: [
                    float(bounds[self._ACTION_BOUND_LOWER]),
                    float(bounds[self._ACTION_BOUND_UPPER]),
                ]
                for key, bounds in self.action_bounds.items()
            },
            "action_projection_metadata": dict(self.action_projection_metadata),
        }


class DummyLearnedLocalPolicyAdapter:
    """Deterministic learned-policy adapter fixture with an explicit contract.

    This class intentionally does not load a model. It mirrors the observation,
    action, and logging gates used by real learned local policies so tests can
    exercise adapter plumbing without introducing checkpoint or training
    provenance.
    """

    policy_id = "dummy_learned_local_policy_adapter"
    claim_boundary = "adapter_fixture_only_not_benchmark_evidence"
    observation_level = "lidar_2d"
    planner_observation_mode = "sensor_fusion_state"
    action_command_space = "unicycle_vw"
    _required_observation_inputs = ("drive_state", "rays")
    _forbidden_observation_inputs = (
        "future_states",
        "future_trajectory",
        "future_collision_label",
        "simulator_outcome",
        "termination_reason",
    )
    _action = {"v": 0.25, "omega": 0.0}
    _action_bounds = {"v": [0.0, 0.5], "omega": [-0.5, 0.5]}
    _ACTION_BOUND_LOWER = 0
    _ACTION_BOUND_UPPER = 1

    def metadata(self) -> dict[str, Any]:
        """Return checklist-style metadata for the dummy adapter fixture."""
        return {
            "policy_id": self.policy_id,
            "claim_boundary": self.claim_boundary,
            "verdict": "eligible_for_adapter",
            "observation_t": "decision step t before action selection",
            "observation_contract": {
                "observation_level": self.observation_level,
                "planner_observation_mode": self.planner_observation_mode,
                "required_inputs": list(self._required_observation_inputs),
                "deployment_observable": ["drive_state", "rays"],
                "training_only": [],
                "rejected_evaluation_time_inputs": list(self._forbidden_observation_inputs),
                "normalization": "raw_fixture_values",
            },
            "observation_fields": {
                "deployment_observable": ["drive_state", "rays"],
                "training_only": [],
                "forbidden_evaluation_time": [],
            },
            "split_provenance": {
                "training_data_source": "none_dummy_fixture",
                "validation_split": "not_applicable",
                "test_split": "not_applicable",
                "checkpoint_or_model_provenance": "none_no_checkpoint_loaded",
                "privileged_training_inputs": "none",
                "privileged_training_inputs_enter_evaluation": False,
                "normalization_statistics": "none",
                "normalization_statistics_fit_on_training_only": True,
                "evidence_source": "Robot SF adapter boundary unit test",
            },
            "action_contract": {
                "output_family": "velocity_command",
                "command_space": self.action_command_space,
                "output_keys": ["v", "omega"],
                "frame": "robot",
                "units": "m/s and rad/s",
                "bounds": dict(self._action_bounds),
                "kinematics_compatibility": "differential-drive unicycle_vw fixture",
                "projection_policy": "none",
                "raw_to_robot_sf_action": "copy deterministic fixture command",
                "guard_or_projection_policy": "no guard; post_guard_action equals adapted_action",
            },
            "per_step_logging": {
                "raw_model_action": "direct deterministic fixture output",
                "adapted_action": "Robot SF action dictionary after adapter conversion",
                "post_guard_action": "final emitted action; equal to adapted_action for fixture",
                "guard_applied": "boolean flag",
                "guard_or_fallback_reason": "stable string; none for fixture",
                "observation_level": self.observation_level,
                "planner_observation_mode": self.planner_observation_mode,
                "action_bounds": dict(self._action_bounds),
                "action_projection_metadata": "projection status and policy metadata",
            },
            "candidate_registry": {
                "entry_planned": False,
                "adapter_path": "robot_sf/planner/learned_policy_adapter.py",
                "smoke_or_validation_command": (
                    "uv run pytest -q tests/planner/test_dummy_learned_policy_adapter.py"
                ),
                "missing_checkpoint_policy": "not_applicable_no_checkpoint",
                "unsupported_observation_policy": "fail closed before action emission",
                "guard_activation_policy": "not_applicable_no_guard",
            },
        }

    def predict(
        self,
        observation: Mapping[str, Any],
        *,
        observation_level: str | None = None,
        action_command_space: str | None = None,
    ) -> LearnedPolicyStepResult:
        """Return the deterministic fixture action after contract validation."""
        self._validate_request(
            observation,
            observation_level=(
                self.observation_level if observation_level is None else observation_level
            ),
            action_command_space=(
                self.action_command_space if action_command_space is None else action_command_space
            ),
        )
        action = dict(self._action)
        return LearnedPolicyStepResult(
            action=action,
            raw_model_action=dict(action),
            adapted_action=dict(action),
            post_guard_action=dict(action),
            guard_applied=False,
            guard_or_fallback_reason="none",
            observation_level=self.observation_level,
            planner_observation_mode=self.planner_observation_mode,
            action_bounds={
                key: [
                    float(bounds[self._ACTION_BOUND_LOWER]),
                    float(bounds[self._ACTION_BOUND_UPPER]),
                ]
                for key, bounds in self._action_bounds.items()
            },
            action_projection_metadata={
                "projected": False,
                "projection_policy": "none",
            },
        )

    def plan(self, observation: Mapping[str, Any]) -> tuple[float, float]:
        """Return the deterministic command as a planner-style ``(v, omega)`` tuple."""
        result = self.predict(observation)
        return result.action["v"], result.action["omega"]

    def step(self, obs: Mapping[str, Any]) -> dict[str, float]:
        """Return the deterministic command as a planner-protocol action dictionary."""
        return dict(self.predict(obs).action)

    def reset(self, *, seed: int | None = None) -> None:
        """Accept seeded planner resets; the fixture has no state to reset."""
        del seed

    def configure(self, config: Any) -> None:
        """Accept empty configuration updates for planner-protocol compatibility."""
        if config not in (None, {}):
            raise LearnedPolicyAdapterContractError(
                "dummy learned-policy fixture does not accept runtime configuration"
            )

    def close(self) -> None:
        """Release fixture resources; no external resources are owned."""

    def _validate_request(
        self,
        observation: Mapping[str, Any],
        *,
        observation_level: str,
        action_command_space: str,
    ) -> None:
        """Validate the requested observation and action contract before prediction."""
        if observation_level != self.observation_level:
            raise LearnedPolicyAdapterContractError(
                "unsupported observation_level "
                f"'{observation_level}'; expected '{self.observation_level}'"
            )
        if action_command_space != self.action_command_space:
            raise LearnedPolicyAdapterContractError(
                "unsupported action_command_space "
                f"'{action_command_space}'; expected '{self.action_command_space}'"
            )

        if observation is None:
            raise LearnedPolicyAdapterContractError("observation must not be None")
        missing = [key for key in self._required_observation_inputs if key not in observation]
        if missing:
            raise LearnedPolicyAdapterContractError(
                "missing required observation inputs: " + ", ".join(missing)
            )
        forbidden = [key for key in self._forbidden_observation_inputs if key in observation]
        if forbidden:
            raise LearnedPolicyAdapterContractError(
                "forbidden evaluation-time observation inputs: " + ", ".join(forbidden)
            )


__all__ = [
    "DummyLearnedLocalPolicyAdapter",
    "LearnedPolicyAdapterContractError",
    "LearnedPolicyStepResult",
]
