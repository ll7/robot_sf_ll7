"""Deterministic, observation-only commitment for maneuver portfolios.

The manager runs in two phases. :meth:`context_for_candidates` is a pure
calculation over the current candidate set and #8062 evaluations. Its allowed
candidate set enforces normal dwell and hysteresis without changing #8062's
safety-first lexicographic ordering. :meth:`observe_selection` records one
selected candidate after arbitration and rejects repeated or stale steps.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from robot_sf.planner.maneuver_candidates import ManeuverCandidate, ManeuverId
from robot_sf.planner.multimodal_trajectory_arbitrator import CandidateEvaluation

_SCHEMA_VERSION = "maneuver_commitment.v1"
_NUMERIC_TOLERANCE = 1.0e-9
_MAX_CANDIDATES = 64


class ManeuverState(StrEnum):
    """Stable semantic maneuver state, independent from candidate identity."""

    FOLLOW = "follow"
    PASS_LEFT = "pass_left"
    PASS_RIGHT = "pass_right"
    YIELD = "yield"
    STOP = "stop"
    RECOVERY = "recovery"


class TransitionReason(StrEnum):
    """Stable reason codes for state transitions and releases."""

    INITIAL_SELECTION = "initial_selection"
    SAME_MANEUVER = "same_maneuver"
    CONTROLLED_STOP_OVERRIDE = "controlled_stop_override"
    NORMAL_ADVANTAGE = "normal_advantage"
    MINIMUM_DWELL = "minimum_dwell"
    HYSTERESIS_HOLD = "hysteresis_hold"
    ACTIVE_HARD_INFEASIBLE = "active_hard_infeasible"
    SAFER_RISK_CLASS = "safer_risk_class"
    SAFETY_OVERRIDE = "safety_override"
    PASS_COMPLETE = "pass_complete"
    INTERACTION_CLEAR = "interaction_clear"
    INTERACTION_CHANGED = "interaction_changed"
    CORRIDOR_INVALID = "corridor_invalid"
    MAXIMUM_DURATION = "maximum_duration"
    CLEAR_PERSISTENCE = "clear_persistence"
    STAGNATION_REEVALUATION = "stagnation_reevaluation"
    ROUTE_CHANGED = "route_changed"
    RESET = "reset"
    UNSUPPORTED_RECOVERY = "unsupported_recovery"
    NO_SELECTION = "no_selection"


_STATE_BY_MANEUVER: Mapping[ManeuverId, ManeuverState] = MappingProxyType(
    {
        ManeuverId.ROUTE_FOLLOW: ManeuverState.FOLLOW,
        ManeuverId.PASS_LEFT: ManeuverState.PASS_LEFT,
        ManeuverId.PASS_RIGHT: ManeuverState.PASS_RIGHT,
        ManeuverId.YIELD_CREEP: ManeuverState.YIELD,
        ManeuverId.CONTROLLED_STOP: ManeuverState.STOP,
    }
)


def _default_min_dwell() -> dict[ManeuverState, int]:
    return {
        ManeuverState.FOLLOW: 0,
        ManeuverState.PASS_LEFT: 3,
        ManeuverState.PASS_RIGHT: 3,
        ManeuverState.YIELD: 2,
        ManeuverState.STOP: 3,
        ManeuverState.RECOVERY: 0,
    }


def _default_max_duration() -> dict[ManeuverState, int | None]:
    return {
        ManeuverState.FOLLOW: None,
        ManeuverState.PASS_LEFT: 120,
        ManeuverState.PASS_RIGHT: 120,
        ManeuverState.YIELD: 60,
        ManeuverState.STOP: 120,
        ManeuverState.RECOVERY: 0,
    }


def _normalize_state_map(
    values: Mapping[ManeuverState | str, int], *, name: str
) -> dict[ManeuverState, int]:
    normalized: dict[ManeuverState, int] = {}
    for raw_state, value in values.items():
        try:
            state = ManeuverState(raw_state)
        except ValueError as exc:
            raise ValueError(f"{name} contains an unknown maneuver state") from exc
        if state in normalized:
            raise ValueError(f"{name} contains duplicate maneuver states")
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} values must be non-negative integers")
        normalized[state] = value
    return normalized


def _normalize_durations(
    min_values: Mapping[ManeuverState | str, int],
    max_values: Mapping[ManeuverState | str, int | None],
) -> tuple[dict[ManeuverState, int], dict[ManeuverState, int | None]]:
    min_dwell = _default_min_dwell()
    min_dwell.update(_normalize_state_map(min_values, name="min_dwell_steps_by_state"))
    max_duration = _default_max_duration()
    for raw_state, value in max_values.items():
        try:
            state = ManeuverState(raw_state)
        except ValueError as exc:
            raise ValueError("max_duration_steps_by_state contains an unknown state") from exc
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("maximum durations must be non-negative integers or None")
        max_duration[state] = value
    for state, minimum in min_dwell.items():
        maximum = max_duration.get(state)
        if maximum is not None and maximum < minimum:
            raise ValueError(f"maximum duration for {state.value} is below its minimum dwell")
    return min_dwell, max_duration


def _normalize_switch_costs(
    values: Mapping[tuple[ManeuverState | str, ManeuverState | str], float],
) -> dict[tuple[ManeuverState, ManeuverState], float]:
    costs: dict[tuple[ManeuverState, ManeuverState], float] = {}
    for raw_pair, raw_value in values.items():
        if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
            raise ValueError("switch_cost_by_transition keys must be state pairs")
        try:
            pair = (ManeuverState(raw_pair[0]), ManeuverState(raw_pair[1]))
        except ValueError as exc:
            raise ValueError("switch_cost_by_transition contains an unknown state") from exc
        if isinstance(raw_value, bool):
            raise ValueError("switch costs must be finite and non-negative numbers")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("switch costs must be finite and non-negative")
        costs[pair] = value
    return costs


def _validate_float_config(config: ManeuverCommitmentConfig) -> None:
    for name in (
        "same_maneuver_candidate_change_cost",
        "pass_side_switch_cost",
        "general_switch_cost",
        "utility_hysteresis_margin",
        "interaction_clear_distance_m",
        "pass_complete_route_margin_m",
        "rejoin_lateral_tolerance_m",
        "progress_epsilon_m",
    ):
        raw_value = getattr(config, name)
        if isinstance(raw_value, bool):
            raise ValueError(f"{name} must be finite and non-negative")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        object.__setattr__(config, name, value)
    if config.progress_epsilon_m <= 0.0:
        raise ValueError("progress_epsilon_m must be positive")


def _validate_count_config(config: ManeuverCommitmentConfig) -> None:
    for name in (
        "clear_persistence_steps",
        "stagnation_steps",
        "max_state_transitions_per_episode",
    ):
        value = getattr(config, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config.unsupported_recovery_policy != "reject":
        raise ValueError("unsupported_recovery_policy must remain 'reject'")


@dataclass(frozen=True, slots=True)
class ManeuverCommitmentConfig:
    """Validated, bounded thresholds and costs for the state machine."""

    min_dwell_steps_by_state: Mapping[ManeuverState | str, int] = field(
        default_factory=_default_min_dwell
    )
    max_duration_steps_by_state: Mapping[ManeuverState | str, int | None] = field(
        default_factory=_default_max_duration
    )
    switch_cost_by_transition: Mapping[tuple[ManeuverState | str, ManeuverState | str], float] = (
        field(default_factory=dict)
    )
    same_maneuver_candidate_change_cost: float = 0.1
    pass_side_switch_cost: float = 5.0
    general_switch_cost: float = 2.0
    utility_hysteresis_margin: float = 0.05
    clear_persistence_steps: int = 3
    interaction_clear_distance_m: float = 1.2
    pass_complete_route_margin_m: float = 0.8
    rejoin_lateral_tolerance_m: float = 0.35
    progress_epsilon_m: float = 0.02
    stagnation_steps: int = 20
    max_state_transitions_per_episode: int = 100
    unsupported_recovery_policy: str = "reject"

    def __post_init__(self) -> None:
        """Normalize and validate all configuration values."""
        min_dwell, max_duration = _normalize_durations(
            self.min_dwell_steps_by_state, self.max_duration_steps_by_state
        )
        costs = _normalize_switch_costs(self.switch_cost_by_transition)
        _validate_float_config(self)
        _validate_count_config(self)
        object.__setattr__(self, "min_dwell_steps_by_state", MappingProxyType(min_dwell))
        object.__setattr__(self, "max_duration_steps_by_state", MappingProxyType(max_duration))
        object.__setattr__(self, "switch_cost_by_transition", MappingProxyType(costs))

    def min_dwell(self, state: ManeuverState) -> int:
        """Return the configured minimum dwell for one semantic state."""
        return self.min_dwell_steps_by_state[state]

    def max_duration(self, state: ManeuverState) -> int | None:
        """Return the configured maximum duration, if any."""
        return self.max_duration_steps_by_state[state]

    @property
    def config_hash(self) -> str:
        """Return a deterministic digest of the validated configuration."""
        payload = {
            "min_dwell": sorted(
                (key.value, value) for key, value in self.min_dwell_steps_by_state.items()
            ),
            "max_duration": sorted(
                (key.value, value) for key, value in self.max_duration_steps_by_state.items()
            ),
            "switch_costs": sorted(
                (left.value, right.value, value)
                for (left, right), value in self.switch_cost_by_transition.items()
            ),
            "costs": [
                self.same_maneuver_candidate_change_cost,
                self.pass_side_switch_cost,
                self.general_switch_cost,
            ],
            "thresholds": [
                self.utility_hysteresis_margin,
                self.clear_persistence_steps,
                self.interaction_clear_distance_m,
                self.pass_complete_route_margin_m,
                self.rejoin_lateral_tolerance_m,
                self.progress_epsilon_m,
                self.stagnation_steps,
                self.max_state_transitions_per_episode,
                self.unsupported_recovery_policy,
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CommitmentObservation:
    """Current observation-derived route and interaction facts."""

    route_fingerprint: str | None = None
    route_progress_m: float | None = None
    lateral_offset_m: float | None = None
    interaction_id: str | None = None
    interaction_distance_m: float | None = None
    interaction_present: bool | None = None
    interaction_behind: bool = False
    route_projection_valid: bool = True
    safety_override: bool = False

    def __post_init__(self) -> None:
        """Validate optional observation fields without interpreting missing data."""
        for name in ("route_fingerprint", "interaction_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty text or None")
        for name in ("route_progress_m", "lateral_offset_m", "interaction_distance_m"):
            raw_value = getattr(self, name)
            if raw_value is None:
                continue
            if isinstance(raw_value, bool):
                raise TypeError(f"{name} must be numeric or None")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite or None")
            object.__setattr__(self, name, value)
        for name in ("interaction_behind", "route_projection_valid", "safety_override"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a boolean")
        if self.interaction_present is not None and type(self.interaction_present) is not bool:
            raise TypeError("interaction_present must be boolean or None")


@dataclass(frozen=True, slots=True)
class CommitmentState:
    """Bounded per-episode semantic commitment state."""

    active: ManeuverState | None = None
    activated_step: int | None = None
    activated_progress_m: float | None = None
    last_transition_step: int | None = None
    last_selected_candidate_id: str | None = None
    stable_interaction_id: str | None = None
    route_fingerprint: str | None = None
    last_route_progress_m: float | None = None
    no_progress_steps: int = 0
    clear_steps: int = 0
    transition_count: int = 0
    last_observed_step: int | None = None
    last_transition_reason: str | None = TransitionReason.RESET.value


@dataclass(frozen=True, slots=True)
class CandidateCommitment:
    """Bounded commitment contribution for one candidate identity."""

    candidate_id: str
    maneuver_state: ManeuverState
    compatible_with_active: bool
    switch_required: bool
    minimum_dwell_remaining_steps: int
    switch_cost: float
    hard_lock_reason: str | None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe diagnostics for this candidate."""
        return {
            "maneuver_state": self.maneuver_state.value,
            "compatible_with_active": self.compatible_with_active,
            "switch_required": self.switch_required,
            "minimum_dwell_remaining_steps": self.minimum_dwell_remaining_steps,
            "switch_cost": self.switch_cost,
            "hard_lock_reason": self.hard_lock_reason,
        }


@dataclass(frozen=True, slots=True)
class ManeuverCommitmentContext:
    """Immutable output from the non-mutating candidate-context phase."""

    step: int
    switch_costs: Mapping[str, float]
    allowed_candidate_ids: tuple[str, ...]
    candidate_context: Mapping[str, CandidateCommitment]
    active_state: ManeuverState | None
    minimum_dwell_remaining_steps: int
    hold_reason: str | None
    release_reason: str | None
    config_hash: str
    observation: CommitmentObservation
    previous_step: int | None
    previous_transition_count: int
    episode_revision: int
    projected_no_progress_steps: int
    projected_clear_steps: int

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe diagnostics without candidate future outcomes."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "step": self.step,
            "active_state": self.active_state.value if self.active_state else None,
            "minimum_dwell_remaining_steps": self.minimum_dwell_remaining_steps,
            "hold_reason": self.hold_reason,
            "release_reason": self.release_reason,
            "allowed_candidate_ids": list(self.allowed_candidate_ids),
            "switch_costs": dict(sorted(self.switch_costs.items())),
            "candidate_context": {
                key: self.candidate_context[key].to_dict() for key in sorted(self.candidate_context)
            },
            "transition_performed": False,
            "config_hash": self.config_hash,
        }


@dataclass(frozen=True, slots=True)
class CommitmentTransition:
    """Outcome of one accepted selection observation."""

    step: int
    previous_state: ManeuverState | None
    new_state: ManeuverState | None
    transition_performed: bool
    transition_reason: str
    safety_release: bool
    last_selected_candidate_id: str | None
    minimum_dwell_remaining_steps: int
    transition_count: int
    transition_limit_reached: bool
    config_hash: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe transition diagnostics."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "step": self.step,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "new_state": self.new_state.value if self.new_state else None,
            "transition_performed": self.transition_performed,
            "transition_reason": self.transition_reason,
            "safety_release": self.safety_release,
            "last_selected_candidate_id": self.last_selected_candidate_id,
            "minimum_dwell_remaining_steps": self.minimum_dwell_remaining_steps,
            "transition_count": self.transition_count,
            "transition_limit_reached": self.transition_limit_reached,
            "config_hash": self.config_hash,
        }


class ManeuverCommitmentManager:
    """Pure-context, single-observation commitment state machine."""

    def __init__(self, config: ManeuverCommitmentConfig | None = None) -> None:
        """Create an episode-local state machine with validated thresholds."""
        self.config = ManeuverCommitmentConfig() if config is None else config
        if not isinstance(self.config, ManeuverCommitmentConfig):
            raise TypeError("config must be a ManeuverCommitmentConfig")
        self._episode_revision = 0
        self._state = CommitmentState()
        self._last_context: ManeuverCommitmentContext | None = None
        self._last_transition: CommitmentTransition | None = None
        self.reset()

    @property
    def state(self) -> CommitmentState:
        """Return the immutable current state."""
        return self._state

    def reset(self, *, seed: int | None = None) -> None:
        """Clear episode-local state; no random source is used."""
        del seed
        self._episode_revision += 1
        self._state = CommitmentState()
        self._last_context = None
        self._last_transition = None

    def context_for_candidates(
        self,
        *,
        step: int,
        candidates: Sequence[ManeuverCandidate],
        evaluations_without_switch_cost: Sequence[CandidateEvaluation],
        observation: CommitmentObservation | None = None,
    ) -> ManeuverCommitmentContext:
        """Return complete costs and an allowed set without changing state.

        A normal hold restricts arbitration to the active semantic maneuver
        and the controlled-stop candidate. A hard feasibility/risk improvement,
        a route/safety release, or an explicit reevaluation admits the full set.
        """
        self._validate_step(step)
        candidate_values = self._validate_candidates(candidates)
        evaluation_by_id = self._validate_evaluations(
            candidate_values, evaluations_without_switch_cost, require_zero_switch=True
        )
        sensed = observation or CommitmentObservation()
        route_changed = self._route_changed(sensed)
        no_progress_steps = 0 if route_changed else self._project_no_progress(sensed)
        clear_steps = 0 if route_changed else self._project_clear_steps(sensed, evaluation_by_id)
        release_reason = self._release_reason(
            step=step,
            evaluations=evaluation_by_id,
            observation=sensed,
            no_progress_steps=no_progress_steps,
            clear_steps=clear_steps,
        )
        active = self._state.active
        active_age = self._active_age(step)
        dwell = max(0, self.config.min_dwell(active) - active_age) if active else 0
        hold_reason: str | None = None
        if active is not None and release_reason is None:
            if dwell > 0:
                hold_reason = TransitionReason.MINIMUM_DWELL.value
            elif self._normal_switch_advantage(active, candidate_values, evaluation_by_id):
                release_reason = TransitionReason.NORMAL_ADVANTAGE.value
            else:
                hold_reason = TransitionReason.HYSTERESIS_HOLD.value

        all_ids = tuple(sorted(candidate.candidate_id for candidate in candidate_values))
        if active is None or release_reason is not None:
            allowed_ids = all_ids
        else:
            allowed_ids = tuple(
                sorted(
                    candidate.candidate_id
                    for candidate in candidate_values
                    if _STATE_BY_MANEUVER[candidate.maneuver] is active
                    or candidate.maneuver is ManeuverId.CONTROLLED_STOP
                )
            )
        allowed_set = frozenset(allowed_ids)
        costs: dict[str, float] = {}
        contexts: dict[str, CandidateCommitment] = {}
        for candidate in candidate_values:
            state = _STATE_BY_MANEUVER[candidate.maneuver]
            compatible = active is not None and state is active
            cost = self._switch_cost(
                active=active,
                active_candidate_id=self._state.last_selected_candidate_id,
                candidate_state=state,
                candidate_id=candidate.candidate_id,
            )
            costs[candidate.candidate_id] = cost
            contexts[candidate.candidate_id] = CandidateCommitment(
                candidate_id=candidate.candidate_id,
                maneuver_state=state,
                compatible_with_active=compatible,
                switch_required=active is not None and state is not active,
                minimum_dwell_remaining_steps=dwell,
                switch_cost=cost,
                hard_lock_reason=(
                    hold_reason if candidate.candidate_id not in allowed_set else None
                ),
            )
        context = ManeuverCommitmentContext(
            step=step,
            switch_costs=MappingProxyType(dict(sorted(costs.items()))),
            allowed_candidate_ids=allowed_ids,
            candidate_context=MappingProxyType(dict(sorted(contexts.items()))),
            active_state=active,
            minimum_dwell_remaining_steps=dwell,
            hold_reason=hold_reason,
            release_reason=release_reason,
            config_hash=self.config.config_hash,
            observation=sensed,
            previous_step=self._state.last_observed_step,
            previous_transition_count=self._state.transition_count,
            episode_revision=self._episode_revision,
            projected_no_progress_steps=no_progress_steps,
            projected_clear_steps=clear_steps,
        )
        return context

    def observe_selection(
        self,
        *,
        step: int,
        selected_candidate_id: str | None,
        evaluations: Sequence[CandidateEvaluation],
        context: ManeuverCommitmentContext,
    ) -> CommitmentTransition:
        """Commit at most one state transition for this planner step.

        Returns:
            An immutable transition record suitable for planner diagnostics.
        """
        self._validate_step(step)
        self._validate_context_for_observation(step, context)
        evaluations_by_id = self._validate_observed_evaluations(evaluations, context)
        selected_state, reason, transitioned, override = self._resolve_observed_selection(
            selected_candidate_id=selected_candidate_id,
            evaluations=evaluations_by_id,
            context=context,
        )
        transition_limit_reached = (
            transitioned
            and self._state.transition_count >= self.config.max_state_transitions_per_episode
        )
        count = self._state.transition_count + int(transitioned)
        state = self._state_after_observation(
            step=step,
            selected_candidate_id=selected_candidate_id,
            selected_state=selected_state,
            reason=reason,
            transitioned=transitioned,
            context=context,
            transition_count=count,
        )
        previous = self._state.active
        self._state = state
        active_age = max(0, step - state.activated_step) if state.activated_step is not None else 0
        dwell_remaining = (
            max(0, self.config.min_dwell(selected_state) - active_age)
            if selected_state is not None
            else 0
        )
        transition = CommitmentTransition(
            step=step,
            previous_state=previous,
            new_state=selected_state,
            transition_performed=transitioned,
            transition_reason=reason,
            safety_release=override
            or reason == TransitionReason.CONTROLLED_STOP_OVERRIDE.value
            or context.release_reason
            in {
                TransitionReason.ACTIVE_HARD_INFEASIBLE.value,
                TransitionReason.SAFER_RISK_CLASS.value,
                TransitionReason.SAFETY_OVERRIDE.value,
                TransitionReason.CORRIDOR_INVALID.value,
            },
            last_selected_candidate_id=state.last_selected_candidate_id,
            minimum_dwell_remaining_steps=dwell_remaining,
            transition_count=count,
            transition_limit_reached=transition_limit_reached,
            config_hash=self.config.config_hash,
        )
        self._last_transition = transition
        self._last_context = context
        return transition

    def snapshot(self, *, step: int | None = None) -> dict[str, object]:
        """Return a JSON-safe snapshot of current state and last candidate context."""
        observed_step = step if step is not None else self._state.last_observed_step
        age = (
            max(0, observed_step - self._state.activated_step)
            if observed_step is not None and self._state.activated_step is not None
            else 0
        )
        active = self._state.active
        dwell = max(0, self.config.min_dwell(active) - age) if active is not None else 0
        candidate_context = (
            self._last_context.to_dict()["candidate_context"] if self._last_context else {}
        )
        transition_performed = (
            self._last_transition.transition_performed if self._last_transition else False
        )
        return {
            "schema_version": _SCHEMA_VERSION,
            "active_state": active.value if active else None,
            "active_age_steps": age,
            "minimum_dwell_remaining_steps": dwell,
            "last_selected_candidate_id": self._state.last_selected_candidate_id,
            "interaction_id": self._state.stable_interaction_id,
            "route_fingerprint": self._state.route_fingerprint,
            "no_progress_steps": self._state.no_progress_steps,
            "transition_count": self._state.transition_count,
            "last_transition_reason": self._state.last_transition_reason,
            "candidate_context": candidate_context,
            "transition_performed": transition_performed,
            "previous_state": (
                self._last_transition.previous_state.value
                if self._last_transition and self._last_transition.previous_state
                else None
            ),
            "new_state": (
                self._last_transition.new_state.value
                if self._last_transition and self._last_transition.new_state
                else None
            ),
            "safety_release": self._last_transition.safety_release
            if self._last_transition
            else False,
            "config_hash": self.config.config_hash,
            "transition_limit_reached": (
                self._last_transition.transition_limit_reached if self._last_transition else False
            ),
        }

    def _validate_step(self, step: int) -> None:
        if type(step) is not int or step < 0:
            raise ValueError("step must be a non-negative integer")
        last = self._state.last_observed_step
        if last is not None and step <= last:
            raise ValueError("step is duplicate or out of order")

    def _validate_context_for_observation(
        self, step: int, context: ManeuverCommitmentContext
    ) -> None:
        if not isinstance(context, ManeuverCommitmentContext):
            raise TypeError("context must be produced by context_for_candidates")
        if (
            context.step != step
            or context.previous_step != self._state.last_observed_step
            or context.previous_transition_count != self._state.transition_count
            or context.episode_revision != self._episode_revision
            or context.config_hash != self.config.config_hash
            or context.active_state is not self._state.active
        ):
            raise ValueError("commitment context is stale or belongs to another manager state")

    def _resolve_observed_selection(
        self,
        *,
        selected_candidate_id: str | None,
        evaluations: Mapping[str, CandidateEvaluation],
        context: ManeuverCommitmentContext,
    ) -> tuple[ManeuverState | None, str, bool, bool]:
        if selected_candidate_id is not None:
            if not isinstance(selected_candidate_id, str) or not selected_candidate_id:
                raise ValueError("selected_candidate_id must be non-empty text or None")
            if selected_candidate_id not in evaluations:
                raise ValueError("selected candidate must have one arbitration evaluation")
            if not evaluations[selected_candidate_id].eligible:
                raise ValueError(
                    "selected candidate is ineligible under the arbitration safety contract"
                )
        selected_state = (
            _STATE_BY_MANEUVER[ManeuverId(evaluations[selected_candidate_id].maneuver)]
            if selected_candidate_id is not None
            else None
        )
        override = context.observation.safety_override
        reason = self._transition_reason(context, selected_state, override)
        transitioned = self._transition_performed(self._state.active, selected_state, reason)
        if (
            selected_candidate_id is None
            and self._state.active is not None
            and context.release_reason is None
        ):
            return self._state.active, TransitionReason.NO_SELECTION.value, False, override
        return selected_state, reason, transitioned, override

    def _state_after_observation(
        self,
        *,
        step: int,
        selected_candidate_id: str | None,
        selected_state: ManeuverState | None,
        reason: str,
        transitioned: bool,
        context: ManeuverCommitmentContext,
        transition_count: int,
    ) -> CommitmentState:
        activated_step = self._state.activated_step
        activated_progress = self._state.activated_progress_m
        if selected_state is not None and transitioned:
            activated_step = step
            activated_progress = context.observation.route_progress_m
        elif selected_state is None and transitioned:
            activated_step = None
            activated_progress = None
        route_changed = self._route_changed(context.observation)
        interaction_id = self._interaction_id_after_selection(
            selected_state, context, route_changed=route_changed
        )
        route_fingerprint = context.observation.route_fingerprint or self._state.route_fingerprint
        if context.observation.route_progress_m is not None:
            route_progress = context.observation.route_progress_m
        elif route_changed:
            route_progress = None
        else:
            route_progress = self._state.last_route_progress_m
        return CommitmentState(
            active=selected_state,
            activated_step=activated_step,
            activated_progress_m=activated_progress,
            last_transition_step=step if transitioned else self._state.last_transition_step,
            last_selected_candidate_id=(
                selected_candidate_id
                if selected_candidate_id is not None
                else self._state.last_selected_candidate_id
            ),
            stable_interaction_id=interaction_id,
            route_fingerprint=route_fingerprint,
            last_route_progress_m=route_progress,
            no_progress_steps=context.projected_no_progress_steps,
            clear_steps=0 if transitioned else context.projected_clear_steps,
            transition_count=transition_count,
            last_observed_step=step,
            last_transition_reason=reason if transitioned else self._state.last_transition_reason,
        )

    def _interaction_id_after_selection(
        self,
        selected_state: ManeuverState | None,
        context: ManeuverCommitmentContext,
        *,
        route_changed: bool,
    ) -> str | None:
        if selected_state not in {ManeuverState.PASS_LEFT, ManeuverState.PASS_RIGHT}:
            return None
        return context.observation.interaction_id or (
            None if route_changed else self._state.stable_interaction_id
        )

    def _validate_candidates(
        self, candidates: Sequence[ManeuverCandidate]
    ) -> tuple[ManeuverCandidate, ...]:
        values = tuple(candidates)
        if len(values) > _MAX_CANDIDATES:
            raise ValueError("candidate set exceeds the bounded commitment capacity")
        if any(not isinstance(candidate, ManeuverCandidate) for candidate in values):
            raise TypeError("candidates must be ManeuverCandidate values")
        ids = [candidate.candidate_id for candidate in values]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        for candidate in values:
            self._validate_candidate(candidate)
        if not any(candidate.maneuver is ManeuverId.CONTROLLED_STOP for candidate in values):
            raise ValueError("candidate set must include a CONTROLLED_STOP candidate")
        if values:
            grid = _candidate_time_grid(values[0])
            if any(_candidate_time_grid(candidate) != grid for candidate in values[1:]):
                raise ValueError("candidate time grids and timestamps must match")
        return tuple(sorted(values, key=lambda candidate: candidate.candidate_id))

    def _validate_candidate(self, candidate: ManeuverCandidate) -> None:
        if candidate.maneuver not in _STATE_BY_MANEUVER:
            raise ValueError(TransitionReason.UNSUPPORTED_RECOVERY.value)
        if not math.isfinite(candidate.dt_s) or candidate.dt_s <= 0.0:
            raise ValueError("candidate time step must be finite and positive")
        if candidate.horizon_steps < 1:
            raise ValueError("candidate horizon must be positive")

    def _validate_evaluations(
        self,
        candidates: Sequence[ManeuverCandidate],
        evaluations: Sequence[CandidateEvaluation],
        *,
        require_zero_switch: bool,
    ) -> dict[str, CandidateEvaluation]:
        values = tuple(evaluations)
        result: dict[str, CandidateEvaluation] = {}
        for evaluation in values:
            if not isinstance(evaluation, CandidateEvaluation):
                raise TypeError("evaluations must be CandidateEvaluation values")
            if evaluation.candidate_id in result:
                raise ValueError("candidate evaluations must have unique IDs")
            result[evaluation.candidate_id] = evaluation
        expected = {candidate.candidate_id for candidate in candidates}
        if set(result) != expected:
            raise ValueError("evaluations must cover every candidate ID exactly once")
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        for candidate_id, evaluation in result.items():
            self._validate_evaluation(
                candidate_by_id[candidate_id], evaluation, require_zero_switch=require_zero_switch
            )
        return result

    def _validate_evaluation(
        self,
        candidate: ManeuverCandidate,
        evaluation: CandidateEvaluation,
        *,
        require_zero_switch: bool,
    ) -> None:
        try:
            maneuver = ManeuverId(evaluation.maneuver)
        except ValueError as exc:
            raise ValueError("evaluation maneuver is not a supported candidate ID") from exc
        if maneuver is not candidate.maneuver:
            raise ValueError("evaluation maneuver disagrees with the candidate")
        if type(evaluation.risk_bucket) is not int or evaluation.risk_bucket < 0:
            raise ValueError("risk_bucket must be a non-negative integer")
        switch_cost = float(evaluation.switch_cost)
        if not math.isfinite(switch_cost):
            raise ValueError("switch_cost must be finite")
        if require_zero_switch and abs(switch_cost) > _NUMERIC_TOLERANCE:
            raise ValueError("context evaluations must not already include switch costs")
        _validate_decision_key(evaluation)

    def _validate_observed_evaluations(
        self,
        evaluations: Sequence[CandidateEvaluation],
        context: ManeuverCommitmentContext,
    ) -> dict[str, CandidateEvaluation]:
        result: dict[str, CandidateEvaluation] = {}
        candidate_states = {
            key: value.maneuver_state for key, value in context.candidate_context.items()
        }
        for evaluation in evaluations:
            if not isinstance(evaluation, CandidateEvaluation):
                raise TypeError("evaluations must be CandidateEvaluation values")
            if evaluation.candidate_id in result:
                raise ValueError("observed candidate evaluations must have unique IDs")
            expected_state = candidate_states.get(evaluation.candidate_id)
            if expected_state is None:
                raise ValueError("observed evaluation is not in the commitment context")
            try:
                maneuver = ManeuverId(evaluation.maneuver)
            except ValueError as exc:
                raise ValueError(TransitionReason.UNSUPPORTED_RECOVERY.value) from exc
            if maneuver not in _STATE_BY_MANEUVER:
                raise ValueError(TransitionReason.UNSUPPORTED_RECOVERY.value)
            if _STATE_BY_MANEUVER[maneuver] is not expected_state:
                raise ValueError("observed evaluation maneuver disagrees with commitment context")
            result[evaluation.candidate_id] = evaluation
        if set(result) != set(context.allowed_candidate_ids):
            raise ValueError("arbitration must evaluate every allowed candidate exactly once")
        return result

    def _active_age(self, step: int) -> int:
        activated = self._state.activated_step
        return max(0, step - activated) if activated is not None else 0

    def _project_no_progress(self, observation: CommitmentObservation) -> int:
        if observation.route_progress_m is None:
            return self._state.no_progress_steps
        previous = self._state.last_route_progress_m
        if previous is None:
            return 0
        if observation.route_progress_m - previous >= self.config.progress_epsilon_m:
            return 0
        return self._state.no_progress_steps + 1

    def _project_clear_steps(
        self,
        observation: CommitmentObservation,
        evaluations: Mapping[str, CandidateEvaluation],
    ) -> int:
        active = self._state.active
        if active in {ManeuverState.PASS_LEFT, ManeuverState.PASS_RIGHT}:
            clear = observation.interaction_present is False or (
                observation.interaction_distance_m is not None
                and observation.interaction_distance_m > self.config.interaction_clear_distance_m
            )
            return self._state.clear_steps + 1 if clear else 0
        if active in {ManeuverState.YIELD, ManeuverState.STOP}:
            active_values = [
                item
                for item in evaluations.values()
                if _STATE_BY_MANEUVER[ManeuverId(item.maneuver)] is active and item.eligible
            ]
            moving_values = [
                item
                for item in evaluations.values()
                if _STATE_BY_MANEUVER[ManeuverId(item.maneuver)] not in {active, ManeuverState.STOP}
                and item.eligible
            ]
            if active_values and moving_values:
                active_class = min(_safety_class(item) for item in active_values)
                if any(_safety_class(item) == active_class for item in moving_values):
                    return self._state.clear_steps + 1
            return 0
        return 0

    def _release_reason(
        self,
        *,
        step: int,
        evaluations: Mapping[str, CandidateEvaluation],
        observation: CommitmentObservation,
        no_progress_steps: int,
        clear_steps: int,
    ) -> str | None:
        active = self._state.active
        if active is None:
            return None
        immediate = self._immediate_release_reason(active, evaluations, observation)
        if immediate is not None:
            return immediate
        completed = self._completion_release_reason(active, step, observation, clear_steps)
        if completed is not None:
            return completed
        duration = self.config.max_duration(active)
        if duration is not None and self._active_age(step) >= duration:
            return TransitionReason.MAXIMUM_DURATION.value
        if no_progress_steps >= self.config.stagnation_steps:
            return TransitionReason.STAGNATION_REEVALUATION.value
        return None

    def _immediate_release_reason(
        self,
        active: ManeuverState,
        evaluations: Mapping[str, CandidateEvaluation],
        observation: CommitmentObservation,
    ) -> str | None:
        if observation.safety_override:
            return TransitionReason.SAFETY_OVERRIDE.value
        if self._route_changed(observation):
            return TransitionReason.ROUTE_CHANGED.value
        if not observation.route_projection_valid:
            return TransitionReason.CORRIDOR_INVALID.value
        if (
            self._state.stable_interaction_id is not None
            and observation.interaction_id is not None
            and observation.interaction_id != self._state.stable_interaction_id
        ):
            return TransitionReason.INTERACTION_CHANGED.value
        active_values = [
            item
            for item in evaluations.values()
            if _STATE_BY_MANEUVER[ManeuverId(item.maneuver)] is active and item.eligible
        ]
        if not active_values:
            return TransitionReason.ACTIVE_HARD_INFEASIBLE.value
        best_active_class = min(_safety_class(item) for item in active_values)
        alternatives = [
            item
            for item in evaluations.values()
            if _STATE_BY_MANEUVER[ManeuverId(item.maneuver)] is not active and item.eligible
        ]
        if any(_safety_class(item) < best_active_class for item in alternatives):
            return TransitionReason.SAFER_RISK_CLASS.value
        return None

    def _route_changed(self, observation: CommitmentObservation) -> bool:
        """Return whether this observation belongs to a different known route."""
        return (
            self._state.route_fingerprint is not None
            and observation.route_fingerprint is not None
            and observation.route_fingerprint != self._state.route_fingerprint
        )

    def _completion_release_reason(
        self,
        active: ManeuverState,
        step: int,
        observation: CommitmentObservation,
        clear_steps: int,
    ) -> str | None:
        if active in {ManeuverState.PASS_LEFT, ManeuverState.PASS_RIGHT}:
            if observation.interaction_behind:
                return TransitionReason.PASS_COMPLETE.value
            if (
                observation.lateral_offset_m is not None
                and abs(observation.lateral_offset_m) <= self.config.rejoin_lateral_tolerance_m
                and observation.route_progress_m is not None
                and self._state.activated_progress_m is not None
                and observation.route_progress_m - self._state.activated_progress_m
                >= self.config.pass_complete_route_margin_m
            ):
                return TransitionReason.PASS_COMPLETE.value
            if clear_steps >= self.config.clear_persistence_steps:
                return TransitionReason.INTERACTION_CLEAR.value
        if active in {ManeuverState.YIELD, ManeuverState.STOP}:
            if clear_steps >= self.config.clear_persistence_steps and self._active_age(
                step
            ) >= self.config.min_dwell(active):
                return TransitionReason.CLEAR_PERSISTENCE.value
        return None

    def _normal_switch_advantage(
        self,
        active: ManeuverState,
        candidates: Sequence[ManeuverCandidate],
        evaluations: Mapping[str, CandidateEvaluation],
    ) -> bool:
        states = {
            candidate.candidate_id: _STATE_BY_MANEUVER[candidate.maneuver]
            for candidate in candidates
        }
        incumbents = [
            item
            for candidate_id, item in evaluations.items()
            if states[candidate_id] is active and item.eligible
        ]
        challengers = [
            item
            for candidate_id, item in evaluations.items()
            if states[candidate_id] is not active and item.eligible
        ]
        if not incumbents or not challengers:
            return False
        best_incumbent = min(incumbents, key=lambda item: item.decision_key)
        best_challenger = min(challengers, key=lambda item: item.decision_key)
        if _safety_class(best_challenger) < _safety_class(best_incumbent):
            return True
        if _safety_class(best_challenger) != _safety_class(best_incumbent):
            return False
        return _relative_decision_advantage(
            best_challenger.decision_key,
            best_incumbent.decision_key,
            margin=self.config.utility_hysteresis_margin,
        )

    def _switch_cost(
        self,
        *,
        active: ManeuverState | None,
        active_candidate_id: str | None,
        candidate_state: ManeuverState,
        candidate_id: str,
    ) -> float:
        if active is None or candidate_state is ManeuverState.STOP:
            return 0.0
        if candidate_state is active:
            if active_candidate_id is not None and candidate_id != active_candidate_id:
                return self.config.same_maneuver_candidate_change_cost
            return 0.0
        pair = (active, candidate_state)
        if pair in self.config.switch_cost_by_transition:
            return self.config.switch_cost_by_transition[pair]
        if {active, candidate_state} == {ManeuverState.PASS_LEFT, ManeuverState.PASS_RIGHT}:
            return self.config.pass_side_switch_cost
        return self.config.general_switch_cost

    def _transition_reason(
        self,
        context: ManeuverCommitmentContext,
        selected_state: ManeuverState | None,
        safety_override: bool,
    ) -> str:
        if safety_override:
            return TransitionReason.SAFETY_OVERRIDE.value
        if context.release_reason is not None:
            return context.release_reason
        if self._state.active is None and selected_state is not None:
            return TransitionReason.INITIAL_SELECTION.value
        if selected_state is None:
            return TransitionReason.NO_SELECTION.value
        if selected_state is ManeuverState.STOP and self._state.active is not ManeuverState.STOP:
            return TransitionReason.CONTROLLED_STOP_OVERRIDE.value
        return TransitionReason.SAME_MANEUVER.value

    def _transition_performed(
        self,
        previous: ManeuverState | None,
        selected: ManeuverState | None,
        reason: str,
    ) -> bool:
        if previous is None and selected is None:
            return False
        if previous is not selected:
            return True
        return reason in {
            TransitionReason.SAFETY_OVERRIDE.value,
            TransitionReason.PASS_COMPLETE.value,
            TransitionReason.INTERACTION_CLEAR.value,
            TransitionReason.INTERACTION_CHANGED.value,
            TransitionReason.CORRIDOR_INVALID.value,
            TransitionReason.MAXIMUM_DURATION.value,
            TransitionReason.CLEAR_PERSISTENCE.value,
            TransitionReason.STAGNATION_REEVALUATION.value,
            TransitionReason.ROUTE_CHANGED.value,
            TransitionReason.SAFER_RISK_CLASS.value,
            TransitionReason.ACTIVE_HARD_INFEASIBLE.value,
        }


def _candidate_time_grid(candidate: ManeuverCandidate) -> tuple[float, int, float]:
    """Return identity-independent timing metadata for portfolio consistency."""
    timestamp = candidate.metadata.get("timestamp_s")
    if timestamp is None or isinstance(timestamp, bool):
        raise ValueError("candidate timestamp_s metadata is required")
    timestamp_s = float(timestamp)
    if not math.isfinite(timestamp_s):
        raise ValueError("candidate timestamp_s must be finite")
    return (candidate.dt_s, candidate.horizon_steps, timestamp_s)


def _validate_decision_key(evaluation: CandidateEvaluation) -> None:
    key = evaluation.decision_key
    if not isinstance(key, tuple) or len(key) < 7:
        raise ValueError("evaluation decision_key is incomplete")
    if not evaluation.eligible:
        return
    for value in key[4:-2]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("post-safety decision-key fields must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError("eligible decision-key values must be finite")


def _safety_class(evaluation: CandidateEvaluation) -> tuple[int, int, int, int]:
    """Return the explicit lexicographic safety prefix used by #8062."""
    return (
        0 if evaluation.hard_feasible else 1,
        0 if evaluation.braking_feasible else 1,
        0 if evaluation.risk_limit_ok else 1,
        evaluation.risk_bucket,
    )


def _relative_decision_advantage(
    challenger: tuple[object, ...],
    incumbent: tuple[object, ...],
    *,
    margin: float,
) -> bool:
    """Compare the first post-safety numeric difference without scalar weights.

    Returns:
        Whether the challenger clears the configured relative margin.
    """
    if len(challenger) != len(incumbent) or len(challenger) < 7:
        raise ValueError("decision keys must have matching post-safety fields")
    for left, right in zip(challenger[4:-2], incumbent[4:-2], strict=True):
        if left == right:
            continue
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, (int, float))
            or not isinstance(right, (int, float))
        ):
            raise ValueError("post-safety decision-key values must be numeric")
        left_value = float(left)
        right_value = float(right)
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            return False
        if left_value >= right_value:
            return False
        relative_gap = (right_value - left_value) / max(
            abs(left_value), abs(right_value), _NUMERIC_TOLERANCE
        )
        return relative_gap > margin
    return False


__all__ = [
    "CandidateCommitment",
    "CommitmentObservation",
    "CommitmentState",
    "CommitmentTransition",
    "ManeuverCommitmentConfig",
    "ManeuverCommitmentContext",
    "ManeuverCommitmentManager",
    "ManeuverState",
    "TransitionReason",
]
