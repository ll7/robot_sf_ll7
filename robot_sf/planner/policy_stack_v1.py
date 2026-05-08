"""Minimal policy_stack_v1 portfolio runtime.

The v1 stack is intentionally conservative: it scores commands from existing
proposal sources, records why unavailable or failed sources did not participate,
and fails closed only for explicitly mandatory sources.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_sf.planner.risk_dwa import (
    RiskDWAPlannerAdapter,
    RiskDWAPlannerConfig,
    build_risk_dwa_config,
)

_STATUS_KEYS = (
    "native",
    "adapter",
    "fallback",
    "degraded",
    "failed",
    "not_available",
    "rejected",
)
_COMMAND_STATUSES = {"native", "adapter", "fallback", "degraded"}


@dataclass(frozen=True)
class PolicyStackV1Config:
    """Configuration for the minimal non-learning policy stack."""

    proposal_sources: tuple[str, ...] = ("goal", "risk_dwa")
    mandatory_sources: tuple[str, ...] = ()
    optional_sources: tuple[str, ...] = ()
    fallback_sources: tuple[str, ...] = ()
    degraded_sources: tuple[str, ...] = ()

    max_linear_speed: float = 1.0
    max_angular_speed: float = 1.2
    goal_tolerance: float = 0.25
    heading_gain: float = 1.4
    hard_stop_clearance: float = 0.35

    goal_progress_weight: float = 4.0
    heading_weight: float = 0.8
    angular_penalty_weight: float = 0.2
    clearance_weight: float = 0.8


@dataclass(frozen=True)
class PolicyStackV1BuildConfig:
    """Composite config for stack-level and child proposal-source settings."""

    policy_stack: PolicyStackV1Config
    risk_dwa: RiskDWAPlannerConfig


@dataclass(frozen=True)
class _Proposal:
    """One candidate command or diagnostic-only source result."""

    key: str
    status: str
    command: tuple[float, float] | None = None
    reason: str | None = None


class PolicyStackV1Adapter:
    """Score a small portfolio of native and adapter proposal sources."""

    def __init__(
        self,
        *,
        config: PolicyStackV1Config | None = None,
        risk_dwa: RiskDWAPlannerAdapter | None = None,
    ) -> None:
        """Create the stack with optional child planner injection for tests."""
        self.config = config or PolicyStackV1Config()
        self.risk_dwa = risk_dwa or RiskDWAPlannerAdapter()
        self._steps = 0
        self._status_counts = _empty_status_counts()
        self._selection_counts: dict[str, int] = {}
        self._failed_count = 0
        self._unavailable_count = 0
        self._rejected_count = 0
        self._shield_intervention_count = 0
        self._last_step: dict[str, Any] | None = None

    def reset(self) -> None:
        """Clear episode-local diagnostics and reset child proposals."""
        self._steps = 0
        self._status_counts = _empty_status_counts()
        self._selection_counts.clear()
        self._failed_count = 0
        self._unavailable_count = 0
        self._rejected_count = 0
        self._shield_intervention_count = 0
        self._last_step = None

        reset = getattr(self.risk_dwa, "reset", None)
        if callable(reset):
            reset()

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Return the selected unicycle command and record step diagnostics."""
        proposals = self._collect_proposals(observation)
        candidates = [
            p for p in proposals if p.command is not None and p.status in _COMMAND_STATUSES
        ]
        if not candidates:
            self._raise_no_candidate(proposals)

        scores = {
            proposal.key: self._score_command(proposal.command, observation)
            for proposal in candidates
            if proposal.command is not None
        }
        selected = max(candidates, key=lambda proposal: scores[proposal.key]["total"])
        command = selected.command
        if command is None:
            self._raise_no_candidate(proposals)

        step_status_counts = _empty_status_counts()
        rejection_reasons: dict[str, str] = {}
        proposal_statuses: dict[str, str] = {}
        proposal_commands: dict[str, list[float]] = {}

        for proposal in proposals:
            step_status_counts[proposal.status] = step_status_counts.get(proposal.status, 0) + 1
            proposal_statuses[proposal.key] = proposal.status
            if proposal.command is not None:
                proposal_commands[proposal.key] = [
                    float(proposal.command[0]),
                    float(proposal.command[1]),
                ]
            if proposal.reason:
                rejection_reasons[proposal.key] = proposal.reason

        selected_key = selected.key
        selected_mode = selected.status
        shield_intervened = False
        if self._violates_hard_stop(observation, command):
            command = (0.0, 0.0)
            selected_key = "shield_stop"
            selected_mode = "fallback"
            shield_intervened = True
            step_status_counts["fallback"] += 1
            rejection_reasons["shield_stop"] = (
                f"hard stop clearance below {float(self.config.hard_stop_clearance):.3f} m"
            )

        additional_rejected_count = max(len(candidates) - 1, 0)
        if shield_intervened:
            additional_rejected_count += 1
        step_status_counts["rejected"] += additional_rejected_count
        rejected_count = step_status_counts["rejected"]

        failed_count = step_status_counts["failed"]
        unavailable_count = step_status_counts["not_available"]

        self._steps += 1
        _accumulate_counts(self._status_counts, step_status_counts)
        self._selection_counts[selected_key] = self._selection_counts.get(selected_key, 0) + 1
        self._failed_count += failed_count
        self._unavailable_count += unavailable_count
        self._rejected_count += rejected_count
        if shield_intervened:
            self._shield_intervention_count += 1

        self._last_step = {
            "selected_proposal_key": selected_key,
            "selected_mode": selected_mode,
            "candidate_count": len(candidates),
            "rejected_count": rejected_count,
            "failed_count": failed_count,
            "unavailable_count": unavailable_count,
            "shield_intervened": shield_intervened,
            "proposal_status_counts": dict(step_status_counts),
            "proposal_statuses": proposal_statuses,
            "proposal_commands": proposal_commands,
            "risk_score_components": {key: dict(value) for key, value in scores.items()},
            "rejection_reasons": rejection_reasons,
        }
        return float(command[0]), float(command[1])

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-safe episode diagnostics for benchmark metadata."""
        return {
            "steps": int(self._steps),
            "proposal_status_counts": dict(self._status_counts),
            "selection_counts": dict(self._selection_counts),
            "failed_count": int(self._failed_count),
            "unavailable_count": int(self._unavailable_count),
            "rejected_count": int(self._rejected_count),
            "shield_intervention_count": int(self._shield_intervention_count),
            "last_step": deepcopy(self._last_step) if self._last_step is not None else None,
        }

    def _collect_proposals(self, observation: dict[str, Any]) -> list[_Proposal]:
        """Collect one normalized proposal from each configured source.

        Returns:
            list[_Proposal]: Proposal records in configured source order.
        """
        proposals: list[_Proposal] = []
        for source in self.config.proposal_sources:
            key = _normalize_source(source)
            proposal = self._proposal_from_source(key, observation)
            if (
                proposal.status in {"failed", "not_available", "rejected"}
                and key in self.config.mandatory_sources
            ):
                reason = proposal.reason or proposal.status.replace("_", " ")
                raise RuntimeError(f"mandatory proposal source '{key}' unavailable: {reason}")
            proposals.append(proposal)
        return proposals

    def _proposal_from_source(self, key: str, observation: dict[str, Any]) -> _Proposal:
        """Build a proposal from one named source.

        Returns:
            _Proposal: Command proposal or diagnostic status for the source.
        """
        status = self._source_mode(key)
        if key == "goal":
            return self._command_proposal(
                key=key, status=status, command=self._goal_command(observation)
            )
        if key == "risk_dwa":
            try:
                command = self.risk_dwa.plan(observation)
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                return _Proposal(key=key, status="failed", reason=str(exc))
            return self._command_proposal(key=key, status=status, command=command)
        return _Proposal(key=key, status="not_available", reason="not available")

    def _command_proposal(
        self,
        *,
        key: str,
        status: str,
        command: object,
    ) -> _Proposal:
        """Return a proposal with a normalized command or a rejected diagnostic."""
        try:
            values = np.asarray(command, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            return _Proposal(key=key, status="rejected", reason=f"invalid command: {exc}")
        if values.size != 2:
            return _Proposal(
                key=key,
                status="rejected",
                reason=f"invalid command shape: expected 2 values, got {values.size}",
            )
        linear, angular = float(values[0]), float(values[1])
        if not np.isfinite([linear, angular]).all():
            return _Proposal(key=key, status="rejected", reason="non-finite command")
        if abs(linear) > float(self.config.max_linear_speed) or abs(angular) > float(
            self.config.max_angular_speed
        ):
            return _Proposal(
                key=key,
                status="rejected",
                reason=(
                    "outside configured command bounds: "
                    f"linear={linear:.3f} max={float(self.config.max_linear_speed):.3f}, "
                    f"angular={angular:.3f} max={float(self.config.max_angular_speed):.3f}"
                ),
            )
        return _Proposal(key=key, status=status, command=(linear, angular))

    def _source_mode(self, key: str) -> str:
        """Resolve benchmark provenance mode for a proposal source.

        Returns:
            str: One of native, adapter, fallback, or degraded.
        """
        if key in self.config.fallback_sources:
            return "fallback"
        if key in self.config.degraded_sources:
            return "degraded"
        if key == "goal":
            return "native"
        return "adapter"

    def _goal_command(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Compute the native goal-seeking proposal command.

        Returns:
            tuple[float, float]: Linear and angular command.
        """
        robot_pos, heading, goal = _robot_goal_heading(observation)
        to_goal = goal - robot_pos
        distance = float(np.linalg.norm(to_goal))
        if distance <= float(self.config.goal_tolerance):
            return 0.0, 0.0

        target_heading = float(np.arctan2(to_goal[1], to_goal[0]))
        heading_error = _wrap_angle(target_heading - heading)
        linear = min(float(self.config.max_linear_speed), distance)
        angular = float(
            np.clip(
                heading_error * float(self.config.heading_gain),
                -float(self.config.max_angular_speed),
                float(self.config.max_angular_speed),
            )
        )
        return linear, angular

    def _score_command(
        self,
        command: tuple[float, float],
        observation: dict[str, Any],
    ) -> dict[str, float]:
        """Score one candidate command by progress, heading, clearance, and smoothness.

        Returns:
            dict[str, float]: Total score and component diagnostics.
        """
        robot_pos, heading, goal = _robot_goal_heading(observation)
        goal_vec = goal - robot_pos
        distance = float(np.linalg.norm(goal_vec))
        if distance <= 1e-9:
            goal_dir = np.array([np.cos(heading), np.sin(heading)], dtype=float)
        else:
            goal_dir = goal_vec / distance
        velocity = np.array(
            [float(command[0]) * np.cos(heading), float(command[0]) * np.sin(heading)],
            dtype=float,
        )
        progress = float(np.dot(velocity, goal_dir))
        command_heading = _wrap_angle(heading + float(command[1]) * 0.2)
        target_heading = float(np.arctan2(goal_dir[1], goal_dir[0]))
        heading_alignment = float(np.cos(_wrap_angle(target_heading - command_heading)))
        min_clearance = _min_ped_clearance(observation)
        clearance_bonus = min(min_clearance, 2.0) if np.isfinite(min_clearance) else 2.0
        angular_penalty = abs(float(command[1]))
        total = (
            float(self.config.goal_progress_weight) * progress
            + float(self.config.heading_weight) * heading_alignment
            + float(self.config.clearance_weight) * clearance_bonus
            - float(self.config.angular_penalty_weight) * angular_penalty
        )
        return {
            "total": float(total),
            "goal_progress": float(progress),
            "heading_alignment": float(heading_alignment),
            "ped_clearance": float(clearance_bonus),
            "angular_penalty": float(angular_penalty),
        }

    def _violates_hard_stop(
        self,
        observation: dict[str, Any],
        command: tuple[float, float],
    ) -> bool:
        """Return whether a moving command violates hard-stop clearance."""
        if float(self.config.hard_stop_clearance) <= 0.0:
            return False
        moving = abs(float(command[0])) > 1e-6 or abs(float(command[1])) > 1e-6
        if not moving:
            return False
        return _min_ped_clearance(observation) < float(self.config.hard_stop_clearance)

    def _raise_no_candidate(self, proposals: list[_Proposal]) -> None:
        """Raise an error summarizing why no proposal could be selected."""
        reasons = {
            proposal.key: proposal.reason or proposal.status
            for proposal in proposals
            if proposal.command is None
        }
        raise RuntimeError(
            "No available policy_stack_v1 proposals."
            if not reasons
            else f"No available policy_stack_v1 proposals: {reasons}"
        )


def build_policy_stack_v1_build_config(
    cfg: dict[str, Any] | None,
) -> PolicyStackV1BuildConfig:
    """Build stack and child proposal-source configs from a mapping payload.

    Returns:
        PolicyStackV1BuildConfig: Parsed stack and Risk-DWA child config.
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    stack_raw = cfg.get("policy_stack", {}) if isinstance(cfg.get("policy_stack"), dict) else {}
    merged_stack = {**cfg, **stack_raw}

    policy_stack = PolicyStackV1Config(
        proposal_sources=_tuple_of_strings(
            merged_stack.get("proposal_sources"),
            default=PolicyStackV1Config.proposal_sources,
        ),
        mandatory_sources=_tuple_of_strings(merged_stack.get("mandatory_sources")),
        optional_sources=_tuple_of_strings(merged_stack.get("optional_sources")),
        fallback_sources=_tuple_of_strings(merged_stack.get("fallback_sources")),
        degraded_sources=_tuple_of_strings(merged_stack.get("degraded_sources")),
        max_linear_speed=float(merged_stack.get("max_linear_speed", 1.0)),
        max_angular_speed=float(merged_stack.get("max_angular_speed", 1.2)),
        goal_tolerance=float(merged_stack.get("goal_tolerance", 0.25)),
        heading_gain=float(merged_stack.get("heading_gain", 1.4)),
        hard_stop_clearance=float(merged_stack.get("hard_stop_clearance", 0.35)),
        goal_progress_weight=float(merged_stack.get("goal_progress_weight", 4.0)),
        heading_weight=float(merged_stack.get("heading_weight", 0.8)),
        angular_penalty_weight=float(merged_stack.get("angular_penalty_weight", 0.2)),
        clearance_weight=float(merged_stack.get("clearance_weight", 0.8)),
    )
    risk_raw = cfg.get("risk_dwa", {}) if isinstance(cfg.get("risk_dwa"), dict) else {}
    return PolicyStackV1BuildConfig(
        policy_stack=policy_stack,
        risk_dwa=build_risk_dwa_config(risk_raw),
    )


def _robot_goal_heading(observation: dict[str, Any]) -> tuple[np.ndarray, float, np.ndarray]:
    """Extract robot position, heading, and active goal from a SocNav observation.

    Returns:
        tuple[np.ndarray, float, np.ndarray]: Robot XY, heading, and goal XY.
    """
    robot = observation.get("robot") if isinstance(observation.get("robot"), dict) else {}
    goal_state = observation.get("goal") if isinstance(observation.get("goal"), dict) else {}
    robot_pos = _as_vector(robot.get("position"), pad=2)[:2]
    heading = float(_as_vector(robot.get("heading"), pad=1)[0])
    goal_next = _as_vector(goal_state.get("next"), pad=2)[:2]
    goal_current = _as_vector(goal_state.get("current"), pad=2)[:2]
    goal = goal_next if float(np.linalg.norm(goal_next - robot_pos)) > 1e-6 else goal_current
    return robot_pos, heading, goal


def _min_ped_clearance(observation: dict[str, Any]) -> float:
    """Return nearest pedestrian distance from the robot.

    Returns:
        float: Minimum pedestrian clearance, or ``inf`` when no valid pedestrians exist.
    """
    robot = observation.get("robot") if isinstance(observation.get("robot"), dict) else {}
    pedestrians = (
        observation.get("pedestrians") if isinstance(observation.get("pedestrians"), dict) else {}
    )
    robot_pos = _as_vector(robot.get("position"), pad=2)[:2]
    positions = np.asarray(pedestrians.get("positions", []), dtype=float)
    if positions.ndim == 1 and positions.size % 2 == 0:
        positions = positions.reshape(-1, 2)
    if positions.ndim != 2 or positions.shape[-1] != 2 or positions.size == 0:
        return float("inf")
    dists = np.linalg.norm(positions - robot_pos[None, :], axis=1)
    return float(np.min(dists)) if dists.size else float("inf")


def _as_vector(value: Any, *, pad: int) -> np.ndarray:
    """Coerce optional payloads to a flat vector with zero padding.

    Returns:
        np.ndarray: One-dimensional vector with at least ``pad`` values.
    """
    arr = np.asarray([] if value is None else value, dtype=float).reshape(-1)
    if arr.size < pad:
        arr = np.pad(arr, (0, pad - arr.size), constant_values=0.0)
    return arr


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to the ``[-pi, pi)`` interval.

    Returns:
        float: Wrapped angle in radians.
    """
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _tuple_of_strings(value: Any, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Normalize scalar or sequence config values into source-key tuples.

    Returns:
        tuple[str, ...]: Normalized source identifiers, or ``default``.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return (_normalize_source(value),)
    if isinstance(value, (list, tuple)):
        return tuple(_normalize_source(item) for item in value)
    return default


def _normalize_source(value: Any) -> str:
    """Normalize one proposal source identifier.

    Returns:
        str: Lowercase stripped source key.
    """
    return str(value).strip().lower()


def _empty_status_counts() -> dict[str, int]:
    """Create a zero-valued status counter mapping.

    Returns:
        dict[str, int]: Status keys initialized to zero.
    """
    return dict.fromkeys(_STATUS_KEYS, 0)


def _accumulate_counts(target: dict[str, int], delta: dict[str, int]) -> None:
    """Add one status counter mapping into another in place."""
    for key in _STATUS_KEYS:
        target[key] = int(target.get(key, 0)) + int(delta.get(key, 0))


__all__ = [
    "PolicyStackV1Adapter",
    "PolicyStackV1BuildConfig",
    "PolicyStackV1Config",
    "build_policy_stack_v1_build_config",
]
