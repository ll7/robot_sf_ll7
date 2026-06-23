"""Minimal ScenarioBelief contract for sensor-agnostic observation projections.

This MVP is intentionally small: it proves that a simulator oracle path and a
visibility-limited perception path can emit the same semantic contract and the
same policy-observation key set while differing through uncertainty,
provenance, visibility, and missing-data fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from robot_sf.common.issue_provenance import SCENARIO_BELIEF_DESIGN_PARENT_ISSUE
from robot_sf.sensor.socnav_observation import _map_position_cap

SCENARIO_BELIEF_SCHEMA_VERSION = "scenario-belief.v1"
DESIGN_PARENT_ISSUE = SCENARIO_BELIEF_DESIGN_PARENT_ISSUE


class VisibilityState(StrEnum):
    """Visibility vocabulary for partial-observability-aware beliefs."""

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    OUT_OF_RANGE = "out_of_range"
    OUTSIDE_FOV = "outside_fov"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TrackedAgentMetadata:
    """Tracking-specific diagnostic metadata for tracked-agent beliefs.

    This is representation/diagnostic-only evidence. It does not claim real-sensor
    calibration, benchmark improvement, or SNQI movement.
    """

    track_id: str
    detection_count: int = 0
    missed_detections: int = 0
    track_age_s: float = 0.0
    last_detection_s: float = 0.0
    is_coasted: bool = False

    def to_debug_dict(self) -> dict[str, Any]:
        """Return deterministic JSON/YAML-ready tracking metadata."""
        return {
            "track_id": self.track_id,
            "detection_count": self.detection_count,
            "missed_detections": self.missed_detections,
            "track_age_s": round(float(self.track_age_s), 6),
            "last_detection_s": round(float(self.last_detection_s), 6),
            "is_coasted": self.is_coasted,
        }


@dataclass(frozen=True)
class BeliefSource:
    """Provenance for a value or entity inside a ScenarioBelief."""

    adapter: str
    sensor_ids: tuple[str, ...] = ()
    calibration_status: str = "synthetic"

    def to_debug_dict(self) -> dict[str, Any]:
        """Return deterministic JSON/YAML-ready source metadata."""
        return {
            "adapter": self.adapter,
            "sensor_ids": list(self.sensor_ids),
            "calibration_status": self.calibration_status,
        }


@dataclass(frozen=True)
class Estimate2D:
    """Two-dimensional estimate plus simple uncertainty metadata."""

    mean_xy: tuple[float, float]
    covariance_xy: tuple[tuple[float, float], tuple[float, float]]
    confidence: float
    frame_id: str = "map"
    units: str = "m"
    covariance_units: str = "m^2"

    @classmethod
    def point(
        cls,
        value: Any,
        *,
        confidence: float,
        variance: float,
        frame_id: str = "map",
        units: str = "m",
        covariance_units: str = "m^2",
    ) -> Estimate2D:
        """Build a point estimate with isotropic covariance.

        Returns:
            Estimate2D: Normalized point estimate and covariance metadata.
        """
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            raise ValueError("Estimate2D requires at least two coordinates")
        variance = float(variance)
        return cls(
            mean_xy=(float(arr[0]), float(arr[1])),
            covariance_xy=((variance, 0.0), (0.0, variance)),
            confidence=float(confidence),
            frame_id=frame_id,
            units=units,
            covariance_units=covariance_units,
        )

    def as_array(self) -> np.ndarray:
        """Return the estimate mean as a float32 XY array."""
        return np.asarray(self.mean_xy, dtype=np.float32)

    def to_debug_dict(self) -> dict[str, Any]:
        """Return deterministic JSON/YAML-ready estimate metadata."""
        return {
            "mean_xy": [round(v, 6) for v in self.mean_xy],
            "covariance_xy": [[round(v, 6) for v in row] for row in self.covariance_xy],
            "confidence": round(self.confidence, 6),
            "frame_id": self.frame_id,
            "units": self.units,
            "covariance_units": self.covariance_units,
        }


@dataclass(frozen=True)
class EntityBelief:
    """Belief over one robot, pedestrian, or goal-like entity."""

    entity_id: str
    entity_type: str
    position: Estimate2D
    velocity: Estimate2D
    radius: float
    heading: float
    existence_probability: float
    visibility_state: VisibilityState
    source: BeliefSource
    last_observed_age_s: float = 0.0
    missing_fields: tuple[str, ...] = ()
    tracking: TrackedAgentMetadata | None = None
    class_probabilities: tuple[tuple[str, float], ...] = ()

    def to_debug_dict(self) -> dict[str, Any]:
        """Return deterministic JSON/YAML-ready entity metadata."""
        class_probabilities = self.class_probabilities or ((self.entity_type, 1.0),)
        result: dict[str, Any] = {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "class_probabilities": {
                label: round(float(probability), 6)
                for label, probability in sorted(class_probabilities)
            },
            "position": self.position.to_debug_dict(),
            "velocity": self.velocity.to_debug_dict(),
            "radius": round(float(self.radius), 6),
            "heading": round(float(self.heading), 6),
            "existence_probability": round(float(self.existence_probability), 6),
            "visibility_state": self.visibility_state.value,
            "source": self.source.to_debug_dict(),
            "last_observed_age_s": round(float(self.last_observed_age_s), 6),
            "missing_fields": list(self.missing_fields),
        }
        if self.tracking is not None:
            result["tracking"] = self.tracking.to_debug_dict()
        return result


@dataclass(frozen=True)
class GoalBelief:
    """Belief over current and next navigation goals."""

    current: Estimate2D
    next: Estimate2D
    source: BeliefSource

    def to_debug_dict(self) -> dict[str, Any]:
        """Return deterministic JSON/YAML-ready goal metadata."""
        return {
            "current": self.current.to_debug_dict(),
            "next": self.next.to_debug_dict(),
            "source": self.source.to_debug_dict(),
        }


@dataclass(frozen=True)
class AgentProjectionDiff:
    """Per-agent comparison entry for projection/diagnostics inspection.

    This is representation/diagnostic-only evidence. It does not claim benchmark
    improvement, SNQI movement, or paper-facing performance.
    """

    entity_id: str
    position_diff: float
    velocity_diff: float
    visibility_oracle: str
    visibility_partial: str
    confidence_oracle: float
    confidence_partial: float
    missing_fields_oracle: tuple[str, ...]
    missing_fields_partial: tuple[str, ...]
    in_policy_oracle: bool
    in_policy_partial: bool

    def to_debug_dict(self) -> dict[str, Any]:
        """Return deterministic JSON/YAML-ready agent diff entry."""
        return {
            "entity_id": self.entity_id,
            "position_diff": round(float(self.position_diff), 6),
            "velocity_diff": round(float(self.velocity_diff), 6),
            "visibility_oracle": self.visibility_oracle,
            "visibility_partial": self.visibility_partial,
            "confidence_oracle": round(float(self.confidence_oracle), 6),
            "confidence_partial": round(float(self.confidence_partial), 6),
            "missing_fields_oracle": list(self.missing_fields_oracle),
            "missing_fields_partial": list(self.missing_fields_partial),
            "in_policy_oracle": self.in_policy_oracle,
            "in_policy_partial": self.in_policy_partial,
        }


@dataclass(frozen=True)
class ProjectionDiff:
    """Compact structured diff between two ScenarioBelief projections.

    This is representation/diagnostic-only evidence. It does not claim benchmark
    improvement, SNQI movement, or paper-facing performance.
    """

    agent_diffs: tuple[AgentProjectionDiff, ...]
    total_agents_oracle: int
    total_agents_partial: int
    visible_agents_oracle: int
    visible_agents_partial: int
    ego_position_diff: float
    ego_heading_diff: float
    goal_current_diff: float
    goal_next_diff: float
    agent_count_match: bool
    policy_key_set_match: bool

    def to_debug_dict(self) -> dict[str, Any]:
        """Return deterministic JSON/YAML-ready projection diff."""
        return {
            "agent_diffs": [d.to_debug_dict() for d in self.agent_diffs],
            "total_agents_oracle": self.total_agents_oracle,
            "total_agents_partial": self.total_agents_partial,
            "visible_agents_oracle": self.visible_agents_oracle,
            "visible_agents_partial": self.visible_agents_partial,
            "ego_position_diff": round(float(self.ego_position_diff), 6),
            "ego_heading_diff": round(float(self.ego_heading_diff), 6),
            "goal_current_diff": round(float(self.goal_current_diff), 6),
            "goal_next_diff": round(float(self.goal_next_diff), 6),
            "agent_count_match": self.agent_count_match,
            "policy_key_set_match": self.policy_key_set_match,
        }


def compute_clear_tracking_metrics(
    ground_truth: ScenarioBelief,
    observed: ScenarioBelief,
) -> dict[str, Any]:
    """Compute CLEAR-style tracking diagnostics between oracle and observed beliefs.

    The diagnostics use pedestrian entity IDs as the correspondence contract. A
    ground-truth pedestrian is detected only when the observed belief contains
    the same entity ID with ``VISIBLE`` state. The precision term is the mean
    matched centroid error in meters, matching the MOTP intuition without
    claiming calibrated detector fidelity.

    Returns:
        Deterministic diagnostic payload with MOTA and MOTP fields.
    """
    truth_agents = {agent.entity_id: agent for agent in ground_truth.agents}
    observed_visible = {
        agent.entity_id: agent
        for agent in observed.agents
        if agent.visibility_state is VisibilityState.VISIBLE
    }
    truth_ids = set(truth_agents)
    observed_ids = set(observed_visible)
    matched_ids = sorted(truth_ids & observed_ids)
    missed = sorted(truth_ids - observed_ids)
    false_positive = sorted(observed_ids - truth_ids)

    centroid_errors = [
        float(
            np.linalg.norm(
                observed_visible[entity_id].position.as_array()
                - truth_agents[entity_id].position.as_array()
            )
        )
        for entity_id in matched_ids
    ]
    denominator = len(truth_agents)
    id_switches = 0
    mota = (
        1.0
        if denominator == 0
        else 1.0 - (len(missed) + len(false_positive) + id_switches) / float(denominator)
    )
    return {
        "schema_version": "clear-tracking-metrics.v1",
        "enabled": True,
        "ground_truth_count": denominator,
        "detection_count": len(matched_ids),
        "missed_detection_count": len(missed),
        "false_positive_count": len(false_positive),
        "id_switch_count": id_switches,
        "mota": float(max(0.0, min(1.0, mota))),
        "motp_m": float(np.mean(centroid_errors)) if centroid_errors else float("nan"),
        "motp_match_count": len(centroid_errors),
    }


@dataclass(frozen=True)
class ScenarioBelief:
    """Sensor-agnostic semantic scenario contract.

    `to_socnav_struct()` is the first policy projection. It deliberately keeps the existing
    `SOCNAV_STRUCT` key layout and leaves richer uncertainty/provenance fields to debug output.
    """

    frame_id: str
    sim_time_s: float
    timestep_s: float
    map_size: tuple[float, float]
    ego: EntityBelief
    ego_speed: tuple[float, ...]
    goals: tuple[GoalBelief, ...]
    agents: tuple[EntityBelief, ...]
    source_summary: BeliefSource
    max_pedestrians: int
    pedestrian_radius: float
    schema_version: str = SCENARIO_BELIEF_SCHEMA_VERSION
    design_parent_issue: int = DESIGN_PARENT_ISSUE

    def diagnostic_summary(self) -> dict[str, Any]:
        """Return compact diagnostic summary for quick representation-quality inspection.

        This is diagnostic-only evidence. It does not claim benchmark improvement,
        SNQI movement, or paper-facing performance.
        """
        vis_counts: dict[str, int] = {}
        for state in VisibilityState:
            vis_counts[state.value] = 0
        for agent in self.agents:
            vis_counts[agent.visibility_state.value] += 1

        return {
            "frame_id": self.frame_id,
            "sim_time_s": round(float(self.sim_time_s), 6),
            "schema_version": self.schema_version,
            "adapter": self.source_summary.adapter,
            "total_agents": len(self.agents),
            "visible_count": vis_counts[VisibilityState.VISIBLE.value],
            "occluded_count": vis_counts[VisibilityState.OCCLUDED.value],
            "out_of_range_count": vis_counts[VisibilityState.OUT_OF_RANGE.value],
            "outside_fov_count": vis_counts[VisibilityState.OUTSIDE_FOV.value],
            "unknown_count": vis_counts[VisibilityState.UNKNOWN.value],
            "agents_with_missing_data": sum(1 for a in self.agents if a.missing_fields),
            "agents_not_observed_this_step": sum(
                1 for a in self.agents if a.last_observed_age_s > 0.0
            ),
            "agents_with_tracking_meta": sum(1 for a in self.agents if a.tracking is not None),
            "coasted_agents": sum(
                1 for a in self.agents if a.tracking is not None and a.tracking.is_coasted
            ),
        }

    def policy_projection_keys(self) -> tuple[str, ...]:
        """Return the stable top-level key set used by the SOCNAV_STRUCT projection."""
        return ("goal", "map", "pedestrians", "robot", "sim")

    def to_socnav_struct(self) -> dict[str, Any]:
        """Project the belief to the existing SocNav structured observation key layout.

        Returns:
            dict[str, Any]: Policy-facing observation with the existing SOCNAV_STRUCT keys.
        """
        visible_agents = tuple(
            agent for agent in self.agents if agent.visibility_state is VisibilityState.VISIBLE
        )
        robot_pos = self._clip_position(self.ego.position.as_array())
        goal = self.goals[0] if self.goals else _empty_goal(self.source_summary)
        current_goal = self._clip_position(goal.current.as_array())
        next_goal = self._clip_position(goal.next.as_array())

        ordered_agents = sorted(
            visible_agents,
            key=lambda agent: (
                float(np.linalg.norm(agent.position.as_array() - robot_pos)),
                agent.entity_id,
            ),
        )[: self.max_pedestrians]
        padded_positions = np.zeros((self.max_pedestrians, 2), dtype=np.float32)
        padded_velocities = np.zeros((self.max_pedestrians, 2), dtype=np.float32)
        if ordered_agents:
            positions = np.asarray(
                [agent.position.mean_xy for agent in ordered_agents], dtype=np.float32
            )
            velocities = np.asarray(
                [agent.velocity.mean_xy for agent in ordered_agents], dtype=np.float32
            )
            padded_positions[: len(ordered_agents)] = self._clip_position(positions)
            padded_velocities[: len(ordered_agents)] = _rotate_world_velocities_to_ego(
                velocities,
                self.ego.heading,
            )

        return {
            "robot": {
                "position": robot_pos,
                "heading": np.array([_wrap_angle(self.ego.heading)], dtype=np.float32),
                "speed": np.asarray(self.ego_speed, dtype=np.float32),
                "velocity_xy": self.ego.velocity.as_array(),
                "angular_velocity": np.array([0.0], dtype=np.float32),
                "radius": np.array([self.ego.radius], dtype=np.float32),
            },
            "goal": {
                "current": current_goal,
                "next": next_goal,
            },
            "pedestrians": {
                "positions": padded_positions,
                "velocities": padded_velocities,
                "radius": np.array([self.pedestrian_radius], dtype=np.float32),
                "count": np.array([float(len(ordered_agents))], dtype=np.float32),
            },
            "map": {
                "size": self._clip_position(np.asarray(self.map_size, dtype=np.float32)),
            },
            "sim": {
                "timestep": np.array([self.timestep_s], dtype=np.float32),
            },
        }

    def to_debug_dict(self) -> dict[str, Any]:
        """Return deterministic debug metadata for JSON/YAML inspection."""
        return {
            "schema_version": self.schema_version,
            "design_parent_issue": self.design_parent_issue,
            "frame_id": self.frame_id,
            "sim_time_s": round(float(self.sim_time_s), 6),
            "timestep_s": round(float(self.timestep_s), 6),
            "map_size": [round(float(v), 6) for v in self.map_size],
            "source_summary": self.source_summary.to_debug_dict(),
            "policy_projection_keys": list(self.policy_projection_keys()),
            "ego": self.ego.to_debug_dict(),
            "goals": [goal.to_debug_dict() for goal in self.goals],
            "agents": [agent.to_debug_dict() for agent in self.agents],
        }

    def to_uncertainty_report(self) -> dict[str, Any]:
        """Project belief into a diagnostic report preserving all uncertainty fields.

        This is the uncertainty-preserving consumer projection for local planner analysis.
        Unlike to_socnav_struct() which drops covariance, class_probabilities, and confidence
        for backward compatibility, this report preserves all uncertainty metadata.

        This is diagnostic-only evidence. It does not claim benchmark improvement,
        SNQI movement, or paper-facing performance.

        Returns:
            dict[str, Any]: Deterministic report with covariance, class probabilities,
                and confidence for ego, goals, and visibility-filtered agents.
        """
        visible_agents = tuple(
            agent for agent in self.agents if agent.visibility_state is VisibilityState.VISIBLE
        )
        robot_pos = self._clip_position(self.ego.position.as_array())

        ordered_agents = sorted(
            visible_agents,
            key=lambda agent: (
                float(np.linalg.norm(agent.position.as_array() - robot_pos)),
                agent.entity_id,
            ),
        )[: self.max_pedestrians]

        return {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "sim_time_s": round(float(self.sim_time_s), 6),
            "ego": {
                "class_probabilities": {
                    lbl: round(float(p), 6) for lbl, p in sorted(_class_probabilities(self.ego))
                },
                "position_covariance_xy": [
                    [round(float(v), 6) for v in row] for row in self.ego.position.covariance_xy
                ],
                "velocity_covariance_xy": [
                    [round(float(v), 6) for v in row] for row in self.ego.velocity.covariance_xy
                ],
                "position_confidence": round(float(self.ego.position.confidence), 6),
                "velocity_confidence": round(float(self.ego.velocity.confidence), 6),
                "heading": round(float(_wrap_angle(self.ego.heading)), 6),
            },
            "goals": [
                {
                    "current_covariance_xy": [
                        [round(float(v), 6) for v in row] for row in g.current.covariance_xy
                    ],
                    "next_covariance_xy": [
                        [round(float(v), 6) for v in row] for row in g.next.covariance_xy
                    ],
                    "current_confidence": round(float(g.current.confidence), 6),
                    "next_confidence": round(float(g.next.confidence), 6),
                }
                for g in self.goals
            ],
            "agents": [
                {
                    "entity_id": a.entity_id,
                    "class_probabilities": {
                        lbl: round(float(p), 6) for lbl, p in sorted(_class_probabilities(a))
                    },
                    "position_covariance_xy": [
                        [round(float(v), 6) for v in row] for row in a.position.covariance_xy
                    ],
                    "velocity_covariance_xy": [
                        [round(float(v), 6) for v in row] for row in a.velocity.covariance_xy
                    ],
                    "position_confidence": round(float(a.position.confidence), 6),
                    "velocity_confidence": round(float(a.velocity.confidence), 6),
                    "existence_probability": round(float(a.existence_probability), 6),
                    "visibility_state": a.visibility_state.value,
                }
                for a in ordered_agents
            ],
        }

    def _clip_position(self, values: np.ndarray) -> np.ndarray:
        """Clip position-like values to the representable map extent.

        Returns:
            np.ndarray: Clipped float32 position array.
        """
        position_cap = np.asarray(self.map_size, dtype=np.float32)
        return np.clip(values, 0.0, position_cap).astype(np.float32)


def _class_probabilities(entity: EntityBelief) -> tuple[tuple[str, float], ...]:
    """Return explicit or default class-probability metadata for an entity."""
    return entity.class_probabilities or ((entity.entity_type, 1.0),)


def _in_policy_projection(
    agents: tuple[EntityBelief, ...],
    robot_pos: np.ndarray,
    max_pedestrians: int,
) -> frozenset[str]:
    """Return entity_ids that pass the visibility + sort + cap policy filter.

    This mirrors the agent-selection logic in ``ScenarioBelief.to_socnav_struct``
    so diagnostic code can check per-agent policy membership without reimplementing
    the full projection.
    """
    visible = [a for a in agents if a.visibility_state is VisibilityState.VISIBLE]
    sorted_visible = sorted(
        visible,
        key=lambda a: (
            float(np.linalg.norm(a.position.as_array() - robot_pos)),
            a.entity_id,
        ),
    )[:max_pedestrians]
    return frozenset(a.entity_id for a in sorted_visible)


def compute_projection_diff(
    oracle: ScenarioBelief,
    partial: ScenarioBelief,
    *,
    max_pedestrians: int | None = None,
) -> ProjectionDiff:
    """Compute a compact structured diff between an oracle and partial-observation belief.

    This is diagnostic-only evidence. It does not claim benchmark improvement,
    SNQI movement, or paper-facing performance.

    Args:
        oracle: High-confidence oracle-source ScenarioBelief.
        partial: Visibility-limited or otherwise degraded ScenarioBelief.
        max_pedestrians: Cap for policy-projection membership (defaults to oracle's).

    Returns:
        ProjectionDiff: Per-agent and summary diff metadata.
    """
    if max_pedestrians is None:
        max_pedestrians = oracle.max_pedestrians

    oracle_by_id = {a.entity_id: a for a in oracle.agents}
    partial_by_id = {a.entity_id: a for a in partial.agents}
    all_ids = sorted(set(oracle_by_id.keys()) | set(partial_by_id.keys()))

    robot_pos_oracle = oracle.ego.position.as_array()
    robot_pos_partial = partial.ego.position.as_array()

    oracle_policy_set = _in_policy_projection(oracle.agents, robot_pos_oracle, max_pedestrians)
    partial_policy_set = _in_policy_projection(partial.agents, robot_pos_partial, max_pedestrians)

    agent_diffs: list[AgentProjectionDiff] = []
    for eid in all_ids:
        oa = oracle_by_id.get(eid)
        pa = partial_by_id.get(eid)

        pos_oa = oa.position.as_array() if oa else np.zeros(2, dtype=np.float32)
        pos_pa = pa.position.as_array() if pa else np.zeros(2, dtype=np.float32)
        vel_oa = oa.velocity.as_array() if oa else np.zeros(2, dtype=np.float32)
        vel_pa = pa.velocity.as_array() if pa else np.zeros(2, dtype=np.float32)

        pos_diff = float(np.linalg.norm(pos_oa - pos_pa))
        vel_diff = float(np.linalg.norm(vel_oa - vel_pa))

        agent_diffs.append(
            AgentProjectionDiff(
                entity_id=eid,
                position_diff=pos_diff,
                velocity_diff=vel_diff,
                visibility_oracle=oa.visibility_state.value
                if oa
                else VisibilityState.UNKNOWN.value,
                visibility_partial=pa.visibility_state.value
                if pa
                else VisibilityState.UNKNOWN.value,
                confidence_oracle=oa.position.confidence if oa else 0.0,
                confidence_partial=pa.position.confidence if pa else 0.0,
                missing_fields_oracle=oa.missing_fields if oa else (),
                missing_fields_partial=pa.missing_fields if pa else (),
                in_policy_oracle=eid in oracle_policy_set,
                in_policy_partial=eid in partial_policy_set,
            )
        )

    ego_pos_diff = float(np.linalg.norm(robot_pos_oracle - robot_pos_partial))
    ego_heading_diff = _wrap_angle(oracle.ego.heading - partial.ego.heading)

    oracle_goal = oracle.goals[0] if oracle.goals else _empty_goal(oracle.source_summary)
    partial_goal = partial.goals[0] if partial.goals else _empty_goal(partial.source_summary)
    goal_current_diff = float(
        np.linalg.norm(
            np.asarray(oracle_goal.current.mean_xy, dtype=np.float32)
            - np.asarray(partial_goal.current.mean_xy, dtype=np.float32),
        )
    )
    goal_next_diff = float(
        np.linalg.norm(
            np.asarray(oracle_goal.next.mean_xy, dtype=np.float32)
            - np.asarray(partial_goal.next.mean_xy, dtype=np.float32),
        )
    )

    oracle_keys = set(oracle.to_socnav_struct().keys())
    partial_keys = set(partial.to_socnav_struct().keys())

    return ProjectionDiff(
        agent_diffs=tuple(agent_diffs),
        total_agents_oracle=len(oracle.agents),
        total_agents_partial=len(partial.agents),
        visible_agents_oracle=sum(
            1 for a in oracle.agents if a.visibility_state is VisibilityState.VISIBLE
        ),
        visible_agents_partial=sum(
            1 for a in partial.agents if a.visibility_state is VisibilityState.VISIBLE
        ),
        ego_position_diff=ego_pos_diff,
        ego_heading_diff=ego_heading_diff,
        goal_current_diff=goal_current_diff,
        goal_next_diff=goal_next_diff,
        agent_count_match=len(oracle.agents) == len(partial.agents),
        policy_key_set_match=oracle_keys == partial_keys,
    )


def scenario_belief_from_simulator_oracle(
    simulator: Any,
    *,
    env_config: Any,
    max_pedestrians: int,
    robot_index: int = 0,
) -> ScenarioBelief:
    """Construct a high-confidence ScenarioBelief from simulator state.

    Returns:
        ScenarioBelief: Oracle-source belief with near-zero synthetic uncertainty.
    """
    source = BeliefSource(adapter="simulator_oracle", sensor_ids=("simulator",))
    return _scenario_belief_from_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=max_pedestrians,
        robot_index=robot_index,
        source=source,
        visibility_states=None,
        apply_tracking_noise=False,
    )


def scenario_belief_from_visibility_limited_simulator(
    simulator: Any,
    *,
    env_config: Any,
    max_pedestrians: int,
    robot_index: int = 0,
) -> ScenarioBelief:
    """Construct a partial-observation ScenarioBelief using synthetic visibility limits.

    Returns:
        ScenarioBelief: Visibility-limited belief with missing/uncertain hidden agents.
    """
    source = BeliefSource(
        adapter="visibility_limited_simulator",
        sensor_ids=("synthetic_visibility",),
        calibration_status="synthetic_not_sensor_calibrated",
    )
    robot_pose = simulator.robots[robot_index].pose
    robot_pos = np.asarray(robot_pose[0], dtype=np.float32)
    robot_heading = float(robot_pose[1])
    ped_positions = np.asarray(simulator.ped_pos, dtype=np.float32)
    visibility_states = _visibility_states(
        ped_positions,
        robot_pos=robot_pos,
        robot_heading=robot_heading,
        env_config=env_config,
    )
    return _scenario_belief_from_simulator(
        simulator,
        env_config=env_config,
        max_pedestrians=max_pedestrians,
        robot_index=robot_index,
        source=source,
        visibility_states=visibility_states,
        apply_tracking_noise=True,
    )


def _scenario_belief_from_simulator(
    simulator: Any,
    *,
    env_config: Any,
    max_pedestrians: int,
    robot_index: int,
    source: BeliefSource,
    visibility_states: tuple[VisibilityState, ...] | None,
    apply_tracking_noise: bool,
) -> ScenarioBelief:
    """Shared simulator-state adapter implementation.

    Returns:
        ScenarioBelief: Belief populated from simulator-like state.
    """
    robot = simulator.robots[robot_index]
    robot_pose = robot.pose
    robot_pos = np.asarray(robot_pose[0], dtype=np.float32)
    robot_heading = float(robot_pose[1])
    robot_speed = _robot_speed(robot)
    robot_velocity = _robot_velocity_xy(robot, robot_heading)
    ped_positions = _pedestrian_positions(simulator)
    ped_velocities = _pedestrian_velocities(simulator, ped_positions)
    map_size = _map_size(simulator)
    timestep_s = float(getattr(simulator.config, "time_per_step_in_secs", 0.0) or 0.0)
    ped_radius = float(getattr(getattr(env_config, "sim_config", None), "ped_radius", 0.0) or 0.0)

    if visibility_states is None:
        visibility_states = (VisibilityState.VISIBLE,) * int(ped_positions.shape[0])
    tracking_noise_std_m = _tracking_noise_std_m(env_config) if apply_tracking_noise else 0.0

    agents = tuple(
        _agent_belief(
            idx=idx,
            position=position,
            velocity=ped_velocities[idx],
            radius=ped_radius,
            source=source,
            visibility_state=visibility_states[idx],
            tracking_noise_std_m=tracking_noise_std_m,
        )
        for idx, position in enumerate(ped_positions)
    )

    goal = np.asarray(simulator.goal_pos[robot_index], dtype=np.float32)
    next_goal = simulator.next_goal_pos[robot_index]
    next_goal_arr = np.zeros(2, dtype=np.float32) if next_goal is None else np.asarray(next_goal)

    return ScenarioBelief(
        frame_id="map",
        sim_time_s=float(getattr(simulator, "sim_time_s", 0.0) or 0.0),
        timestep_s=timestep_s,
        map_size=(float(map_size[0]), float(map_size[1])),
        ego=EntityBelief(
            entity_id=f"robot_{robot_index}",
            entity_type="ego_robot",
            position=Estimate2D.point(robot_pos, confidence=1.0, variance=1e-6),
            velocity=Estimate2D.point(
                robot_velocity,
                confidence=1.0,
                variance=1e-6,
                units="m/s",
                covariance_units="(m/s)^2",
            ),
            radius=float(getattr(robot.config, "radius", 0.0)),
            heading=robot_heading,
            existence_probability=1.0,
            visibility_state=VisibilityState.VISIBLE,
            source=source,
            class_probabilities=(("ego_robot", 1.0),),
        ),
        ego_speed=robot_speed,
        goals=(
            GoalBelief(
                current=Estimate2D.point(goal, confidence=1.0, variance=1e-6),
                next=Estimate2D.point(next_goal_arr, confidence=1.0, variance=1e-6),
                source=source,
            ),
        ),
        agents=agents,
        source_summary=source,
        max_pedestrians=int(max_pedestrians),
        pedestrian_radius=ped_radius,
    )


def _agent_belief(
    *,
    idx: int,
    position: np.ndarray,
    velocity: np.ndarray,
    radius: float,
    source: BeliefSource,
    visibility_state: VisibilityState,
    tracking_noise_std_m: float = 0.0,
) -> EntityBelief:
    """Build one pedestrian belief, degrading uncertainty for non-visible agents.

    Returns:
        EntityBelief: Pedestrian belief with source and visibility metadata.
    """
    visible = visibility_state is VisibilityState.VISIBLE
    confidence = 0.98 if visible else 0.35
    position_noise = float(max(0.0, tracking_noise_std_m)) if visible else 0.0
    variance = max(0.01, position_noise**2) if visible else 1.0
    observed_position = np.asarray(position, dtype=np.float32)
    if position_noise > 0.0:
        direction = 1.0 if idx % 2 == 0 else -1.0
        observed_position = observed_position + np.asarray(
            [direction * position_noise, 0.0],
            dtype=np.float32,
        )
    missing_fields = () if visible else ("policy_position", "policy_velocity")
    tracking = (
        None
        if visible
        else TrackedAgentMetadata(
            track_id=f"track_ped_{idx:03d}",
            detection_count=0,
            missed_detections=3,
            track_age_s=1.0,
            last_detection_s=1.0,
            is_coasted=True,
        )
    )
    return EntityBelief(
        entity_id=f"ped_{idx:03d}",
        entity_type="pedestrian",
        position=Estimate2D.point(observed_position, confidence=confidence, variance=variance),
        velocity=Estimate2D.point(
            velocity,
            confidence=confidence,
            variance=variance,
            units="m/s",
            covariance_units="(m/s)^2",
        ),
        radius=radius,
        heading=0.0,
        existence_probability=confidence,
        visibility_state=visibility_state,
        source=source,
        last_observed_age_s=0.0 if visible else 1.0,
        missing_fields=missing_fields,
        tracking=tracking,
        class_probabilities=(("pedestrian", confidence),),
    )


def _tracking_noise_std_m(env_config: Any) -> float:
    """Return non-negative synthetic tracking centroid noise from config."""
    settings = getattr(env_config, "observation_visibility", None)
    if settings is None:
        return 0.0
    try:
        value = float(getattr(settings, "tracking_noise_std_m", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) and value > 0.0 else 0.0


def _visibility_states(
    ped_positions: np.ndarray,
    *,
    robot_pos: np.ndarray,
    robot_heading: float,
    env_config: Any,
) -> tuple[VisibilityState, ...]:
    """Return synthetic visibility states for each pedestrian."""
    if ped_positions.size == 0:
        return ()
    settings = getattr(env_config, "observation_visibility", None)
    if settings is None or not bool(getattr(settings, "enabled", False)):
        return (VisibilityState.VISIBLE,) * int(ped_positions.shape[0])

    rel = ped_positions - robot_pos
    dists = np.linalg.norm(rel, axis=1)
    max_range_m = getattr(settings, "max_range_m", None)
    fov_degrees = float(getattr(settings, "fov_degrees", 360.0))
    states: list[VisibilityState] = []
    for rel_xy, dist in zip(rel, dists, strict=True):
        if max_range_m is not None and float(dist) > float(max_range_m):
            states.append(VisibilityState.OUT_OF_RANGE)
            continue
        if fov_degrees < 360.0:
            bearing = float(np.arctan2(rel_xy[1], rel_xy[0]))
            delta = _wrap_angle(bearing - robot_heading)
            if abs(delta) > float(np.deg2rad(fov_degrees) / 2.0):
                states.append(VisibilityState.OUTSIDE_FOV)
                continue
        states.append(VisibilityState.VISIBLE)
    return tuple(states)


def _pedestrian_positions(simulator: Any) -> np.ndarray:
    """Return pedestrian positions as a stable two-column array."""
    ped_pos = getattr(simulator, "ped_pos", None)
    if ped_pos is None:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.asarray(ped_pos, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return arr.reshape(-1, 2)


def _pedestrian_velocities(simulator: Any, ped_positions: np.ndarray) -> np.ndarray:
    """Return pedestrian velocities or a zero fallback matching positions."""
    try:
        ped_vel = getattr(simulator, "ped_vel", None)
        if ped_vel is not None:
            vel = np.asarray(ped_vel, dtype=np.float32).reshape(-1, 2)
            if vel.shape == ped_positions.shape:
                return vel
    except (AttributeError, TypeError, ValueError):
        pass
    return np.zeros_like(ped_positions, dtype=np.float32)


def _robot_speed(robot: Any) -> tuple[float, ...]:
    """Return the existing SocNav robot speed field from current_speed."""
    current_speed = getattr(robot, "current_speed", None)
    if current_speed is None:
        return (0.0, 0.0)
    arr = np.asarray(current_speed, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return (0.0, 0.0)
    return tuple(float(v) for v in arr)


def _robot_velocity_xy(robot: Any, heading: float) -> np.ndarray:
    """Return robot world-frame XY velocity from state or current speed."""
    state = getattr(robot, "state", None)
    velocity_xy = getattr(state, "velocity_xy", None)
    if velocity_xy is None:
        velocity_xy = getattr(state, "robot_velocity_xy", None)
    if velocity_xy is None:
        velocity_xy = getattr(robot, "robot_velocity_xy", None)
    if velocity_xy is not None:
        arr = np.asarray(velocity_xy, dtype=np.float32).reshape(-1)
        if arr.size >= 2:
            return arr[:2]
    current_speed = np.asarray(_robot_speed(robot), dtype=np.float32).reshape(-1)
    linear_speed = float(current_speed[0]) if current_speed.size > 0 else 0.0
    return np.array(
        [linear_speed * float(np.cos(heading)), linear_speed * float(np.sin(heading))],
        dtype=np.float32,
    )


def _map_size(simulator: Any) -> np.ndarray:
    """Return map size clipped to the existing SocNav position cap."""
    map_def = getattr(simulator, "map_def", None)
    if map_def is None:
        return np.array([50.0, 50.0], dtype=np.float32)
    raw_size = np.asarray(
        [
            float(getattr(map_def, "width", 50.0) or 50.0),
            float(getattr(map_def, "height", 50.0) or 50.0),
        ],
        dtype=np.float32,
    )
    return np.minimum(raw_size, _map_position_cap(map_def))


def _empty_goal(source: BeliefSource) -> GoalBelief:
    """Return a zero goal placeholder for defensive projections."""
    zero = Estimate2D.point((0.0, 0.0), confidence=0.0, variance=1.0)
    return GoalBelief(current=zero, next=zero, source=source)


def _rotate_world_velocities_to_ego(velocities: np.ndarray, heading: float) -> np.ndarray:
    """Rotate world-frame pedestrian velocities into the ego frame.

    Returns:
        np.ndarray: Ego-frame XY velocities.
    """
    cos_h = float(np.cos(heading))
    sin_h = float(np.sin(heading))
    vx = velocities[:, 0]
    vy = velocities[:, 1]
    return np.stack([cos_h * vx + sin_h * vy, -sin_h * vx + cos_h * vy], axis=1).astype(np.float32)


def _wrap_angle(value: float) -> float:
    """Wrap an angle in radians to [-pi, pi].

    Returns:
        float: Wrapped angle in radians.
    """
    return float(((value + np.pi) % (2.0 * np.pi)) - np.pi)
