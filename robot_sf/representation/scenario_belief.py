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

from robot_sf.sensor.socnav_observation import _map_position_cap

SCENARIO_BELIEF_SCHEMA_VERSION = "scenario-belief.v1"
DESIGN_PARENT_ISSUE = 1966


class VisibilityState(StrEnum):
    """Visibility vocabulary for partial-observability-aware beliefs."""

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    OUT_OF_RANGE = "out_of_range"
    OUTSIDE_FOV = "outside_fov"
    UNKNOWN = "unknown"


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

    @classmethod
    def point(cls, value: Any, *, confidence: float, variance: float) -> Estimate2D:
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

    def to_debug_dict(self) -> dict[str, Any]:
        """Return deterministic JSON/YAML-ready entity metadata."""
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
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

    def _clip_position(self, values: np.ndarray) -> np.ndarray:
        """Clip position-like values to the representable map extent.

        Returns:
            np.ndarray: Clipped float32 position array.
        """
        position_cap = np.asarray(self.map_size, dtype=np.float32)
        return np.clip(values, 0.0, position_cap).astype(np.float32)


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
    )


def _scenario_belief_from_simulator(
    simulator: Any,
    *,
    env_config: Any,
    max_pedestrians: int,
    robot_index: int,
    source: BeliefSource,
    visibility_states: tuple[VisibilityState, ...] | None,
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

    agents = tuple(
        _agent_belief(
            idx=idx,
            position=position,
            velocity=ped_velocities[idx],
            radius=ped_radius,
            source=source,
            visibility_state=visibility_states[idx],
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
            velocity=Estimate2D.point(robot_velocity, confidence=1.0, variance=1e-6),
            radius=float(getattr(robot.config, "radius", 0.0)),
            heading=robot_heading,
            existence_probability=1.0,
            visibility_state=VisibilityState.VISIBLE,
            source=source,
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
) -> EntityBelief:
    """Build one pedestrian belief, degrading uncertainty for non-visible agents.

    Returns:
        EntityBelief: Pedestrian belief with source and visibility metadata.
    """
    visible = visibility_state is VisibilityState.VISIBLE
    confidence = 0.98 if visible else 0.35
    variance = 0.01 if visible else 1.0
    missing_fields = () if visible else ("policy_position", "policy_velocity")
    return EntityBelief(
        entity_id=f"ped_{idx:03d}",
        entity_type="pedestrian",
        position=Estimate2D.point(position, confidence=confidence, variance=variance),
        velocity=Estimate2D.point(velocity, confidence=confidence, variance=variance),
        radius=radius,
        heading=0.0,
        existence_probability=confidence,
        visibility_state=visibility_state,
        source=source,
        last_observed_age_s=0.0 if visible else 1.0,
        missing_fields=missing_fields,
    )


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
