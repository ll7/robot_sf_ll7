"""Inject a ScenarioBelief uncertainty sidecar into the benchmark stream_gap policy (#3556).

Plain-language summary: #3471 (PR #3553) showed, in a controlled scripted scenario, that *dropping*
uncertain agents from the ``stream_gap`` planner's reasoning is less safe than *retaining* them. This
hook lets the **real benchmark runner** (`robot_sf.benchmark.map_runner`) reproduce that
``oracle`` / ``uncertain_retained`` / ``uncertain_dropped`` contrast: it builds a ScenarioBelief from
each benchmark observation, applies a configurable field-of-view / range uncertainty source, projects
it through the production ``project_scenario_belief_for_planner`` consumer, and merges the resulting
uncertainty sidecar into the observation the planner receives.

The three modes share one ground-truth observation and differ only in what the planner trusts:

* ``oracle`` -- certain belief; planner reacts to every agent; gate off.
* ``uncertain_retained`` -- agents outside the configured FOV/range are marked uncertain, but the
  planner's uncertainty gate is OFF, so it fail-closed *keeps* them (conservative).
* ``uncertain_dropped`` -- same uncertain belief, gate ON, so low-confidence agents are dropped.

Fail-closed: if the observation cannot be projected, the sidecar is omitted (the planner then keeps
every deterministic agent — the conservative default), never silently dropping agents.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from robot_sf.planner.scenario_belief_adapter import project_scenario_belief_for_planner
from robot_sf.representation import scenario_belief_from_simulator_oracle

PLANNER_KEY = "stream_gap"

#: Existence confidence assigned to an out-of-view agent (below the 0.5 gate threshold).
_DEGRADED_EXISTENCE = 0.2

#: Belief modes -> (use the visibility-limited uncertainty source?, enable the planner gate?).
BELIEF_MODES: dict[str, dict[str, bool]] = {
    "oracle": {"visibility_limited": False, "gate": False},
    "uncertain_retained": {"visibility_limited": True, "gate": False},
    "uncertain_dropped": {"visibility_limited": True, "gate": True},
}

# Defaults for the configurable FOV/range uncertainty source (overridable via the algo config).
DEFAULT_FOV_DEGREES = 120.0
DEFAULT_MAX_RANGE_M: float | None = None
DEFAULT_PED_RADIUS = 0.3
DEFAULT_MAX_PEDESTRIANS = 16


def _as_xy(value: Any) -> np.ndarray:
    """Coerce a value into an ``(n, 2)`` float array, or an empty array.

    Returns:
        np.ndarray: An ``(n, 2)`` float32 array, empty when the value is not 2-D xy data.
    """
    arr = np.asarray(value, dtype=np.float32) if value is not None else np.zeros((0, 2), np.float32)
    if arr.ndim == 1 and arr.size >= 2:
        arr = arr.reshape(1, -1)[:, :2]
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float32)
    return arr[:, :2].astype(np.float32)


def _env_config_for_belief(*, ped_radius: float) -> Any:
    """Build the minimal env-config object the oracle ScenarioBelief constructor reads.

    Returns:
        Any: A namespace exposing ``sim_config.ped_radius``.
    """
    return SimpleNamespace(sim_config=SimpleNamespace(ped_radius=ped_radius))


def _is_out_of_view(
    point_xy: np.ndarray,
    *,
    robot_pos: np.ndarray,
    robot_heading: float,
    fov_degrees: float,
    max_range_m: float | None,
) -> bool:
    """Return whether an agent is outside the robot's field of view or sensing range.

    Returns:
        bool: True when the agent is beyond ``max_range_m`` or outside the FOV cone.
    """
    rel = np.asarray(point_xy, dtype=float) - np.asarray(robot_pos, dtype=float)
    dist = float(np.linalg.norm(rel))
    if max_range_m is not None and dist > float(max_range_m):
        return True
    if fov_degrees < 360.0 and dist > 1e-6:
        bearing = float(np.arctan2(rel[1], rel[0]))
        delta = (bearing - robot_heading + np.pi) % (2.0 * np.pi) - np.pi
        if abs(delta) > float(np.deg2rad(fov_degrees) / 2.0):
            return True
    return False


def _degrade_out_of_view_agents(
    belief: Any,
    *,
    robot_pos: np.ndarray,
    robot_heading: float,
    fov_degrees: float,
    max_range_m: float | None,
) -> Any:
    """Lower the existence confidence of agents outside the FOV/range (keeping all agents aligned).

    This keeps the uncertainty rows 1:1 with the observation's pedestrians (unlike dropping agents),
    so the planner's uncertainty gate can drop the low-confidence ones in the dropped mode.

    Returns:
        Any: A belief with out-of-view agents degraded; the original belief when none changed.
    """
    agents = list(belief.agents)
    changed = False
    for idx, agent in enumerate(agents):
        if _is_out_of_view(
            np.asarray(agent.position.mean_xy, dtype=float),
            robot_pos=robot_pos,
            robot_heading=robot_heading,
            fov_degrees=fov_degrees,
            max_range_m=max_range_m,
        ):
            agents[idx] = replace(agent, existence_probability=_DEGRADED_EXISTENCE)
            changed = True
    return replace(belief, agents=tuple(agents)) if changed else belief


def simulator_from_observation(obs: dict[str, Any], *, ped_radius: float) -> SimpleNamespace:
    """Build the lightweight simulator stand-in the belief constructor consumes from an observation.

    Returns:
        SimpleNamespace: A simulator-like object exposing ped/robot/goal state for belief construction.
    """
    robot = obs.get("robot") if isinstance(obs.get("robot"), dict) else {}
    goal = obs.get("goal") if isinstance(obs.get("goal"), dict) else {}
    peds = obs.get("pedestrians") if isinstance(obs.get("pedestrians"), dict) else {}

    robot_pos = np.asarray(robot.get("position", [0.0, 0.0]), dtype=np.float32).reshape(-1)[:2]
    heading = float(np.asarray(robot.get("heading", [0.0]), dtype=np.float32).reshape(-1)[0])
    goal_cur = np.asarray(
        goal.get("current", goal.get("next", [0.0, 0.0])), dtype=np.float32
    ).reshape(-1)[:2]

    ped_pos = _as_xy(peds.get("positions"))
    ped_vel = _as_xy(peds.get("velocities"))
    if ped_vel.shape != ped_pos.shape:
        ped_vel = np.zeros_like(ped_pos)

    return SimpleNamespace(
        ped_pos=ped_pos,
        ped_vel=ped_vel,
        robots=[
            SimpleNamespace(
                pose=((float(robot_pos[0]), float(robot_pos[1])), heading),
                current_speed=np.array([0.0, 0.0], dtype=np.float32),
                config=SimpleNamespace(radius=ped_radius),
            )
        ],
        goal_pos=[goal_cur],
        next_goal_pos=[goal_cur],
        map_def=SimpleNamespace(width=40.0, height=40.0, obstacles=[]),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )


def augment_observation_with_belief(
    obs: dict[str, Any],
    *,
    mode: str,
    fov_degrees: float = DEFAULT_FOV_DEGREES,
    max_range_m: float | None = DEFAULT_MAX_RANGE_M,
    ped_radius: float = DEFAULT_PED_RADIUS,
    max_pedestrians: int = DEFAULT_MAX_PEDESTRIANS,
) -> dict[str, Any]:
    """Return ``obs`` with a ScenarioBelief uncertainty sidecar merged in for ``mode``.

    Fail-closed: any construction/projection failure returns the observation unchanged (no sidecar),
    so the planner keeps every deterministic agent rather than dropping one on bad data.
    """
    if mode not in BELIEF_MODES:
        return obs
    peds = obs.get("pedestrians")
    if not isinstance(peds, dict) or _as_xy(peds.get("positions")).shape[0] == 0:
        return obs

    try:
        simulator = simulator_from_observation(obs, ped_radius=ped_radius)
        env_config = _env_config_for_belief(ped_radius=ped_radius)
        belief = scenario_belief_from_simulator_oracle(
            simulator, env_config=env_config, max_pedestrians=max_pedestrians
        )
        if BELIEF_MODES[mode]["visibility_limited"]:
            robot_pose = simulator.robots[0].pose
            belief = _degrade_out_of_view_agents(
                belief,
                robot_pos=np.asarray(robot_pose[0], dtype=float),
                robot_heading=float(robot_pose[1]),
                fov_degrees=fov_degrees,
                max_range_m=max_range_m,
            )
        projection = project_scenario_belief_for_planner(belief, planner_key=PLANNER_KEY)
    except (ValueError, IndexError, KeyError, TypeError):
        return obs

    proj_peds = projection.observation.get("pedestrians", {})
    rows = proj_peds.get("uncertainty") if isinstance(proj_peds, dict) else None
    if not isinstance(rows, list) or not rows:
        return obs

    # Merge only the uncertainty sidecar; keep the runner's real positions/velocities/count.
    new_peds = deepcopy(peds)
    new_peds["uncertainty"] = rows
    new_peds["uncertainty_compatibility"] = proj_peds.get("uncertainty_compatibility")
    new_obs = deepcopy(obs)
    new_obs["pedestrians"] = new_peds
    return new_obs


class BeliefModeStreamGapAdapter:
    """Wrap a ``StreamGapPlannerAdapter`` to inject the belief uncertainty sidecar before planning.

    Delegates every other attribute (reset/diagnostics/bind_env/close) to the inner adapter so the
    benchmark policy wiring is unchanged.
    """

    def __init__(
        self,
        inner: Any,
        *,
        mode: str,
        fov_degrees: float = DEFAULT_FOV_DEGREES,
        max_range_m: float | None = DEFAULT_MAX_RANGE_M,
        ped_radius: float = DEFAULT_PED_RADIUS,
        max_pedestrians: int = DEFAULT_MAX_PEDESTRIANS,
    ) -> None:
        """Store the inner adapter and the belief-mode parameters."""
        if mode not in BELIEF_MODES:
            raise ValueError(f"unknown belief mode: {mode}")
        self._inner = inner
        self._mode = mode
        self._params = {
            "fov_degrees": fov_degrees,
            "max_range_m": max_range_m,
            "ped_radius": ped_radius,
            "max_pedestrians": max_pedestrians,
        }

    def plan(self, observation: dict[str, Any]) -> tuple[float, float]:
        """Augment the observation with the belief sidecar, then delegate to the inner planner.

        Returns:
            tuple[float, float]: The inner planner's (linear, angular) command.
        """
        augmented = augment_observation_with_belief(observation, mode=self._mode, **self._params)
        return self._inner.plan(augmented)

    def __getattr__(self, name: str) -> Any:
        """Delegate any non-overridden attribute to the inner adapter.

        Returns:
            Any: The attribute resolved on the inner adapter.
        """
        return getattr(self._inner, name)
