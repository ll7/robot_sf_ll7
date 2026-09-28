"""Shared-world multi-robot contract: one world, synchronized actions (issue #9344).

This module is the versioned multi-agent adapter that preserves existing
single-robot behavior while making shared-world semantics explicit:

- one :class:`~robot_sf.sim.simulator.Simulator` per world (``world_id``),
  stable per-episode ``agent_id`` values, and an explicit robot count;
- all observations come from one state snapshot, all actions are collected
  before any mutation, and one step advances the world exactly once;
- action packets are validated (complete, duplicate-free, known agents,
  finite) *before* any state mutation; invalid packets leave the world
  unchanged and raise :class:`SharedWorldPacketError` with a diagnostic;
- per-agent termination/truncation is recorded without resetting any other
  agent; only an explicit :meth:`SharedWorldRunner.reset` resets the world;
- finished agents remain in the world under a zero hold action; the episode
  ends under an explicit any/all completion policy.

A requested world that exceeds spawn capacity is rejected with
:class:`SharedWorldAdmissionError`: worlds are never silently duplicated and
starts are never overlapped. Independent Gym environments, vectorized
throughput, and a shared interacting world stay distinguishable: this runner
is the shared interacting world. It performs no training and redesigns no
reward. Silent fallback from shared-world mode to independent worlds is a
contract violation, not a degraded mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from robot_sf.gym_env.env_util import (
    global_reset_seed,
    init_collision_and_sensors,
    init_spaces,
    reset_episode_counter_for_seed,
)
from robot_sf.robot.robot_state import RobotState
from robot_sf.sim.simulator import init_simulators

if TYPE_CHECKING:
    from collections.abc import Sequence

    from robot_sf.gym_env.env_config import EnvSettings
    from robot_sf.gym_env.unified_config import MultiRobotConfig
    from robot_sf.nav.map_config import MapDefinition

__all__ = [
    "CONTRACT_VERSION",
    "SCHEMA_VERSION",
    "AgentStatus",
    "CompletionPolicy",
    "SharedWorldAdmission",
    "SharedWorldAdmissionError",
    "SharedWorldPacketError",
    "SharedWorldRunner",
    "SharedWorldStepResult",
    "admit_shared_world",
    "assign_agent_ids",
    "validate_action_packet",
]

SCHEMA_VERSION = "shared_world_contract.v1"
CONTRACT_VERSION = "1.0.0"

CompletionPolicy = Literal["any", "all"]


class SharedWorldAdmissionError(ValueError):
    """Raised when a shared world cannot be admitted without guessing capacity."""


class SharedWorldPacketError(ValueError):
    """Raised when an action packet is incomplete, foreign, or non-finite.

    The error names the offending agent IDs so the caller can repair the
    packet. The world is never partially mutated before this error.
    """


@dataclass(frozen=True)
class SharedWorldAdmission:
    """Receipt for one admitted shared world."""

    world_id: str
    requested_robots: int
    admitted_robots: int
    spawn_capacity: int
    schema_version: str = SCHEMA_VERSION


def _require_count(name: str, value: Any) -> int:
    """Return ``value`` as a positive int or raise an admission error."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SharedWorldAdmissionError(f"{name} must be a positive integer, got {value!r}")
    return value


def assign_agent_ids(world_id: str, num_robots: int) -> tuple[str, ...]:
    """Assign stable per-episode agent IDs for one world.

    Returns:
        ``agent_000`` .. ``agent_{n-1:03d}`` in spawn order.
    """
    count = _require_count("num_robots", num_robots)
    if not isinstance(world_id, str) or not world_id:
        raise SharedWorldAdmissionError(f"world_id must be a non-empty string, got {world_id!r}")
    return tuple(f"agent_{index:03d}" for index in range(count))


def admit_shared_world(
    *,
    world_id: str,
    requested_robots: int,
    spawn_capacity: int,
) -> SharedWorldAdmission:
    """Admit a shared world or reject it without guessing capacity.

    Requesting more robots than spawn capacity is rejected: the caller must
    shrink the request or provision a bigger map. Worlds are never duplicated
    and starts never overlapped to satisfy a request.

    Returns:
        Admission receipt with ``admitted_robots == requested_robots``.
    """
    requested = _require_count("requested_robots", requested_robots)
    capacity = _require_count("spawn_capacity", spawn_capacity)
    if not isinstance(world_id, str) or not world_id:
        raise SharedWorldAdmissionError(f"world_id must be a non-empty string, got {world_id!r}")
    if requested > capacity:
        raise SharedWorldAdmissionError(
            f"rejected: spawn capacity insufficient (requested {requested} robots, "
            f"capacity {capacity} starts in world {world_id!r}); refusing to "
            "duplicate worlds or overlap starts"
        )
    return SharedWorldAdmission(
        world_id=world_id,
        requested_robots=requested,
        admitted_robots=requested,
        spawn_capacity=capacity,
    )


def _packet_action_vector(agent_id: str, action: Any) -> np.ndarray:
    """Coerce one packet action to a finite 2-vector or raise a packet error.

    Returns:
        The validated ``[linear, angular]`` command vector.
    """
    if isinstance(action, bool):
        raise SharedWorldPacketError(f"agent {agent_id!r}: action must be numeric, got bool")
    try:
        vector = np.asarray(action, dtype=float).reshape(2)
    except (TypeError, ValueError) as error:
        raise SharedWorldPacketError(
            f"agent {agent_id!r}: action must be a 2-element [linear, angular] "
            f"command, got {action!r}"
        ) from error
    if not np.all(np.isfinite(vector)):
        raise SharedWorldPacketError(f"agent {agent_id!r}: action must be finite, got {action!r}")
    return vector


def validate_action_packet(
    packet: Sequence[tuple[str, Any]],
    expected_ids: Sequence[str],
) -> list[np.ndarray]:
    """Validate a complete action packet before any world mutation.

    The packet is a sequence of ``(agent_id, action)`` pairs so duplicate IDs
    stay detectable (a plain ``dict`` would silently collapse them). Order is
    irrelevant: actions are remapped to stable spawn order, so reordering
    packets never changes the result.

    Returns:
        Action vectors in ``expected_ids`` order.

    Raises:
        SharedWorldPacketError: On missing, duplicate, or foreign agent IDs,
            or on missing/non-finite actions.
    """
    expected = list(expected_ids)
    if not expected:
        raise SharedWorldPacketError("packet validation requires at least one expected agent")
    entries = list(packet)
    seen: set[str] = set()
    by_id: dict[str, np.ndarray] = {}
    for entry in entries:
        try:
            agent_id, action = entry
        except (TypeError, ValueError) as error:
            raise SharedWorldPacketError(
                f"malformed packet entry {entry!r}: expected (agent_id, action)"
            ) from error
        if agent_id in seen:
            raise SharedWorldPacketError(f"duplicate action for agent {agent_id!r}")
        if agent_id not in expected:
            raise SharedWorldPacketError(f"foreign agent {agent_id!r}: expected one of {expected}")
        seen.add(agent_id)
        by_id[agent_id] = _packet_action_vector(agent_id, action)
    missing = [agent_id for agent_id in expected if agent_id not in by_id]
    if missing:
        raise SharedWorldPacketError(
            f"incomplete action packet: missing agents {missing}; refusing partial mutation"
        )
    return [by_id[agent_id] for agent_id in expected]


@dataclass
class AgentStatus:
    """Per-agent lifecycle state after one shared step."""

    agent_id: str
    goal_reached: bool
    collided: bool
    timed_out: bool
    done: bool
    holding: bool


@dataclass(frozen=True)
class SharedWorldStepResult:
    """Outcome of one synchronized world transition."""

    world_id: str
    step_index: int
    statuses: tuple[AgentStatus, ...]
    episode_done: bool
    episode_truncated: bool


class SharedWorldRunner:
    """Own exactly one simulator and step all its robots synchronously.

    Construction uses :func:`create` so admission, ID assignment, and the
    single-simulator invariant stay in one audited place. Use
    :func:`admit_shared_world` first when capacity is not already proven.
    """

    def __init__(self) -> None:
        """Initialize an unbound runner; use :func:`create` instead."""
        raise NotImplementedError("use SharedWorldRunner.create()")

    @classmethod
    def _bind(
        cls,
        *,
        world_id: str,
        agent_ids: tuple[str, ...],
        simulator: Any,
        states: list[RobotState],
        hold_action: np.ndarray,
        completion: CompletionPolicy,
        dt: float,
    ) -> SharedWorldRunner:
        """Bind a validated single-simulator world without re-running admission.

        Returns:
            A runner bound to the given simulator and states.
        """
        runner = cls.__new__(cls)
        runner._world_id = world_id
        runner._agent_ids = agent_ids
        runner._sim = simulator
        runner._states = states
        runner._hold_action = hold_action
        runner._completion = completion
        runner._dt = dt
        runner._step_index = 0
        runner._episode_done = False
        runner._holding = [False] * len(agent_ids)
        return runner

    @classmethod
    def create(
        cls,
        *,
        world_id: str,
        env_config: EnvSettings | MultiRobotConfig,
        map_def: MapDefinition | None = None,
        num_robots: int,
        completion: CompletionPolicy = "any",
        seed: int | None = None,
    ) -> SharedWorldRunner:
        """Create one shared world with ``num_robots`` interacting robots.

        The map must offer at least ``num_robots`` start positions; otherwise
        creation is rejected instead of splitting robots across worlds.
        ``completion="any"`` ends the episode when the first agent finishes;
        ``"all"`` waits for every agent. Finished agents remain in the world
        under a zero hold action until an explicit :meth:`reset`.

        Returns:
            A runner bound to exactly one simulator.
        """
        admission = admit_shared_world(
            world_id=world_id,
            requested_robots=num_robots,
            spawn_capacity=_spawn_capacity(env_config, map_def),
        )
        resolved_map = _resolve_map_def(env_config, map_def)
        with global_reset_seed(seed):
            simulators = init_simulators(
                env_config,
                resolved_map,
                admission.admitted_robots,
                random_start_pos=False,
            )
        if len(simulators) != 1:
            raise SharedWorldAdmissionError(
                f"world {world_id!r}: expected exactly one simulator for "
                f"{admission.admitted_robots} robots, got {len(simulators)}"
            )
        sim = simulators[0]
        _, _, orig_obs_space = init_spaces(env_config, resolved_map)
        occupancies, sensors = init_collision_and_sensors(sim, env_config, orig_obs_space)
        d_t = float(env_config.sim_config.time_per_step_in_secs)
        max_ep_time = float(env_config.sim_config.sim_time_in_secs)
        states = [
            RobotState(nav, occ, sen, d_t, max_ep_time)
            for nav, occ, sen in zip(sim.robot_navs, occupancies, sensors, strict=False)
        ]
        if len(states) != admission.admitted_robots:
            raise SharedWorldAdmissionError(
                f"world {world_id!r}: simulator hosts {len(states)} robots, "
                f"admitted {admission.admitted_robots}"
            )
        hold_action = np.zeros(2, dtype=float)
        runner = cls._bind(
            world_id=world_id,
            agent_ids=assign_agent_ids(world_id, admission.admitted_robots),
            simulator=sim,
            states=states,
            hold_action=hold_action,
            completion=completion,
            dt=d_t,
        )
        runner.reset(seed=seed)
        return runner

    @property
    def world_id(self) -> str:
        """Return the stable world identity."""
        return self._world_id

    @property
    def agent_ids(self) -> tuple[str, ...]:
        """Return stable agent IDs in spawn order."""
        return self._agent_ids

    @property
    def dt(self) -> float:
        """Return the logical timestep advanced by every shared step."""
        return self._dt

    @property
    def simulator(self) -> Any:
        """Return the owned simulator (diagnostic accessor).

        Inspect freely, but mutate only through :meth:`step` and
        :meth:`reset` so the one-step-per-transition invariant holds.
        """
        return self._sim

    @property
    def step_index(self) -> int:
        """Return the count of completed shared transitions."""
        return self._step_index

    def robot_positions(self) -> np.ndarray:
        """Return a copy of current robot positions, one row per agent."""
        return np.array([np.asarray(robot.pos, dtype=float) for robot in self._sim.robots])

    def ped_positions(self) -> np.ndarray:
        """Return a copy of current pedestrian positions."""
        return np.asarray(self._sim.ped_pos, dtype=float).copy()

    def reset(self, seed: int | None = None) -> None:
        """Reset the whole world explicitly; per-agent resets do not exist."""
        with global_reset_seed(seed):
            self._sim.reset_state()
            for state in self._states:
                reset_episode_counter_for_seed(state, seed)
                state.reset()
        self._step_index = 0
        self._episode_done = False
        self._holding = [False] * len(self._agent_ids)

    def _statuses(self) -> tuple[AgentStatus, ...]:
        """Snapshot per-agent lifecycle flags from the current state.

        Returns:
            One status per agent in spawn order.
        """
        statuses = []
        for agent_id, state, holding in zip(
            self._agent_ids, self._states, self._holding, strict=True
        ):
            collided = bool(
                state.is_collision_with_robot
                or state.is_collision_with_ped
                or state.is_collision_with_obst
            )
            statuses.append(
                AgentStatus(
                    agent_id=agent_id,
                    goal_reached=bool(state.is_route_complete),
                    collided=collided,
                    timed_out=bool(state.is_timeout),
                    done=bool(state.is_terminal),
                    holding=bool(holding),
                )
            )
        return tuple(statuses)

    def step(self, packet: Sequence[tuple[str, Any]]) -> SharedWorldStepResult:
        """Apply one validated packet and advance the world exactly one ``dt``.

        Validation (completeness, duplicates, foreign IDs, finiteness) runs
        before any mutation. Actions for already-finished agents are replaced
        by the zero hold action; those agents remain in the world.

        Returns:
            Step result with per-agent statuses and episode flags.
        """
        if self._episode_done:
            raise SharedWorldPacketError(
                f"world {self._world_id!r}: episode is complete; call reset() before stepping again"
            )
        ordered = validate_action_packet(packet, self._agent_ids)
        parsed = []
        for robot, action in zip(self._sim.robots, ordered, strict=True):
            parsed.append(robot.parse_action(action))
        for index, done in enumerate(self._done_flags()):
            if done:
                parsed[index] = self._sim.robots[index].parse_action(self._hold_action)
                self._holding[index] = True
        self._sim.step_once(parsed)
        for state in self._states:
            state.step()
        self._step_index += 1
        statuses = self._statuses()
        done_flags = [status.done for status in statuses]
        episode_done = any(done_flags) if self._completion == "any" else all(done_flags)
        self._episode_done = bool(episode_done)
        return SharedWorldStepResult(
            world_id=self._world_id,
            step_index=self._step_index,
            statuses=statuses,
            episode_done=bool(episode_done),
            episode_truncated=bool(any(status.timed_out for status in statuses)),
        )

    def _done_flags(self) -> list[bool]:
        """Return cached per-agent done flags without advancing state."""
        return [bool(state.is_terminal) for state in self._states]


def _resolve_map_def(
    env_config: EnvSettings | MultiRobotConfig,
    map_def: MapDefinition | None,
) -> MapDefinition:
    """Resolve the map the same way the multi-robot environment does.

    Returns:
        The caller-supplied map, or the configured map pool's selection.
    """
    if map_def is not None:
        return map_def
    pool = env_config.map_pool.map_defs
    map_id = getattr(env_config, "map_id", None)
    if map_id in pool:
        return pool[map_id]
    return next(iter(pool.values()))


def _spawn_capacity(
    env_config: EnvSettings | MultiRobotConfig,
    map_def: MapDefinition | None,
) -> int:
    """Return the robot start-position capacity of the resolved map."""
    return int(_resolve_map_def(env_config, map_def).num_start_pos)
