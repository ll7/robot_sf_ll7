"""Production-simulator ``CounterfactualModel`` adapter for frozen-state replay (#5442).

This module wires the real Robot SF :class:`~robot_sf.sim.simulator.Simulator` into
the simulator-agnostic frozen-state counterfactual-replay engine
(:mod:`robot_sf.benchmark.last_avoidable_replay`). It is the production-simulator
slice named as remaining in the issue #5442 thread: a ``CounterfactualModel`` that
implements the *smallest snapshot/restore seam* over a live simulator, including the
RNG-capture seam the prior controlled-fixture slice flagged as out of scope.

Scope and determinism contract
------------------------------
A faithful mid-episode snapshot of the production simulator must capture everything
that affects future steps, including the random-number generator state. The engine's
:class:`~robot_sf.benchmark.last_avoidable_replay.CounterfactualModel` contract
requires that restoring a snapshot and applying the same actions reproduce the same
contact outcome bit-for-bit.

This adapter captures:

* the pedestrian PySocialForce state buffer (positions, velocities, goals) via the
  simulator's ``pysf_state`` accessor;
* per-pedestrian mutable behavior runtimes (single-pedestrian waypoint/hold state and
  route-group navigators) so a resumed replay follows the recorded path;
* the robot pose/velocity state;
* the **global** numpy RNG via :func:`numpy.random.get_state` /
  :func:`numpy.random.set_state` — this is the seam the issue's stop rule flagged:
  ``robot_sf`` pedestrian goal/zone sampling draws from the global RNG, so a
  deterministic branch replay must restore it around every branch.

The adapter is constructed for a single robot in a robot-only or pedestrian interaction
scenario and replays a recorded ``baseline_actions`` list of ``RobotAction`` tuples.
It is diagnostic/offline only: it assigns no fault and is not a real-episode root-cause
claim (see the engine's fail-closed determination vocabulary).

The snapshot/restore seam here is intentionally scoped to the actual mutable state of a
running ``Simulator``. Broader state (PySF force internal buffers, obstacle KD-trees)
is reproducible from the captured actor state and the immutable config/map, so it is not
independently snapshotted. The determinism check in the engine is the safeguard: if a
replay diverges, the engine abstains to ``unknown`` rather than guessing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.ped_npc.ped_behavior import SinglePedestrianRuntime

if TYPE_CHECKING:
    from collections.abc import Sequence

    from robot_sf.sim.simulator import Simulator


@dataclass
class _SimulatorSnapshot:
    """Opaque, restorable copy of the live simulator state at one decision point.

    Attributes:
        step_index: Number of ``step_once`` calls applied since construction.
        pysf_state: Copy of the pedestrian PySocialForce state buffer.
        ped_headings: Copy of per-pedestrian body headings (HSFM models).
        ped_angular_velocities: Copy of per-pedestrian angular velocities.
        robot_poses: Copy of each robot's ``((x, y), heading)`` pose.
        robot_velocities: Copy of each robot's ``(linear, angular)`` velocity.
        single_runtimes: Deep copy of single-pedestrian behavior runtimes.
        route_navigators: Deep copy of route-group navigators' mutable state.
        global_rng_state: Numpy global RNG state captured via ``get_state``.
        peds_have_obstacle_forces: Simulator obstacle-force flag (affects stepping).
    """

    step_index: int
    pysf_state: np.ndarray
    ped_headings: np.ndarray
    ped_angular_velocities: np.ndarray
    robot_poses: list[Any]
    robot_velocities: list[Any]
    single_runtimes: list[Any]
    route_navigators: dict[int, Any]
    global_rng_state: Any
    peds_have_obstacle_forces: bool


def _copy_global_rng_state() -> tuple[Any, ...]:
    """Return a deep copy of the numpy global RNG state.

    ``numpy.random.get_state`` returns a tuple whose internal key array is a *view*
    backed by the generator; restoring it later can mutate that shared buffer and
    corrupt the captured snapshot. Copying the key array up front makes each snapshot
    independent, so restoring it reproduces the same RNG stream.
    """
    key, state, pos, has_gauss, cached_gauss = np.random.get_state()
    return (key, state.copy(), pos, has_gauss, cached_gauss)


def _capture_single_runtimes(peds_behaviors: list[Any]) -> list[Any]:
    """Deep-copy single-pedestrian behavior runtime state for a snapshot.

    Only :class:`~robot_sf.ped_npc.ped_behavior.SinglePedestrianBehavior` carries
    per-pedestrian mutable runtime state that changes during a step; route/crowd
    behaviors advance navigators that are captured separately.

    Returns:
        A list aligned with ``peds_behaviors``; each entry is a deep-copied list of
        runtime field dicts, or ``None`` when the behavior has no single-pedestrian
        runtimes.
    """
    runtimes: list[Any] = []
    for behavior in peds_behaviors:
        rt = getattr(behavior, "_runtimes", None)
        if rt is not None:
            runtimes.append(deepcopy([vars(r) for r in rt]))
        else:
            runtimes.append(None)
    return runtimes


def _restore_single_runtimes(peds_behaviors: list[Any], runtimes: list[Any]) -> None:
    """Restore single-pedestrian behavior runtime state from a snapshot."""
    for behavior, saved in zip(peds_behaviors, runtimes, strict=True):
        if saved is None or not hasattr(behavior, "_runtimes"):
            continue
        behavior._runtimes = [SinglePedestrianRuntime(**fields) for fields in saved]


def _capture_route_navigators(peds_behaviors: list[Any]) -> dict[int, Any]:
    """Deep-copy route-group navigator mutable state for a snapshot.

    Returns:
        A mapping from ``id(behavior)`` to the captured per-group navigator state.
    """
    navigators: dict[int, Any] = {}
    for behavior in peds_behaviors:
        navs = getattr(behavior, "navigators", None)
        if navs:
            # RouteNavigator exposes a few mutable fields; capture the ones that
            # advance during step() (waypoint index, current/next waypoint caches).
            captured = {}
            for gid, nav in navs.items():
                # ``current_waypoint``/``next_waypoint`` are derived from ``waypoint_id``;
                # only ``waypoint_id`` and ``reached_waypoint`` are mutable state.
                captured[gid] = {
                    "waypoint_id": int(nav.waypoint_id),
                    "reached_waypoint": bool(nav.reached_waypoint),
                }
            navigators[id(behavior)] = captured
    return navigators


def _restore_route_navigators(peds_behaviors: list[Any], navigators: dict[int, Any]) -> None:
    """Restore route-group navigator mutable state from a snapshot."""
    for behavior in peds_behaviors:
        navs = getattr(behavior, "navigators", None)
        if not navs:
            continue
        captured = navigators.get(id(behavior))
        if not captured:
            continue
            for gid, nav in navs.items():
                state = captured.get(gid)
                if state is None:
                    continue
                nav.waypoint_id = state["waypoint_id"]
                nav.reached_waypoint = state["reached_waypoint"]


class SimulatorCounterfactualModel:
    """``CounterfactualModel`` adapter over a live Robot SF ``Simulator``.

    The adapter is positioned at step 0 of a recorded baseline episode. It drives the
    simulator via ``step_once`` with the robot's single action, and exposes the
    snapshot/restore seam the replay engine needs. The global numpy RNG is captured and
    restored so pedestrian goal/zone resampling replays deterministically.

    Args:
        simulator: A constructed ``Simulator`` (robot-only or with pedestrians).
        collision_fn: Optional callable ``(model) -> bool``; defaults to Euclidean
            proximity between the first robot and any pedestrian within
            ``collision_radius``.
        collision_radius: Contact distance (m) used by the default collision predicate.
        capture_rng: When ``True`` (default) the global numpy RNG state is captured and
            restored so pedestrian goal/zone resampling replays deterministically. When
            ``False`` the RNG seam is intentionally omitted, exercising the engine's
            fail-closed ``unknown`` path on a nondeterministic baseline.
    """

    def __init__(
        self,
        simulator: Simulator,
        collision_fn: Any | None = None,
        collision_radius: float = 0.5,
        capture_rng: bool = True,
    ) -> None:
        """Initialize the adapter at step 0 of the simulator."""
        self.sim = simulator
        self.collision_radius = float(collision_radius)
        self.capture_rng = bool(capture_rng)
        self._step_index = 0
        self._collision_fn = collision_fn

    def _default_collision(self) -> bool:
        """Return whether the first robot is within ``collision_radius`` of any ped."""
        robot_pos = np.asarray(self.sim.robot_pos[0], dtype=float)
        ped_positions = np.asarray(self.sim.ped_pos, dtype=float)
        if ped_positions.size == 0:
            return False
        distances = np.linalg.norm(ped_positions - robot_pos, axis=-1)
        return bool(np.any(distances <= self.collision_radius))

    def snapshot(self) -> _SimulatorSnapshot:
        """Capture the full live simulator state including the global RNG.

        Returns:
            A :class:`_SimulatorSnapshot` restorable via :meth:`restore`.
        """
        return _SimulatorSnapshot(
            step_index=self._step_index,
            pysf_state=self.sim.pysf_state.pysf_states().copy(),
            ped_headings=self.sim.ped_headings.copy(),
            ped_angular_velocities=self.sim.ped_angular_velocities.copy(),
            robot_poses=[deepcopy(r.pose) for r in self.sim.robots],
            robot_velocities=[deepcopy(getattr(r, "state", None)) for r in self.sim.robots],
            single_runtimes=_capture_single_runtimes(self.sim.peds_behaviors),
            route_navigators=_capture_route_navigators(self.sim.peds_behaviors),
            global_rng_state=_copy_global_rng_state() if self.capture_rng else None,
            peds_have_obstacle_forces=bool(self.sim.peds_have_obstacle_forces),
        )

    def restore(self, snapshot: _SimulatorSnapshot) -> None:
        """Restore the live simulator to a previously captured snapshot."""
        self._step_index = snapshot.step_index
        self.sim.pysf_state.pysf_states()[...] = snapshot.pysf_state
        self.sim.ped_headings = snapshot.ped_headings.copy()
        self.sim.ped_angular_velocities = snapshot.ped_angular_velocities.copy()
        for robot, pose, vel_state in zip(
            self.sim.robots, snapshot.robot_poses, snapshot.robot_velocities, strict=True
        ):
            robot.state = deepcopy(vel_state)
        _restore_single_runtimes(self.sim.peds_behaviors, snapshot.single_runtimes)
        _restore_route_navigators(self.sim.peds_behaviors, snapshot.route_navigators)
        if snapshot.global_rng_state is not None:
            key, state, pos, has_gauss, cached_gauss = snapshot.global_rng_state
            np.random.set_state((key, state.copy(), pos, has_gauss, cached_gauss))

    def step(self, action: Any) -> None:
        """Advance the simulator one control tick applying the robot action."""
        self.sim.step_once([action])
        self._step_index += 1

    def collision(self) -> bool:
        """Return whether the robot is in contact at the current state."""
        if self._collision_fn is not None:
            return bool(self._collision_fn(self))
        return self._default_collision()

    def feasible_actions(self) -> Sequence[Any]:
        """Return the admissible robot action set at the current state.

        The default feasible set is a lattice of constant-velocity (maintain-speed)
        commands along ``+x`` and a set of decelerating/idle commands, sufficient to
        probe whether a slower or halted robot avoids contact. Callers requiring a
        planner-specific action lattice may subclass and override this method.
        """
        return (
            (1.0, 0.0),
            (0.5, 0.0),
            (0.1, 0.0),
            (0.0, 0.0),
        )

    def action_label(self, action: Any) -> str:
        """Return a stable label for a robot action (for provenance)."""
        vx = float(getattr(action, "__getitem__", lambda i: action[i])(0))
        vy = float(getattr(action, "__getitem__", lambda i: action[i])(1))
        return f"robot_cmd=(vx={vx:g},vy={vy:g})"
