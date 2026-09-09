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
* robot route-navigator progress;
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

import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sf.nav.occupancy import circle_collides_any_lines
from robot_sf.ped_npc.ped_behavior import SinglePedestrianRuntime

if TYPE_CHECKING:
    from collections.abc import Sequence

    from robot_sf.sim.simulator import Simulator


COLLISION_SCOPE_PEDESTRIAN_ONLY = "pedestrian_only"
COLLISION_SCOPE_ALL = "all"
_COLLISION_SCOPES = (COLLISION_SCOPE_PEDESTRIAN_ONLY, COLLISION_SCOPE_ALL)
PED_RESPONSE_REPLAYED = "replayed"
PED_RESPONSE_CLOSED_LOOP = "closed_loop"


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
        robot_navigators: Deep copy of robot route-navigator progress.
        single_runtimes: Deep copy of single-pedestrian behavior runtimes.
        route_navigators: Deep copy of route-group navigators' mutable state.
        pedestrian_groups: Deep copy of mutable pedestrian group membership.
        pedestrian_group_by_ped: Deep copy of the pedestrian-to-group reverse lookup.
        behavior_rng_states: Per-behavior NumPy generator states.
        residual_adversary: Deep copy of the stateful residual controller, if active.
        global_rng_state: Numpy global RNG state captured via ``get_state``.
        peds_have_obstacle_forces: Simulator obstacle-force flag (affects stepping).
        ped_max_speeds: Copy of pedestrian speed caps used by future force updates.
        python_random_state: State of the stdlib ``random`` stream used by route sampling.
        residual_adversary_state: Mutable residual-controller state, when instantiated.
        absolute_time_s: Pre-step world time derived from the simulator timestep.
        remaining_budget_steps: Remaining episode steps under ``sim_time_in_secs``.
    """

    step_index: int
    pysf_state: np.ndarray
    ped_headings: np.ndarray
    ped_angular_velocities: np.ndarray
    robot_poses: list[Any]
    robot_velocities: list[Any]
    robot_navigators: list[Any]
    single_runtimes: list[Any]
    route_navigators: dict[Any, Any]
    global_rng_state: Any
    peds_have_obstacle_forces: bool
    # These fields were added by the production replay hardening slice. Defaults
    # keep older in-memory snapshots readable while the typed artifact layer
    # serializes the stable fields below.
    pedestrian_groups: dict[Any, Any] | None = None
    pedestrian_group_by_ped: dict[Any, Any] | None = None
    behavior_rng_states: dict[str, Any] | None = None
    residual_adversary: Any = None
    ped_max_speeds: np.ndarray | None = None
    python_random_state: Any = None
    residual_adversary_state: dict[str, Any] | None = None
    absolute_time_s: float | None = None
    remaining_budget_steps: int | None = None


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
            captured: list[dict[str, Any]] = []
            for runtime in rt:
                definition = runtime.definition
                captured.append(
                    {
                        "ped_id": int(runtime.ped_id),
                        "definition_id": str(getattr(definition, "id", runtime.ped_id)),
                        "trajectory": [
                            [float(point[0]), float(point[1])] for point in runtime.trajectory
                        ],
                        "waypoint_index": int(runtime.waypoint_index),
                        "pending_waits": {
                            str(int(index)): float(wait)
                            for index, wait in runtime.pending_waits.items()
                        },
                        "start_delay_remaining_s": float(runtime.start_delay_remaining_s),
                        "wait_remaining_s": float(runtime.wait_remaining_s),
                        "waiting_for_advance": bool(runtime.waiting_for_advance),
                        "joined_group_id": (
                            None
                            if runtime.joined_group_id is None
                            else int(runtime.joined_group_id)
                        ),
                        "left_group": bool(runtime.left_group),
                        "hold_waypoint_index": (
                            None
                            if runtime.hold_waypoint_index is None
                            else int(runtime.hold_waypoint_index)
                        ),
                        "proximity_hold_engaged": bool(runtime.proximity_hold_engaged),
                        "proximity_hold_released": bool(runtime.proximity_hold_released),
                        "proximity_hold_elapsed_s": float(runtime.proximity_hold_elapsed_s),
                        "hold_released_by": runtime.hold_released_by,
                    }
                )
            runtimes.append(captured)
        else:
            runtimes.append(None)
    return runtimes


def _restore_single_runtimes(peds_behaviors: list[Any], runtimes: list[Any]) -> None:  # noqa: C901
    """Restore single-pedestrian behavior runtime state from a complete snapshot."""
    if not isinstance(runtimes, list) or len(runtimes) != len(peds_behaviors):
        raise ValueError("single-pedestrian runtime snapshot does not match behavior count")
    for behavior, saved in zip(peds_behaviors, runtimes, strict=True):
        current_runtimes = getattr(behavior, "_runtimes", None)
        if current_runtimes is None:
            if saved is not None:
                raise ValueError("snapshot contains runtimes for a behavior without runtimes")
            continue
        if saved is None:
            raise ValueError("snapshot is missing single-pedestrian runtime state")
        if not isinstance(saved, list) or len(saved) != len(current_runtimes):
            raise ValueError("single-pedestrian runtime snapshot does not match actor count")
        current_by_ped_id = {int(runtime.ped_id): runtime for runtime in current_runtimes}
        current_by_definition_id = {
            str(getattr(runtime.definition, "id", runtime.ped_id)): runtime
            for runtime in current_runtimes
        }
        restored: list[SinglePedestrianRuntime] = []
        matched_runtime_ids: set[int] = set()
        for fields in saved:
            if not isinstance(fields, Mapping):
                raise ValueError("single-pedestrian runtime state must be a mapping")
            # Accept snapshots written before the typed runtime representation.
            if "definition" in fields:
                legacy_current = current_by_ped_id.get(int(fields["ped_id"]))
                if legacy_current is None:
                    raise ValueError(
                        "single-pedestrian snapshot identity does not match destination"
                    )
                if id(legacy_current) in matched_runtime_ids:
                    raise ValueError("single-pedestrian snapshot contains a duplicate runtime")
                matched_runtime_ids.add(id(legacy_current))
                restored.append(SinglePedestrianRuntime(**fields))
                continue
            current = current_by_ped_id.get(int(fields["ped_id"])) or current_by_definition_id.get(
                str(fields["definition_id"])
            )
            if current is None:
                raise ValueError(
                    "single-pedestrian snapshot identity does not match the destination "
                    f"behavior (ped_id={fields.get('ped_id')!r})"
                )
            if id(current) in matched_runtime_ids:
                raise ValueError("single-pedestrian snapshot contains a duplicate runtime")
            matched_runtime_ids.add(id(current))
            mutable_fields = {
                key: value
                for key, value in fields.items()
                if key
                not in {
                    "ped_id",
                    "definition_id",
                    "trajectory",
                }
            }
            mutable_fields["pending_waits"] = {
                int(index): float(wait) for index, wait in mutable_fields["pending_waits"].items()
            }
            trajectory = [
                (float(point[0]), float(point[1])) for point in fields.get("trajectory", [])
            ]
            restored.append(
                SinglePedestrianRuntime(
                    ped_id=int(fields["ped_id"]),
                    definition=current.definition,
                    trajectory=trajectory or list(current.trajectory),
                    **mutable_fields,
                )
            )
        if len(matched_runtime_ids) != len(current_runtimes):
            raise ValueError("single-pedestrian snapshot omits a destination runtime")
        behavior._runtimes = restored


def _stable_behavior_identity(index: int, behavior: Any) -> str:
    """Return a process-independent identity for one behavior controller."""
    behavior_type = type(behavior).__name__.lower()
    if hasattr(behavior, "single_offset"):
        return f"single:{int(behavior.single_offset)}"
    navs = getattr(behavior, "navigators", None)
    if navs is not None:
        global_offset = int(getattr(behavior, "global_ped_offset", 0))
        groups = ",".join(sorted(str(int(group_id)) for group_id in navs))
        return f"route:{global_offset}:{groups}"
    return f"behavior:{index}:{behavior_type}"


def _capture_behavior_rng_states(peds_behaviors: list[Any]) -> dict[str, Any]:
    """Capture owned NumPy generator states using stable behavior identities.

    Returns:
        Mapping from stable behavior/attribute identity to generator state.
    """
    captured: dict[str, Any] = {}
    for index, behavior in enumerate(peds_behaviors):
        identity = _stable_behavior_identity(index, behavior)
        for attribute in ("rng", "_rng"):
            generator = getattr(behavior, attribute, None)
            bit_generator = getattr(generator, "bit_generator", None)
            if bit_generator is not None:
                captured[f"{identity}:{attribute}"] = deepcopy(bit_generator.state)
    return captured


def _restore_behavior_rng_states(peds_behaviors: list[Any], saved: dict[str, Any] | None) -> None:
    """Restore owned NumPy generator states, rejecting identity drift."""
    if saved is not None and not isinstance(saved, Mapping):
        raise ValueError("behavior RNG snapshot must be a mapping")
    expected_keys: set[str] = set()
    for index, behavior in enumerate(peds_behaviors):
        identity = _stable_behavior_identity(index, behavior)
        for attribute in ("rng", "_rng"):
            key = f"{identity}:{attribute}"
            generator = getattr(behavior, attribute, None)
            bit_generator = getattr(generator, "bit_generator", None)
            if bit_generator is None:
                continue
            expected_keys.add(key)
            if saved is None:
                raise ValueError(f"snapshot is missing behavior RNG {key!r}")
            if key in saved:
                state = saved[key]
            elif id(behavior) in saved:
                # Read-only compatibility for snapshots written by the first
                # hardening slice, which keyed generators by process-local object id.
                state = saved[id(behavior)]
            else:
                raise ValueError(f"snapshot is missing behavior RNG {key!r}")
            bit_generator.state = deepcopy(state)
    if saved is not None:
        unexpected = {key for key in saved if isinstance(key, str) and key not in expected_keys}
        if unexpected:
            raise ValueError(
                f"snapshot contains unknown behavior RNG identities: {sorted(unexpected)}"
            )


def _capture_pedestrian_groups(groups: Any) -> tuple[dict[Any, Any], dict[Any, Any]]:
    """Capture mutable pedestrian group membership and reverse lookup.

    Returns:
        Deep-copied group membership and its pedestrian-to-group reverse lookup.
    """
    return deepcopy(groups.groups), deepcopy(groups.group_by_ped_id)


def _restore_pedestrian_groups(
    groups: Any,
    memberships: dict[Any, Any],
    group_by_ped: dict[Any, Any],
) -> None:
    """Restore pedestrian group membership and invalidate the derived list cache."""
    groups.groups = deepcopy(memberships)
    groups.group_by_ped_id = deepcopy(group_by_ped)
    invalidate_cache = getattr(groups, "_invalidate_groups_as_lists_cache", None)
    if callable(invalidate_cache):
        invalidate_cache()
    elif hasattr(groups, "_groups_as_lists_cache"):
        groups._groups_as_lists_cache = None


def _synchronize_pysf_groups(simulator: Any) -> None:
    """Synchronize the PySocialForce group list with restored public membership.

    ``Simulator.step_once`` computes forces before it passes the current public
    grouping to the pedestrian integrator.  The force objects therefore read
    ``simulator.pysf_sim.peds.groups`` during that first phase; restoring only
    ``simulator.groups`` would leave branch-specific backend membership active.
    Small test doubles may omit the backend, so they remain compatible with the
    public grouping snapshot seam.
    """
    pysf_peds = getattr(getattr(simulator, "pysf_sim", None), "peds", None)
    if pysf_peds is not None:
        groups = getattr(simulator, "groups", None)
        if groups is None:
            return
        groups_as_lists = getattr(groups, "groups_as_lists", None)
        if groups_as_lists is None:
            groups_as_lists = [list(ped_ids) for ped_ids in groups.groups.values()]
        pysf_peds.groups = deepcopy(groups_as_lists)


_RESIDUAL_STATE_FIELDS = (
    "_last_residual",
    "_held_proposal",
    "_step_index",
    "_macro_action_index",
    "_macro_steps",
    "_target_mask",
    "_target_indices",
    "_summary_norm_sum",
    "_summary_norm_max",
    "_summary_norm_sample_count",
    "_summary_nonzero_sample_count",
    "_summary_adjusted_proposal_count",
    "_summary_finite",
    "_summary_bound_safe",
    "_summary_invalid",
)


def _capture_residual_adversary_state(simulator: Any) -> dict[str, Any] | None:
    """Capture mutable state of an instantiated residual adversary, if any.

    Returns:
        A copied field mapping, or ``None`` when no adversary is instantiated.
    """
    adversary = getattr(simulator, "_residual_adversary", None)
    if adversary is None:
        return None
    return {
        field: deepcopy(getattr(adversary, field))
        for field in _RESIDUAL_STATE_FIELDS
        if hasattr(adversary, field)
    }


def _restore_residual_adversary_state(simulator: Any, saved: dict[str, Any] | None) -> None:
    """Restore a captured residual-adversary state without loading executable objects."""
    if saved is None:
        if getattr(simulator, "_residual_adversary", None) is not None:
            raise ValueError("snapshot is missing residual-adversary state")
        return
    if not isinstance(saved, Mapping):
        raise ValueError("residual-adversary snapshot must be a mapping")
    adversary = getattr(simulator, "_residual_adversary", None)
    if adversary is None:
        builder = getattr(simulator, "_build_residual_adversary", None)
        if not callable(builder):
            raise ValueError("snapshot requires a residual-adversary instance")
        adversary = builder()
        if adversary is None:
            raise ValueError("snapshot contains residual-adversary state but config is inactive")
        simulator._residual_adversary = adversary
    expected_fields = {field for field in _RESIDUAL_STATE_FIELDS if hasattr(adversary, field)}
    unsupported = set(saved) - set(_RESIDUAL_STATE_FIELDS)
    if unsupported:
        raise ValueError(f"unsupported residual-adversary state field {sorted(unsupported)}")
    missing = expected_fields - set(saved)
    if missing:
        raise ValueError(f"snapshot is missing residual-adversary state fields: {sorted(missing)}")
    for field, value in saved.items():
        setattr(adversary, field, deepcopy(value))


def _capture_route_navigators(peds_behaviors: list[Any]) -> dict[str, Any]:
    """Deep-copy route-group navigator mutable state for a snapshot.

    Returns:
        A mapping from stable behavior identity to the captured per-group navigator state.
    """
    navigators: dict[str, Any] = {}
    for index, behavior in enumerate(peds_behaviors):
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
            navigators[_stable_behavior_identity(index, behavior)] = captured
    return navigators


def _restore_route_navigators(  # noqa: C901
    peds_behaviors: list[Any], navigators: dict[str, Any]
) -> None:
    """Restore route-group navigator state, rejecting missing identities or fields."""
    if not isinstance(navigators, Mapping):
        raise ValueError("route navigator snapshot must be a mapping")
    expected_identities: set[str] = set()
    for index, behavior in enumerate(peds_behaviors):
        navs = getattr(behavior, "navigators", None)
        if not navs:
            continue
        identity = _stable_behavior_identity(index, behavior)
        expected_identities.add(identity)
        captured = navigators.get(identity)
        if captured is None:
            # Read-only compatibility for an in-memory snapshot from the old
            # process-local representation. Durable snapshots never emit this key.
            captured = navigators.get(id(behavior))  # type: ignore[arg-type]
        if captured is None:
            raise ValueError(f"snapshot is missing route navigator identity {identity!r}")
        if not isinstance(captured, Mapping):
            raise ValueError(f"route navigator state for {identity!r} must be a mapping")
        captured_group_ids = {str(group_id) for group_id in captured}
        expected_group_ids = {str(group_id) for group_id in navs}
        if captured_group_ids != expected_group_ids:
            raise ValueError(
                f"route navigator group identities for {identity!r} do not match destination"
            )
        for gid, nav in navs.items():
            state = captured.get(gid, captured.get(str(gid)))
            if not isinstance(state, Mapping):
                raise ValueError(f"route navigator state for group {gid!r} is incomplete")
            if "waypoint_id" not in state or "reached_waypoint" not in state:
                raise ValueError(f"route navigator state for group {gid!r} is incomplete")
            nav.waypoint_id = state["waypoint_id"]
            nav.reached_waypoint = state["reached_waypoint"]
    unexpected = {
        key for key in navigators if isinstance(key, str) and key not in expected_identities
    }
    if unexpected:
        raise ValueError(
            f"snapshot contains unknown route navigator identities: {sorted(unexpected)}"
        )


def _restore_robot_navigators(robot_navigators: list[Any], saved: list[Any]) -> None:
    """Restore mutable robot route-navigator state from a snapshot."""
    for navigator, saved_navigator in zip(robot_navigators, saved, strict=True):
        navigator.waypoints = deepcopy(saved_navigator.waypoints)
        navigator.waypoint_id = saved_navigator.waypoint_id
        navigator.proximity_threshold = saved_navigator.proximity_threshold
        navigator.pos = deepcopy(saved_navigator.pos)
        navigator.reached_waypoint = saved_navigator.reached_waypoint


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
            ``collision_radius``. This deliberately narrow predicate is labelled
            ``pedestrian_only`` in the adapter metadata.
        collision_radius: Contact distance (m) used by the default collision predicate.
        collision_scope: ``pedestrian_only`` preserves the historical diagnostic
            predicate; ``all`` uses the simulator's robot footprint against map
            bounds, obstacle segments, and pedestrian footprints.
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
        collision_scope: str = COLLISION_SCOPE_PEDESTRIAN_ONLY,
    ) -> None:
        """Initialize the adapter at step 0 of the simulator."""
        if len(simulator.robots) != 1:
            raise ValueError(
                "SimulatorCounterfactualModel requires a simulator with exactly one robot."
            )
        self.sim = simulator
        self.collision_radius = float(collision_radius)
        self.capture_rng = bool(capture_rng)
        scope = str(collision_scope).strip().lower()
        if scope not in _COLLISION_SCOPES:
            raise ValueError(
                f"collision_scope must be one of {_COLLISION_SCOPES} (got {collision_scope!r})"
            )
        self.collision_scope = scope
        self._step_index = 0
        self._collision_fn = collision_fn

    def _default_collision(self) -> bool:
        """Return the selected collision predicate for the live simulator."""
        if self.collision_scope == COLLISION_SCOPE_ALL:
            return self._all_collisions()
        return self._pedestrian_only_collision()

    def _pedestrian_only_collision(self) -> bool:
        """Return the historical narrow robot-centre/pedestrian-centre predicate."""
        robot_pos = np.asarray(self.sim.robot_pos[0], dtype=float)
        ped_positions = np.asarray(self.sim.ped_pos, dtype=float)
        if ped_positions.size == 0:
            return False
        distances = np.linalg.norm(ped_positions - robot_pos, axis=-1)
        return bool(np.any(distances <= self.collision_radius))

    def _all_collisions(self) -> bool:
        """Return canonical footprint collision against bounds, walls, and peds.

        ``ContinuousOccupancy`` is the source of truth for circle/segment and
        circle/circle geometry. The adapter keeps this mode explicit because the
        legacy default above intentionally reports only pedestrian proximity.
        """
        robot = self.sim.robots[0]
        robot_pos = tuple(float(value) for value in robot.pos)
        robot_radius = float(getattr(robot.config, "radius", self.collision_radius))
        map_def = self.sim.map_def
        if not (0.0 <= robot_pos[0] <= float(map_def.width)) or not (
            0.0 <= robot_pos[1] <= float(map_def.height)
        ):
            return True

        get_obstacles = getattr(self.sim, "get_obstacle_lines", None)
        obstacle_lines = get_obstacles() if callable(get_obstacles) else ()
        if circle_collides_any_lines((robot_pos, robot_radius), obstacle_lines):
            return True

        ped_positions = np.asarray(self.sim.ped_pos, dtype=float)
        if ped_positions.size == 0:
            return False
        ped_radius = float(
            getattr(
                getattr(self.sim, "config", None),
                "ped_radius",
                getattr(
                    getattr(getattr(self.sim, "pysf_sim", None), "peds", None), "agent_radius", 0.4
                ),
            )
        )
        distances = np.linalg.norm(ped_positions - np.asarray(robot_pos), axis=-1)
        return bool(np.any(distances <= robot_radius + ped_radius))

    @property
    def replay_source_kind(self) -> str:
        """Identify this adapter as a live-episode replay source.

        The adapter does not yet carry the map, scenario, seed, episode, and
        software-commit receipt needed by the causal-report join. The replay
        engine records this source kind so that the join can abstain instead of
        relabelling a native result as a synthetic fixture.
        """
        return "live_episode"

    @property
    def action_set_id(self) -> str:
        """Return an ID bound to the native drivetrain and its limits."""
        config = self.sim.robots[0].config
        if hasattr(config, "max_linear_decel"):
            return (
                "simulator_native_action_lattice_v1:diff_drive_acceleration:"
                f"max_linear_decel={float(config.max_linear_decel):g}:"
                f"max_angular_accel={float(config.max_angular_accel):g}"
            )
        if hasattr(config, "max_decel") and hasattr(config, "max_steer"):
            return (
                "simulator_native_action_lattice_v1:bicycle_acceleration:"
                f"max_decel={float(config.max_decel):g}:max_steer={float(config.max_steer):g}"
            )
        if getattr(config, "command_mode", None) == "vx_vy":
            return (
                "simulator_native_action_lattice_v1:holonomic_velocity:"
                f"max_speed={float(config.max_speed):g}"
            )
        if getattr(config, "command_mode", None) == "unicycle_vw":
            return (
                "simulator_native_action_lattice_v1:unicycle_velocity:"
                f"max_angular_speed={float(config.max_angular_speed):g}"
            )
        return "simulator_native_action_lattice_v1:unsupported"

    @property
    def feasibility_filter(self) -> str:
        """Return the native feasible-action rule and its current cardinality."""
        return f"native_all_supported_actions_v1:n={len(self.feasible_actions())}"

    @property
    def collision_predicate(self) -> str:
        """Return a stable provenance identifier for the executed predicate."""
        if self._collision_fn is not None:
            return "custom_collision_fn"
        if self.collision_scope == COLLISION_SCOPE_ALL:
            return "continuous_occupancy_robot_bounds_obstacles_pedestrians_v1"
        return "robot_pedestrian_center_distance_v1"

    @property
    def pedestrian_response(self) -> str:
        """Return whether pedestrian stepping is independent or robot-reactive.

        Robot-aware forces and the residual adversary read the live robot pose on
        every simulator step. Their presence therefore makes a branch closed-loop;
        with all such forces disabled, the pedestrian path is independent of the
        substituted robot action and can be treated as replayed.
        """
        config = self.sim.config
        if any(
            bool(getattr(getattr(config, name, None), "is_active", False))
            for name in ("prf_config", "apf_config", "residual_adversary")
        ):
            return PED_RESPONSE_CLOSED_LOOP
        for behavior in self.sim.peds_behaviors:
            definitions = getattr(behavior, "single_pedestrians", ())
            for definition in definitions:
                role = getattr(definition, "role", None)
                if role in {"follow", "accompany"}:
                    return PED_RESPONSE_CLOSED_LOOP
                if role == "lead" and not (
                    getattr(definition, "goal", None) or getattr(definition, "trajectory", None)
                ):
                    return PED_RESPONSE_CLOSED_LOOP
                if getattr(definition, "hold_until_robot_within_m", None) is not None:
                    return PED_RESPONSE_CLOSED_LOOP
        return PED_RESPONSE_REPLAYED

    @property
    def replay_state_complete(self) -> str:
        """Report whether the adapter captured every declared RNG stream."""
        return "true" if self.capture_rng else "false"

    def snapshot(self) -> _SimulatorSnapshot:
        """Capture the full live simulator state including the global RNG.

        Returns:
            A :class:`_SimulatorSnapshot` restorable via :meth:`restore`.
        """
        groups = getattr(self.sim, "groups", None)
        if groups is None:
            pedestrian_groups = None
            pedestrian_group_by_ped = None
        else:
            pedestrian_groups, pedestrian_group_by_ped = _capture_pedestrian_groups(groups)
        peds = getattr(getattr(self.sim, "pysf_sim", None), "peds", None)
        max_speeds = getattr(peds, "max_speeds", None)
        config = getattr(self.sim, "config", None)
        time_per_step = getattr(config, "time_per_step_in_secs", None)
        sim_time = getattr(config, "sim_time_in_secs", None)
        if time_per_step is None:
            absolute_time_s = None
            remaining_budget_steps = None
        else:
            absolute_time_s = self._step_index * float(time_per_step)
            remaining_budget_steps = (
                None
                if sim_time is None
                else max(0, int(np.ceil(float(sim_time) / float(time_per_step))) - self._step_index)
            )
        behavior_rng_states = _capture_behavior_rng_states(self.sim.peds_behaviors)
        return _SimulatorSnapshot(
            step_index=self._step_index,
            pysf_state=self.sim.pysf_state.pysf_states().copy(),
            ped_headings=self.sim.ped_headings.copy(),
            ped_angular_velocities=self.sim.ped_angular_velocities.copy(),
            robot_poses=[deepcopy(r.pose) for r in self.sim.robots],
            robot_velocities=[deepcopy(getattr(r, "state", None)) for r in self.sim.robots],
            robot_navigators=deepcopy(self.sim.robot_navs),
            single_runtimes=_capture_single_runtimes(self.sim.peds_behaviors),
            route_navigators=_capture_route_navigators(self.sim.peds_behaviors),
            global_rng_state=_copy_global_rng_state() if self.capture_rng else None,
            peds_have_obstacle_forces=bool(self.sim.peds_have_obstacle_forces),
            pedestrian_groups=pedestrian_groups,
            pedestrian_group_by_ped=pedestrian_group_by_ped,
            residual_adversary=deepcopy(getattr(self.sim, "_residual_adversary", None)),
            behavior_rng_states=behavior_rng_states,
            ped_max_speeds=None if max_speeds is None else np.asarray(max_speeds).copy(),
            python_random_state=deepcopy(random.getstate()) if self.capture_rng else None,
            residual_adversary_state=_capture_residual_adversary_state(self.sim),
            absolute_time_s=absolute_time_s,
            remaining_budget_steps=remaining_budget_steps,
        )

    def restore(self, snapshot: _SimulatorSnapshot) -> None:  # noqa: C901
        """Restore the live simulator to a previously captured snapshot."""
        self._step_index = snapshot.step_index
        self.sim.pysf_state.pysf_states()[...] = snapshot.pysf_state
        self.sim.ped_headings = snapshot.ped_headings.copy()
        self.sim.ped_angular_velocities = snapshot.ped_angular_velocities.copy()
        for robot, pose, vel_state in zip(
            self.sim.robots, snapshot.robot_poses, snapshot.robot_velocities, strict=True
        ):
            if vel_state is not None:
                robot.state = deepcopy(vel_state)
                if hasattr(robot.state, "pose"):
                    robot.state.pose = deepcopy(pose)
            else:
                robot.reset_state(deepcopy(pose))
        _restore_robot_navigators(self.sim.robot_navs, snapshot.robot_navigators)
        _restore_single_runtimes(self.sim.peds_behaviors, snapshot.single_runtimes)
        _restore_route_navigators(self.sim.peds_behaviors, snapshot.route_navigators)
        groups = getattr(self.sim, "groups", None)
        if (
            groups is not None
            and snapshot.pedestrian_groups is not None
            and snapshot.pedestrian_group_by_ped is not None
        ):
            _restore_pedestrian_groups(
                groups,
                snapshot.pedestrian_groups,
                snapshot.pedestrian_group_by_ped,
            )
            _synchronize_pysf_groups(self.sim)
        _restore_behavior_rng_states(self.sim.peds_behaviors, snapshot.behavior_rng_states)
        if snapshot.residual_adversary is not None:
            self.sim._residual_adversary = deepcopy(snapshot.residual_adversary)
        elif (
            snapshot.residual_adversary_state is None
            and getattr(self.sim, "_residual_adversary", None) is not None
        ):
            raise ValueError("snapshot is missing residual-adversary state")
        self.sim.peds_have_obstacle_forces = snapshot.peds_have_obstacle_forces
        peds = getattr(getattr(self.sim, "pysf_sim", None), "peds", None)
        if snapshot.ped_max_speeds is not None and peds is not None:
            current_max_speeds = getattr(peds, "max_speeds", None)
            if current_max_speeds is None or np.shape(current_max_speeds) != np.shape(
                snapshot.ped_max_speeds
            ):
                peds.max_speeds = snapshot.ped_max_speeds.copy()
            else:
                peds.max_speeds[...] = snapshot.ped_max_speeds
        _restore_residual_adversary_state(self.sim, snapshot.residual_adversary_state)
        if snapshot.global_rng_state is not None:
            key, state, pos, has_gauss, cached_gauss = snapshot.global_rng_state
            np.random.set_state((key, state.copy(), pos, has_gauss, cached_gauss))
        if snapshot.python_random_state is not None:
            random.setstate(deepcopy(snapshot.python_random_state))

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

        The default set follows the native action semantics of the configured robot.
        Acceleration-controlled robots receive neutral, partial-braking, and full-
        braking controls; velocity-controlled robots receive the current velocity
        and a stop control. Callers requiring a planner-specific action lattice may
        subclass and override this method. Unknown robot action semantics return an
        empty set so the replay engine fails closed with incomplete coverage.
        """
        robot = self.sim.robots[0]
        config = robot.config
        if hasattr(config, "max_linear_decel"):
            max_decel = float(config.max_linear_decel)
            max_angular_accel = float(config.max_angular_accel)
            # Differential-drive actions are accelerations (m/s^2, rad/s^2),
            # integrated by DifferentialDriveMotion over the simulator timestep.
            return (
                (0.0, 0.0),
                (-0.5 * max_decel, 0.0),
                (-max_decel, 0.0),
                (0.0, -max_angular_accel),
                (0.0, max_angular_accel),
            )
        if hasattr(config, "max_decel") and hasattr(config, "max_steer"):
            max_decel = float(config.max_decel)
            max_steer = float(config.max_steer)
            # Bicycle actions are (linear acceleration, steering angle), not
            # velocity/turn-rate commands.
            return (
                (0.0, 0.0),
                (-0.5 * max_decel, 0.0),
                (-max_decel, 0.0),
                (0.0, -max_steer),
                (0.0, max_steer),
            )
        command_mode = getattr(config, "command_mode", None)
        if command_mode == "vx_vy":
            vx, vy = robot.state.velocity_xy
            max_speed = float(config.max_speed)
            return (
                (float(vx), float(vy)),
                (0.0, 0.0),
                (0.0, -max_speed),
                (0.0, max_speed),
            )
        if command_mode == "unicycle_vw":
            velocity, angular_velocity = robot.state.velocity_vw
            max_angular_speed = float(config.max_angular_speed)
            return (
                (float(velocity), float(angular_velocity)),
                (0.0, 0.0),
                (float(velocity), -max_angular_speed),
                (float(velocity), max_angular_speed),
            )
        return ()

    def action_label(self, action: Any) -> str:
        """Return a stable label for a robot action (for provenance)."""
        first = float(action[0])
        second = float(action[1])
        config = self.sim.robots[0].config
        if hasattr(config, "max_linear_decel"):
            return f"robot_accel=(linear_mps2={first:g},angular_radps2={second:g})"
        if hasattr(config, "max_decel") and hasattr(config, "max_steer"):
            return f"robot_accel=(linear_mps2={first:g},steering_angle_rad={second:g})"
        if getattr(config, "command_mode", None) == "vx_vy":
            return f"robot_velocity=(vx_mps={first:g},vy_mps={second:g})"
        if getattr(config, "command_mode", None) == "unicycle_vw":
            return f"robot_velocity=(linear_mps={first:g},angular_radps={second:g})"
        return f"robot_cmd=(first={first:g},second={second:g})"
