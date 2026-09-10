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
from math import isfinite
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


def _restore_int(value: Any, name: str) -> int:
    """Validate an integer restore field without truncating or coercing strings.

    Returns:
        The validated integer.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _restore_float(value: Any, name: str) -> float:
    """Validate a finite numeric restore field without accepting strings/bools.

    Returns:
        The validated finite float.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _restore_bool(value: Any, name: str) -> bool:
    """Validate a boolean restore field without truthiness coercion.

    Returns:
        The validated boolean.
    """
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _restore_string(value: Any, name: str) -> str:
    """Validate a string restore field without string coercion.

    Returns:
        The validated string.
    """
    if type(value) is not str:
        raise ValueError(f"{name} must be a string")
    return value


def _restore_optional_int(value: Any, name: str) -> int | None:
    """Validate an optional integer restore field.

    Returns:
        The validated integer or ``None``.
    """
    if value is None:
        return None
    return _restore_int(value, name)


def _restore_optional_string(value: Any, name: str) -> str | None:
    """Validate an optional string restore field.

    Returns:
        The validated string or ``None``.
    """
    if value is None:
        return None
    return _restore_string(value, name)


def _restore_mapping_index(value: Any, name: str) -> int:
    """Validate an integer mapping key, including its JSON string form.

    Returns:
        The validated integer key.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if type(value) is str:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer") from exc
        if str(parsed) != value:
            raise ValueError(f"{name} must use the canonical integer string form")
        return parsed
    raise ValueError(f"{name} must be an integer")


_MODERN_RUNTIME_FIELDS = frozenset(
    {
        "ped_id",
        "definition_id",
        "trajectory",
        "waypoint_index",
        "pending_waits",
        "start_delay_remaining_s",
        "wait_remaining_s",
        "waiting_for_advance",
        "joined_group_id",
        "left_group",
        "hold_waypoint_index",
        "proximity_hold_engaged",
        "proximity_hold_released",
        "proximity_hold_elapsed_s",
        "hold_released_by",
    }
)
_LEGACY_RUNTIME_FIELDS = (_MODERN_RUNTIME_FIELDS - {"definition_id"}) | {"definition"}
_MISSING = object()


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


def _validate_restore_array(snapshot_array: Any, destination_array: Any, name: str) -> None:
    """Require an owned numeric array to match the destination exactly."""
    if not isinstance(snapshot_array, np.ndarray):
        raise ValueError(f"{name} must be a numpy array")
    if not isinstance(destination_array, np.ndarray):
        raise ValueError(f"destination {name} must be a numpy array")
    if snapshot_array.dtype == object or not np.issubdtype(snapshot_array.dtype, np.number):
        raise ValueError(f"{name} must have a numeric dtype")
    if not np.all(np.isfinite(snapshot_array)):
        raise ValueError(f"{name} must contain only finite values")
    if snapshot_array.shape != destination_array.shape:
        raise ValueError(
            f"{name} shape mismatch: snapshot={snapshot_array.shape}, "
            f"destination={destination_array.shape}"
        )
    if snapshot_array.dtype != destination_array.dtype:
        raise ValueError(
            f"{name} dtype mismatch: snapshot={snapshot_array.dtype}, "
            f"destination={destination_array.dtype}"
        )


def _validate_raw_snapshot(simulator: Any, snapshot: _SimulatorSnapshot) -> None:
    """Validate raw restore arrays and booleans before any destination mutation."""
    if not isinstance(snapshot, _SimulatorSnapshot):
        raise ValueError("snapshot must be a SimulatorSnapshot")
    _restore_int(snapshot.step_index, "snapshot step_index")
    _restore_bool(snapshot.peds_have_obstacle_forces, "snapshot peds_have_obstacle_forces")
    _validate_restore_array(
        snapshot.pysf_state,
        simulator.pysf_state.pysf_states(),
        "pysf_state",
    )
    _validate_restore_array(snapshot.ped_headings, simulator.ped_headings, "ped_headings")
    _validate_restore_array(
        snapshot.ped_angular_velocities,
        simulator.ped_angular_velocities,
        "ped_angular_velocities",
    )

    peds = getattr(getattr(simulator, "pysf_sim", None), "peds", None)
    destination_max_speeds = getattr(peds, "max_speeds", None)
    if (snapshot.ped_max_speeds is None) != (destination_max_speeds is None):
        raise ValueError("ped_max_speeds presence mismatch between snapshot and destination")
    if snapshot.ped_max_speeds is not None:
        _validate_restore_array(
            snapshot.ped_max_speeds,
            destination_max_speeds,
            "ped_max_speeds",
        )


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


def _restore_runtime_trajectory(value: Any) -> list[tuple[float, float]]:
    """Validate and normalize one captured single-pedestrian trajectory.

    Returns:
        A destination-owned list of finite coordinate pairs.
    """
    if not isinstance(value, list):
        raise ValueError("single-pedestrian runtime trajectory must be a list")
    trajectory: list[tuple[float, float]] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"single-pedestrian runtime trajectory[{index}] must be a pair")
        trajectory.append(
            (
                _restore_float(point[0], f"single-pedestrian trajectory[{index}][0]"),
                _restore_float(point[1], f"single-pedestrian trajectory[{index}][1]"),
            )
        )
    return trajectory


def _restore_runtime_pending_waits(value: Any) -> dict[int, float]:
    """Validate and normalize captured pending-wait entries.

    Returns:
        A destination-owned integer-keyed wait mapping.
    """
    if not isinstance(value, Mapping):
        raise ValueError("single-pedestrian runtime pending_waits must be a mapping")
    pending_waits: dict[int, float] = {}
    for raw_index, raw_wait in value.items():
        index = _restore_mapping_index(raw_index, "single-pedestrian runtime pending_waits key")
        if index in pending_waits:
            raise ValueError("single-pedestrian runtime contains duplicate pending wait")
        pending_waits[index] = _restore_float(
            raw_wait, f"single-pedestrian runtime pending_waits[{index}]"
        )
    return pending_waits


def _restore_single_runtime_entry(
    fields: Mapping[str, Any],
    current_by_ped_id: Mapping[int, Any],
    current_by_definition_id: Mapping[str, Any],
) -> tuple[Any, SinglePedestrianRuntime]:
    """Validate and rebuild one single-pedestrian runtime entry.

    Returns:
        The matched destination runtime and its restored replacement.
    """
    is_legacy = "definition" in fields
    expected_fields = _LEGACY_RUNTIME_FIELDS if is_legacy else _MODERN_RUNTIME_FIELDS
    if "ped_id" not in fields or (not is_legacy and "definition_id" not in fields):
        raise ValueError("single-pedestrian runtime contains incomplete or unknown fields")
    ped_id = _restore_int(fields["ped_id"], "single-pedestrian runtime ped_id")
    if is_legacy:
        current = current_by_ped_id.get(ped_id)
        saved_definition = fields["definition"]
        if current is not None and getattr(saved_definition, "id", None) != getattr(
            current.definition, "id", None
        ):
            raise ValueError("single-pedestrian snapshot definition does not match destination")
    else:
        definition_id = _restore_string(
            fields["definition_id"], "single-pedestrian runtime definition_id"
        )
        current = current_by_ped_id.get(ped_id) or current_by_definition_id.get(definition_id)
        if (
            current is not None
            and str(getattr(current.definition, "id", current.ped_id)) != definition_id
        ):
            raise ValueError("single-pedestrian snapshot definition does not match destination")
    if current is None:
        identity_message = (
            "does not match destination" if is_legacy else "does not match the destination"
        )
        raise ValueError(
            "single-pedestrian snapshot identity "
            f"{identity_message} behavior (ped_id={fields.get('ped_id')!r})"
        )
    if set(fields) != expected_fields:
        raise ValueError("single-pedestrian runtime contains incomplete or unknown fields")
    restored = SinglePedestrianRuntime(
        ped_id=ped_id,
        definition=current.definition,
        trajectory=_restore_runtime_trajectory(fields["trajectory"]),
        waypoint_index=_restore_int(
            fields["waypoint_index"], "single-pedestrian runtime waypoint_index"
        ),
        pending_waits=_restore_runtime_pending_waits(fields["pending_waits"]),
        start_delay_remaining_s=_restore_float(
            fields["start_delay_remaining_s"],
            "single-pedestrian runtime start_delay_remaining_s",
        ),
        wait_remaining_s=_restore_float(
            fields["wait_remaining_s"],
            "single-pedestrian runtime wait_remaining_s",
        ),
        waiting_for_advance=_restore_bool(
            fields["waiting_for_advance"], "single-pedestrian runtime waiting_for_advance"
        ),
        joined_group_id=_restore_optional_int(
            fields["joined_group_id"], "single-pedestrian runtime joined_group_id"
        ),
        left_group=_restore_bool(fields["left_group"], "single-pedestrian runtime left_group"),
        hold_waypoint_index=_restore_optional_int(
            fields["hold_waypoint_index"], "single-pedestrian runtime hold_waypoint_index"
        ),
        proximity_hold_engaged=_restore_bool(
            fields["proximity_hold_engaged"],
            "single-pedestrian runtime proximity_hold_engaged",
        ),
        proximity_hold_released=_restore_bool(
            fields["proximity_hold_released"],
            "single-pedestrian runtime proximity_hold_released",
        ),
        proximity_hold_elapsed_s=_restore_float(
            fields["proximity_hold_elapsed_s"],
            "single-pedestrian runtime proximity_hold_elapsed_s",
        ),
        hold_released_by=_restore_optional_string(
            fields["hold_released_by"], "single-pedestrian runtime hold_released_by"
        ),
    )
    return current, restored


def _restore_single_runtimes(  # noqa: C901
    peds_behaviors: list[Any], runtimes: list[Any]
) -> None:
    """Restore single-pedestrian behavior state from a complete snapshot."""
    if not isinstance(runtimes, list) or len(runtimes) != len(peds_behaviors):
        raise ValueError("single-pedestrian runtime snapshot does not match behavior count")
    assignments: list[tuple[Any, list[SinglePedestrianRuntime]]] = []
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
            current, restored_runtime = _restore_single_runtime_entry(
                fields, current_by_ped_id, current_by_definition_id
            )
            if id(current) in matched_runtime_ids:
                raise ValueError("single-pedestrian snapshot contains a duplicate runtime")
            matched_runtime_ids.add(id(current))
            restored.append(restored_runtime)
        if len(matched_runtime_ids) != len(current_runtimes):
            raise ValueError("single-pedestrian snapshot omits a destination runtime")
        assignments.append((behavior, restored))
    for behavior, restored in assignments:
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


def _behavior_rng_targets(peds_behaviors: list[Any]) -> list[tuple[str, int, Any]]:
    """Collect stable and legacy identities for owned behavior generators.

    Returns:
        Tuples of stable identity, legacy object identity, and bit generator.
    """
    targets: list[tuple[str, int, Any]] = []
    for index, behavior in enumerate(peds_behaviors):
        identity = _stable_behavior_identity(index, behavior)
        for attribute in ("rng", "_rng"):
            generator = getattr(behavior, attribute, None)
            bit_generator = getattr(generator, "bit_generator", None)
            if bit_generator is not None:
                targets.append((f"{identity}:{attribute}", id(behavior), bit_generator))
    return targets


def _select_behavior_rng_state(saved: Mapping[Any, Any], key: str, legacy_key: int) -> Any:
    """Select one stable or legacy behavior RNG state, rejecting duplicates.

    Returns:
        The selected state mapping.
    """
    has_stable = key in saved
    has_legacy = legacy_key in saved
    if has_stable and has_legacy:
        raise ValueError(f"snapshot contains duplicate behavior RNG identity {key!r}")
    if has_stable:
        state = saved[key]
    elif has_legacy:
        # Read-only compatibility for snapshots written by the first hardening
        # slice, which keyed generators by process-local object id.
        state = saved[legacy_key]
    else:
        raise ValueError(f"snapshot is missing behavior RNG {key!r}")
    if not isinstance(state, Mapping):
        raise ValueError(f"behavior RNG state for {key!r} must be a mapping")
    return state


def _restore_behavior_rng_value(  # noqa: C901 - mirrors the supported NumPy state scalar shapes
    value: Any, template: Any, name: str
) -> Any:
    """Validate one behavior RNG value against the destination state shape.

    Returns:
        A destination-owned RNG value with no scalar coercion or unknown fields.
    """
    if isinstance(template, Mapping):
        if not isinstance(value, Mapping) or set(value) != set(template):
            raise ValueError(f"behavior RNG state for {name} has incomplete or unknown fields")
        return {
            key: _restore_behavior_rng_value(value[key], template[key], f"{name}.{key}")
            for key in template
        }
    if isinstance(template, np.ndarray):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype == object
            or value.dtype != template.dtype
            or value.shape != template.shape
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(f"behavior RNG state for {name} has an incompatible array")
        return value.copy()
    if isinstance(template, bool):
        return _restore_bool(value, f"behavior RNG state {name}")
    if isinstance(template, (int, np.integer)) and not isinstance(template, bool):
        return _restore_int(value, f"behavior RNG state {name}")
    if isinstance(template, (float, np.floating)):
        return _restore_float(value, f"behavior RNG state {name}")
    if type(template) is str:
        return _restore_string(value, f"behavior RNG state {name}")
    if template is None:
        if value is not None:
            raise ValueError(f"behavior RNG state {name} must be null")
        return None
    if type(value) is not type(template):
        raise ValueError(f"behavior RNG state {name} has an incompatible type")
    return deepcopy(value)


def _restore_behavior_rng_states(peds_behaviors: list[Any], saved: dict[str, Any] | None) -> None:
    """Restore owned NumPy generator states, rejecting identity drift."""
    if saved is not None and not isinstance(saved, Mapping):
        raise ValueError("behavior RNG snapshot must be a mapping")
    targets = _behavior_rng_targets(peds_behaviors)
    if saved is None:
        if targets:
            raise ValueError(f"snapshot is missing behavior RNG {targets[0][0]!r}")
        return
    invalid_keys = [key for key in saved if type(key) not in (str, int)]
    if invalid_keys:
        raise ValueError(
            f"snapshot contains invalid behavior RNG identities: {sorted(invalid_keys, key=str)}"
        )
    expected_keys = {key for key, _, _ in targets}
    legacy_keys = {legacy_key for _, legacy_key, _ in targets}
    pending = []
    for key, legacy_key, bit_generator in targets:
        state = _select_behavior_rng_state(saved, key, legacy_key)
        pending.append(
            (
                bit_generator,
                _restore_behavior_rng_value(state, bit_generator.state, key),
            )
        )
    unexpected = set(saved) - expected_keys - legacy_keys
    if unexpected:
        raise ValueError(
            f"snapshot contains unknown behavior RNG identities: {sorted(unexpected, key=str)}"
        )
    for bit_generator, state in pending:
        bit_generator.state = deepcopy(state)


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


def _resolve_residual_adversary(simulator: Any) -> tuple[Any, bool]:
    """Return the current residual adversary and whether this call created it.

    Returns:
        The adversary object and a creation flag.
    """
    adversary = getattr(simulator, "_residual_adversary", None)
    if adversary is not None:
        return adversary, False
    builder = getattr(simulator, "_build_residual_adversary", None)
    if not callable(builder):
        raise ValueError("snapshot requires a residual-adversary instance")
    adversary = builder()
    if adversary is None:
        raise ValueError("snapshot contains residual-adversary state but config is inactive")
    return adversary, True


def _validate_residual_fields(adversary: Any, saved: Mapping[str, Any]) -> set[str]:
    """Validate residual field names against the supported and destination sets.

    Returns:
        The supported field names present on the destination adversary.
    """
    if any(type(field) is not str for field in saved):
        raise ValueError("residual-adversary snapshot fields must be strings")
    expected_fields = {field for field in _RESIDUAL_STATE_FIELDS if hasattr(adversary, field)}
    unsupported = set(saved) - set(_RESIDUAL_STATE_FIELDS)
    if unsupported:
        raise ValueError(
            f"unsupported residual-adversary state field {sorted(unsupported, key=str)}"
        )
    destination_unsupported = set(saved) - expected_fields
    if destination_unsupported:
        raise ValueError(
            "residual-adversary state fields are unsupported by destination: "
            f"{sorted(destination_unsupported)}"
        )
    missing = expected_fields - set(saved)
    if missing:
        raise ValueError(f"snapshot is missing residual-adversary state fields: {sorted(missing)}")
    return expected_fields


def _restore_residual_value(current: Any, value: Any, field: str) -> Any:
    """Validate one residual-adversary value against its destination field.

    Returns:
        A destination-owned validated value.
    """
    if isinstance(current, np.ndarray):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype == object
            or not np.issubdtype(value.dtype, np.number)
        ):
            raise ValueError(f"residual-adversary field {field} must be a numeric array")
        if value.shape != current.shape or value.dtype != current.dtype:
            raise ValueError(f"residual-adversary field {field} shape or dtype mismatch")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"residual-adversary field {field} contains non-finite values")
        return value.copy()
    if isinstance(current, bool):
        return _restore_bool(value, f"residual-adversary field {field}")
    if isinstance(current, (int, np.integer)) and not isinstance(current, bool):
        return _restore_int(value, f"residual-adversary field {field}")
    if isinstance(current, (float, np.floating)):
        return _restore_float(value, f"residual-adversary field {field}")
    if type(value) is not type(current):
        raise ValueError(f"residual-adversary field {field} has an incompatible type")
    return deepcopy(value)


def _restore_residual_adversary_state(simulator: Any, saved: dict[str, Any] | None) -> None:
    """Restore a captured residual-adversary state without loading executable objects."""
    if saved is None:
        if getattr(simulator, "_residual_adversary", None) is not None:
            raise ValueError("snapshot is missing residual-adversary state")
        return
    if not isinstance(saved, Mapping):
        raise ValueError("residual-adversary snapshot must be a mapping")
    adversary, created = _resolve_residual_adversary(simulator)
    _validate_residual_fields(adversary, saved)
    restored = {
        field: _restore_residual_value(getattr(adversary, field), value, field)
        for field, value in saved.items()
    }
    if created:
        simulator._residual_adversary = adversary
    for field, value in restored.items():
        setattr(adversary, field, value)


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


def _select_route_navigator_capture(
    behavior: Any, identity: str, navigators: Mapping[Any, Any]
) -> tuple[Any, int]:
    """Select a stable or legacy capture for one route behavior.

    Returns:
        The captured mapping and its accepted legacy identity.
    """
    legacy_identity = id(behavior)
    has_stable = identity in navigators
    has_legacy = legacy_identity in navigators
    if has_stable and has_legacy:
        raise ValueError(f"snapshot contains duplicate route navigator identity {identity!r}")
    if has_stable:
        captured = navigators[identity]
    elif has_legacy:
        # Read-only compatibility for an in-memory snapshot from the old
        # process-local representation. Durable snapshots never emit this key.
        captured = navigators[legacy_identity]
    else:
        raise ValueError(f"snapshot is missing route navigator identity {identity!r}")
    if not isinstance(captured, Mapping):
        raise ValueError(f"route navigator state for {identity!r} must be a mapping")
    return captured, legacy_identity


def _route_navigator_assignments(
    navs: Mapping[Any, Any], captured: Mapping[Any, Any], identity: str
) -> list[tuple[Any, int, bool]]:
    """Validate one route behavior's captured group states.

    Returns:
        Pending navigator assignments as ``(navigator, waypoint_id, reached)`` tuples.
    """
    captured_by_group: dict[int, Mapping[str, Any]] = {}
    for raw_group_id, state in captured.items():
        group_id = _restore_mapping_index(
            raw_group_id, f"route navigator identity {identity} group id"
        )
        if group_id in captured_by_group:
            raise ValueError(
                f"route navigator state for {identity!r} contains duplicate group identity"
            )
        if not isinstance(state, Mapping):
            raise ValueError(f"route navigator state for group {group_id!r} is incomplete")
        captured_by_group[group_id] = state
    expected_group_ids = {
        _restore_mapping_index(group_id, "destination route navigator group id")
        for group_id in navs
    }
    if set(captured_by_group) != expected_group_ids:
        raise ValueError(
            f"route navigator group identities for {identity!r} do not match destination"
        )
    assignments: list[tuple[Any, int, bool]] = []
    for gid, nav in navs.items():
        group_id = _restore_mapping_index(gid, "destination route navigator group id")
        state = captured_by_group[group_id]
        if set(state) != {"waypoint_id", "reached_waypoint"}:
            raise ValueError(
                f"route navigator state for group {gid!r} is incomplete or contains unknown fields"
            )
        waypoint_id = _restore_int(
            state["waypoint_id"], f"route navigator waypoint_id for group {gid!r}"
        )
        reached_waypoint = _restore_bool(
            state["reached_waypoint"], f"route navigator reached_waypoint for group {gid!r}"
        )
        if nav.waypoints and not 0 <= waypoint_id < len(nav.waypoints):
            raise ValueError(f"route navigator waypoint_id for group {gid!r} is out of range")
        assignments.append((nav, waypoint_id, reached_waypoint))
    return assignments


def _restore_route_navigators(peds_behaviors: list[Any], navigators: dict[str, Any]) -> None:
    """Restore route-group navigator state, rejecting missing identities or fields."""
    if not isinstance(navigators, Mapping):
        raise ValueError("route navigator snapshot must be a mapping")
    invalid_keys = [key for key in navigators if type(key) not in (str, int)]
    if invalid_keys:
        raise ValueError(
            f"snapshot contains invalid route navigator identities: {sorted(invalid_keys, key=str)}"
        )
    expected_identities: set[str] = set()
    legacy_identities: set[int] = set()
    assignments: list[tuple[Any, int, bool]] = []
    for index, behavior in enumerate(peds_behaviors):
        navs = getattr(behavior, "navigators", None)
        if not navs:
            continue
        identity = _stable_behavior_identity(index, behavior)
        expected_identities.add(identity)
        captured, legacy_identity = _select_route_navigator_capture(behavior, identity, navigators)
        legacy_identities.add(legacy_identity)
        assignments.extend(_route_navigator_assignments(navs, captured, identity))
    unexpected = {
        key for key in navigators if key not in expected_identities and key not in legacy_identities
    }
    if unexpected:
        raise ValueError(
            f"snapshot contains unknown route navigator identities: {sorted(unexpected, key=str)}"
        )
    for navigator, waypoint_id, reached_waypoint in assignments:
        navigator.waypoint_id = waypoint_id
        navigator.reached_waypoint = reached_waypoint


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
        """Return the declared finite action lattice and its current cardinality.

        The production robot command space is generally continuous. This adapter
        evaluates only the explicit finite lattice returned by
        :meth:`feasible_actions`; the label must not claim exhaustive coverage of
        the underlying continuous space.
        """
        return f"native_declared_action_lattice_v1:n={len(self.feasible_actions())}"

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
            peds_have_obstacle_forces=_restore_bool(
                self.sim.peds_have_obstacle_forces,
                "simulator peds_have_obstacle_forces",
            ),
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

    def restore(self, snapshot: _SimulatorSnapshot) -> None:
        """Restore the live simulator atomically to a captured snapshot."""
        before = self.snapshot()
        before_residual = getattr(self.sim, "_residual_adversary", _MISSING)
        try:
            self._restore_unchecked(snapshot)
        except Exception:
            try:
                if before_residual is _MISSING:
                    if hasattr(self.sim, "_residual_adversary"):
                        delattr(self.sim, "_residual_adversary")
                else:
                    self.sim._residual_adversary = deepcopy(before_residual)
                self._restore_unchecked(before)
            except Exception as rollback_exc:
                raise RuntimeError(
                    "simulator snapshot restore failed and rollback failed; "
                    "destination state may be partial"
                ) from rollback_exc
            raise

    def _restore_unchecked(self, snapshot: _SimulatorSnapshot) -> None:  # noqa: C901
        """Apply a snapshot, leaving rollback responsibility to :meth:`restore`."""
        _validate_raw_snapshot(self.sim, snapshot)
        self._step_index = _restore_int(snapshot.step_index, "snapshot step_index")
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
        self.sim.peds_have_obstacle_forces = _restore_bool(
            snapshot.peds_have_obstacle_forces, "snapshot peds_have_obstacle_forces"
        )
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
