"""Controlled deterministic fixtures for frozen-state counterfactual replay (#5442).

These fixtures implement :class:`~robot_sf.benchmark.last_avoidable_replay.CounterfactualModel`
with a minimal 2D kinematic robot/pedestrian interaction. They are *controlled
fixtures*, not the production simulator: the full robot_sf simulator draws
pedestrian goals/zones from the **global** numpy RNG and exposes no snapshot API,
so a faithful mid-episode snapshot/restore seam there would require a broad
simulator change (out of scope for #5442; see
``docs/context/issue_5442_last_avoidable_replay.md``). This kinematic model gives
a fully deterministic, snapshot-restorable state — including its own RNG — so the
counterfactual-replay engine can be validated end to end on CPU.

The robot travels along ``+x`` toward a crossing pedestrian and may command a
deceleration each tick. In ``replayed`` pedestrian mode the pedestrian follows a
path that is a function of time and the fixture RNG only (independent of the
robot), so branching robot actions never change the pedestrian. In
``closed_loop`` mode the pedestrian additionally reacts to the robot (a repulsion
that couples the two bodies), which is the two-action interaction case.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from robot_sf.benchmark.last_avoidable_replay import ReplayConfig

PED_RESPONSE_REPLAYED = "replayed"
PED_RESPONSE_CLOSED_LOOP = "closed_loop"


@dataclass(frozen=True)
class KinematicScenario:
    """Parameters of a controlled kinematic collision fixture.

    Attributes:
        robot_x0: Initial robot ``x`` position (robot ``y`` is fixed at 0).
        robot_speed0: Initial robot forward speed (m/s along ``+x``).
        ped_pos0: Initial pedestrian ``(x, y)`` position.
        ped_vel0: Nominal pedestrian ``(vx, vy)`` velocity (the recorded path in
            ``replayed`` mode).
        dt: Control tick duration (s).
        collision_radius: Contact distance between robot and pedestrian (m).
        decel_levels: Admissible deceleration lattice (m/s^2); ``0.0`` is the
            maintain-speed baseline action.
        pedestrian_response: ``replayed`` or ``closed_loop``.
        ped_repulsion_gain: Closed-loop repulsion strength (ignored when replayed).
        ped_influence_radius: Distance under which the closed-loop pedestrian
            reacts to the robot.
        rng_jitter_std: Std of per-tick pedestrian velocity jitter drawn from the
            fixture RNG; ``0.0`` disables jitter.
        seed: Seed for the fixture's own ``numpy.random.Generator``.
        include_rng_in_snapshot: When ``False`` the snapshot omits the RNG state,
            making a jittered replay nondeterministic — used to exercise the
            engine's ``unknown`` (nondeterministic baseline) path.
        feasible_from_step: Steps strictly before this return an *empty* feasible
            action set, used to exercise the ``unknown`` coverage-gap path.
            ``0`` (default) means every step is testable.
        physical_collision_radius: True geometric contact distance. When ``None``
            (default) it equals :attr:`collision_radius`, so the reported and
            physical collision predicates coincide. When set *smaller* than
            ``collision_radius`` the model's :meth:`KinematicCollisionModel.collision`
            uses the (inflated) ``collision_radius`` while
            :meth:`KinematicCollisionModel.physical_collision` uses the physical
            radius — used to model a metric-artifact fixture (a footprint-inflation
            quirk that flags a collision with no physical contact).
    """

    robot_x0: float
    robot_speed0: float
    ped_pos0: tuple[float, float]
    ped_vel0: tuple[float, float]
    dt: float = 0.1
    collision_radius: float = 0.5
    decel_levels: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)
    pedestrian_response: str = PED_RESPONSE_REPLAYED
    ped_repulsion_gain: float = 0.0
    ped_influence_radius: float = 1.5
    rng_jitter_std: float = 0.0
    seed: int = 0
    include_rng_in_snapshot: bool = True
    feasible_from_step: int = 0
    physical_collision_radius: float | None = None


@dataclass
class _KinematicState:
    """Mutable per-step state of the kinematic model."""

    step: int
    robot_x: float
    robot_speed: float
    ped_pos: np.ndarray
    ped_vel: np.ndarray
    rng_state: dict[str, Any]


class KinematicCollisionModel:
    """Deterministic 2D kinematic fixture implementing ``CounterfactualModel``.

    The model holds its own ``numpy.random.Generator`` and exposes snapshot/restore
    that (optionally) includes the RNG bit-generator state, so the replay engine
    can prove baseline determinism and branch over decelerations.
    """

    def __init__(self, scenario: KinematicScenario) -> None:
        """Initialize the model at step 0 from ``scenario``."""
        self.scenario = scenario
        self._rng = np.random.default_rng(scenario.seed)
        self.step_index = 0
        self.robot_x = float(scenario.robot_x0)
        self.robot_speed = float(scenario.robot_speed0)
        self.ped_pos = np.asarray(scenario.ped_pos0, dtype=float)
        self.ped_vel = np.asarray(scenario.ped_vel0, dtype=float)

    # -- CounterfactualModel protocol -------------------------------------
    def snapshot(self) -> _KinematicState:
        """Return a restorable copy of the full state (incl. RNG when configured)."""
        rng_state = self._rng.bit_generator.state if self.scenario.include_rng_in_snapshot else {}
        return _KinematicState(
            step=self.step_index,
            robot_x=self.robot_x,
            robot_speed=self.robot_speed,
            ped_pos=self.ped_pos.copy(),
            ped_vel=self.ped_vel.copy(),
            rng_state=_deep_copy_state(rng_state),
        )

    def restore(self, snapshot: _KinematicState) -> None:
        """Restore the model to ``snapshot`` (RNG restored only if it was captured)."""
        self.step_index = snapshot.step
        self.robot_x = snapshot.robot_x
        self.robot_speed = snapshot.robot_speed
        self.ped_pos = snapshot.ped_pos.copy()
        self.ped_vel = snapshot.ped_vel.copy()
        if snapshot.rng_state:
            self._rng.bit_generator.state = _deep_copy_state(snapshot.rng_state)

    def step(self, action: Any) -> None:
        """Advance one control tick applying deceleration ``action`` (m/s^2)."""
        decel = float(action)
        self.robot_speed = max(0.0, self.robot_speed - decel * self.scenario.dt)
        self.robot_x += self.robot_speed * self.scenario.dt

        effective_vel = self.ped_vel.copy()
        if self.scenario.pedestrian_response == PED_RESPONSE_CLOSED_LOOP:
            effective_vel = effective_vel + self._closed_loop_repulsion()
        if self.scenario.rng_jitter_std > 0.0:
            effective_vel = effective_vel + self._rng.normal(
                0.0, self.scenario.rng_jitter_std, size=2
            )
        self.ped_pos = self.ped_pos + effective_vel * self.scenario.dt
        self.step_index += 1

    def collision(self) -> bool:
        """Return whether robot and pedestrian are within ``collision_radius``.

        This is the *reported* collision predicate (it may use an inflated
        footprint via ``collision_radius``); see :meth:`physical_collision` for
        the true geometric predicate.
        """
        return self._within(self.scenario.collision_radius)

    def physical_distance(self) -> float:
        """Return the true Euclidean robot/pedestrian distance at the current state."""
        robot_pos = np.array([self.robot_x, 0.0])
        return float(np.linalg.norm(robot_pos - self.ped_pos))

    def physical_collision(self) -> bool:
        """Return whether the true geometric distance is within the physical radius.

        When ``physical_collision_radius`` is ``None`` this matches
        :meth:`collision`. When it is set smaller than ``collision_radius``, a
        state can report a collision (:meth:`collision` ``True``) without physical
        contact — the metric-artifact case.
        """
        radius = (
            self.scenario.physical_collision_radius
            if self.scenario.physical_collision_radius is not None
            else self.scenario.collision_radius
        )
        return self._within(radius)

    def _within(self, radius: float) -> bool:
        """Return whether the robot/pedestrian distance is within ``radius``."""
        robot_pos = np.array([self.robot_x, 0.0])
        return bool(np.linalg.norm(robot_pos - self.ped_pos) <= radius)

    def feasible_actions(self) -> Sequence[float]:
        """Return the admissible deceleration lattice at the current step.

        Returns an empty sequence before ``scenario.feasible_from_step`` to model
        a decision point where no admissible substitution can be evaluated.
        """
        if self.step_index < self.scenario.feasible_from_step:
            return ()
        return self.scenario.decel_levels

    def action_label(self, action: Any) -> str:
        """Return a stable label for a deceleration action."""
        return f"decel={float(action):g}"

    # -- internals --------------------------------------------------------
    def _closed_loop_repulsion(self) -> np.ndarray:
        """Return the pedestrian's reactive velocity increment away from the robot."""
        robot_pos = np.array([self.robot_x, 0.0])
        offset = self.ped_pos - robot_pos
        distance = float(np.linalg.norm(offset))
        if distance >= self.scenario.ped_influence_radius or distance == 0.0:
            return np.zeros(2)
        direction = offset / distance
        magnitude = self.scenario.ped_repulsion_gain * (
            1.0 - distance / self.scenario.ped_influence_radius
        )
        return direction * magnitude


def _deep_copy_state(state: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a numpy bit-generator state mapping (arrays copied).

    Returns:
        A deep copy of ``state`` with any nested arrays and mappings copied.
    """
    copied: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, dict):
            copied[key] = _deep_copy_state(value)
        else:
            copied[key] = value
    return copied


def find_contact_step(scenario: KinematicScenario, max_steps: int = 200) -> int | None:
    """Roll the maintain-speed baseline forward and return the first contact step.

    This helper derives ``t_contact`` for a scenario so fixtures need not hard-code
    it.

    Returns:
        The first step index at which contact occurs, or ``None`` if no contact
        occurs within ``max_steps``.
    """
    model = KinematicCollisionModel(scenario)
    if model.collision():
        return 0
    for step in range(max_steps):
        model.step(0.0)
        if model.collision():
            return step
    return None


def maintain_baseline_actions(n_steps: int) -> list[float]:
    """Return ``n_steps`` maintain-speed (zero-deceleration) baseline actions.

    Returns:
        A list of ``n_steps`` zero-deceleration actions.
    """
    return [0.0] * n_steps


# -- Named acceptance-criteria fixtures ------------------------------------
#
# Each builder returns a scenario whose maintain-speed baseline collides. The
# numbers are tuned so the replay engine reports a distinct determination; the
# tests assert the engine's computed t_uca / t_inevitable rather than re-deriving
# the kinematics by hand.


def preventable_late_braking_scenario() -> KinematicScenario:
    """A robot that can avoid contact by braking early but not late.

    The pedestrian crosses the robot's path; braking early enough lets the
    pedestrian clear first (or stops the robot short), while braking only near
    contact cannot. Expected determination: ``avoidable`` with a finite ``t_uca``
    strictly before ``t_inevitable``.

    Returns:
        The configured :class:`KinematicScenario`.
    """
    return KinematicScenario(
        robot_x0=0.0,
        robot_speed0=5.0,
        ped_pos0=(5.0, -1.2),
        ped_vel0=(0.0, 1.0),
        dt=0.1,
        collision_radius=0.5,
        decel_levels=(0.0, 2.0, 4.0, 8.0),
        pedestrian_response=PED_RESPONSE_REPLAYED,
    )


def already_unavoidable_scenario() -> KinematicScenario:
    """A robot already too close and fast for any admissible brake to avoid contact.

    Expected determination: ``already_unavoidable`` (deterministic baseline, full
    feasible-action coverage, no action prevents contact).

    Returns:
        The configured :class:`KinematicScenario`.
    """
    return KinematicScenario(
        robot_x0=3.2,
        robot_speed0=6.0,
        ped_pos0=(5.0, -0.4),
        ped_vel0=(0.0, 0.6),
        dt=0.1,
        collision_radius=0.5,
        decel_levels=(0.0, 2.0, 4.0, 8.0),
        pedestrian_response=PED_RESPONSE_REPLAYED,
    )


def two_action_interaction_scenario() -> KinematicScenario:
    """A closed-loop pedestrian that reacts to the robot (two-body interaction).

    Branching a robot deceleration changes the pedestrian's reactive path too, so
    avoidance is a genuine two-action interaction. Expected determination:
    ``avoidable``.

    Returns:
        The configured :class:`KinematicScenario`.
    """
    return KinematicScenario(
        robot_x0=0.0,
        robot_speed0=5.0,
        ped_pos0=(5.0, -1.2),
        ped_vel0=(0.0, 1.0),
        dt=0.1,
        collision_radius=0.5,
        decel_levels=(0.0, 2.0, 4.0, 8.0),
        pedestrian_response=PED_RESPONSE_CLOSED_LOOP,
        ped_repulsion_gain=1.5,
        ped_influence_radius=1.5,
    )


def nondeterministic_baseline_scenario() -> KinematicScenario:
    """A jittered pedestrian whose RNG is *not* snapshotted, so replay diverges.

    Expected determination: ``unknown`` (nondeterministic baseline).

    Returns:
        The configured :class:`KinematicScenario`.
    """
    return replace(
        preventable_late_braking_scenario(),
        rng_jitter_std=0.5,
        include_rng_in_snapshot=False,
        seed=7,
    )


def missing_feasible_action_scenario() -> KinematicScenario:
    """A scenario with no feasible actions in the danger window (coverage gap).

    Expected determination: ``unknown`` (incomplete feasible-action coverage),
    never ``unavoidable``.

    Returns:
        The configured :class:`KinematicScenario`.
    """
    return replace(
        preventable_late_braking_scenario(),
        feasible_from_step=10_000,  # every in-window step returns an empty set
    )


# ===========================================================================
# Issue #5443: fault-injection cause-attribution fixtures
# ===========================================================================
# The 14 builders below construct :class:`CollisionCauseFixture` objects for the
# frozen manifest ``collision_cause_attribution_manifest_5443.json``. Each
# fixture pairs a faulted kinematic scenario with observable :class:`InjectedFault`
# signatures and a counterfactual ``repair_scenario`` per fault. The fixtures
# intentionally carry **no ground-truth cause_class label**: the rule-based
# analyser (``collision_cause_analyser.py``) attributes a cause from the
# observable evidence and *computed* counterfactuals (does repairing a fault
# remove contact?), never from the answer key.
#
# Geometric primitives (tuned so the decisive pattern is deterministic):
#   * ``_collision_avoidable_scenario``: robot and crossing pedestrian collide on
#     the maintain-speed baseline at ~step 9, but the contact is avoidable by an
#     admissible brake (replay verdict ``avoidable``).
#   * decisive single repair: a faster pedestrian clears before the robot arrives
#     -> the repaired scenario no longer collides (the fault is decisive).
#   * partial repairs (ambiguous fixtures): each alone still collides, only both
#     together avoid contact.
#   * ``already_unavoidable_scenario``: no admissible brake avoids (replay verdict
#     ``already_unavoidable``).


def _repair_pedestrian_clears(scenario: KinematicScenario) -> KinematicScenario:
    """Repair that makes the pedestrian clear before the robot arrives (no contact).

    Used as the *decisive* single repair for the eight avoidable single-cause
    fixtures: removing the decisive fault lets the pedestrian clear in time.

    Returns:
        The repaired scenario in which the pedestrian clears before contact.
    """
    return replace(scenario, ped_vel0=(0.0, 2.0))


def _repair_pedestrian_partially_clears(scenario: KinematicScenario) -> KinematicScenario:
    """Partial repair (faster pedestrian) that *still* collides on its own.

    Used for ambiguous fixtures: this repair alone does not avoid contact, so the
    fault is not decisive by itself.

    Returns:
        The partially repaired scenario that still collides.
    """
    return replace(scenario, ped_vel0=(0.0, 1.6))


def _repair_robot_slows_slightly(scenario: KinematicScenario) -> KinematicScenario:
    """Partial repair (slightly slower robot) that *still* collides on its own.

    Used for ambiguous fixtures as the second candidate repair.

    Returns:
        The partially repaired scenario that still collides.
    """
    return replace(scenario, robot_speed0=4.2)


def _repair_remove_footprint_inflation(scenario: KinematicScenario) -> KinematicScenario:
    """Repair that sets the reported radius back to the physical radius.

    Returns:
        The repaired scenario with the reported radius equal to the physical radius.
    """
    return replace(scenario, collision_radius=scenario.physical_collision_radius or 0.5)


@dataclass(frozen=True)
class InjectedFault:
    """One observable fault signature injected into a controlled fixture.

    This is the *trace evidence* the analyser detects, not the answer key: it
    records what kind of pipeline fault occurred and the control-step window in
    which it is observable. Decisiveness is computed by the analyser via
    ``repair_scenario``; it is not declared here.

    Attributes:
        fault_type: Observable fault category (one of the single cause-class
            labels). For a negative control this is the *suspicious* signal type.
        activation_window: Inclusive ``(start, end)`` control-step window in which
            the fault is observable. Negative controls that never truly activate
            use ``(-1, -1)``.
        repair_scenario: Callable returning the kinematic scenario with this fault
            removed, used to *compute* whether repairing the fault avoids contact.
            ``None`` when no repair is testable.
        gates_applied_command: Whether the fault ever gated the applied command on
            this trace. A suspicious signal that never gates the command is
            correlation without causal effect (negative-control guard flap).
    """

    fault_type: str
    activation_window: tuple[int, int]
    repair_scenario: Callable[[KinematicScenario], KinematicScenario] | None = None
    gates_applied_command: bool = True


@dataclass(frozen=True)
class CollisionCauseFixture:
    """A controlled fault-injection fixture: scenario + observable fault evidence.

    The fixture intentionally does **not** carry the manifest's ground-truth
    ``cause_class``; the analyser attributes from the observable faults and
    computed counterfactuals. ``fixture_id`` keys the fixture to the frozen
    manifest for scoring only.

    Attributes:
        fixture_id: Stable identifier matching a manifest entry.
        scenario: The faulted kinematic scenario (baseline collides, or reports a
            collision for the metric-artifact case).
        faults: Observable injected fault signatures.
        replay_config: Optional replay configuration override; when ``None`` the
            analyser derives a deterministic config from the contact step.
        metric_artifact: When ``True`` the scenario reports a collision that has
            no physical contact (footprint inflation).
    """

    fixture_id: str
    scenario: KinematicScenario
    faults: tuple[InjectedFault, ...] = ()
    replay_config: ReplayConfig | None = None
    metric_artifact: bool = False


def _avoidable_collision_scenario() -> KinematicScenario:
    """Base avoidable collision scenario: collides on baseline, avoidable by braking.

    Returns:
        The base avoidable collision scenario.
    """
    return KinematicScenario(
        robot_x0=0.0,
        robot_speed0=5.0,
        ped_pos0=(5.0, -1.2),
        ped_vel0=(0.0, 1.0),
        dt=0.1,
        collision_radius=0.5,
        decel_levels=(0.0, 2.0, 4.0, 8.0),
        pedestrian_response=PED_RESPONSE_REPLAYED,
    )


# --- Eight avoidable single-cause fixtures (decisive repair avoids contact) --


def obs_omission_01_fixture() -> CollisionCauseFixture:
    """Fixture ``obs_omission_01``: an observation omission drops a detection.

    The decisive repair (restore the dropped detection) lets the pedestrian clear
    in time. Observable onset at step 12 (the manifest activation window).

    Returns:
        The ``obs_omission_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="obs_omission_01",
        scenario=_avoidable_collision_scenario(),
        faults=(
            InjectedFault(
                fault_type="observation_omission",
                activation_window=(12, 12),
                repair_scenario=_repair_pedestrian_clears,
            ),
        ),
    )


def obs_delay_01_fixture() -> CollisionCauseFixture:
    """Fixture ``obs_delay_01``: the observation stream is delayed by three steps.

    Returns:
        The ``obs_delay_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="obs_delay_01",
        scenario=_avoidable_collision_scenario(),
        faults=(
            InjectedFault(
                fault_type="observation_delay",
                activation_window=(8, 11),
                repair_scenario=_repair_pedestrian_clears,
            ),
        ),
    )


def prediction_miss_01_fixture() -> CollisionCauseFixture:
    """Fixture ``prediction_miss_01``: the pedestrian forecast omits a turn.

    Returns:
        The ``prediction_miss_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="prediction_miss_01",
        scenario=_avoidable_collision_scenario(),
        faults=(
            InjectedFault(
                fault_type="prediction_miss",
                activation_window=(15, 15),
                repair_scenario=_repair_pedestrian_clears,
            ),
        ),
    )


def candidate_omission_01_fixture() -> CollisionCauseFixture:
    """Fixture ``candidate_omission_01``: the evasive candidate is pruned.

    Returns:
        The ``candidate_omission_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="candidate_omission_01",
        scenario=_avoidable_collision_scenario(),
        faults=(
            InjectedFault(
                fault_type="candidate_omission",
                activation_window=(9, 9),
                repair_scenario=_repair_pedestrian_clears,
            ),
        ),
    )


def bad_selection_01_fixture() -> CollisionCauseFixture:
    """Fixture ``bad_selection_01``: the selector picks a colliding candidate.

    Returns:
        The ``bad_selection_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="bad_selection_01",
        scenario=_avoidable_collision_scenario(),
        faults=(
            InjectedFault(
                fault_type="bad_selection",
                activation_window=(10, 10),
                repair_scenario=_repair_pedestrian_clears,
            ),
        ),
    )


def guard_omission_01_fixture() -> CollisionCauseFixture:
    """Fixture ``guard_omission_01``: the safety guard is disabled across a window.

    Returns:
        The ``guard_omission_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="guard_omission_01",
        scenario=_avoidable_collision_scenario(),
        faults=(
            InjectedFault(
                fault_type="guard_omission",
                activation_window=(13, 14),
                repair_scenario=_repair_pedestrian_clears,
            ),
        ),
    )


def infeasible_command_01_fixture() -> CollisionCauseFixture:
    """Fixture ``infeasible_command_01``: the commanded deceleration saturates.

    Returns:
        The ``infeasible_command_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="infeasible_command_01",
        scenario=_avoidable_collision_scenario(),
        faults=(
            InjectedFault(
                fault_type="infeasible_applied_command",
                activation_window=(11, 11),
                repair_scenario=_repair_pedestrian_clears,
            ),
        ),
    )


def route_trap_01_fixture() -> CollisionCauseFixture:
    """Fixture ``route_trap_01``: the route commits to a corridor with no evasion.

    Returns:
        The ``route_trap_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="route_trap_01",
        scenario=_avoidable_collision_scenario(),
        faults=(
            InjectedFault(
                fault_type="route_trap",
                activation_window=(3, 3),
                repair_scenario=_repair_pedestrian_clears,
            ),
        ),
    )


# --- Already-unavoidable contact (no decisive fault; pure inevitability) ----


def already_unavoidable_01_fixture() -> CollisionCauseFixture:
    """Fixture ``already_unavoidable_01``: contact was already unavoidable.

    No fault signature is injected; the cause is the inevitability itself, which
    the analyser recovers from the replay verdict ``already_unavoidable``.

    Returns:
        The ``already_unavoidable_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="already_unavoidable_01",
        scenario=already_unavoidable_scenario(),
        faults=(),
    )


# --- Metric artifact (reported collision with no physical contact) ----------


def metric_artifact_01_fixture() -> CollisionCauseFixture:
    """Fixture ``metric_artifact_01``: footprint inflation flags a phantom contact.

    The reported collision radius is inflated beyond the physical radius, and the
    geometry sits in the quirk band: the reported predicate fires while the true
    geometric distance stays beyond the physical radius. The analyser detects the
    reported/physical mismatch directly from the model.

    Returns:
        The ``metric_artifact_01`` fault-injection fixture.
    """
    scenario = KinematicScenario(
        robot_x0=0.0,
        robot_speed0=5.0,
        ped_pos0=(5.0, -0.9),
        ped_vel0=(0.0, 0.0),
        dt=0.1,
        collision_radius=1.2,  # inflated reported radius
        physical_collision_radius=0.4,  # true physical radius
        decel_levels=(0.0, 2.0, 4.0, 8.0),
        pedestrian_response=PED_RESPONSE_REPLAYED,
    )
    return CollisionCauseFixture(
        fixture_id="metric_artifact_01",
        scenario=scenario,
        faults=(
            InjectedFault(
                fault_type="metric_artifact",
                activation_window=(20, 20),
                repair_scenario=_repair_remove_footprint_inflation,
            ),
        ),
        metric_artifact=True,
    )


# --- Ambiguous interacting fixtures (no single decisive repair) -------------


def ambiguous_pred_guard_01_fixture() -> CollisionCauseFixture:
    """Fixture ``ambiguous_pred_guard_01``: prediction miss + guard omission.

    Two candidate faults overlap; each single repair still collides, so neither
    alone is decisive and the analyser abstains as ``interacting_ambiguous``.

    Returns:
        The ``ambiguous_pred_guard_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="ambiguous_pred_guard_01",
        scenario=_avoidable_collision_scenario(),
        faults=(
            InjectedFault(
                fault_type="prediction_miss",
                activation_window=(10, 14),
                repair_scenario=_repair_pedestrian_partially_clears,
            ),
            InjectedFault(
                fault_type="guard_omission",
                activation_window=(10, 14),
                repair_scenario=_repair_robot_slows_slightly,
            ),
        ),
    )


def ambiguous_route_selection_01_fixture() -> CollisionCauseFixture:
    """Fixture ``ambiguous_route_selection_01``: route trap + bad selection.

    Two candidate faults jointly cause contact; neither single repair is
    decisive, so the analyser abstains as ``interacting_ambiguous``.

    Returns:
        The ``ambiguous_route_selection_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="ambiguous_route_selection_01",
        scenario=_avoidable_collision_scenario(),
        faults=(
            InjectedFault(
                fault_type="route_trap",
                activation_window=(5, 12),
                repair_scenario=_repair_robot_slows_slightly,
            ),
            InjectedFault(
                fault_type="bad_selection",
                activation_window=(5, 12),
                repair_scenario=_repair_pedestrian_partially_clears,
            ),
        ),
    )


# --- Negative controls (suspicious signal, no causal effect) ---------------


def negative_control_jitter_01_fixture() -> CollisionCauseFixture:
    """Fixture ``negative_control_jitter_01``: observation jitter, no effect.

    A suspicious observation-jitter signal is present but the collision is
    already unavoidable and the jitter never gates the applied command; its repair
    does not avoid contact, so the analyser abstains as ``none``.

    Returns:
        The ``negative_control_jitter_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="negative_control_jitter_01",
        scenario=already_unavoidable_scenario(),
        faults=(
            InjectedFault(
                fault_type="observation_omission",
                activation_window=(-1, -1),
                repair_scenario=None,
                gates_applied_command=False,
            ),
        ),
    )


def negative_control_guard_flap_01_fixture() -> CollisionCauseFixture:
    """Fixture ``negative_control_guard_flap_01``: guard toggles, never gates.

    A guard toggles briefly but its state never gates the applied command on this
    trace; correlation without causal effect. The analyser abstains as ``none``.

    Returns:
        The ``negative_control_guard_flap_01`` fault-injection fixture.
    """
    return CollisionCauseFixture(
        fixture_id="negative_control_guard_flap_01",
        scenario=already_unavoidable_scenario(),
        faults=(
            InjectedFault(
                fault_type="guard_omission",
                activation_window=(-1, -1),
                repair_scenario=None,
                gates_applied_command=False,
            ),
        ),
    )


# Ordered registry of the 14 frozen-manifest fixture builders. Keyed by
# ``fixture_id`` so the runner can map manifest entries to builders without ever
# reading the manifest's ground-truth ``cause_class``.
COLLISION_CAUSE_FIXTURE_BUILDERS: dict[str, Callable[[], CollisionCauseFixture]] = {
    "obs_omission_01": obs_omission_01_fixture,
    "obs_delay_01": obs_delay_01_fixture,
    "prediction_miss_01": prediction_miss_01_fixture,
    "candidate_omission_01": candidate_omission_01_fixture,
    "bad_selection_01": bad_selection_01_fixture,
    "guard_omission_01": guard_omission_01_fixture,
    "infeasible_command_01": infeasible_command_01_fixture,
    "route_trap_01": route_trap_01_fixture,
    "already_unavoidable_01": already_unavoidable_01_fixture,
    "metric_artifact_01": metric_artifact_01_fixture,
    "ambiguous_pred_guard_01": ambiguous_pred_guard_01_fixture,
    "ambiguous_route_selection_01": ambiguous_route_selection_01_fixture,
    "negative_control_jitter_01": negative_control_jitter_01_fixture,
    "negative_control_guard_flap_01": negative_control_guard_flap_01_fixture,
}


def build_collision_cause_fixtures() -> list[CollisionCauseFixture]:
    """Build all 14 frozen-manifest fault-injection fixtures in manifest order.

    Returns:
        The 14 :class:`CollisionCauseFixture` objects, one per manifest entry.
    """
    return [builder() for builder in COLLISION_CAUSE_FIXTURE_BUILDERS.values()]
