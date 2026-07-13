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
    from collections.abc import Sequence

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
        """Return whether robot and pedestrian are within ``collision_radius``."""
        robot_pos = np.array([self.robot_x, 0.0])
        return bool(np.linalg.norm(robot_pos - self.ped_pos) <= self.scenario.collision_radius)

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
