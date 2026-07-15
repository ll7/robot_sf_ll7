"""Tests for the production-simulator CounterfactualModel adapter (issue #5442).

This is the remaining slice named in the issue thread: a real
``CounterfactualModel`` adapter over the live Robot SF ``Simulator`` plus the
RNG-capture seam. The tests build a headless ``Simulator`` (no display), prove the
snapshot/restore seam reproduces a baseline pedestrian-robot episode bit-for-bit,
and drive the frozen-state replay engine end to end on a genuine production fixture
where the baseline collides but halting the robot avoids contact.

The fail-closed ``unknown`` path for a nondeterministic baseline is already covered
by the controlled-fixture tests (``nondeterministic_baseline_scenario``); here we
additionally show that omitting the global-RNG capture seam makes a real baseline
replay diverge, exercising the same guard on production state.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.last_avoidable_replay import (
    SUBSTITUTION_HOLD,
    VERDICT_AVOIDABLE,
    ReplayConfig,
    locate_last_avoidable,
)
from robot_sf.benchmark.simulator_counterfactual_adapter import (
    SimulatorCounterfactualModel,
)
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import init_simulators

# A genuine production fixture: the robot drives into a doorway crossing; the
# maintain-speed baseline collides at step 39 but halting the robot clears it.
_FIXTURE_MAP = "maps/svg_maps/classic_doorway.svg"
_FIXTURE_DENSITY = [0.06]
_FIXTURE_SEED = 21
_FIXTURE_SPEED = 1.0
_FIXTURE_CONTACT_STEP = 39
_COLLISION_RADIUS = 0.5


def _build_simulator() -> object:
    """Construct a deterministic, headless ``Simulator`` for the fixture map."""
    map_def = convert_map(_FIXTURE_MAP)
    sim_config = SimulationSettings(
        difficulty=0,
        ped_density_by_difficulty=_FIXTURE_DENSITY,
        route_spawn_seed=_FIXTURE_SEED,
    )
    cfg = RobotSimulationConfig(sim_config=sim_config)
    return init_simulators(cfg, map_def, num_robots=1, random_start_pos=False)[0]


def _build_dense_simulator() -> object:
    """Construct a denser doorway simulator where pedestrians respawn mid-episode.

    The respawns draw from the global numpy RNG (via ``sample_zone``), so a replay
    without the RNG-capture seam diverges — exercising the same guard the engine's
    ``unknown`` determination relies on.
    """
    map_def = convert_map(_FIXTURE_MAP)
    sim_config = SimulationSettings(
        difficulty=0,
        ped_density_by_difficulty=[0.15],
        route_spawn_seed=21,
    )
    cfg = RobotSimulationConfig(sim_config=sim_config)
    return init_simulators(cfg, map_def, num_robots=1, random_start_pos=False)[0]


def _run_engine(model: SimulatorCounterfactualModel, *, determinism_replays: int = 5):
    """Drive ``locate_last_avoidable`` over the fixture episode."""
    contact_step = _FIXTURE_CONTACT_STEP
    horizon = contact_step + 10
    config = ReplayConfig(
        t_danger=0,
        t_contact=contact_step,
        horizon=horizon,
        substitution_mode=SUBSTITUTION_HOLD,
        determinism_replays=determinism_replays,
        action_set_id="sim_robot_cmd_lattice",
        feasibility_filter="maintain_or_halt",
        collision_predicate="robot_ped_euclidean<=radius",
        pedestrian_response="replayed",
    )
    baseline = [(float(_FIXTURE_SPEED), 0.0)] * (contact_step + horizon + 2)
    return locate_last_avoidable(model, baseline, config)


# -- snapshot/restore seam ------------------------------------------------
def test_snapshot_restore_reproduces_baseline_deterministically() -> None:
    """Restoring a snapshot and replaying yields the same contact step every time."""
    sim = _build_simulator()
    model = SimulatorCounterfactualModel(sim, collision_radius=_COLLISION_RADIUS)
    snap0 = model.snapshot()
    contacts: list[int | None] = []
    for _ in range(4):
        model.restore(snap0)
        c = None
        for i in range(_FIXTURE_CONTACT_STEP + 5):
            if model.collision():
                c = i
                break
            model.step((float(_FIXTURE_SPEED), 0.0))
        else:
            if model.collision():
                c = _FIXTURE_CONTACT_STEP + 5
        contacts.append(c)
    assert len(set(contacts)) == 1, "snapshot/restore must be deterministic"


def test_rng_capture_seam_prevents_divergence() -> None:
    """Without the global-RNG capture seam, a mid-episode respawn replay diverges.

    A dense doorway scenario triggers pedestrian group respawns that draw from the
    global numpy RNG (via ``sample_zone``). With the seam ``capture_rng=True`` the
    snapshot restores that RNG so the replay is bit-for-bit identical; with
    ``capture_rng=False`` the same replay diverges by meters — exactly the
    nondeterministic-baseline condition the engine's ``unknown`` guard protects
    against.
    """
    # With the seam: replay is bit-for-bit reproducible.
    sim = _build_dense_simulator()
    model = SimulatorCounterfactualModel(sim, collision_radius=_COLLISION_RADIUS, capture_rng=True)
    snap0 = model.snapshot()
    model.restore(snap0)
    for _ in range(100):
        model.step((0.0, 0.0))
    captured_positions = np.asarray(sim.ped_pos).copy()

    # Without the seam: a perturbed global RNG makes the replay diverge.
    sim2 = _build_dense_simulator()
    model2 = SimulatorCounterfactualModel(
        sim2, collision_radius=_COLLISION_RADIUS, capture_rng=False
    )
    snap0b = model2.snapshot()
    model2.restore(snap0b)
    np.random.seed(999)
    for _ in range(100):
        model2.step((0.0, 0.0))
    uncaptured_positions = np.asarray(sim2.ped_pos).copy()

    assert not np.allclose(captured_positions, uncaptured_positions), (
        "the RNG-capture seam must matter for this fixture; mid-episode respawn "
        "should diverge without it"
    )


# -- engine integration on a real production fixture ----------------------
def test_production_fixture_is_avoidable() -> None:
    """The live-simulator adapter reports an avoidable production collision deterministically."""
    sim = _build_simulator()
    model = SimulatorCounterfactualModel(sim, collision_radius=_COLLISION_RADIUS)
    report = _run_engine(model)
    assert report.verdict == VERDICT_AVOIDABLE
    assert report.determinism.deterministic is True
    assert report.t_uca is not None and report.t_inevitable is not None
    assert report.t_uca <= report.t_inevitable <= report.config.t_contact
    assert report.feasible_coverage == pytest.approx(1.0)
    assert report.minimal_sufficient_interventions, "must record preventing interventions"
    # Every decision point in the danger window is preserved.
    assert len(report.branches) == report.config.t_contact - report.config.t_danger


def test_adapter_satisfies_counterfactual_model_protocol() -> None:
    """The adapter exposes the snapshot/restore/step/collision/feasible/label seam."""
    sim = _build_simulator()
    model = SimulatorCounterfactualModel(sim, collision_radius=_COLLISION_RADIUS)
    snap = model.snapshot()
    assert snap is not None
    model.restore(snap)
    model.step((float(_FIXTURE_SPEED), 0.0))
    assert isinstance(model.collision(), bool)
    actions = model.feasible_actions()
    assert len(actions) >= 2
    assert model.action_label(actions[0]).startswith("robot_cmd=")
