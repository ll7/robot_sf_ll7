"""Replay/no-op fidelity for the first nominated #8568 case.

Nominated case: ``classic_head_on_corridor_medium``, seed 115, first canonical
planner arm in manifest order. The September payload retains no source-complete
planner continuation (all 14 arms disable both trace recorders), so the native
seam drives the simulator with a deterministic constant action — the ruling's
post-hoc new-diagnostic-trace path, not a planner shootout.

The test proves the distributable capture the ruling requires: run natively to
a boundary, capture a typed snapshot, restore it into a fresh model, no-op
continue both, and compare every continuation step with the canonical
``compare_continuation_traces`` comparator. A divergence names its first
channel instead of passing silently.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from robot_sf.benchmark.simulator_counterfactual_adapter import SimulatorCounterfactualModel
from robot_sf.benchmark.typed_snapshot import (
    NoOpStep,
    SnapshotBoundary,
    SnapshotCompatibility,
    capture_typed_snapshot,
    compare_continuation_traces,
    restore_typed_snapshot,
)
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import init_simulators

_SCENARIO_ID = "classic_head_on_corridor_medium"
_SEED = 115
_PREFIX_STEPS = 6
_CONTINUATION_STEPS = 6
_FIXTURE_MAP = Path(__file__).resolve().parents[2] / "maps/svg_maps/classic_head_on_corridor.svg"


def _build_model() -> SimulatorCounterfactualModel:
    """Build the nominated native corridor simulator at the ruling seed."""
    np.random.seed(_SEED)
    map_def = convert_map(str(_FIXTURE_MAP))
    sim_config = SimulationSettings(
        difficulty=0,
        ped_density_by_difficulty=[0.06],
        route_spawn_seed=_SEED,
    )
    config = RobotSimulationConfig(sim_config=sim_config)
    simulator = init_simulators(config, map_def, num_robots=1, random_start_pos=False)[0]
    return SimulatorCounterfactualModel(simulator, collision_radius=0.5, capture_rng=True)


def _compatibility(model: SimulatorCounterfactualModel) -> SnapshotCompatibility:
    """Return deterministic identity metadata for the nominated case."""
    return SnapshotCompatibility(
        map_sha256=hashlib.sha256(_FIXTURE_MAP.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(f"{_SCENARIO_ID}:{_SEED}".encode()).hexdigest(),
        code_revision=hashlib.sha256(b"issue-8568-fidelity").hexdigest(),
        dt_s=float(model.sim.config.time_per_step_in_secs),
        planner_id="issue-8568.constant-action",
    )


def _observe(model: SimulatorCounterfactualModel, step: int) -> NoOpStep:
    """Record the distributable per-step state the ruling requires."""
    return NoOpStep(
        step=step,
        state={
            "pysf_state": np.asarray(model.sim.pysf_state.pysf_states()).copy(),
            "robot_pose": tuple(model.sim.robots[0].pose[0]),
            "robot_velocity": tuple(model.sim.robots[0].state.velocity),
            "waypoint_id": int(model.sim.robot_navs[0].waypoint_id),
        },
        applied_action=(0.2, 0.0),
    )


def _continue(model: SimulatorCounterfactualModel, steps: int, start: int) -> list[NoOpStep]:
    """No-op continue with the deterministic constant action and record traces."""
    trace = [_observe(model, start)]
    for offset in range(steps):
        model.step((0.2, 0.0))
        trace.append(_observe(model, start + offset + 1))
    return trace


def test_nominated_case_snapshot_restore_preserves_noop_continuation() -> None:
    """Snapshot/restore at the ruling boundary reproduces the native suffix exactly."""
    model = _build_model()
    compatibility = _compatibility(model)
    _continue(model, _PREFIX_STEPS, 0)
    boundary = SnapshotBoundary(
        step_index=_PREFIX_STEPS,
        absolute_time_s=float(model.sim.config.time_per_step_in_secs * _PREFIX_STEPS),
        remaining_budget_steps=2000 - _PREFIX_STEPS,
    )
    typed = capture_typed_snapshot(model, compatibility, boundary=boundary)
    assert typed.boundary.remaining_budget_steps == 2000 - _PREFIX_STEPS

    baseline = _continue(model, _CONTINUATION_STEPS, _PREFIX_STEPS)

    fresh = _build_model()
    _continue(fresh, _PREFIX_STEPS, 0)
    restore_typed_snapshot(fresh, typed, compatibility)
    replay = _continue(fresh, _CONTINUATION_STEPS, _PREFIX_STEPS)

    comparison = compare_continuation_traces(baseline, replay)
    assert comparison.equivalent, (
        f"fidelity_divergence scenario={_SCENARIO_ID} seed={_SEED} "
        f"step={comparison.first_divergence_step} "
        f"field={comparison.first_divergence_field}"
    )
    assert comparison.compared_steps == _CONTINUATION_STEPS + 1
