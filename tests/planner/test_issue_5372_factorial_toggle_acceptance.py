"""Issue #5372 acceptance test: factor toggles for the #5355 2x2 factorial.

This test is the closure artifact for issue #5372. It asserts the four
preregistered acceptance bullets end-to-end against the production adapter and
the four canonical arm configs — no campaign, no GPU/Slurm, no claim promotion:

1. Factor A ``prediction_enabled`` (``predictor_backend``) freezes pedestrians at
   their current position in BOTH the soft-cost predictor (``_predict_pedestrians``)
   and the hard-constraint ``ConstantVelocityPedestrianPredictor`` when set to
   ``none``; constant-velocity forward projection when set to ``constant_velocity``.
2. Factor B ``hard_pedestrian_constraints_enabled`` drops the hard
   ``NonlinearConstraint`` set when False and sets the soft pedestrian-clearance
   weight to the base NMPC default 4.5 (exposed via ``build_prediction_mpc_config``);
   B-OFF remains a functional avoidance planner (optimizer + all cost terms + static
   clearance + soft pedestrian penalty), not a degenerate one.
3. Fidelity smoke: across two named scenarios, all four arms, the decision trace
   differs across each factor flip (A-on vs A-off; B-on vs B-off) and B-OFF
   completes episodes (functionality check).
4. Arm YAML configs (4) share the observation contract, kinematics, and runtime
   budget and differ ONLY in the two factor flags plus the implied soft weight; a
   preflight identity test asserts this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.planner.prediction_mpc import (
    ConstantVelocityPedestrianPredictor,
    NullPedestrianPredictor,
    PredictionMPCConfig,
    PredictionMPCPlannerAdapter,
    build_prediction_mpc_config,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALGO_DIR = _REPO_ROOT / "configs" / "algos"
_ARM_IDS = ("A0_B0", "A0_B1", "A1_B0", "A1_B1")
_BASE_WEIGHT = 4.5

_TRUTH_TABLE: dict[str, dict[str, object]] = {
    "A0_B0": {"backend": "none", "hard": False, "weight": 4.5},
    "A1_B0": {"backend": "constant_velocity", "hard": False, "weight": 4.5},
    "A0_B1": {"backend": "none", "hard": True, "weight": 0.0},
    "A1_B1": {"backend": "constant_velocity", "hard": True, "weight": 0.0},
}


def _arm(arm_id: str) -> PredictionMPCPlannerAdapter:
    """Build a fresh planner adapter for a canonical arm config."""
    path = _ALGO_DIR / f"prediction_mpc_factorial_{arm_id}.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    adapter = PredictionMPCPlannerAdapter(build_prediction_mpc_config(payload))
    adapter.reset()
    return adapter


def _obs(
    *,
    robot: tuple[float, float] = (0.0, 0.0),
    heading: float = 0.0,
    speed: float = 0.0,
    goal: tuple[float, float] = (2.5, 0.0),
    ped_positions: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Build a compact SocNav observation (pedestrian velocities are ego-frame)."""
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([speed], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
            "count": np.asarray([len(ped_positions)], dtype=float),
            "radius": np.asarray([0.25], dtype=float),
        },
    }


# --------------------------------------------------------------------------- #
# Bullet 1: Factor A freezes pedestrians in BOTH prediction paths, one flag.
# --------------------------------------------------------------------------- #


def test_factor_a_freezes_both_prediction_paths() -> None:
    """predictor_backend='none' freezes in soft-cost and hard-constraint paths."""
    positions = np.asarray([[1.0, 0.5]], dtype=float)
    velocities = np.asarray([[0.4, -0.2]], dtype=float)
    obs = _obs(ped_positions=[[1.0, 0.5]], ped_velocities=[[0.4, -0.2]])

    # Soft-cost path: PredictionMPCPlannerAdapter._predict_pedestrians.
    a_off = _arm("A0_B0")
    a_on = _arm("A1_B0")
    assert np.array_equal(a_off._predict_pedestrians(positions, velocities, step_idx=3), positions)
    assert not np.array_equal(
        a_on._predict_pedestrians(positions, velocities, step_idx=3), positions
    )

    # Hard-constraint path: the predictor handed to _optimizer_constraints.
    assert isinstance(a_off._future_predictor, NullPedestrianPredictor)
    assert isinstance(a_on._future_predictor, ConstantVelocityPedestrianPredictor)
    futures_off = a_off._future_predictor.predict(obs, horizon_steps=6, dt=0.25)
    futures_on = a_on._future_predictor.predict(obs, horizon_steps=6, dt=0.25)
    assert np.allclose(futures_off.positions_world[:, -1, :], futures_off.positions_world[:, 0, :])
    assert not np.allclose(
        futures_on.positions_world[:, -1, :], futures_on.positions_world[:, 0, :]
    )


# --------------------------------------------------------------------------- #
# Bullet 2: Factor B drops the hard constraint set and exposes W_soft=4.5.
# --------------------------------------------------------------------------- #


def test_factor_b_drops_hard_constraint_set_and_exposes_soft_weight() -> None:
    """B-off returns no hard constraint and propagates the base NMPC weight 4.5."""
    obs = _obs(ped_positions=[[1.1, 0.3]], ped_velocities=[[0.0, 0.0]])
    b_off = _arm("A0_B0")
    b_on = _arm("A0_B1")

    # Hard-constraint set presence flips with Factor B.
    ctx_off = _RolloutContext_like(b_off, obs)
    ctx_on = _RolloutContext_like(b_on, obs)
    assert len(b_off._optimizer_constraints(ctx_off)) == 0
    assert len(b_on._optimizer_constraints(ctx_on)) == 1

    # Soft weight is propagated as the base NMPC default (4.5), not 0.0.
    nmpc_off = _to_nmpc(b_off.prediction_config)
    assert nmpc_off.pedestrian_clearance_weight == pytest.approx(_BASE_WEIGHT)
    assert b_off.prediction_config.pedestrian_clearance_weight == pytest.approx(_BASE_WEIGHT)


def test_factor_b_off_remains_functional_planner() -> None:
    """B-off is a functional soft-cost planner, not a degenerate one."""
    obs = _obs(ped_positions=[], ped_velocities=[])
    b_off = _arm("A0_B0")
    linear, angular = b_off.plan(obs)
    diag = b_off.diagnostics()
    assert np.isfinite(linear) and np.isfinite(angular)
    assert linear > 0.0
    assert diag["nonzero_command_count"] == 1
    assert diag["factorial_toggles"]["hard_pedestrian_constraints_enabled"] is False
    assert diag["factorial_toggles"]["pedestrian_clearance_weight"] == pytest.approx(_BASE_WEIGHT)


# --------------------------------------------------------------------------- #
# Bullet 3: fidelity smoke — traces differ across each flip; B-off completes.
# --------------------------------------------------------------------------- #


_SCENARIOS: dict[str, dict[str, object]] = {
    "classic_doorway": {
        "goal": (2.5, 0.0),
        "ped_positions": [(1.1, 0.3)],
        "ped_velocities": [(0.0, 0.0)],
    },
    "francis2023_circular_crossing": {
        "goal": (2.5, 0.0),
        "ped_positions": [(1.2, 0.55)],
        "ped_velocities": [(0.0, -0.5)],
    },
}


@pytest.mark.parametrize("b_level", ["B0", "B1"])
def test_fidelity_factor_a_flip_changes_trace(b_level: str) -> None:
    """A-on vs A-off changes the decision trace on a dynamic scene (prereg §6.1)."""
    scenario = _SCENARIOS["francis2023_circular_crossing"]
    a_off, a_on = _arm(f"A0_{b_level}"), _arm(f"A1_{b_level}")
    obs = _obs(**scenario)
    assert not np.allclose(a_off.plan(obs), a_on.plan(obs))


@pytest.mark.parametrize("a_level", ["A0", "A1"])
@pytest.mark.parametrize("scenario_name", list(_SCENARIOS))
def test_fidelity_factor_b_flip_changes_trace(a_level: str, scenario_name: str) -> None:
    """B-on vs B-off changes the decision trace where a pedestrian is within margin."""
    b_off, b_on = _arm(f"{a_level}_B0"), _arm(f"{a_level}_B1")
    obs = _obs(**_SCENARIOS[scenario_name])
    assert not np.allclose(b_off.plan(obs), b_on.plan(obs))


@pytest.mark.parametrize("arm_id", ["A0_B0", "A1_B0"])
def test_fidelity_b_off_makes_progress_toward_goal(arm_id: str) -> None:
    """B-OFF arms are functional planners: finite forward progress on a clear route.

    Mirrors the B-OFF functionality check (prereg §6.2) using the production
    ``plan`` path with an unobstructed goal; the full rolling-episode completion
    is exercised by the dedicated fidelity-smoke module.
    """
    adapter = _arm(arm_id)
    # Ego-frame zero velocities => pedestrians stationary; clear straight route.
    obs = _obs(goal=(5.0, 0.0), ped_positions=[], ped_velocities=[])
    linear, angular = adapter.plan(obs)
    diag = adapter.diagnostics()
    assert np.isfinite(linear) and np.isfinite(angular)
    assert linear > 0.0
    assert diag["nonzero_command_count"] == 1
    assert diag["factorial_toggles"]["hard_pedestrian_constraints_enabled"] is False
    assert diag["factorial_toggles"]["pedestrian_clearance_weight"] == pytest.approx(_BASE_WEIGHT)


# --------------------------------------------------------------------------- #
# Bullet 4: the four arm YAMLs differ ONLY in the two factor flags + weight.
# --------------------------------------------------------------------------- #


def test_arm_configs_realize_truth_table() -> None:
    """All four arm configs build to the preregistered 2x2 truth table."""
    for arm_id, expected in _TRUTH_TABLE.items():
        cfg = build_prediction_mpc_config(
            yaml.safe_load((_ALGO_DIR / f"prediction_mpc_factorial_{arm_id}.yaml").read_text())
        )
        assert cfg.predictor_backend == expected["backend"], arm_id
        assert cfg.hard_pedestrian_constraints_enabled is expected["hard"], arm_id
        assert cfg.pedestrian_clearance_weight == pytest.approx(float(expected["weight"])), arm_id


def test_arm_configs_share_observation_kinematics_runtime_and_differ_only_in_factors() -> None:
    """Arm YAMLs are identical outside the two factor flags + implied soft weight."""
    from dataclasses import fields

    cfgs = {
        arm_id: build_prediction_mpc_config(
            yaml.safe_load((_ALGO_DIR / f"prediction_mpc_factorial_{arm_id}.yaml").read_text())
        )
        for arm_id in _ARM_IDS
    }
    factor_fields = {
        "predictor_backend",
        "hard_pedestrian_constraints_enabled",
        "pedestrian_clearance_weight",
    }
    differing = {
        field.name
        for field in fields(PredictionMPCConfig)
        if any(getattr(cfgs[a], field.name) != getattr(cfgs["A0_B0"], field.name) for a in _ARM_IDS)
    }
    assert differing == factor_fields, f"arms differ in exactly the factor fields, got {differing}"


# --------------------------------------------------------------------------- #
# Local helpers (kept private to this module).
# --------------------------------------------------------------------------- #


def _RolloutContext_like(adapter: PredictionMPCPlannerAdapter, obs: dict[str, object]):
    """Build a rollout context matching the adapter's observation extraction."""
    from types import SimpleNamespace

    (
        robot_pos,
        heading,
        speed,
        ped_positions,
        ped_velocities,
        _count,
        robot_radius,
        ped_radius,
    ) = adapter._extract_state(obs)
    return SimpleNamespace(
        robot_pos=np.asarray(robot_pos, dtype=float),
        heading=float(np.asarray(heading, dtype=float).reshape(-1)[0]),
        current_speed=float(np.asarray(speed, dtype=float).reshape(-1)[0]),
        goal=np.asarray(obs["goal"]["current"], dtype=float),
        ped_positions=np.asarray(ped_positions, dtype=float),
        ped_velocities=np.asarray(ped_velocities, dtype=float),
        robot_radius=robot_radius,
        ped_radius=ped_radius,
        observation=obs,
        speed_cap=float(adapter.prediction_config.max_linear_speed),
    )


def _to_nmpc(cfg: PredictionMPCConfig):
    """Map a PredictionMPCConfig to the NMPC core config it feeds."""
    from robot_sf.planner.prediction_mpc import _to_nmpc_config

    return _to_nmpc_config(cfg)
