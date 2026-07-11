"""Three-arm diagnostic comparison for issue #4013 learned prediction MPC.

Claim boundary: diagnostic-only interface comparison. No training, no campaign,
no real scenario simulation, no navigation-quality claim. This proves the
comparison framework runs all three arms and each emits bounded finite commands.

The three arms mirror the paired comparison runner
(``scripts/benchmark/run_issue_4013_model_based_comparison.py``):

1. ``learned_prediction_mpc`` — learned short-horizon pedestrian prediction + MPC
   (untrained smoke model; source labeled ``diagnostic_untrained_smoke``).
2. ``cv_prediction_mpc`` — constant-velocity pedestrian prediction + MPC
   (deterministic; source labeled ``constant_velocity``).
3. ``model_free_baseline`` — trivial inline goal-seeker with no pedestrian model.

Running the representative evaluation with real map scenarios and seeds
is the remaining campaign work that stays out of this PR.
"""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.planner.learned_prediction_mpc import build_learned_prediction_mpc_adapter
from robot_sf.planner.prediction_mpc import (
    PredictionMPCConfig,
    PredictionMPCPlannerAdapter,
)

_SMOKE_CFG = {
    "allow_untrained_smoke": True,
    "max_pedestrians": 2,
    "horizon_steps": 2,
    "hidden_dim": 8,
    "solver_max_iterations": 8,
}


def _obs(
    *,
    goal: tuple[float, float] = (3.0, 0.0),
    ped_positions: list[tuple[float, float]] | None = None,
    ped_velocities: list[tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Build minimal SocNav observation for three-arm comparison tests."""
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray([0.0, 0.0], dtype=float),
            "heading": np.asarray([0.0], dtype=float),
            "speed": np.asarray([0.0], dtype=float),
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


class _TrivialGoalSeeker:
    """Model-free baseline: emit max_speed straight ahead without any pedestrian model."""

    max_linear_speed: float = 0.9
    max_angular_speed: float = 1.1

    def plan(self, observation: dict) -> tuple[float, float]:
        del observation
        return self.max_linear_speed, 0.0


def test_learned_and_cv_planners_emit_bounded_commands_on_same_observations() -> None:
    """Learned MPC and CV MPC both emit bounded commands on identical observations.

    Proves the two model-based arms share the same MPC interface and neither
    raises on the same synthetic observation payload.
    Claim boundary: diagnostic-only interface proof; no navigation quality claim.
    """
    pytest.importorskip("torch")
    learned = build_learned_prediction_mpc_adapter(_SMOKE_CFG)
    cv = PredictionMPCPlannerAdapter(PredictionMPCConfig(horizon_steps=2, solver_max_iterations=8))

    for obs in [
        _obs(goal=(2.0, 0.0)),
        _obs(goal=(2.0, 0.0), ped_positions=[(0.8, 0.0)], ped_velocities=[(0.0, 0.0)]),
    ]:
        l_lin, l_ang = learned.plan(obs)
        cv_lin, cv_ang = cv.plan(obs)

        for linear, angular, name in [
            (l_lin, l_ang, "learned"),
            (cv_lin, cv_ang, "cv"),
        ]:
            assert np.isfinite(linear), f"{name}: non-finite linear"
            assert np.isfinite(angular), f"{name}: non-finite angular"
            assert 0.0 <= linear <= cv.config.max_linear_speed, f"{name}: linear out of bounds"
            assert abs(angular) <= cv.config.max_angular_speed, f"{name}: angular out of bounds"


def test_learned_and_cv_predictors_report_distinct_sources_on_same_observation() -> None:
    """Learned and CV backends label predicted futures with distinct source identifiers.

    Proves the two model-based arms use different prediction backends that can
    be distinguished at runtime via the source tag on PredictedPedestrianFutures.
    """
    pytest.importorskip("torch")
    learned = build_learned_prediction_mpc_adapter(_SMOKE_CFG)
    cv = PredictionMPCPlannerAdapter(PredictionMPCConfig(horizon_steps=2))
    obs = _obs(goal=(2.0, 0.0), ped_positions=[(0.8, 0.0)], ped_velocities=[(0.0, 0.0)])

    learned_futures = learned._future_predictor.predict(obs, horizon_steps=2, dt=0.2)
    cv_futures = cv._future_predictor.predict(obs, horizon_steps=2, dt=0.2)

    assert learned_futures.source == "diagnostic_untrained_smoke"
    assert cv_futures.source == "constant_velocity"
    assert learned_futures.source != cv_futures.source
    assert np.all(np.isfinite(learned_futures.positions_world))
    assert np.all(np.isfinite(cv_futures.positions_world))


def test_three_arm_diagnostic_comparison_all_emit_bounded_commands() -> None:
    """All three comparison arms emit bounded finite commands on the same observations.

    Claim boundary: diagnostic-only proof that the comparison framework supports
    all three arms simultaneously. No navigation quality, training, campaign, or
    real scenario.

    This test mirrors the intent of the paired campaign runner
    (``scripts/benchmark/run_issue_4013_model_based_comparison.py``) at the
    minimum viable evidence tier: it proves the interface is plumbed correctly
    without running a full map-runner scenario.
    """
    pytest.importorskip("torch")

    learned = build_learned_prediction_mpc_adapter(_SMOKE_CFG)
    cv = PredictionMPCPlannerAdapter(PredictionMPCConfig(horizon_steps=2, solver_max_iterations=8))
    model_free = _TrivialGoalSeeker()

    observations = [
        _obs(goal=(3.0, 0.0)),
        _obs(goal=(3.0, 0.0), ped_positions=[(0.8, 0.0)], ped_velocities=[(0.0, 0.0)]),
        _obs(goal=(3.0, 0.0), ped_positions=[(1.5, 0.5)], ped_velocities=[(0.5, 0.0)]),
    ]

    results: dict[str, list[tuple[float, float]]] = {
        "learned_prediction_mpc": [],
        "cv_prediction_mpc": [],
        "model_free_baseline": [],
    }
    for obs in observations:
        results["learned_prediction_mpc"].append(learned.plan(obs))
        results["cv_prediction_mpc"].append(cv.plan(obs))
        results["model_free_baseline"].append(model_free.plan(obs))

    max_lin = cv.config.max_linear_speed
    max_ang = cv.config.max_angular_speed
    for arm, commands in results.items():
        for linear, angular in commands:
            assert np.isfinite(linear), f"{arm}: non-finite linear"
            assert np.isfinite(angular), f"{arm}: non-finite angular"
            assert 0.0 <= linear <= max_lin, f"{arm}: linear {linear!r} out of [0, {max_lin}]"
            assert abs(angular) <= max_ang, f"{arm}: angular {angular!r} out of bounds"

    assert all(results[arm] for arm in results), "each arm must produce at least one action"
