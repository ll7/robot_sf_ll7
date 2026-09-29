"""Release windows distinguish control cadence from checkpoint forecast cadence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.planner.predictive_mppi import PredictiveMPPIAdapter, build_predictive_mppi_config
from robot_sf.planner.socnav import PredictionPlannerAdapter

ROOT = Path(__file__).parents[2] / "configs/algos"


@pytest.mark.parametrize(
    ("historical", "release", "nested", "dt_key", "steps_key"),
    [
        (
            "risk_dwa_camera_ready_goal_v2.yaml",
            "risk_dwa_release_v0_0_8.yaml",
            None,
            "rollout_dt",
            "rollout_steps",
        ),
        (
            "guarded_ppo_camera_ready_cpu_goal_v2.yaml",
            "guarded_ppo_release_v0_0_8.yaml",
            None,
            "guard_rollout_dt",
            "guard_rollout_steps",
        ),
        (
            "guarded_ppo_camera_ready_cpu_goal_v2.yaml",
            "guarded_ppo_release_v0_0_8.yaml",
            "fallback_risk_dwa",
            "rollout_dt",
            "rollout_steps",
        ),
    ],
)
def test_release_step_count_retains_historical_horizon_seconds(
    historical: str, release: str, nested: str | None, dt_key: str, steps_key: str
) -> None:
    old = yaml.safe_load((ROOT / historical).read_text(encoding="utf-8"))
    new = yaml.safe_load((ROOT / release).read_text(encoding="utf-8"))
    if nested:
        old, new = old[nested], new[nested]
        assert new["dynamic_window_version"] == "drive_limited_v2"
    assert new[dt_key] == pytest.approx(0.1)
    assert new[dt_key] * new[steps_key] == pytest.approx(old[dt_key] * old[steps_key])


@pytest.fixture(params=["prediction_planner", "predictive_mppi"])
def checkpoint_window(request):
    """Load the actual release model without fallback, then obtain its capped window."""
    arm = request.param
    payload = yaml.safe_load((ROOT / f"{arm}_release_v0_0_8.yaml").read_text())
    config = build_predictive_mppi_config(payload)
    observation = {
        "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0]},
        "goal": {"current": [5.0, 0.0], "next": [5.0, 0.0]},
        "pedestrians": {"positions": [[0.5, 12.0]], "velocities": [[0.0, -12.0]], "count": [1]},
    }
    if arm == "predictive_mppi":
        planner = PredictiveMPPIAdapter(config, allow_fallback=False)
        future, mask, steps = planner._predict_future(observation)
        predictor = planner._predictor
        dt = planner.config.rollout_dt
    else:
        planner = PredictionPlannerAdapter(config.socnav, allow_fallback=False)
        state, mask, *_ = planner._build_model_input(observation)
        future = planner._predict_trajectories(state, mask)
        steps = planner._effective_rollout_steps(future_peds=future, mask=mask)
        predictor = planner
        dt = planner.config.predictive_rollout_dt
    assert predictor._model is not None  # Never accept the constant-velocity fallback.
    assert future.shape[1] == 8  # Both registry-pinned decoder artifacts have eight outputs.
    assert int(np.count_nonzero(mask)) == 1
    return arm, planner, predictor, future, mask, steps, dt, observation


def test_release_checkpoint_effective_window_is_one_point_six_seconds(checkpoint_window):
    """The real eight-output checkpoint and robot rollout cover the historical 1.6 s."""
    arm, planner, predictor, future, _mask, steps, dt, _obs = checkpoint_window
    assert steps * dt == pytest.approx(1.6)
    assert future.shape[1] * predictor.config.predictive_rollout_dt == pytest.approx(1.6)
    if arm == "predictive_mppi":
        assert planner.config.rollout_dt == predictor.config.predictive_rollout_dt
    trajectory = predictor._rollout_robot(v=0.5, w=0.0, dt=dt, steps=steps)
    assert trajectory[-1, 0] == pytest.approx(0.8)  # 0.5 m/s for 1.6 s.


def test_release_checkpoint_window_scores_crossing_at_one_second(checkpoint_window):
    """Score a hand crossing on the real model's effective grid; no learned-path claim."""
    arm, planner, predictor, future, mask, steps, dt, observation = checkpoint_window
    # Pedestrian crosses x=0.5 at t=1.0 s; robot at 0.5 m/s reaches it then.
    # At t<=0.8 s the lateral distance is >=2.4 m: safe for 1.0/0.4 m bodies.
    times = np.arange(1, future.shape[1] + 1) * dt
    crossing = np.zeros_like(future)
    crossing[0, :, 0] = 0.5
    crossing[0, :, 1] = 12.0 * (1.0 - times)
    if arm == "prediction_planner":
        collision, _ = predictor._collision_cost(
            future_peds=crossing,
            mask=mask,
            v=0.5,
            w=0.0,
            steps=steps,
        )
        assert collision > 0.0, "1.0 s crossing must be inside the effective checkpoint window"
        short_collision, _ = predictor._collision_cost(
            future_peds=crossing,
            mask=mask,
            v=0.5,
            w=0.0,
            steps=round(0.8 / dt),
        )
        assert short_collision == 0.0
    else:

        def score(n):
            return planner._sequence_rollout(
                np.tile([0.5, 0.0], (n, 1)),
                robot_pos=np.zeros(2),
                heading=0.0,
                goal=np.asarray([5.0, 0.0]),
                future=crossing,
                mask=mask,
                observation=observation,
                anchor_action=(0.5, 0.0),
            )

        assert score(steps) >= planner.config.invalid_sequence_cost, (
            "1.0 s crossing must be inside the effective checkpoint window"
        )
        assert score(round(0.8 / dt)) < planner.config.invalid_sequence_cost
