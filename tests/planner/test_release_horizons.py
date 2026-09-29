"""The 0.0.8 control cadence retains every historical planning lookahead."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
            "prediction_planner_camera_ready.yaml",
            "prediction_planner_release_v0_0_8.yaml",
            None,
            "predictive_rollout_dt",
            "predictive_horizon_steps",
        ),
        (
            "predictive_mppi_camera_ready_goal_v2.yaml",
            "predictive_mppi_release_v0_0_8.yaml",
            None,
            "rollout_dt",
            "horizon_steps",
        ),
        (
            "predictive_mppi_camera_ready_goal_v2.yaml",
            "predictive_mppi_release_v0_0_8.yaml",
            None,
            "predictive_rollout_dt",
            "predictive_horizon_steps",
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


@pytest.mark.parametrize(
    ("historical", "release"),
    [
        ("prediction_planner_camera_ready.yaml", "prediction_planner_release_v0_0_8.yaml"),
        ("predictive_mppi_camera_ready_goal_v2.yaml", "predictive_mppi_release_v0_0_8.yaml"),
    ],
)
def test_release_predictor_boost_retains_extra_lookahead_seconds(
    historical: str, release: str
) -> None:
    old = yaml.safe_load((ROOT / historical).read_text(encoding="utf-8"))
    new = yaml.safe_load((ROOT / release).read_text(encoding="utf-8"))
    assert new["predictive_horizon_boost_steps"] * new["predictive_rollout_dt"] == pytest.approx(
        old["predictive_horizon_boost_steps"] * old["predictive_rollout_dt"]
    )
