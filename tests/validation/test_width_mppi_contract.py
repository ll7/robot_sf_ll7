"""Exercise the documented width route with its real, registry-pinned predictor."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.planner.predictive_mppi import PredictiveMPPIAdapter, build_predictive_mppi_config
from scripts.validation import run_empty_world_sweep as sweep

OBSERVATION = {
    "robot": {"position": [0.0, 0.0], "heading": [0.0], "speed": [0.0]},
    "goal": {"current": [5.0, 0.0], "next": [5.0, 0.0]},
    "pedestrians": {"positions": [[2.0, 1.0]], "velocities": [[0.0, 0.0]], "count": [1]},
}


def _derive(tmp_path: Path) -> Path:
    """Use the production input builder for all three widths."""
    cfg_path, scenarios = sweep.build_derived_inputs(
        "width",
        seeds=[1001],
        arms=["predictive_mppi"],
        scenarios_filter=None,
        workers=1,
        out_dir=tmp_path,
        step_trace=True,
    )
    assert len(scenarios) == 3
    return cfg_path


def test_selected_width_mppi_runs_real_forecast_without_fallback(tmp_path: Path) -> None:
    """The documented route constructs and scores its actual eight-output checkpoint."""
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

    campaign = load_campaign_config(_derive(tmp_path))
    profile = campaign.planners[0].algo_config_path
    settings = yaml.safe_load(Path(profile).read_text())
    planner = PredictiveMPPIAdapter(build_predictive_mppi_config(settings), allow_fallback=False)
    future, mask, steps = planner._predict_future(OBSERVATION)
    assert planner._predictor._model is not None
    assert not planner.foresight_degraded()
    assert future.shape == (16, 8, 2)
    assert np.count_nonzero(mask) == 1
    assert steps == 8
    assert planner.config.rollout_dt == pytest.approx(0.1)
    assert planner.config.socnav.predictive_rollout_dt == pytest.approx(0.1)
    assert np.isfinite(planner.plan(OBSERVATION)).all()


def test_width_preflight_rejects_historical_twelve_step_profile(
    tmp_path: Path, monkeypatch
) -> None:
    """An incompatible selected arm must fail before a campaign can start."""
    source = yaml.safe_load((sweep.REPO_ROOT / sweep.SUITES["width"]).read_text())
    arm = next(p for p in source["planners"] if p["key"] == "predictive_mppi")
    arm["algo_config"] = "configs/algos/predictive_mppi_camera_ready.yaml"
    incompatible = tmp_path / "incompatible.yaml"
    incompatible.write_text(yaml.safe_dump(source))
    monkeypatch.setitem(sweep.SUITES, "width", str(incompatible))
    with pytest.raises(ValueError, match=r"horizon_steps=12.*8 steps"):
        _derive(tmp_path / "derived")
