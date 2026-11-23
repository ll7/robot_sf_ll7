"""Keep the historical twelve-step request fail-closed against the real checkpoint."""

from pathlib import Path

import pytest
import yaml

from robot_sf.planner.predictive_mppi import PredictiveMPPIAdapter, build_predictive_mppi_config


def test_historical_mppi_guard_rejects_twelve_real_forecast_steps() -> None:
    """A longer robot rollout must not silently cap or extrapolate eight model outputs."""
    root = Path(__file__).resolve().parents[2]
    payload = yaml.safe_load((root / "configs/algos/predictive_mppi_camera_ready.yaml").read_text())
    planner = PredictiveMPPIAdapter(build_predictive_mppi_config(payload), allow_fallback=False)
    with pytest.raises(ValueError, match=r"horizon_steps=12.*8 steps"):
        planner._predict_future({})
    assert planner._predictor._model is not None
    assert not planner.foresight_degraded()
