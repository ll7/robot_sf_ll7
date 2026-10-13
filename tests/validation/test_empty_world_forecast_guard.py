"""Receipt preflight must reject unsupported real forecasts before collection."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.validation import run_empty_world_sweep as sweep


@pytest.fixture
def checkpoint(tmp_path):
    """A genuine eight-step checkpoint, loaded through the production loader."""
    from robot_sf.planner.predictive_model import (
        PredictiveModelConfig,
        PredictiveTrajectoryModel,
        save_predictive_checkpoint,
    )

    path = tmp_path / "predictive.pt"
    model = PredictiveTrajectoryModel(
        PredictiveModelConfig(hidden_dim=8, message_passing_steps=1, horizon_steps=8)
    )
    save_predictive_checkpoint(path, model=model, optimizer=None, epoch=0)
    return path


@pytest.fixture
def campaign_factory(tmp_path, checkpoint):
    cfg_path, scenarios = sweep.build_derived_inputs(
        "width",
        seeds=[1001],
        arms=["goal"],
        scenarios_filter=None,
        workers=1,
        out_dir=tmp_path / "derived",
        step_trace=False,
    )
    base = yaml.safe_load(cfg_path.read_text())

    def make(name, *, horizon=8, predictor_horizon=8, algo="predictive_mppi", missing=False):
        settings_path = tmp_path / f"{name}_algo.yaml"
        settings_path.write_text(
            yaml.safe_dump(
                {
                    "allow_testing_algorithms": True,
                    "predictive_checkpoint_path": str(
                        checkpoint if not missing else tmp_path / "missing.pt"
                    ),
                    "predictive_horizon_steps": predictor_horizon,
                    "predictive_rollout_dt": 0.1,
                    "predictive_adaptive_horizon_enabled": False,
                    "predictive_horizon_boost_steps": 0,
                    "horizon_steps": horizon,
                    "rollout_dt": 0.1,
                }
            )
        )
        payload = copy.deepcopy(base)
        payload["planners"].append(
            {
                "key": name,
                "algo": algo,
                "benchmark_profile": "experimental",
                "algo_config": str(settings_path),
            }
        )
        path = tmp_path / f"{name}_campaign.yaml"
        path.write_text(yaml.safe_dump(payload))
        return path, scenarios

    return make


@pytest.mark.parametrize("check_only", [False, True])
@pytest.mark.parametrize(("horizon", "predictor_horizon", "requested"), [(12, 8, 12), (8, 9, 9)])
def test_invalid_width_forecast_aborts_before_main_collection(
    tmp_path,
    monkeypatch,
    campaign_factory,
    check_only,
    horizon,
    predictor_horizon,
    requested,
):
    """A valid later suite requesting an unsupported forecast blocks all runs."""
    from robot_sf.benchmark import camera_ready_campaign

    main_cfg = campaign_factory("main_ok")
    width_cfg = campaign_factory("width_bad", horizon=horizon, predictor_horizon=predictor_horizon)
    collected = []
    monkeypatch.setattr(sweep, "verify_head", lambda _: "fixture-head")
    monkeypatch.setattr(
        sweep, "build_derived_inputs", lambda suite, **_: main_cfg if suite == "main" else width_cfg
    )
    monkeypatch.setattr(
        camera_ready_campaign,
        "run_campaign",
        lambda cfg, **_: (
            collected.append(cfg)
            or {
                "campaign_execution_status": "completed",
                "exit_code": 0,
                "unexpected_failed_runs": 0,
                "total_episodes": 0,
            }
        ),
    )
    argv = [
        "--head-sha",
        "HEAD",
        "--suite",
        "both",
        "--seeds",
        "1001",
        "--no-step-trace",
        "--output-dir",
        str(tmp_path / "out"),
    ]
    if check_only:
        argv.append("--check-only")
    with pytest.raises(
        ValueError, match=rf"forecast guard.*width_bad.*requested.*{requested}.*supported.*8"
    ):
        sweep.main(argv)
    assert collected == []


@pytest.mark.parametrize("algo", ["predictive_mppi", "prediction_planner"])
def test_predictor_request_nine_is_rejected_against_actual_eight(campaign_factory, algo):
    """Predictor requests are checked even when the MPPI optimizer asks for eight."""
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

    path, _ = campaign_factory("predictor_bad", predictor_horizon=9, algo=algo)
    with pytest.raises(
        ValueError, match=r"forecast guard.*predictor_bad.*requested.*9.*supported.*8"
    ):
        sweep.assert_forecast_support(load_campaign_config(path))


def test_supported_eight_and_nonpredictive_goal_need_no_extra_fields(campaign_factory):
    """Normal eight-step prediction loads, while a goal arm has no forecast settings."""
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

    path, _ = campaign_factory("supported")
    sweep.assert_forecast_support(load_campaign_config(path))


def test_missing_real_checkpoint_refuses_preflight(campaign_factory, monkeypatch, tmp_path):
    """An unavailable model cannot pass based on a declared horizon alone."""
    derived = campaign_factory("missing_model", missing=True)
    monkeypatch.setattr(sweep, "verify_head", lambda _: "fixture-head")
    monkeypatch.setattr(sweep, "build_derived_inputs", lambda suite, **_: derived)
    with pytest.raises(RuntimeError, match=r"forecast guard.*missing_model.*checkpoint"):
        sweep.main(
            [
                "--head-sha",
                "HEAD",
                "--suite",
                "width",
                "--seeds",
                "1001",
                "--check-only",
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_actual_output_shape_overrules_declared_checkpoint_horizon(campaign_factory, monkeypatch):
    """A model returning seven steps cannot claim eight through its config."""
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
    from robot_sf.planner.predictive_model import PredictiveTrajectoryModel

    original_forward = PredictiveTrajectoryModel.forward

    def shorter_forecast(model, state, mask):
        output = original_forward(model, state, mask)
        output["future_positions"] = output["future_positions"][:, :, :7]
        return output

    monkeypatch.setattr(PredictiveTrajectoryModel, "forward", shorter_forecast)
    path, _ = campaign_factory("actual_shorter")
    with pytest.raises(
        ValueError, match=r"forecast guard.*actual_shorter.*requested.*8.*supported.*7"
    ):
        sweep.assert_forecast_support(load_campaign_config(path))


def test_requested_adaptive_boost_is_checked(campaign_factory):
    """An enabled boost must be supported, even when the base horizon fits."""
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

    path, _ = campaign_factory("boost_bad", algo="prediction_planner")
    payload = yaml.safe_load(path.read_text())
    settings_path = Path(payload["planners"][-1]["algo_config"])
    settings = yaml.safe_load(settings_path.read_text())
    settings.update(predictive_adaptive_horizon_enabled=True, predictive_horizon_boost_steps=4)
    settings_path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match=r"forecast guard.*boost_bad.*requested.*12.*supported.*8"):
        sweep.assert_forecast_support(load_campaign_config(path))
