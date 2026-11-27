"""Regression coverage for the frozen development comparison."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark import timestep_sensitivity as study
from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config
from robot_sf.common.artifact_paths import get_repository_root
from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from robot_sf.benchmark.camera_ready._config_types import CampaignConfig


def test_expansion_preserves_the_release_roster_and_matched_development_grid(
    tmp_path: Path,
) -> None:
    """Changing a step size must preserve the scenario, seed and planner bindings."""
    study.main(["--output-dir", str(tmp_path)])
    manifest = json.loads((tmp_path / "timestep_manifest.json").read_text())
    root = get_repository_root()
    reference = yaml.safe_load(
        (
            root
            / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml"
        ).read_text()
    )
    expected_scenarios = {
        "classic_bottleneck_medium",
        "classic_doorway_medium",
        "classic_head_on_corridor_low",
        "classic_urban_crossing_medium",
        "francis2023_perpendicular_traffic",
    }
    assert len(manifest["campaigns"]) == 3
    for campaign, dt, steps in zip(
        manifest["campaigns"], (0.05, 0.1, 0.2), (1200, 600, 300), strict=True
    ):
        assert campaign["episode_count"] == 700
        assert campaign["analysis_only"]
        arms = campaign["arms"]
        assert {(arm["planner"], arm["scenario"]) for arm in arms} == {
            (planner["key"], scenario)
            for planner in reference["planners"]
            for scenario in expected_scenarios
        }
        assert len(arms) == 70
        for arm in arms:
            planner = next(p for p in reference["planners"] if p["key"] == arm["planner"])
            assert arm["algorithm"] == planner["algo"]
            assert arm["planner_config"] == planner.get("algo_config")
            assert arm["kinematics"] == "differential_drive"
            assert arm["seeds"] == list(range(1111, 1121))
            assert arm["simulation"] == {"interval_s": dt, "steps": steps, "horizon_s": 60.0}
            assert arm["controller_interval_s"] == arm["close_clearance_sample_interval_s"] == dt
            if arm["algorithm"] in {"prediction_planner", "predictive_mppi"}:
                assert arm["planner_timing"]["forecast"] == {
                    "interval_s": 0.1,
                    "steps": 8,
                    "horizon_s": 0.8,
                }
            if arm["algorithm"] == "socnav_sampling":
                assert arm["planner_timing"]["sampled_controls"] == {
                    "interval_s": dt,
                    "steps": {0.05: 40, 0.1: 20, 0.2: 10}[dt],
                    "horizon_s": 2.0,
                }


def test_witness_retains_native_steps_and_existing_runner_refusals(tmp_path: Path) -> None:
    """Make a native goal witness and retain a normal missing-planner error in the report."""
    cfg = load_campaign_config(get_repository_root() / study.CONFIG_PATTERN.format(ms=200))
    goal = next(planner for planner in cfg.planners if planner.key == "goal")
    unavailable = replace(goal, key="unavailable", algo_config_path=tmp_path / "missing.yaml")
    cfg = replace(cfg, planners=(goal, unavailable))
    results = study._run_witness(cfg, _load_campaign_scenarios(cfg), tmp_path)

    native, refused = results
    assert native["status"] == "qualified"
    assert native["execution_mode"] == "native"
    assert native["timestamps_s"] == [0.2, 0.4]
    assert native["seed"] == 1111
    assert refused["status"] == "refused"
    assert "Algorithm config file not found" in refused["reason"]


def test_expansion_identifies_an_omitted_campaign_step() -> None:
    """A campaign with no step size cannot report its timing contract."""
    cfg = load_campaign_config(get_repository_root() / study.CONFIG_PATTERN.format(ms=100))
    cfg = replace(cfg, dt=None)
    with pytest.raises(ValueError, match="requires an explicit campaign dt"):
        study.expand_timestep_campaign(cfg, _load_campaign_scenarios(cfg))


@pytest.mark.parametrize("mismatch", ["planner_dt", "planner_limit", "other_scenario_limit"])
def test_witness_refuses_mismatched_campaign_timing(tmp_path: Path, mismatch: str) -> None:
    """The short witness must reject a bad full-run cap even outside the witness scenario."""
    cfg = load_campaign_config(get_repository_root() / study.CONFIG_PATTERN.format(ms=200))
    goal = next(planner for planner in cfg.planners if planner.key == "goal")
    scenarios = _load_campaign_scenarios(cfg)
    if mismatch == "planner_dt":
        goal = replace(goal, dt_override=0.1)
    elif mismatch == "planner_limit":
        goal = replace(goal, horizon_override=150)
    else:
        other = next(s for s in scenarios if s["name"] != study.WITNESS_SCENARIO)
        other["simulation_config"]["max_episode_steps"] = 150
    results = study._run_witness(replace(cfg, planners=(goal,)), scenarios, tmp_path)
    assert results[0]["status"] == "timing_mismatch"
    assert "expected" in results[0]["reason"]
    assert not list(tmp_path.rglob("episodes.jsonl"))


def test_reports_include_step_count_timers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both outputs must carry the resolved timers on the grid where they advance."""

    def load_timer_planners(path: Path) -> CampaignConfig:
        cfg = load_campaign_config(path)
        return replace(
            cfg,
            planners=tuple(
                planner
                for planner in cfg.planners
                if planner.algo in {"orca", "hybrid_rule_local_planner"}
            ),
        )

    monkeypatch.setattr(study, "load_campaign_config", load_timer_planners)
    study.main(["--output-dir", str(tmp_path), "--witness"])
    campaigns = json.loads((tmp_path / "timestep_manifest.json").read_text())["campaigns"]
    for campaign, dt, durations in zip(
        campaigns,
        (0.05, 0.1, 0.2),
        ((0.15, 0.5, 0.3, 12.75), (0.3, 1.0, 0.6, 25.5), (0.6, 2.0, 1.2, 51.0)),
        strict=True,
    ):
        stall_s, persistence_s, oscillation_s, capacity_s = durations
        witnesses = {witness["planner"]: witness for witness in campaign["witness"]}
        assert len(campaign["arms"]) == 25
        for arm in campaign["arms"]:
            timers = arm["planner_timing"]["step_timers"]
            if arm["scenario"] == study.WITNESS_SCENARIO:
                assert witnesses[arm["planner"]]["planner_timing"]["step_timers"] == timers
            if arm["algorithm"] == "orca":
                assert timers["stall_before_commit"]["steps"] == 3
                assert timers["stall_before_commit"]["horizon_s"] == pytest.approx(stall_s)
                assert timers["commit_persistence"]["steps"] == 10
                assert timers["commit_persistence"]["horizon_s"] == persistence_s
                assert timers["stall_progress_sample"]["horizon_s"] == dt
                continue
            assert timers["oscillation_history"]["steps"] == 6
            assert timers["oscillation_history"]["horizon_s"] == pytest.approx(oscillation_s)
            assert timers["oscillation_history"]["weight"] == 0.5
            assert timers["heading_smoothness"] == {
                "interval_s": dt,
                "horizon_s": dt,
                "steps": 1,
                "weight": 0.4,
            }
            assert timers["braking_reaction"]["steps"] == 1
            assert timers["braking_reaction"]["horizon_s"] == dt
            assert timers["braking_reaction"]["speed_cap_reaction_s"] == dt
            assert timers["progress_history_capacity"]["capacity_samples"] == 256
            assert timers["progress_history_capacity"]["steps"] == 255
            assert timers["progress_history_capacity"]["horizon_s"] == capacity_s
            assert timers["route_trace_recovery_hold"] == {
                "interval_s": dt,
                "steps": 2,
                "horizon_s": 2 * dt,
                "enabled": False,
            }


@pytest.mark.parametrize(
    ("field", "expected_status"),
    [
        ("canonical_algorithm", "binding_mismatch"),
        ("requested_model_id", "binding_mismatch"),
        ("forecast_grid", "timing_mismatch"),
        ("effective_budget_steps", "timing_mismatch"),
    ],
)
def test_witness_refuses_mismatched_runtime_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, expected_status: str
) -> None:
    """An episode that ran on the right simulator grid can report the wrong binding."""
    pytest.importorskip("torch")
    checkpoint = tmp_path / "test-predictor"
    checkpoint.touch()

    class FixedForecast:
        """Return a fixed eight-sample forecast through the real inference adapter."""

        config = SimpleNamespace(input_dim=4)

        def __call__(self, state: Any, _mask: Any) -> dict[str, Any]:
            return {"future_positions": state[..., :2].unsqueeze(2).expand(-1, -1, 8, -1)}

    monkeypatch.setattr(
        PredictionPlannerAdapter, "_resolve_checkpoint_path", lambda _self: checkpoint
    )
    monkeypatch.setattr(PredictionPlannerAdapter, "_build_model", lambda _self: FixedForecast())
    cfg = load_campaign_config(get_repository_root() / study.CONFIG_PATTERN.format(ms=200))
    predictor = next(planner for planner in cfg.planners if planner.key == "prediction_planner")
    original = study.run_map_batch

    def change_report(scenarios: list[dict[str, Any]], output: Path, *args: Any, **kw: Any) -> Any:
        summary = original(scenarios, output, *args, **kw)
        row = json.loads(output.read_text())
        if field == "canonical_algorithm":
            row["algorithm_metadata"][field] = "goal"
        elif field == "effective_budget_steps":
            row[field] = 3
        else:
            foresight = row["algorithm_metadata"]["foresight_prediction"]
            if field == "requested_model_id":
                foresight[field] = "unexpected_model"
            else:
                foresight[field] = {"interval_s": 0.1, "steps": 7}
        output.write_text(json.dumps(row) + "\n")
        return summary

    monkeypatch.setattr(study, "run_map_batch", change_report)
    results = study._run_witness(
        replace(cfg, planners=(predictor,)), _load_campaign_scenarios(cfg), tmp_path
    )
    assert results[0]["status"] == expected_status, results[0]["reason"]
