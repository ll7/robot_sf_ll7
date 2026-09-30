"""Exercise emitted 0.0.8 sweep inputs and execution accounting without a campaign."""

import json

import pytest

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.scenario_certification.v1 import scenario_actor_source_census
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from scripts.validation import run_empty_world_sweep as sweep


@pytest.mark.parametrize("suite", ["main", "width"])
def test_unchanged_emitted_scenarios_load_from_arbitrary_output_dir(tmp_path, suite):
    cfg_path, scenarios = sweep.build_derived_inputs(
        suite,
        seeds=[1001, 1002],
        arms=["goal"],
        scenarios_filter=None,
        workers=1,
        out_dir=tmp_path / "outside" / "inputs",
        step_trace=True,
    )
    cfg = load_campaign_config(cfg_path)
    consumed = load_scenarios(cfg.scenario_matrix_path)
    assert len(consumed) == len(scenarios)
    for scenario in consumed:
        config = build_robot_config_from_scenario(scenario, scenario_path=cfg.scenario_matrix_path)
        assert scenario_actor_source_census(config)["verified_empty"] is True
    # Campaigns deliberately normalize references against repository-root scoped_scenarios.
    for scenario in _load_campaign_scenarios(cfg):
        config = build_robot_config_from_scenario(
            scenario, scenario_path=sweep.REPO_ROOT / "scoped_scenarios.json"
        )
        assert scenario_actor_source_census(config)["verified_empty"] is True


@pytest.mark.parametrize("case", ["failed", "missing", "incomplete_trace", "complete", "raised"])
def test_cli_reports_failed_missing_and_incomplete_trace_slots(tmp_path, monkeypatch, case):
    import robot_sf.benchmark.camera_ready_campaign as campaign

    matrix = (
        sweep.REPO_ROOT / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    )
    output = tmp_path / "outside"
    root = output / "campaigns" / "empty_world_main"
    run = root / "runs" / "goal__differential_drive"
    run.mkdir(parents=True)
    if case == "failed":
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "failures": [
                        {
                            "scenario_id": "classic_bottleneck_low",
                            "seed": 1001,
                            "error": "RuntimeError('documented runner failure')",
                        }
                    ]
                }
            )
        )
    elif case in {"incomplete_trace", "complete"}:
        from robot_sf.benchmark.map_runner.map_runner import _run_map_episode
        from robot_sf.training.scenario_loader import load_scenarios

        scenario = next(s for s in load_scenarios(matrix) if s["name"] == "classic_bottleneck_low")
        scenario = sweep.remove_pedestrians(scenario, [1001])
        row = _run_map_episode(
            scenario,
            1001,
            horizon=2,
            dt=0.1,
            record_forces=False,
            snqi_weights=None,
            snqi_baseline=None,
            algo="goal",
            scenario_path=matrix,
            record_simulation_step_trace=True,
        )
        if case == "incomplete_trace":
            row["algorithm_metadata"]["simulation_step_trace"]["steps"].pop()
        (run / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    result = {
        "campaign_root": str(root),
        "campaign_execution_status": "failed" if case in {"failed", "raised"} else "completed",
        "status": "failed" if case in {"failed", "raised"} else "ok",
        "unexpected_failed_runs": int(case in {"failed", "raised"}),
        "exit_code": int(case in {"failed", "raised"}),
        "total_episodes": None
        if case == "raised"
        else int(case in {"incomplete_trace", "complete"}),
    }

    def simulated_campaign(*args, **kwargs):
        if case == "raised":
            raise RuntimeError("documented campaign failure")
        return result

    monkeypatch.setattr(campaign, "run_campaign", simulated_campaign)
    monkeypatch.setattr(sweep, "verify_head", lambda head: "owned-test-head")
    code = sweep.main(
        [
            "--head-sha",
            "HEAD",
            "--suite",
            "main",
            "--arms",
            "goal",
            "--scenarios",
            "classic_bottleneck_low",
            "--workers",
            "1",
            "--seeds",
            "1001",
            "--output-dir",
            str(output),
        ]
    )
    assert code == int(case != "complete")
    rows = [json.loads(line) for line in (output / "episodes_main.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert (
        rows[0]["execution_status"]
        == {
            "failed": "execution_failed",
            "missing": "missing_episode",
            "incomplete_trace": "incomplete_trace",
            "complete": "written",
            "raised": "execution_failed",
        }[case]
    )
    meta = json.loads((output / "execution_main.json").read_text())
    assert meta["campaign_execution_status"] == result["campaign_execution_status"]
    assert meta["unexpected_failed_runs"] == result["unexpected_failed_runs"]
    assert meta["exit_code"] == result["exit_code"]
    assert meta["complete"] is (case == "complete")
    assert rows[0]["execution_status"] in (output / "summary_main.csv").read_text()
    # Re-analysis retains exactly the same unavailable slot and nonzero completion status.
    assert sweep.main(
        [
            "--head-sha",
            "HEAD",
            "--suite",
            "main",
            "--arms",
            "goal",
            "--scenarios",
            "classic_bottleneck_low",
            "--workers",
            "1",
            "--seeds",
            "1001",
            "--output-dir",
            str(output),
            "--summarize-only",
        ]
    ) == int(case != "complete")
