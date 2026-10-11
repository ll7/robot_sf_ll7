"""Protect the development pair, its tradeoff table and full stopping-window audit."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = (
    ROOT
    / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_1_0_predictive_braking_v1.yaml"
)
DEFAULT = (
    ROOT
    / "configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release.yaml"
)
ARMS = ("hybrid_v4_default", "hybrid_v4_predictive_braking")


def _analysis():
    """Require the new entrypoint without turning base proof into a collection error."""
    spec = importlib.util.find_spec("scripts.analysis.analyze_predictive_braking_pair")
    assert spec is not None, "paired tradeoff analysis is not implemented"
    return importlib.import_module(spec.name)


def _rows():
    """Two scenarios with hand-counted, aligned development episodes."""
    rows = []
    for scenario in ("station", "crossing"):
        for seed in (1001, 1002):
            for arm in ARMS:
                enabled = arm == ARMS[1]
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "arm": arm,
                        "success": int(enabled and seed == 1001),
                        "collision": 0,
                        "duration_s": 10.0,
                        "near_miss_onsets": 3 if enabled else 1,
                        "near_miss_exposure_s": 2.0 if enabled else 1.0,
                        "bound_2s_windows": 3 if seed == 1001 else 1,
                        "bound_2s_violations": int(enabled and seed == 1001),
                        "fallback_count": 0,
                        "degraded_count": 0,
                        "prediction_speed_error_m_s": 0.2,
                        "dt_s": 0.1,
                    }
                )
    return rows


def test_pair_roster_preserves_default_bytes_and_changes_only_opt_in():
    """Detect missing/aliased arms and silent changes to an existing comparator."""
    assert CAMPAIGN.is_file(), "0.1.0 paired braking roster is missing"
    payload = yaml.safe_load(CAMPAIGN.read_text())
    assert payload["protocol_version"] == "0.1.0"
    assert payload["seed_policy"]["seeds"] == list(range(1001, 1031))
    assert [arm["key"] for arm in payload["planners"]] == list(ARMS)
    assert ROOT / payload["planners"][0]["algo_config"] == DEFAULT
    for path, expected in (
        (DEFAULT, "d4fbb7a9186b8950249cc2b4a24aec49dc49f0bb0829cce89b8ebab8508045e3"),
        (
            ROOT / "configs/algos/hybrid_rule_v4_clearance_braking.yaml",
            "5dbc4f17a1ca8b44c2f78dc4e5c83e94a60615ebd7e97892109124323bc9c4ab",
        ),
        (
            ROOT
            / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_1_0.yaml",
            "1a1914260eed088ded603b6f9e92104f161534d73f1c6b5b9c13c048f30ef29c",
        ),
    ):
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    baseline = yaml.safe_load(DEFAULT.read_text())
    opt_in = yaml.safe_load((ROOT / payload["planners"][1]["algo_config"]).read_text())
    opt_in.pop("name")
    baseline.pop("name")
    assert opt_in["params"].pop("v4_predictive_braking_enabled") is True
    assert opt_in["params"].pop("v4_prediction_speed_error") == 0.2
    assert opt_in == baseline
    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
    from robot_sf.benchmark.map_runner_policies.map_runner_policy_resolution import (
        _resolve_policy_search_candidate_runtime,
    )

    campaign = load_campaign_config(CAMPAIGN)
    assert [arm.key for arm in campaign.planners] == list(ARMS)
    for scenario in (
        "classic_station_platform_medium",
        "francis2023_blind_corner",
        "francis2023_perpendicular_traffic",
    ):
        configs = []
        for arm in campaign.planners:
            algo, config = _resolve_policy_search_candidate_runtime(
                default_algo=arm.algo,
                algo_config_path=str(arm.algo_config_path),
                scenario={"name": scenario},
                config_root=ROOT,
            )
            assert algo == "hybrid_rule_local_planner"
            configs.append(config)
        assert configs[1].pop("v4_predictive_braking_enabled") is True
        assert configs[1].pop("v4_prediction_speed_error") == 0.2
        assert not configs[0].get("v4_predictive_braking_enabled", False)
        assert configs[0] == configs[1]


@pytest.mark.parametrize("missing_simulator", [False, True])
def test_native_diagnostics_require_simulator_after_reset(monkeypatch, missing_simulator):
    """Reject missing native state, close the env, and preserve valid episode metrics."""
    from scripts.validation import run_predictive_braking_diagnostics as runner

    simulator = SimpleNamespace(
        robot_pos=np.array([[0.0, 0.0]]),
        goal_pos=np.array([[1.0, 0.0]]),
        ped_pos=np.array([[1.5, 0.0]]),
        ped_vel=np.zeros((1, 2)),
        config=SimpleNamespace(time_per_step_in_secs=0.1),
    )
    env = SimpleNamespace(simulator=None, close=Mock())

    def reset(*, seed):
        assert seed == 1001
        env.simulator = None if missing_simulator else simulator
        return {}, {}

    def step(_action):
        simulator.robot_pos[0] = [0.1, 0.0]
        return {}, 0.0, False, True, {"meta": {}}

    env.reset = reset
    env.step = step
    policy = Mock(return_value=np.zeros(2))
    policy._planner_adapter.diagnostics.return_value = {}
    monkeypatch.setattr(runner, "_build_env_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "hybrid_config", lambda _scenario: {})
    monkeypatch.setattr(runner, "_build_policy", lambda *_args, **_kwargs: (policy, {}))
    monkeypatch.setattr(runner, "make_robot_env", lambda **_kwargs: env)
    monkeypatch.setattr(runner, "_policy_command_to_env_action", lambda **_kwargs: np.zeros(2))
    task = ({"name": "native-test"}, 1001, True, False, 0)
    if missing_simulator:
        with pytest.raises(RuntimeError, match="initialized simulator"):
            runner.run_cell(task)
        policy.assert_not_called()
    else:
        result = runner.run_cell(task)
        assert result["steps"] == 1
        assert result["travel_m"] == pytest.approx(0.1)
        assert result["duration_s"] == pytest.approx(0.1)
        assert result["final_goal_distance_m"] == pytest.approx(0.9)
        assert result["one_step_tube_checks"] == 1
        assert result["one_step_tube_violations"] == 0
        policy.assert_called_once()
    env.close.assert_called_once()


def test_analysis_keeps_success_gain_next_to_near_miss_cost_and_bound_rate():
    """Catch aggregation across scenarios and averaging rates with unequal denominators."""
    report = _analysis().summarize_pairs(_rows(), expected_seeds=[1001, 1002])
    assert [row["scenario"] for row in report] == ["crossing", "station"]
    for row in report:
        assert row["episodes_per_arm"] == 2
        assert row["success_gain"] == 0.5
        assert row["near_miss_onsets_delta"] == 4
        assert row["near_miss_exposure_s_delta"] == 2.0
        for prefix, successes, onsets, exposure, rate in (
            ("default", 0.0, 2, 2.0, 0.0),
            ("predictive", 0.5, 6, 4.0, 0.25),
        ):
            assert row[f"{prefix}_success_rate"] == successes
            assert row[f"{prefix}_collision_rate"] == 0.0
            assert row[f"{prefix}_near_miss_onsets"] == onsets
            assert row[f"{prefix}_near_miss_exposure_s"] == exposure
            assert row[f"{prefix}_bound_2s_violation_rate"] == rate
            assert row[f"{prefix}_bound_2s_windows"] == 4


@pytest.mark.parametrize(
    "column",
    [
        "success",
        "collision",
        "near_miss_onsets",
        "near_miss_exposure_s",
        "bound_2s_windows",
        "bound_2s_violations",
    ],
)
def test_missing_metric_fails(column):
    """An incomplete producer must not yield a reassuring zero-filled comparison."""
    rows = _rows()
    del rows[0][column]
    with pytest.raises(ValueError, match=column):
        _analysis().summarize_pairs(rows, expected_seeds=[1001, 1002])


@pytest.mark.parametrize(
    "defect",
    ["unpaired", "duplicate", "sealed_seed", "fallback", "nan", "impossible_count", "missing_seed"],
)
def test_invalid_pair_is_rejected(defect):
    """Protect alignment, development-only inputs and valid metric denominators."""
    rows = _rows()
    if defect == "unpaired":
        rows.pop()
    elif defect == "duplicate":
        rows.append(dict(rows[0]))
    elif defect == "sealed_seed":
        rows[0]["seed"] = 111
    elif defect == "fallback":
        rows[0]["fallback_count"] = 1
    elif defect == "nan":
        rows[0]["near_miss_exposure_s"] = float("nan")
    elif defect == "impossible_count":
        rows[0]["bound_2s_violations"] = 4
    else:
        rows = [row for row in rows if row["seed"] != 1002]
    with pytest.raises(ValueError):
        _analysis().summarize_pairs(rows, expected_seeds=[1001, 1002])


def test_no_eligible_windows_is_undefined_not_zero():
    """Empty-world data cannot prove prediction-bound coverage."""
    rows = _rows()
    for row in rows:
        row["bound_2s_windows"] = row["bound_2s_violations"] = 0
    assert (
        _analysis().summarize_pairs(rows, expected_seeds=[1001, 1002])[0][
            "default_bound_2s_violation_rate"
        ]
        is None
    )


def test_full_window_audit_catches_interior_error_and_excludes_distant_pedestrians():
    """Two complete 2 s windows: one interior violation, one valid at the bound."""
    module = _analysis()
    robot = np.zeros((3, 2))
    positions = np.array(
        [[[1.5, 0.0], [3.0, 0.0]], [[1.2, 0.0], [4.0, 0.0]], [[1.5, 0.0], [3.0, 0.0]]]
    )
    velocities = np.zeros_like(positions)
    windows, violations = module.audit_prediction_windows(
        robot, positions, velocities, dt_s=1.0, speed_error_m_s=0.2
    )
    assert (windows, violations) == (1, 1)  # endpoint error is zero; 1 s error is 0.3 > 0.2.
    positions[1, 0, 0] = 1.3  # equality at 0.2 m is valid.
    assert module.audit_prediction_windows(
        robot, positions, velocities, dt_s=1.0, speed_error_m_s=0.2
    ) == (1, 0)


def test_audit_uses_start_velocity_and_requires_whole_two_seconds():
    """Constant observed retreat stays in its tube; short windows are not counted."""
    module = _analysis()
    positions = np.array([[[1.5, 0.0]], [[2.5, 0.0]], [[3.5, 0.0]]])
    velocities = np.broadcast_to([1.0, 0.0], positions.shape)
    robot = np.zeros((3, 2))
    assert module.audit_prediction_windows(
        robot, positions, velocities, dt_s=1.0, speed_error_m_s=0.2
    ) == (1, 0)
    assert module.audit_prediction_windows(
        robot[:2], positions[:2], velocities[:2], dt_s=1.0, speed_error_m_s=0.2
    ) == (0, 0)


def test_cli_writes_a_tradeoff_report_and_rejects_incomplete_producer(tmp_path):
    """Exercise CLI CSV output and its producer completeness admission."""
    module = _analysis()
    path = tmp_path / "episodes.csv"
    rows = _rows()
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "diagnostic-only",
        "bound_audit": "2s_all_sampled_offsets_euclidean_nearby_2m_v1",
        "seeds": [1001, 1002],
        "scenarios": ["station", "crossing"],
        "arms": list(ARMS),
        "head": "synthetic-fixture",
        "sha256": {},
        "empty": False,
        "horizon_override": 100,
        "prediction_speed_error_m_s": 0.2,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    cmd = [
        sys.executable,
        module.__file__,
        "--episodes",
        str(path),
        "--seeds",
        "1001",
        "1002",
        "--output",
        str(tmp_path / "report"),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with (tmp_path / "report/paired.csv").open(newline="") as stream:
        table = list(csv.DictReader(stream))
    assert len(table) == 2
    assert table[0]["predictive_bound_2s_violation_rate"] == "0.25"
    assert table[0]["near_miss_onsets_delta"] == "4"
    assert "not a safety guarantee" in (tmp_path / "report/paired.md").read_text()
    manifest["scenarios"].append("missing_scenario")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Missing or unexpected paired scenarios" in result.stderr
