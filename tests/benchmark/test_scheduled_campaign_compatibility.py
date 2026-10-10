"""Pre-0.0.8 schedules preserve main's actual scenario and episode bytes."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config
from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode

ROOT = Path(__file__).resolve().parents[2]
ORACLE = json.loads(
    (ROOT / "tests/benchmark/fixtures/scheduled_campaign_main_93ba0d75.json").read_text()
)


def _historical_scenarios(protocol):
    cfg = load_campaign_config(ROOT / ORACLE["config"], repository_root=ROOT)
    cfg = replace(
        cfg,
        protocol_version=protocol,
        seed_policy=replace(cfg.seed_policy, mode="fixed-list", seeds=(1001,)),
    )
    return cfg, _load_campaign_scenarios(cfg, ROOT)


@pytest.mark.parametrize("protocol", [None, "0.0.7"])
def test_historical_schedule_preserves_all_main_scenario_bytes(protocol):
    """Preserve the main oracle while independently checking D-085 metadata changes."""
    _, scenarios = _historical_scenarios(protocol)
    overtaking = next(s for s in scenarios if s["name"] == "francis2023_pedestrian_overtaking")
    plausibility = overtaking["metadata"]["plausibility"]
    assert "D-085" in plausibility["notes"]
    assert plausibility["metrics"] is None
    assert plausibility["metrics_updated_on"] is None
    # D-085 intentionally replaces metadata; compare all other loaded bytes with
    # the immutable main oracle using its original source_revision metadata.
    plausibility.update(
        notes=None,
        metrics={
            "min_distance": 2.2223542321899186,
            "mean_distance": 8.737510755793421,
            "robot_ped_within_5m_frac": 0.28587024062018307,
            "ped_force_mean": 0.24676126309566468,
            "force_q95": 1.208259633473192,
        },
        metrics_updated_on="2026-01-30T14:19:28.225110+01:00",
    )
    assert len(scenarios) == ORACLE["scenario_count"] == 48
    scenario = next(s for s in scenarios if s["name"] == "francis2023_blind_corner")
    assert scenario["metadata"]["scenario_horizon"] == {
        "source": "configs/policy_search/scenario_horizons_h500.yaml",
        "recommended_horizon_steps": 316,
        "status": "recommended",
        "bucket": "long",
    }
    canonical = json.dumps(scenarios, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == ORACLE["scenarios_sha256"]


@pytest.mark.slow
@pytest.mark.parametrize("protocol", [None, "0.0.7"])
@pytest.mark.parametrize("name", ["francis2023_blind_corner", "classic_cross_trap_low"])
def test_historical_scheduled_episode_matches_main_row_contract(protocol, name):
    """Native goal identity and a real stationary timeout retain main's row fields."""
    from robot_sf.benchmark.map_runner.map_runner import _build_policy

    cfg, scenarios = _historical_scenarios(protocol)
    scenario = next(s for s in scenarios if s["name"] == name)
    assert scenario["seeds"] == [1001]

    def stationary(algo, config, **kwargs):
        return lambda obs: np.zeros(2), {"algorithm": algo, "config": config}

    row = run_map_episode(
        scenario,
        1001,
        horizon=None,
        dt=cfg.dt,
        algo="goal",
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        scenario_path=ROOT / "scoped_scenarios.json",
        provenance_scenario_path=cfg.scenario_matrix_path,
        policy_builder=_build_policy if name == "francis2023_blind_corner" else stationary,
    )
    expected = ORACLE["rows"][name]
    # Check the real stop first so the timeout case exposes the label regression.
    assert row["steps"] == expected["steps"]
    assert row["termination_reason"] == expected["termination_reason"]
    # Main's metric-v2 rows add this schema field; retain the historical oracle bytes.
    assert sorted(row) == sorted([*expected["row_fields"], "metric_schema_version"])
    assert row["metric_schema_version"] == "robot-sf-metrics.v2"
    assert row["config_hash"] == expected["config_hash"]
    assert row["episode_id"] == expected["episode_id"]
    assert "run_horizon" not in row["scenario_params"]
    assert json.dumps(row["scenario_params"], sort_keys=True) == json.dumps(
        expected["scenario_params"], sort_keys=True
    )
