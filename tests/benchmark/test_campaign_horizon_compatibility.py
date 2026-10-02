"""Main's complete config inventory and native identity guard fixed-horizon compatibility."""

import copy
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios, load_campaign_config
from robot_sf.benchmark.map_runner.map_runner_episode import run_map_episode
from robot_sf.training.scenario_loader import load_scenarios
from scripts.tools.snapshot_campaign_horizons import campaign_admission, tracked_campaign_yaml

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/benchmark/fixtures/campaign_horizons_main_93ba0d75.json"
THREE_WIDTH = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_three_width_doorway_v1.yaml"
)


def test_every_tracked_campaign_preserves_main_admission_and_simulator_limits():
    """Preserve main admission and limits, with D-084's explicit source-budget amendment."""
    fixture = json.loads(FIXTURE.read_text())
    assert fixture["source_revision"] == "93ba0d75fbecc69ddeb62bbf77a435de385caa3b"
    expected = fixture["configs"]
    tracked = tracked_campaign_yaml(ROOT)
    assert set(expected) <= set(tracked), "a main-era input disappeared from the inventory"
    differences = {}
    for path in tracked:
        actual = campaign_admission(ROOT / path, ROOT)
        if path in expected:
            oracle = copy.deepcopy(expected[path])
            if oracle["admitted"]:
                cfg = load_campaign_config(ROOT / path, repository_root=ROOT)
                # D-084 changes only the authored overtaking source H400 -> H600.
                # Fixed runner caps preserve that source limit. D-084 also
                # amends the authoritative release schedule; other historical
                # schedules keep their original entries, even below H600.
                if cfg.scenario_horizons_path is None or cfg.scenario_horizons_path == (
                    ROOT / "configs/benchmarks/horizon_schedules/release_0_0_8_authored_v1.yaml"
                ):
                    for limit in oracle["limits"]:
                        if limit["scenario"] == "francis2023_pedestrian_overtaking":
                            assert limit["max_episode_steps"] == 400
                            limit["max_episode_steps"] = 600
            if actual != oracle:
                differences[path] = {"main_with_d084": oracle, "head": actual}
        elif actual["admitted"]:
            # New scheduled inputs have no main revision. Their independent authored
            # matrix is the oracle, rather than a value calculated by the binding.
            cfg = load_campaign_config(ROOT / path, repository_root=ROOT)
            authored = load_scenarios(cfg.scenario_matrix_path)
            limits = {
                s["name"]: s.get("simulation_config", {}).get("max_episode_steps") for s in authored
            }
            assert all(s["max_episode_steps"] == limits[s["scenario"]] for s in actual["limits"])
    assert not differences, json.dumps(differences, indent=2)


def test_unversioned_smoke_identity_equals_literal_main_oracle():
    """Native goal, blind corner, dev seed 1001: the historical H30 identity survives."""
    from robot_sf.benchmark.map_runner.map_runner import _build_policy

    cfg = load_campaign_config(ROOT / "configs/benchmarks/camera_ready_smoke_all_planners.yaml")
    scenario = next(
        s for s in _load_campaign_scenarios(cfg, ROOT) if s["name"] == "francis2023_blind_corner"
    )
    scenario["seeds"] = [1001]
    row = run_map_episode(
        scenario,
        1001,
        horizon=cfg.horizon,
        dt=cfg.dt,
        algo="goal",
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        scenario_path=ROOT / "scoped_scenarios.json",
        policy_builder=_build_policy,
    )
    assert row["config_hash"] == "cfd4afbfb80f7d05"
    assert row["episode_id"] == "francis2023_blind_corner--1001--cfd4afbfb80f7d05"
    assert row["scenario_params"]["simulation_config"]["max_episode_steps"] == 400
    assert "campaign_horizon" not in row["scenario_params"]["metadata"]


def test_unversioned_authored_timeout_keeps_main_label():
    """An unversioned authored-H600 timeout is terminated, just like main."""
    from dataclasses import replace

    import numpy as np

    cfg = replace(
        load_campaign_config(ROOT / "configs/benchmarks/paper_experiment_matrix_v1.yaml"),
        horizon=None,
    )
    scenario = next(
        s for s in _load_campaign_scenarios(cfg, ROOT) if s["name"] == "classic_cross_trap_low"
    )
    scenario["seeds"] = [1001]

    def stationary(algo, config, **kwargs):
        return (lambda obs: np.zeros(2)), {"algorithm": algo, "config": config}

    row = run_map_episode(
        scenario,
        1001,
        horizon=600,
        dt=0.1,
        algo="goal",
        record_forces=True,
        snqi_weights=None,
        snqi_baseline=None,
        scenario_path=ROOT / "scoped_scenarios.json",
        policy_builder=stationary,
    )
    assert row["steps"] == 600
    assert row["termination_reason"] == "terminated"


def test_three_width_release_declares_authored_0_0_8_budgets_without_execution():
    """The sealed evaluation input is inspected only; all three budgets are 400."""
    cfg = load_campaign_config(THREE_WIDTH)
    assert cfg.protocol_version == "0.0.8"
    assert cfg.horizon is None
    assert cfg.scenario_horizons_path is not None
    scenarios = _load_campaign_scenarios(cfg, ROOT)
    assert len(scenarios) == 3
    assert {s["simulation_config"]["max_episode_steps"] for s in scenarios} == {400}
    assert all(
        s["metadata"]["scenario_horizon"]["recommended_horizon_steps"] == 400 for s in scenarios
    )


@pytest.mark.parametrize("protocol", [None, "0.0.2", "0.0.7", "0.0.8", "0.0.9", "0.1.0", "1.0.0"])
def test_fixed_binding_is_scoped_to_explicit_current_protocols(protocol):
    """An explicit current protocol opts into refusal; older/unidentified inputs stay untouched."""
    from robot_sf.benchmark.camera_ready._config import _apply_fixed_campaign_horizon

    scenario = {"name": "scope", "simulation_config": {"max_episode_steps": 400}}
    if protocol in (None, "0.0.2", "0.0.7"):
        assert _apply_fixed_campaign_horizon(
            [scenario], horizon=600, protocol_version=protocol
        ) == [scenario]
    else:
        with pytest.raises(ValueError, match="below fixed horizon"):
            _apply_fixed_campaign_horizon([scenario], horizon=600, protocol_version=protocol)
