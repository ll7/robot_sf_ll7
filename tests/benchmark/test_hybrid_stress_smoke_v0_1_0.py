"""Successor stress inputs must execute the target release definitions and arms."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.policy_search_manifest import resolve_candidate_manifest_runtime
from robot_sf.benchmark.release_protocol import load_release_manifest, validate_release_manifest
from robot_sf.training.scenario_loader import load_scenarios

ROOT = Path(__file__).resolve().parents[2]
SET = ROOT / "configs/scenarios/sets/paper_matrix_v2_h600_hybrid_stress_smoke_v0_1_0.yaml"
CAMPAIGN = (
    ROOT / "configs/benchmarks/paper_experiment_matrix_v2_h600_hybrid_stress_smoke_v0_1_0.yaml"
)
TARGET = (
    ROOT
    / "configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_0_8_candidate_authored.yaml"
)
MANIFEST = (
    ROOT
    / "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_hybrid_stress_smoke_v0_1_0.yaml"
)
IDENTITIES = (
    "classic_urban_crossing_medium",
    "classic_cross_trap_high",
    "classic_doorway_high",
    "francis2023_exiting_elevator",
    "francis2023_robot_crowding",
)


def _yaml(path):
    return yaml.safe_load(path.read_text())


def _definitions(path):
    # Public loader rebases every map reference into the same comparison root.
    return {s["name"]: dict(s) for s in load_scenarios(path, base_dir=ROOT)}


def _assert_definitions_match(path):
    stress = _definitions(path)
    target = _definitions(ROOT / _yaml(TARGET)["scenario_matrix"])
    assert tuple(stress) == IDENTITIES
    for identity, definition in stress.items():
        assert definition == target[identity], identity


def _hybrids(config):
    return {
        p.key: p for p in config.planners if p.enabled and p.algo == "hybrid_rule_local_planner"
    }


def _effective(planner, scenario):
    def load(value):
        return _yaml(ROOT / str(value)) if value else {}

    return resolve_candidate_manifest_runtime(
        default_algo=planner.algo,
        manifest=_yaml(planner.algo_config_path),
        scenario=scenario,
        load_config=load,
    )


def _assert_arms_match(config):
    target = load_campaign_config(TARGET)
    actual, expected = _hybrids(config), _hybrids(target)
    assert actual.keys() == expected.keys()
    # Cover every target scenario, including the ORCA branch omitted from the slice.
    for scenario in _definitions(target.scenario_matrix_path).values():
        for key in expected:
            assert _effective(actual[key], scenario) == _effective(expected[key], scenario), (
                key,
                scenario["name"],
            )


def test_stress_definitions_equal_release_through_public_loader():
    _assert_definitions_match(SET)


@pytest.mark.parametrize(
    "override",
    [
        {"map_file": str(ROOT / "maps/svg_maps/francis2023/francis2023_exiting_elevator.svg")},
        {"simulation_config": {"ped_density": 0.123}},
        {"simulation_config": {"social_force_kernel_version": "legacy_v1"}},
    ],
)
def test_scenario_drift_is_detected(tmp_path, override):
    payload = _yaml(SET)
    payload["includes"] = [str((SET.parent / p).resolve()) for p in payload["includes"]]
    payload["scenario_overrides_by_name"] = {"francis2023_exiting_elevator": override}
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(payload))
    with pytest.raises(AssertionError, match="francis2023_exiting_elevator"):
        _assert_definitions_match(mutated)


def test_hybrid_roster_and_effective_configs_equal_release():
    _assert_arms_match(load_campaign_config(CAMPAIGN))


@pytest.mark.parametrize("mutation", ["stale_key", "parameter", "branch_parameter"])
def test_hybrid_drift_is_detected(tmp_path, mutation):
    config = load_campaign_config(CAMPAIGN)
    planner = next(iter(_hybrids(config).values()))
    if mutation == "stale_key":
        changed = replace(planner, key="scenario_adaptive_hybrid_orca_v2_bottleneck_yield")
    else:
        payload = deepcopy(_yaml(planner.algo_config_path))
        params = (
            payload["params"]
            if mutation == "parameter"
            else payload["scenario_algo_overrides"]["francis2023_leave_group"]["params"]
        )
        params["max_linear_speed"] = 0.123
        path = tmp_path / "mutated_algo.yaml"
        path.write_text(yaml.safe_dump(payload))
        changed = replace(planner, algo_config_path=path)
    config = replace(
        config, planners=tuple(changed if p.key == planner.key else p for p in config.planners)
    )
    with pytest.raises(AssertionError):
        _assert_arms_match(config)


def test_manifest_pins_axes_branch_witnesses_and_inputs():
    config = load_campaign_config(CAMPAIGN)
    manifest = load_release_manifest(MANIFEST)
    report = validate_release_manifest(manifest, campaign_config=config)
    assert report["status"] == "valid", report["problems"]
    payload = _yaml(MANIFEST)
    contract = payload["stress_smoke_contract"]
    assert contract["diagnostic_algorithm_differences"] == []
    assert contract["required_hybrid_arms"] == list(_hybrids(config))
    for field in ("scenario_sources", "map_sources", "algorithm_sources", "hybrid_configs"):
        for pin in contract[field]:
            assert sha256((MANIFEST.parent / pin["path"]).read_bytes()).hexdigest() == pin["sha256"]
    assert sha256(TARGET.read_bytes()).hexdigest() == contract["target_release_config_sha256"]
    for witness in contract["branch_witnesses"]:
        assert witness["arm"] in _hybrids(config)
        assert witness["branch_key"] == f"{witness['arm']}|francis2023_leave_group|orca"
        assert (
            _effective(_hybrids(config)[witness["arm"]], {"name": witness["scenario"]})[0]
            == witness["algorithm"]
        )
    scenarios = _load_campaign_scenarios(config)
    assert {s["name"]: s["simulation_config"]["max_episode_steps"] for s in scenarios} == contract[
        "effective_horizon_steps"
    ]
    assert {seed for s in scenarios for seed in s["seeds"]} == {1001}
    assert config.protocol_version == "0.1.0"
    assert config.dt == 0.1
    assert config.workers == 4
    assert config.horizon is None
    assert payload["matrix"]["expected_episode_cells"] == 70
    assert payload["claim_boundary"]["benchmark_data_release"] is False
    assert payload["claim_boundary"]["snqi"] == "advisory-no-ranking"
    assert payload["provenance"]["publication_authorized"] is False
    assert _yaml(CAMPAIGN)["snqi_v2_spec"] == _yaml(TARGET)["snqi_v2_spec"]
