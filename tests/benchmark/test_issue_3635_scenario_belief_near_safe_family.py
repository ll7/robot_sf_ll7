"""Contract tests for the issue #3635 ScenarioBelief screening family."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from robot_sf.benchmark.scenario_schema import (
    validate_scenario_list,
    validate_scenario_matrix_metadata,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_SET = (
    REPO_ROOT / "configs/scenarios/sets/issue_3635_near_safe_occlusion_bearing_crossing.yaml"
)
BENCHMARK_CONFIG = REPO_ROOT / "configs/benchmarks/scenario_belief_drop_vs_retain_issue_3556.yaml"
CAMPAIGN_SCRIPT = REPO_ROOT / "scripts/benchmark/run_belief_mode_safety_campaign_issue_3556.py"
campaign = None


def _load_campaign_module():
    spec = importlib.util.spec_from_file_location(
        "run_belief_mode_safety_campaign_issue_3556",
        CAMPAIGN_SCRIPT,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


campaign = _load_campaign_module()


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_issue_3635_family_loads_as_single_occlusion_bearing_crossing() -> None:
    """The new family is a reproducible single-scenario surface, not a result claim."""
    manifest = _load_yaml(SCENARIO_SET)
    scenarios = load_scenarios(SCENARIO_SET)

    assert validate_scenario_matrix_metadata(manifest) == []
    assert validate_scenario_list([dict(scenario) for scenario in scenarios]) == []
    assert len(scenarios) == 1

    scenario = dict(scenarios[0])
    assert scenario["name"] == "issue_3635_near_safe_occlusion_bearing_crossing"
    assert scenario["scenario_family"] == "near_safe_occlusion_bearing_crossing"
    assert scenario["seeds"] == [363501, 363502, 363503]

    visibility = scenario["observation_visibility"]
    assert visibility == {
        "enabled": True,
        "fov_degrees": 120.0,
        "max_range_m": 8.0,
        "static_occlusion": True,
    }

    metadata = scenario["metadata"]
    assert metadata["issue"] == 3635
    assert metadata["benchmark_evidence"] is False
    assert metadata["occlusion_bearing"] is True
    assert "does not report benchmark results" in metadata["claim_boundary"]
    assert metadata["scenario_belief_contract"]["required_modes"] == list(campaign.MODES)


def test_issue_3635_family_builds_visibility_config_without_running_campaign() -> None:
    """Scenario loader must materialize the visibility and single-pedestrian contract."""
    scenario = dict(load_scenarios(SCENARIO_SET)[0])
    runtime_config = build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_SET)

    visibility = runtime_config.observation_visibility
    assert visibility.enabled is True
    assert visibility.fov_degrees == 120.0
    assert visibility.max_range_m == 8.0
    assert visibility.static_occlusion is True

    map_def = next(iter(runtime_config.map_pool.map_defs.values()))
    assert [ped.id for ped in map_def.single_pedestrians] == ["h1"]
    assert map_def.single_pedestrians[0].metadata["role"] == "occluded_crossing_pedestrian"


def test_issue_3635_campaign_manifest_distinguishes_three_belief_arms(tmp_path: Path) -> None:
    """The launch packet and generated algo configs keep oracle/retain/drop arms distinct."""
    payload = _load_yaml(BENCHMARK_CONFIG)

    assert payload["scenario_family"] == SCENARIO_SET.relative_to(REPO_ROOT).as_posix()
    assert payload["seed_set"] == "issue_3635_s3_contract_only"
    assert payload["seed_sets"][payload["seed_set"]]["seeds"] == [363501, 363502, 363503]
    assert payload["no_benchmark_result_claim"] is True
    assert set(payload["belief_modes"]) == set(campaign.MODES)

    for mode in campaign.MODES:
        arm = payload["belief_modes"][mode]
        assert arm["algo"] == "stream_gap"
        assert arm["belief_mode"] == mode
        assert arm["belief_fov_degrees"] == 120.0

        algo_config = yaml.safe_load(
            campaign.write_algo_config(mode, tmp_path, fov_degrees=120.0).read_text(
                encoding="utf-8"
            )
        )
        assert algo_config["algo"] == "stream_gap"
        assert algo_config["belief_mode"] == mode
        assert algo_config["belief_fov_degrees"] == 120.0


def test_issue_3635_runner_default_targets_new_family_and_applies_seed_matrix() -> None:
    """The #3556 runner default resolves the #3635 family and bounded seed fixture."""
    payload = _load_yaml(BENCHMARK_CONFIG)
    expected_family = SCENARIO_SET.relative_to(REPO_ROOT).as_posix()
    assert campaign.DEFAULT_SCENARIO_SET == expected_family
    assert payload["scenario_family"] == expected_family
    assert campaign.DEFAULT_SEED_SET == payload["seed_set"]
    assert campaign.DEFAULT_SEEDS == payload["seed_sets"][payload["seed_set"]]["seeds"]

    scenarios = campaign.load_campaign_scenarios(SCENARIO_SET, campaign.DEFAULT_SEEDS)
    assert len(scenarios) == 1
    scenario = scenarios[0]

    assert scenario["name"] == "issue_3635_near_safe_occlusion_bearing_crossing"
    assert scenario["seeds"] == [363501, 363502, 363503]
    assert Path(scenario["map_file"]).is_absolute()


def test_issue_3635_synthetic_near_safe_discriminating_classification() -> None:
    """Synthetic aggregates prove the three named arms feed the screened decision contract."""
    decision = campaign.classify_screened_decision(
        {
            "oracle": {"episodes": 3, "collision_rate": 0.0, "total_near_misses": 0},
            "uncertain_retained": {
                "episodes": 3,
                "collision_rate": 0.0,
                "total_near_misses": 0,
            },
            "uncertain_dropped": {
                "episodes": 3,
                "collision_rate": 1.0 / 3.0,
                "total_near_misses": 1,
            },
        }
    )

    assert decision["decision"] == "revise"
    assert decision["screening_status"] == "near_safe_discriminating"
    assert decision["oracle_near_safe"] is True
    assert decision["mode_is_discriminating"] is True


def test_issue_3635_launch_packet_arm_contract_is_preflight_checked() -> None:
    """Committed packet arms stay aligned with #3635 scenario and seed contract."""
    ok, detail = campaign.check_launch_packet_arm_contract(
        BENCHMARK_CONFIG,
        set_path=SCENARIO_SET,
        seeds=campaign.DEFAULT_SEEDS,
        fov_degrees=120.0,
    )

    assert ok, detail
    readiness = campaign.check_campaign_readiness(
        SCENARIO_SET,
        campaign.DEFAULT_SEEDS,
        fov_degrees=120.0,
        horizon=300,
        dt=0.1,
        workers=1,
        launch_packet=BENCHMARK_CONFIG,
    )

    arm_check = next(
        check for check in readiness["checks"] if check["name"] == "launch_packet_arm_contract"
    )
    assert arm_check["passed"] is True
    assert "runner arms" in arm_check["detail"]


def test_issue_3635_launch_packet_arm_contract_fails_closed_on_drift(tmp_path: Path) -> None:
    """Preflight rejects retained/dropped arm drift before any campaign episodes run."""
    payload = _load_yaml(BENCHMARK_CONFIG)
    payload["runner_arms"][2]["expected_gate_enabled"] = False
    drifted_packet = tmp_path / "drifted_launch_packet.yaml"
    drifted_packet.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    ok, detail = campaign.check_launch_packet_arm_contract(
        drifted_packet,
        set_path=SCENARIO_SET,
        seeds=campaign.DEFAULT_SEEDS,
        fov_degrees=120.0,
    )

    assert ok is False
    assert "uncertain_dropped arm expected_gate_enabled drifted" in detail
