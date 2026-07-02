"""Tests for issue #3556 ScenarioBelief screening report contracts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from robot_sf.benchmark.scenario_belief_screening import (
    build_input_screening_report,
    build_screening_report,
    classify_screened_decision,
)
from robot_sf.benchmark.scenario_schema import (
    validate_scenario_list,
    validate_scenario_matrix_metadata,
)
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_SET = (
    REPO_ROOT / "configs/scenarios/sets/issue_3556_near_safe_occlusion_bearing_crossing.yaml"
)
BENCHMARK_CONFIG = (
    REPO_ROOT / "configs/benchmarks/scenario_belief_drop_vs_retain_issue_3556_near_safe.yaml"
)
CAMPAIGN_SCRIPT = REPO_ROOT / "scripts/benchmark/run_belief_mode_safety_campaign_issue_3556.py"


def _load_campaign_module():
    spec = importlib.util.spec_from_file_location(
        "run_belief_mode_safety_campaign_issue_3556", CAMPAIGN_SCRIPT
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


def _mode(collision_rate: float, near_misses: int) -> dict[str, float | int]:
    return {
        "episodes": 3,
        "collision_rate": collision_rate,
        "total_near_misses": near_misses,
    }


def test_issue_3556_scenario_config_loads_and_exposes_occlusion_contract() -> None:
    """The #3556 scenario pair is a loadable opt-in real-runner smoke surface."""
    manifest = _load_yaml(SCENARIO_SET)
    scenarios = [dict(scenario) for scenario in load_scenarios(SCENARIO_SET)]

    assert validate_scenario_matrix_metadata(manifest) == []
    assert validate_scenario_list(scenarios) == []
    assert [scenario["name"] for scenario in scenarios] == [
        "issue_3556_near_safe_occlusion_bearing_crossing"
    ]
    scenario = scenarios[0]
    assert scenario["metadata"]["issue"] == 3556
    assert scenario["metadata"]["benchmark_evidence"] is False
    assert scenario["observation_visibility"]["enabled"] is True
    assert scenario["observation_visibility"]["static_occlusion"] is True
    assert scenario["single_pedestrians"][0]["metadata"]["role"] == "occluded_crossing_pedestrian"


def test_screening_input_report_requires_out_of_fov_sidecar_contract() -> None:
    """Static screening verifies scenario IDs, seeds, modes, and occlusion-bearing FOV."""
    scenarios = campaign.load_campaign_scenarios(SCENARIO_SET, campaign.DEFAULT_SEEDS)
    report = build_input_screening_report(
        scenarios=scenarios,
        seeds=campaign.DEFAULT_SEEDS,
        fov_degrees=120.0,
        scenario_set=SCENARIO_SET,
        launch_packet=BENCHMARK_CONFIG,
    )

    assert report["ready"] is True
    assert report["scenario_ids"] == ["issue_3556_near_safe_occlusion_bearing_crossing"]
    assert report["checks"]["out_of_fov_sidecar_contract"]["passed"] is True

    no_visibility = [dict(scenarios[0], observation_visibility={"enabled": False})]
    blocked = build_input_screening_report(
        scenarios=no_visibility,
        seeds=campaign.DEFAULT_SEEDS,
        fov_degrees=120.0,
    )
    assert blocked["ready"] is False
    assert "out_of_fov_sidecar_contract" in blocked["failed_checks"]


def test_screened_decision_uses_only_issue_allowed_labels() -> None:
    """Missing rows and unsafe oracle both use issue #3556 decision labels."""
    missing = classify_screened_decision({}, oracle_near_safe_threshold=0.25)
    assert missing["decision"] == "blocked_no_near_safe_family"

    unsafe_oracle = classify_screened_decision(
        {
            "oracle": _mode(1.0, 3),
            "uncertain_retained": _mode(0.0, 0),
            "uncertain_dropped": _mode(1.0, 3),
        },
        oracle_near_safe_threshold=0.25,
    )
    assert unsafe_oracle["decision"] == "inconclusive_oracle_unsafe"

    revise = classify_screened_decision(
        {
            "oracle": _mode(0.0, 0),
            "uncertain_retained": _mode(0.0, 0),
            "uncertain_dropped": _mode(0.0, 2),
        },
        oracle_near_safe_threshold=0.25,
    )
    assert revise["decision"] == "revise"


def test_input_screening_treats_null_scenario_fov_as_default() -> None:
    """Explicit null FOV in scenario visibility falls back to run FOV."""

    report = build_input_screening_report(
        scenarios=[
            {
                "name": "null-fov-scenario",
                "observation_visibility": {
                    "enabled": True,
                    "fov_degrees": None,
                    "static_occlusion": True,
                },
                "single_pedestrians": [{"id": "ped"}],
            }
        ],
        seeds=[1, 2, 3],
        fov_degrees=120.0,
    )

    assert report["checks"]["out_of_fov_sidecar_contract"]["passed"] is True


def test_runner_preflight_checks_launch_packet_and_screening_inputs() -> None:
    """The runner preflight now gates the #3556 launch packet and screening module."""
    payload = _load_yaml(BENCHMARK_CONFIG)
    assert payload["issue"] == 3556
    assert payload["scenario_family"] == campaign.DEFAULT_SCENARIO_SET
    assert payload["seed_set"] == campaign.DEFAULT_SEED_SET
    assert payload["seed_sets"][campaign.DEFAULT_SEED_SET]["seeds"] == campaign.DEFAULT_SEEDS

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
        horizon=240,
        dt=0.1,
        workers=1,
        launch_packet=BENCHMARK_CONFIG,
    )
    assert readiness["ready"] is True
    assert "scenario_belief_screening_inputs" not in readiness["failed_checks"]
    assert {check["name"] for check in readiness["checks"]} >= {
        "scenario_belief_screening_inputs",
        "launch_packet_arm_contract",
        "oracle_near_safety_contract",
    }


def test_final_screening_report_propagates_decision_and_provenance() -> None:
    """Campaign reports carry one durable screening surface beside aggregate rows."""
    scenarios = campaign.load_campaign_scenarios(SCENARIO_SET, campaign.DEFAULT_SEEDS)
    by_mode = {
        "oracle": _mode(0.0, 0),
        "uncertain_retained": _mode(0.0, 0),
        "uncertain_dropped": _mode(0.0, 2),
    }
    report = build_screening_report(
        scenarios=scenarios,
        seeds=campaign.DEFAULT_SEEDS,
        by_mode=by_mode,
        oracle_near_safe_threshold=0.25,
        fov_degrees=120.0,
        scenario_set=SCENARIO_SET,
        launch_packet=BENCHMARK_CONFIG,
    )

    assert report["ready"] is True
    assert report["decision"]["decision"] == "revise"
    assert report["decision"]["oracle_near_safe_threshold"] == 0.25
    assert report["scenario_ids"] == ["issue_3556_near_safe_occlusion_bearing_crossing"]
    assert set(report["allowed_decision_labels"]) == {
        "revise",
        "retention_dominates",
        "inconclusive",
        "inconclusive_oracle_unsafe",
        "blocked_no_near_safe_family",
    }
