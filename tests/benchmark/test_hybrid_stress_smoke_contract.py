"""Contract tests for the pre-release S30/H600 hybrid stress smoke."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark.camera_ready._config import (
    _load_campaign_scenarios,
    _resolved_seed_inventory,
)
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.release_acceptance import _emergency_stop_marker, _status_markers
from robot_sf.benchmark.release_protocol import load_release_manifest, validate_release_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / (
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_hybrid_stress_smoke_v0_1.yaml"
)
SOURCE_COMMIT = "3651b343ecb4b56f7723a08c16e8b12d8dbd5080"
EXPECTED_SCENARIOS = (
    "classic_urban_crossing_medium",
    "classic_cross_trap_high",
    "classic_doorway_high",
    "francis2023_exiting_elevator",
    "francis2023_robot_crowding",
)
EXPECTED_HYBRID_ARMS = (
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
)
EXPECTED_PLANNER_ARMS = (
    "prediction_planner",
    "goal",
    "social_force",
    "orca",
    "ppo",
    "socnav_sampling",
    "sacadrl",
    *EXPECTED_HYBRID_ARMS,
    "guarded_ppo",
    "predictive_mppi",
    "risk_dwa",
)
FORBIDDEN_STATUSES = ("fallback", "degraded", "unavailable", "failed")
FORBIDDEN_RUNTIME_MARKERS = (
    "selected_source=all_candidates_rejected",
    "selected_source=static_reorient",
    "planner_mode=EMERGENCY_STOP",
    "planner_mode=REORIENT",
    "fallback_count>0",
    "emergency_stop_count>0",
)


def _payload() -> dict[str, Any]:
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (MANIFEST_PATH.parent / path).resolve()


def test_manifest_and_campaign_are_valid_and_resolve_exact_stress_axes() -> None:
    manifest = load_release_manifest(MANIFEST_PATH)
    campaign_config = load_campaign_config(manifest.canonical_campaign_config_path)
    report = validate_release_manifest(manifest, campaign_config=campaign_config)

    assert report["status"] == "valid"
    scenarios = _load_campaign_scenarios(campaign_config)
    scenario_ids = tuple(str(scenario["name"]) for scenario in scenarios)
    assert scenario_ids == EXPECTED_SCENARIOS
    assert _resolved_seed_inventory(scenarios) == [116]
    assert campaign_config.horizon == 600
    assert campaign_config.dt == pytest.approx(0.1)
    assert tuple(planner.key for planner in campaign_config.planners) == EXPECTED_PLANNER_ARMS
    assert len(EXPECTED_PLANNER_ARMS) * len(scenarios) == 70


def test_stress_contract_pins_source_axes_and_fail_closed_policy() -> None:
    payload = _payload()
    contract = payload["stress_smoke_contract"]

    assert contract["schema_version"] == "hybrid-release-stress-smoke.v1"
    assert re.fullmatch(r"[0-9a-f]{40}", contract["source_commit"])
    assert contract["source_commit"] == SOURCE_COMMIT
    assert contract["source_commit_policy"] == "exact-immutable-worktree-sha-required"
    assert contract["expected_episode_cells"] == 70
    assert contract["expected_horizon_steps"] == 600
    assert contract["expected_dt"] == pytest.approx(0.1)
    assert contract["expected_kinematics"] == "differential_drive"
    assert tuple(contract["required_hybrid_arms"]) == EXPECTED_HYBRID_ARMS
    assert tuple(contract["forbidden_row_markers"]) == FORBIDDEN_STATUSES
    assert tuple(contract["forbidden_runtime_markers"]) == FORBIDDEN_RUNTIME_MARKERS
    assert contract["fail_closed"] is True
    assert tuple(contract["historical_sources"]) == (
        "issue7742/job14730",
        "issue4365/job13376-original",
        "issue4365/job13378-original",
    )

    cells = contract["representative_cells"]
    assert [(cell["scenario_id"], cell["seed"]) for cell in cells] == [
        (scenario, 116) for scenario in EXPECTED_SCENARIOS
    ]
    assert {cell["mechanism"] for cell in cells} == {
        "urban-crossing",
        "cross-trap",
        "doorway",
        "elevator-exit",
        "robot-crowding",
    }
    assert payload["claim_boundary"]["snqi"] == "advisory-no-ranking"
    assert payload["claim_boundary"]["fallback_or_degraded_evidence"] == "prohibited"


def test_all_manifest_and_stress_asset_hashes_are_exact() -> None:
    payload = _payload()
    campaign_path = _resolve_manifest_path(payload["canonical_campaign_config"])
    scenario_path = _resolve_manifest_path(payload["scenario"]["matrix_path"])
    assert sha256_file(campaign_path) == payload["campaign_config_sha256"]
    assert sha256_file(scenario_path) == payload["scenario"]["matrix_sha256"]

    suite_policy_path = _resolve_manifest_path(payload["scenario"]["suite_policy_path"])
    route_certification_path = _resolve_manifest_path(
        payload["scenario"]["route_certification_path"]
    )
    seed_sets_path = _resolve_manifest_path(payload["seed_policy"]["seed_sets_path"])
    assert sha256_file(suite_policy_path) == payload["scenario"]["suite_policy_sha256"]
    assert (
        sha256_file(route_certification_path) == payload["scenario"]["route_certification_sha256"]
    )
    assert sha256_file(seed_sets_path) == payload["seed_policy"]["seed_sets_sha256"]
    assert payload["metrics"]["snqi_claim_policy"] == "advisory_no_ranking"

    contract = payload["stress_smoke_contract"]
    pinned_assets = contract["pinned_assets"]
    for key in ("seed_sets", "route_certification"):
        path = _resolve_manifest_path(pinned_assets[f"{key}_path"])
        assert sha256_file(path) == pinned_assets[f"{key}_sha256"]

    for asset in contract["scenario_sources"]:
        path = _resolve_manifest_path(asset["path"])
        assert sha256_file(path) == asset["sha256"]

    observed_hybrid_keys: list[str] = []
    for asset in contract["hybrid_configs"]:
        path = _resolve_manifest_path(asset["path"])
        assert sha256_file(path) == asset["sha256"]
        observed_hybrid_keys.append(asset["planner_key"])
    assert tuple(observed_hybrid_keys) == EXPECTED_HYBRID_ARMS


@pytest.mark.parametrize("status_field", ("status", "readiness_status", "availability_status"))
@pytest.mark.parametrize("status", FORBIDDEN_STATUSES)
def test_forbidden_row_statuses_are_never_admitted(status_field: str, status: str) -> None:
    markers = _status_markers({status_field: status}, "stress-row")
    assert markers
    assert markers[0][1] == status


def test_runtime_fallback_marker_is_never_admitted() -> None:
    fallback_row = {
        "status": "ok",
        "algorithm_metadata": {
            "status": "ok",
            "planner_runtime": {"fallback_count": 1},
        },
    }
    assert _status_markers(fallback_row, "stress-row")
    assert runtime_fallback_or_degraded_marker(
        fallback_row["algorithm_metadata"]["planner_runtime"]
    ) == ("fallback_count", "1")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selected_source", "all_candidates_rejected"),
        ("selected_source", "static_reorient"),
        ("planner_mode", "EMERGENCY_STOP"),
        ("planner_mode", "REORIENT"),
    ],
)
def test_legacy_emergency_paths_are_never_admitted(field: str, value: str) -> None:
    runtime = {"fallback_count": 0, "emergency_stop_count": 0, field: value}

    assert _emergency_stop_marker(runtime) is not None
    assert _status_markers(
        {"status": "ok", "algorithm_metadata": {"planner_runtime": runtime}},
        "stress-row",
    )


def test_positive_emergency_stop_count_is_rejected() -> None:
    runtime = {"fallback_count": 0, "emergency_stop_count": 1}

    assert _emergency_stop_marker(runtime) == ("emergency_stop_count", "1")
    assert _status_markers(
        {"status": "ok", "algorithm_metadata": {"planner_runtime": runtime}},
        "stress-row",
    )


def test_positive_emergency_stop_count_is_rejected_even_for_evaluated_candidates() -> None:
    runtime = {
        "fallback_count": 0,
        "emergency_stop_count": 1,
        "last_decision": {
            "planner_mode": "NORMAL",
            "selected_source": "dynamic_window",
            "candidate_evaluated": True,
            "candidate_count": 3,
        },
    }

    assert _emergency_stop_marker(runtime) == ("emergency_stop_count", "1")
    assert _status_markers(
        {"status": "ok", "algorithm_metadata": {"planner_runtime": runtime}},
        "stress-row",
    )


def test_positive_fallback_count_remains_forbidden() -> None:
    runtime = {
        "fallback_count": 1,
        "emergency_stop_count": 0,
        "last_decision": {"planner_mode": "NORMAL", "selected_source": "dynamic_window"},
    }

    assert runtime_fallback_or_degraded_marker(runtime) == ("fallback_count", "1")
    assert _status_markers(
        {"status": "ok", "algorithm_metadata": {"planner_runtime": runtime}},
        "stress-row",
    )
