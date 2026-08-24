"""Contract tests for the pre-release S30/H600 hybrid stress smoke."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.benchmark import release_protocol
from robot_sf.benchmark.camera_ready._config import (
    _load_campaign_scenarios,
    _resolved_seed_inventory,
)
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.fallback_policy import runtime_fallback_or_degraded_marker
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.release_acceptance import (
    _emergency_stop_marker,
    _status_markers,
    validate_diagnostic_stress_smoke_source_provenance,
)
from robot_sf.benchmark.release_protocol import (
    build_release_provenance,
    build_resolved_release_manifest,
    load_release_manifest,
    validate_release_manifest,
    validate_stress_smoke_runtime_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / (
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_hybrid_stress_smoke_v0_1.yaml"
)
SOURCE_COMMIT = "bc1294a19a837c4d4b9ef9086a2be4ca42dd85f3"
EXPECTED_SCENARIOS = (
    "classic_urban_crossing_medium",
    "classic_cross_trap_high",
    "classic_doorway_high",
    "francis2023_exiting_elevator",
    "francis2023_robot_crowding",
)
EXPECTED_HYBRID_ARMS = (
    "scenario_adaptive_hybrid_orca_v2_bottleneck_yield",
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
    assert re.fullmatch(r"[0-9a-f]{40}", contract["review_base_commit"])
    assert contract["review_base_commit"] == SOURCE_COMMIT
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


def test_release_validation_rejects_stress_asset_hash_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = load_release_manifest(MANIFEST_PATH)
    campaign_config = load_campaign_config(manifest.canonical_campaign_config_path)
    original_sha256_file = release_protocol._sha256_file
    target = manifest.stress_smoke_scenario_source_pins[0].path.resolve()

    def _tampered_hash(path: Path) -> str:
        if path.resolve() == target:
            return "0" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(release_protocol, "_sha256_file", _tampered_hash)
    report = validate_release_manifest(manifest, campaign_config=campaign_config)

    assert report["status"] == "invalid"
    assert any(
        "stress_smoke_contract.scenario_sources hash does not match pinned asset" in problem
        for problem in report["problems"]
    )


def test_stress_asset_resolver_rejects_symlink_escape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scenario/hybrid pins cannot escape the checkout through a symlink."""
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "asset.yaml").write_text("payload: outside\n", encoding="utf-8")
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(release_protocol, "get_repository_root", lambda: repo)

    with pytest.raises(ValueError, match="must not contain symlink components"):
        release_protocol._resolve_stress_contract_file(
            repo / "manifest.yaml", "escape/asset.yaml", "scenario source"
        )


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


def test_nested_emergency_stop_count_is_never_admitted() -> None:
    runtime = {
        "fallback_count": 0,
        "last_decision": {
            "planner_mode": "NORMAL",
            "selected_source": "dynamic_window",
            "emergency_stop_count": 1,
        },
    }

    assert _emergency_stop_marker(runtime) == ("last_decision.emergency_stop_count", "1")
    assert _status_markers(
        {"status": "ok", "algorithm_metadata": {"planner_runtime": runtime}},
        "stress-row",
    )


def test_unwrapped_planner_runtime_emergency_marker_is_never_admitted() -> None:
    payload = {
        "status": "ok",
        "planner_runtime": {
            "last_decision": {
                "planner_mode": "NORMAL",
                "selected_source": "dynamic_window",
                "emergency_stop_count": 1,
            }
        },
    }

    assert _emergency_stop_marker(payload) == (
        "planner_runtime.last_decision.emergency_stop_count",
        "1",
    )
    assert _status_markers(payload, "stress-row")


def test_unwrapped_planner_runtime_fallback_marker_is_never_admitted() -> None:
    payload = {"status": "ok", "planner_runtime": {"fallback_count": 1}}

    assert _status_markers(payload, "stress-row") == [
        ("stress-row.planner_runtime.fallback_count", "1")
    ]


@pytest.mark.parametrize(
    "payload",
    (
        {"planner_runtime": {"fallback_count": "1"}},
        {"planner_runtime": {"fallback_count": "not-a-number"}},
        {"planner_runtime": {"fallback_triggered": "true"}},
        {"planner_runtime": {"degraded": "false"}},
        {"planner_runtime": {"fallback_or_degraded": 1}},
        {"planner_runtime": {"emergency_stop": "true"}},
        {"planner_runtime": {"emergency_stop_count": "0"}},
        {"fallback": "true"},
        {"fallback_used": "false"},
    ),
)
def test_malformed_runtime_marker_encodings_are_never_admitted(
    payload: dict[str, Any],
) -> None:
    """String and non-boolean marker encodings fail closed instead of disappearing."""
    assert _status_markers(payload, "stress-row")


def test_native_rows_preserve_metric_level_missing_values() -> None:
    """Metric availability values are not runtime fallback/degraded markers."""
    payload = {
        "status": "success",
        "metrics": {
            "route_efficiency": "unavailable",
            "snqi": None,
            "availability_status": "missing",
        },
        "planner_runtime": {
            "fallback_count": 0,
            "fallback_triggered": False,
            "degraded": False,
            "fallback_or_degraded": False,
        },
    }

    assert _status_markers(payload, "stress-row") == []


def test_stress_runtime_identity_uses_launch_pin_not_review_base() -> None:
    manifest = load_release_manifest(MANIFEST_PATH)
    runtime_commit = "a" * 40
    admitted = validate_stress_smoke_runtime_identity(
        manifest,
        current_source_commit=runtime_commit,
        launch_expected_source_commit=runtime_commit,
    )
    assert admitted["status"] == "valid"
    assert admitted["runtime_source_commit"] == runtime_commit
    assert admitted["review_base_commit"] == SOURCE_COMMIT

    rejected = validate_stress_smoke_runtime_identity(
        manifest,
        current_source_commit=runtime_commit,
        launch_expected_source_commit="b" * 40,
    )
    assert rejected["status"] == "invalid"


def test_stress_resolved_provenance_contains_pinned_assets_and_runtime_commit() -> None:
    manifest = load_release_manifest(MANIFEST_PATH)
    runtime_commit = "a" * 40
    resolved = build_resolved_release_manifest(
        manifest,
        source_commit=runtime_commit,
    )
    contract = resolved["provenance"]["stress_smoke_contract"]
    assert resolved["provenance"]["source_commit"] == runtime_commit
    assert contract["review_base_commit"] == SOURCE_COMMIT
    assert len(contract["scenario_sources"]) == 5
    assert len(contract["hybrid_configs"]) == 4

    provenance = build_release_provenance(
        manifest,
        campaign_root=REPO_ROOT / "output" / "stress-smoke",
        invoked_command="test",
        source_commit=runtime_commit,
    )
    assert provenance["source_commit"] == runtime_commit
    assert provenance["stress_smoke_contract"] == contract


def test_stress_source_provenance_rejects_mixed_campaign_rows(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    (root / "reports").mkdir(parents=True)
    (root / "runs" / "goal__differential_drive").mkdir(parents=True)
    commit = "a" * 40
    wrong = "b" * 40
    for name, payload in (
        ("campaign_manifest.json", {"git": {"commit": commit}}),
        ("manifest.json", {"git_hash": commit}),
        ("run_meta.json", {"repo": {"commit": commit}}),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    episodes = root / "runs" / "goal__differential_drive" / "episodes.jsonl"
    episodes.write_text(
        json.dumps({"git_hash": wrong, "scenario_id": "s", "seed": 116}) + "\n",
        encoding="utf-8",
    )
    (root / "reports" / "campaign_summary.json").write_text(
        json.dumps(
            {
                "campaign": {"git_hash": commit},
                "runs": [
                    {
                        "episodes_path": "runs/goal__differential_drive/episodes.jsonl",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = validate_diagnostic_stress_smoke_source_provenance(
        root,
        expected_source_commit=commit,
    )
    assert report["status"] == "invalid"
    assert any("campaign provenance" in blocker for blocker in report["blockers"])
    assert report["observed_source_commits"] == [commit, wrong]
