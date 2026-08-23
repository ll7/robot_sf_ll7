"""Acceptance-level regression tests for the fixed hybrid stress contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.release_acceptance import validate_diagnostic_stress_smoke_acceptance
from robot_sf.benchmark.release_protocol import (
    STRESS_SMOKE_EXPECTED_SCENARIO_IDS,
    load_release_manifest,
    validate_stress_smoke_runtime_identity,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / (
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_hybrid_stress_smoke_v0_1.yaml"
)
SOURCE_COMMIT = "a" * 40
CONFIG_HASH = "c" * 16


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _row(*, algo: str, scenario_id: str, seed: int) -> dict[str, Any]:
    return {
        "algo": algo,
        "config_hash": CONFIG_HASH,
        "git_hash": SOURCE_COMMIT,
        "horizon": 600,
        "scenario_id": scenario_id,
        "seed": seed,
        "status": "success",
        "algorithm_metadata": {
            "algorithm": algo,
            "canonical_algorithm": algo,
            "planner_kinematics": {
                "robot_kinematics": "differential_drive",
                "scenario_kinematics": ["differential_drive"],
            },
            "planner_contract": {
                "planner_id": algo,
                "action_contract": {"active_robot_kinematics": "differential_drive"},
            },
        },
        "provenance": {
            "git_hash": SOURCE_COMMIT,
            "config_hash": CONFIG_HASH,
            "config_identity": {"algo": algo},
        },
        "result_provenance": {
            "config_hash": CONFIG_HASH,
            "repo_commit": SOURCE_COMMIT,
            "scenario_id": scenario_id,
            "seed": seed,
            "simulator_settings": {"dt": 0.1, "horizon": 600},
        },
    }


@pytest.fixture
def stress_fixture(tmp_path: Path) -> tuple[Path, Any, Any]:
    """Build a complete accepted 14-arm stress campaign with tiny JSONL files."""
    manifest = load_release_manifest(MANIFEST_PATH)
    campaign_config = load_campaign_config(manifest.canonical_campaign_config_path)
    scenarios = _load_campaign_scenarios(campaign_config)
    scenario_ids = tuple(str(scenario["name"]) for scenario in scenarios)
    assert scenario_ids == STRESS_SMOKE_EXPECTED_SCENARIO_IDS

    root = tmp_path / "campaign"
    _write_json(root / "campaign_manifest.json", {"git": {"commit": SOURCE_COMMIT}})
    _write_json(root / "manifest.json", {"git_hash": SOURCE_COMMIT})
    _write_json(root / "run_meta.json", {"repo": {"commit": SOURCE_COMMIT}})
    runs: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    for planner in campaign_config.planners:
        arm = f"{planner.key}__differential_drive"
        episodes_path = root / "runs" / arm / "episodes.jsonl"
        summary_path = root / "runs" / arm / "summary.json"
        rows = [
            _row(algo=planner.algo, scenario_id=scenario_id, seed=116)
            for scenario_id in scenario_ids
        ]
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        _write_json(summary_path, {"status": "ok"})
        runs.append(
            {
                "planner": {
                    "key": planner.key,
                    "algo": planner.algo,
                    "kinematics": "differential_drive",
                    "horizon": 600,
                    "dt": 0.1,
                },
                "status": "ok",
                "episodes_path": f"runs/{arm}/episodes.jsonl",
                "summary_path": f"runs/{arm}/summary.json",
                "summary": {
                    "status": "ok",
                    "total_jobs": 5,
                    "written": 5,
                    "successful_jobs": 5,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                    "failures": [],
                },
            }
        )
        planner_rows.append(
            {
                "planner_key": planner.key,
                "kinematics": "differential_drive",
                "status": "ok",
                "benchmark_success": "true",
                "episodes": 5,
                "failed_jobs": 0,
            }
        )
    _write_json(
        root / "reports" / "campaign_summary.json",
        {
            "campaign": {
                "git_hash": SOURCE_COMMIT,
                "benchmark_success": True,
                "status": "benchmark_success",
                "evidence_status": "valid",
                "campaign_execution_status": "completed",
            },
            "runs": runs,
            "planner_rows": planner_rows,
        },
    )
    return root, manifest, campaign_config


def _acceptance(root: Path, manifest: Any, campaign_config: Any) -> dict[str, Any]:
    return validate_diagnostic_stress_smoke_acceptance(
        root,
        manifest=manifest,
        campaign_config=campaign_config,
        expected_source_commit=SOURCE_COMMIT,
    )


def _first_row_path(root: Path, planner_key: str) -> Path:
    return root / "runs" / f"{planner_key}__differential_drive" / "episodes.jsonl"


def test_complete_stress_campaign_is_admitted(stress_fixture: tuple[Path, Any, Any]) -> None:
    root, manifest, campaign_config = stress_fixture

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "valid", report["blockers"]
    assert report["diagnostic_success"] is True
    assert report["observed_episode_rows"] == 70


def test_real_episode_schema_may_omit_repeated_kinematics_aliases(
    stress_fixture: tuple[Path, Any, Any],
) -> None:
    root, manifest, campaign_config = stress_fixture
    path = _first_row_path(root, "prediction_planner")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        planner_metadata = row["algorithm_metadata"]
        planner_metadata.pop("planner_kinematics")
        planner_metadata["planner_contract"]["action_contract"].pop("active_robot_kinematics")
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "valid", report["blockers"]


@pytest.mark.parametrize(
    "mutation",
    ("planner", "kinematics", "config", "source", "emergency", "horizon", "dt", "status"),
)
def test_row_alias_and_runtime_bypasses_fail_closed(
    stress_fixture: tuple[Path, Any, Any], mutation: str
) -> None:
    root, manifest, campaign_config = stress_fixture
    path = _first_row_path(root, "prediction_planner")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    if mutation == "planner":
        row["algorithm_metadata"]["canonical_algorithm"] = "goal"
    elif mutation == "kinematics":
        row["algorithm_metadata"]["planner_kinematics"]["robot_kinematics"] = "holonomic"
    elif mutation == "config":
        row["result_provenance"]["config_hash"] = "d" * 16
    elif mutation == "source":
        row["provenance"]["git_hash"] = "b" * 40
    elif mutation == "emergency":
        row["algorithm_metadata"]["runtime"] = {"last_decision": {"planner_mode": "EMERGENCY_STOP"}}
    elif mutation == "horizon":
        row["result_provenance"]["simulator_settings"]["horizon"] = 599
    elif mutation == "dt":
        row["result_provenance"]["simulator_settings"]["dt"] = 0.2
    else:
        row["status"] = "yes"
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"
    assert report["diagnostic_success"] is False


def test_external_and_cross_campaign_paths_fail_closed(
    stress_fixture: tuple[Path, Any, Any], tmp_path: Path
) -> None:
    root, manifest, campaign_config = stress_fixture
    summary_path = root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"][0]["episodes_path"] = str(tmp_path / "other-campaign" / "episodes.jsonl")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"
    assert any("episodes_path rejected" in blocker for blocker in report["blockers"])


def test_unrelated_nested_business_metadata_does_not_trigger_emergency_marker(
    stress_fixture: tuple[Path, Any, Any],
) -> None:
    root, manifest, campaign_config = stress_fixture
    path = _first_row_path(root, "prediction_planner")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["scenario_params"] = {"planner_mode": "REORIENT", "selected_source": "static_reorient"}
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "valid", report["blockers"]


@pytest.mark.parametrize(
    ("field", "value"),
    (("campaign_execution_status", "interrupted"), ("evidence_status", "blocked")),
)
def test_campaign_status_gate_requires_completed_valid_evidence(
    stress_fixture: tuple[Path, Any, Any], field: str, value: str
) -> None:
    root, manifest, campaign_config = stress_fixture
    summary_path = root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["campaign"][field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"


def test_private_runtime_identity_requires_exact_launch_pin_and_clean_worktree() -> None:
    manifest = load_release_manifest(MANIFEST_PATH)
    runtime_commit = "b" * 40

    missing_pin = validate_stress_smoke_runtime_identity(
        manifest,
        current_source_commit=runtime_commit,
        require_launch_pin=True,
        worktree_clean=True,
        require_clean_worktree=True,
    )
    dirty = validate_stress_smoke_runtime_identity(
        manifest,
        current_source_commit=runtime_commit,
        launch_expected_source_commit=runtime_commit,
        require_launch_pin=True,
        worktree_clean=False,
        require_clean_worktree=True,
    )

    assert missing_pin["status"] == "invalid"
    assert dirty["status"] == "invalid"
