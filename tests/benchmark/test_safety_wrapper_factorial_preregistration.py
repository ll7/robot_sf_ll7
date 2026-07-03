"""Tests for issue #3501 safety-wrapper factorial pre-registration harness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.safety_wrapper_factorial_preregistration import (
    EXPECTED_WRAPPER_ARMS,
    build_preregistration_plan,
    check_planned_rows,
    load_factorial_preregistration_config,
    validate_factorial_preregistration_config,
    write_preregistration_plan,
)

CONFIG_PATH = Path(
    "configs/benchmarks/issue_3501_safety_wrapper_factorial_preregistration_cpu_smoke.yaml"
)


def _config() -> dict[str, object]:
    return load_factorial_preregistration_config(CONFIG_PATH)


def test_preregistration_plan_has_paired_off_on_rows_with_identical_policy() -> None:
    """The CPU-smoke row pair differs only by wrapper arm/config."""

    plan = build_preregistration_plan(_config())

    assert plan["benchmark_evidence"] is False
    assert plan["status"] == "planned_rows_cpu_smoke_only"
    assert plan["row_count"] == 2
    assert plan["pair_check"]["complete"] is True
    assert plan["pair_check"]["expected_wrapper_arms"] == list(EXPECTED_WRAPPER_ARMS)

    rows = plan["rows"]
    assert {row["wrapper_arm"] for row in rows} == {"wrapper_off", "wrapper_on"}
    assert {row["planner"] for row in rows} == {"orca"}
    assert {row["scenario_family"] for row in rows} == {"francis2023_blind_corner_cpu_smoke"}
    assert {row["scenario_id"] for row in rows} == {"francis2023_blind_corner"}
    assert {row["seed"] for row in rows} == {111}
    assert len({row["policy_fingerprint"] for row in rows}) == 1


def test_wrapper_on_thresholds_are_checked_by_runtime_validator() -> None:
    """Threshold drift fails closed through validate_runtime_config."""

    config = _config()
    config["wrapper_arms"][1]["safety_wrapper"]["ttc_veto_threshold_s"] = 1.25

    with pytest.raises(ValueError, match="predeclared ablation config"):
        build_preregistration_plan(config)


def test_wrapper_off_on_arm_identity_must_match_enabled_flag() -> None:
    """Swapping arm identity fails before any planned rows are emitted."""

    config = _config()
    config["wrapper_arms"][0]["safety_wrapper"]["arm_key"] = "wrapper_on"

    with pytest.raises(ValueError, match="enabled=False requires arm_key='wrapper_off'"):
        build_preregistration_plan(config)


def test_pair_check_rejects_policy_drift_within_wrapper_pair() -> None:
    """A paired contrast is invalid if policy identity changes across arms."""

    plan = build_preregistration_plan(_config())
    rows = list(plan["rows"])
    rows[1] = dict(rows[1])
    rows[1]["policy_fingerprint"] = "different-policy"

    report = check_planned_rows(rows)

    assert report["complete"] is False
    assert report["policy_mismatches"] == [
        {
            "pairing_key": {
                "study_id": "issue_3501_safety_wrapper_factorial_preregistration_cpu_smoke_v1",
                "planner": "orca",
                "scenario_family": "francis2023_blind_corner_cpu_smoke",
                "scenario_id": "francis2023_blind_corner",
                "seed": 111,
            },
            "field": "policy_fingerprint",
        }
    ]


def test_pair_check_reports_invalid_and_incomplete_rows() -> None:
    """Planned rows must contain wrapper arm and complete off/on pairs."""

    invalid_report = check_planned_rows(
        [
            {
                "study_id": "study",
                "planner": "orca",
                "scenario_family": "family",
                "scenario_id": "scenario",
                "seed": 111,
            }
        ]
    )
    assert invalid_report["complete"] is False
    assert invalid_report["invalid_rows"] == [{"row_index": 0, "fields": ["wrapper_arm"]}]

    incomplete_report = check_planned_rows(
        [
            {
                "study_id": "study",
                "planner": "orca",
                "policy_fingerprint": "same",
                "scenario_family": "family",
                "scenario_id": "scenario",
                "seed": 111,
                "wrapper_arm": "wrapper_off",
            }
        ]
    )
    assert incomplete_report["complete"] is False
    assert incomplete_report["incomplete_pairs"] == [
        {
            "pairing_key": {
                "study_id": "study",
                "planner": "orca",
                "scenario_family": "family",
                "scenario_id": "scenario",
                "seed": 111,
            },
            "wrapper_arms": ["wrapper_off"],
        }
    ]


def test_write_preregistration_plan_outputs_deterministic_json(tmp_path: Path) -> None:
    """The CLI helper writes the reviewable planned-row packet."""

    plan = build_preregistration_plan(_config())
    path = write_preregistration_plan(plan, tmp_path)

    assert path.name == "issue_3501_safety_wrapper_factorial_preregistration_plan.json"
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert '"pair_check"' in path.read_text(encoding="utf-8")


def test_cli_accepts_output_directory_outside_repo(tmp_path: Path) -> None:
    """The CLI can write validation output outside the repository."""
    out_dir = tmp_path / "issue3501-plan"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark/build_issue3501_safety_wrapper_factorial_preregistration.py",
            "--config",
            str(CONFIG_PATH),
            "--out",
            str(out_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "pair_check.complete=True" in completed.stdout
    assert (out_dir / "issue_3501_safety_wrapper_factorial_preregistration_plan.json").is_file()


def test_config_rejects_transient_queue_routing_state() -> None:
    """Tracked config must not encode target host or packet lineage state."""

    config = _config()
    config["target_host"] = "imech156-u"

    with pytest.raises(ValueError, match="transient queue-routing state"):
        validate_factorial_preregistration_config(config, config_path=CONFIG_PATH)


def test_config_rejects_missing_source_contract_path() -> None:
    """Source contracts must point at tracked public files."""

    config = _config()
    config["source_contracts"]["ablation_design"] = "configs/research/missing_3501.yaml"

    with pytest.raises(ValueError, match="source_contracts.ablation_design path does not exist"):
        validate_factorial_preregistration_config(config, config_path=CONFIG_PATH)


def test_config_rejects_planner_not_declared_by_source() -> None:
    """CPU-smoke planner identity must resolve against the declared planner source."""

    config = _config()
    config["policy"]["planner_key"] = "missing_planner"

    with pytest.raises(ValueError, match="not declared by planner_source"):
        validate_factorial_preregistration_config(config, config_path=CONFIG_PATH)
