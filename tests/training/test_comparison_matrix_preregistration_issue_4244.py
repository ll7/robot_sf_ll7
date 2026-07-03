"""Tests for the issue #4244 seven-arm comparison pre-registration contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.training.run_comparison_matrix_preregistration import (
    EXPECTED_ARM_IDS,
    MatrixValidationError,
    build_dry_run_manifest,
    load_matrix,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT / "configs/training/comparison_matrix/issue_4244_seven_arm_preregistration.yaml"
)


def test_issue_4244_matrix_declares_exact_seven_arm_roster() -> None:
    """The matrix freezes the seven intended learned-policy arms."""
    matrix = load_matrix(CONFIG_PATH)

    assert tuple(arm.arm_id for arm in matrix.arms) == EXPECTED_ARM_IDS
    assert matrix.payload["comparison"]["arms_expected"] == 7
    assert matrix.payload["comparison"]["stop_rules"]["early_stop_allowed"] is False
    assert matrix.payload["comparison"]["queue_plan"]["submit_in_this_pr"] is False


def test_issue_4244_matrix_uses_identical_budget_refs_for_every_arm() -> None:
    """Every arm points at the same shared budget instead of per-arm overrides."""
    matrix = load_matrix(CONFIG_PATH)

    budgets = {arm.arm_id: arm.budget_ref for arm in matrix.arms}
    assert set(budgets.values()) == {"shared_budget"}
    assert matrix.shared_budget["seeds"] == [123, 231, 777, 992, 1337]
    assert matrix.shared_budget["total_timesteps"] == 15000000


def test_issue_4244_matrix_bounds_placeholder_training_config() -> None:
    """Only PPO-Mamba lacks a checked-in training config on origin/main."""
    matrix = load_matrix(CONFIG_PATH)

    placeholder_arms = {
        arm.arm_id for arm in matrix.arms if "/placeholders/" in arm.training_config
    }
    assert placeholder_arms == {"ppo_mamba"}


def test_issue_4244_matrix_requires_sac_after_issue_4245_gate() -> None:
    """Offline-online SAC stays excluded until the standalone offline-pretraining gate exists."""
    matrix = load_matrix(CONFIG_PATH)

    sac_arm = next(arm for arm in matrix.arms if arm.arm_id == "offline_online_sac")
    assert "sac_after_issue_4245_standalone_offline_pretraining" in sac_arm.required_gates


def test_issue_4244_matrix_rejects_missing_sac_gate(tmp_path: Path) -> None:
    """The checker fails closed if SAC can enter without the issue #4245 gate."""
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    edited = copy.deepcopy(payload)
    for arm in edited["arms"]:
        if arm["id"] == "offline_online_sac":
            arm["required_gates"].remove("sac_after_issue_4245_standalone_offline_pretraining")
    edited_path = tmp_path / "bad_matrix.yaml"
    edited_path.write_text(yaml.safe_dump(edited), encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="issue #4245 SAC gate"):
        load_matrix(edited_path)


def test_issue_4244_matrix_rejects_symlinked_scenario_config(tmp_path: Path) -> None:
    """Matrix source paths must not traverse symlinks out of the repo root."""

    payload = _temp_matrix_payload(tmp_path)
    external = tmp_path / "external.yaml"
    external.write_text("scenario: external\n", encoding="utf-8")
    scenario_link = tmp_path / "configs/scenarios/symlink.yaml"
    scenario_link.parent.mkdir(parents=True, exist_ok=True)
    scenario_link.symlink_to(external)
    payload["comparison"]["shared_budget"]["scenario_config"] = "configs/scenarios/symlink.yaml"
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="symlinks"):
        load_matrix(matrix_path, repo_root=tmp_path)


def test_issue_4244_matrix_rejects_directory_training_config(tmp_path: Path) -> None:
    """Directory-valued arm configs are not silently accepted as source files."""

    payload = _temp_matrix_payload(tmp_path)
    bad_dir = tmp_path / "configs/training/bad_dir"
    bad_dir.mkdir(parents=True)
    payload["arms"][0]["training_config"] = "configs/training/bad_dir"
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="training_config not found"):
        load_matrix(matrix_path, repo_root=tmp_path)


def test_issue_4244_dry_run_limits_to_two_arms_and_does_not_execute_training() -> None:
    """CPU dry-run produces a bounded manifest and records explicit inclusion-gate exclusions."""
    matrix = load_matrix(CONFIG_PATH)

    manifest = build_dry_run_manifest(matrix, limit_arms=2)

    assert manifest["training_executed"] is False
    assert manifest["slurm_or_gpu_submitted"] is False
    assert manifest["selected_arm_count"] == 2
    assert [arm["id"] for arm in manifest["arms"]] == ["ppo_baseline", "recurrent_ppo_lstm"]
    assert all(arm["queue_entry"]["status"] == "planned_not_submitted" for arm in manifest["arms"])
    assert all(arm["excluded_reasons"] for arm in manifest["arms"])


def test_issue_4244_cli_writes_two_arm_dry_run_manifest(tmp_path: Path) -> None:
    """The entry point writes the requested dry-run manifest without Slurm submission."""
    output = tmp_path / "dry_run_manifest.json"

    exit_code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--dry-run",
            "--limit-arms",
            "2",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "training_comparison_matrix_dry_run.v1"
    assert payload["selected_arm_count"] == 2
    assert payload["training_executed"] is False


def _temp_matrix_payload(tmp_path: Path) -> dict[str, object]:
    payload = copy.deepcopy(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    scenario_path = tmp_path / payload["comparison"]["shared_budget"]["scenario_config"]
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text("scenario: unit\n", encoding="utf-8")
    for arm in payload["arms"]:
        training_config = arm["training_config"]
        if "/placeholders/" in training_config:
            continue
        config_path = tmp_path / training_config
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("policy_id: unit\n", encoding="utf-8")
    return payload
