"""Tests for the issue #4244 seven-arm comparison pre-registration contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts.training.run_comparison_matrix_preregistration import (
    EXPECTED_ARM_IDS,
    FINETUNE_MANIFEST_SCHEMA,
    PRETRAIN_MANIFEST_SCHEMA,
    SAC_GATE,
    SAC_GATE_REQUIRED_ISSUE,
    MatrixValidationError,
    build_dry_run_manifest,
    evaluate_sac_offline_pretraining_gate,
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


def test_issue_4244_matrix_has_no_placeholder_training_configs() -> None:
    """All matrix arms point at executable training configs after the PPO-Mamba slice."""
    matrix = load_matrix(CONFIG_PATH)

    placeholder_arms = {
        arm.arm_id for arm in matrix.arms if "/placeholders/" in arm.training_config
    }
    assert placeholder_arms == set()

    ppo_mamba = next(arm for arm in matrix.arms if arm.arm_id == "ppo_mamba")
    assert ppo_mamba.training_config == "configs/training/ppo/issue_4014_ppo_mamba_smoke.yaml"


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


def test_issue_4244_sac_gate_consumes_issue_4245_evidence_and_passes() -> None:
    """The SAC arm consumes the merged issue #4245 evidence and reports a passing gate."""
    matrix = load_matrix(CONFIG_PATH)

    manifest = build_dry_run_manifest(matrix)

    sac_arm = next(arm for arm in manifest["arms"] if arm["id"] == "offline_online_sac")
    gate = sac_arm["offline_pretraining_gate"]
    assert gate["passed"] is True
    assert gate["required_issue"] == SAC_GATE_REQUIRED_ISSUE
    assert gate["offline_checkpoint_sha256"]
    assert gate["finetune_checkpoint_sha256"]
    # With #4245 evidence now valid, SAC is no longer permanently blocked; the only
    # remaining exclusion is the shared missing smoke-contract output (as for other
    # arms), and no #4245 gate reason is emitted.
    assert not any("issue #4245" in reason for reason in sac_arm["excluded_reasons"])


def test_issue_4244_sac_gate_excludes_arm_when_evidence_missing(tmp_path: Path) -> None:
    """A missing issue #4245 evidence file fails closed with a clear exclusion reason."""
    gate = evaluate_sac_offline_pretraining_gate(
        {
            "provenance_chain": "docs/context/evidence/does_not_exist.json",
            "pretrain_manifest_summary": "docs/context/evidence/does_not_exist.json",
            "finetune_manifest_summary": "docs/context/evidence/does_not_exist.json",
        },
        repo_root=tmp_path,
    )

    assert gate["passed"] is False
    assert "not found" in gate["reason"]


def test_issue_4244_sac_gate_fails_closed_on_broken_provenance_chain(tmp_path: Path) -> None:
    """A provenance chain whose checkpoint SHA does not match the manifest fails closed."""
    evidence = _write_sac_gate_evidence(tmp_path, offline_sha_in_chain="tampered")

    gate = evaluate_sac_offline_pretraining_gate(evidence, repo_root=tmp_path)

    assert gate["passed"] is False
    assert "offline checkpoint SHA" in gate["reason"]


def test_issue_4244_sac_gate_passes_on_consistent_synthetic_evidence(tmp_path: Path) -> None:
    """A well-formed, internally consistent synthetic evidence chain passes the gate."""
    evidence = _write_sac_gate_evidence(tmp_path)

    gate = evaluate_sac_offline_pretraining_gate(evidence, repo_root=tmp_path)

    assert gate["passed"] is True
    assert gate["offline_checkpoint_sha256"] == "offline-sha"
    assert gate["finetune_checkpoint_sha256"] == "finetune-sha"


def test_issue_4244_matrix_requires_sac_gate_evidence_pointers(tmp_path: Path) -> None:
    """Matrix validation fails closed if the SAC gate drops its issue #4245 evidence block."""
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    edited = copy.deepcopy(payload)
    for gate in edited["inclusion_gates"]:
        if gate["id"] == SAC_GATE:
            gate.pop("evidence", None)
    edited_path = tmp_path / "no_evidence_matrix.yaml"
    edited_path.write_text(yaml.safe_dump(edited), encoding="utf-8")

    with pytest.raises(MatrixValidationError, match="evidence"):
        load_matrix(edited_path)


def _write_sac_gate_evidence(
    tmp_path: Path, *, offline_sha_in_chain: str = "offline-sha"
) -> dict[str, str]:
    """Write a synthetic issue #4245 evidence triplet and return relative pointers."""
    evidence_dir = Path("docs/context/evidence/issue_4245_synthetic")
    (tmp_path / evidence_dir).mkdir(parents=True, exist_ok=True)
    chain = {
        "issue": SAC_GATE_REQUIRED_ISSUE,
        "offline_checkpoint_sha256": offline_sha_in_chain,
        "finetune_checkpoint_sha256": "finetune-sha",
    }
    pretrain = {
        "issue": SAC_GATE_REQUIRED_ISSUE,
        "schema_version": PRETRAIN_MANIFEST_SCHEMA,
        "algorithm": "sac",
        "checkpoint_sha256": "offline-sha",
    }
    finetune = {
        "issue": SAC_GATE_REQUIRED_ISSUE,
        "schema_version": FINETUNE_MANIFEST_SCHEMA,
        "algorithm": "sac",
        "parent_checkpoint_sha256": "offline-sha",
        "checkpoint_sha256": "finetune-sha",
    }
    files = {
        "provenance_chain": ("provenance_chain.json", chain),
        "pretrain_manifest_summary": ("pretrain_manifest_summary.json", pretrain),
        "finetune_manifest_summary": ("finetune_manifest_summary.json", finetune),
    }
    pointers: dict[str, str] = {}
    for key, (name, body) in files.items():
        rel = evidence_dir / name
        (tmp_path / rel).write_text(json.dumps(body), encoding="utf-8")
        pointers[key] = rel.as_posix()
    return pointers


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
