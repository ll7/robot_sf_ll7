"""Tests for the oracle-imitation warm-start readiness preflight (issue #1496)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.training.oracle_imitation_warm_start_readiness import (
    PrerequisitesNotReadyError,
    WarmStartReadinessError,
    check_warm_start_readiness,
    load_readiness_manifest,
)
from scripts.validation.check_oracle_imitation_warm_start_readiness import main as check_cli_main

# Real, checked-in #1496 readiness manifest. It references the intentionally NOT
# training-ready #1397 dataset launch packet (a `:pending` durable trace URI), which is
# exactly the current state issue #1496 is gated on until #1470 publishes the durable dataset.
_REAL_MANIFEST = "configs/training/ppo_imitation/oracle_warm_start_readiness_issue_1496.yaml"


def _write_manifest(tmp_path: Path, manifest: dict[str, object]) -> Path:
    path = tmp_path / "readiness.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _ready_manifest(tmp_path: Path) -> dict[str, object]:
    """A fully-satisfied manifest backed by a training-ready packet and local config files.

    The dataset packet is a minimal stand-in with concrete durable trace URIs so the
    canonical launch-packet validator reports ``training_ready`` without needing the real,
    intentionally-pending #1397 packet.
    """
    packet_path = _write_training_ready_packet(tmp_path)
    trace_registry_path = _write_training_ready_trace_registry(tmp_path)
    warm_start = tmp_path / "bc.yaml"
    warm_start.write_text("policy_id: bc\n", encoding="utf-8")
    baseline = tmp_path / "rl_only.yaml"
    baseline.write_text("policy_id: rl_only\n", encoding="utf-8")
    contract = tmp_path / "split_contract.md"
    contract.write_text("# split contract\n", encoding="utf-8")
    return {
        "schema_version": "oracle-imitation-warm-start-readiness.v1",
        "experiment_id": "unit_test_warm_start",
        "dataset_launch_packet": str(packet_path),
        "trace_uri_registry": str(trace_registry_path),
        "warm_start_config": str(warm_start),
        "baseline_config": str(baseline),
        "split_contract": str(contract),
    }


def _write_training_ready_packet(tmp_path: Path) -> Path:
    """Write a launch packet that passes the canonical validator with training_ready=True."""
    packet = {
        "schema_version": "oracle-imitation-launch-packet.v1",
        "dataset_id": "unit_test_oracle_imitation",
        "source_candidate": "hybrid_rule_v3_static_margin0_waypoint2",
        "source_candidate_config": "configs/policy_search/candidates/"
        "hybrid_rule_v3_static_margin0_waypoint2.yaml",
        "source_report": "docs/context/policy_search/reports/"
        "2026-04-30_best_non_learning_local_policy_report.md",
        "split_contract": "docs/context/policy_search/contracts/oracle_imitation_dataset_split.md",
        "scenario_source": "configs/policy_search/nominal_sanity_matrix.yaml",
        "scenario_ids": ["planner_sanity_simple", "classic_crossing_low"],
        "seeds_by_split": {
            "train": [201, 202],
            "validation": [101, 102, 103],
            "evaluation": [111, 112, 113],
        },
        "episode_ids_by_split": {
            "train": ["train__planner_sanity_simple__seed201"],
            "validation": ["validation__planner_sanity_simple__seed101"],
            "evaluation": ["evaluation__planner_sanity_simple__seed111"],
        },
        "hard_slice_assignment": [],
        "relabeling_policy": None,
        "generating_commit": "e14e2f8bc2058d9f0e071219629915dd5b5dd5a8",
        "artifact_paths": {
            "train_trace_jsonl_uri": "wandb-artifact://robot-sf/train:v1",
            "validation_trace_jsonl_uri": "wandb-artifact://robot-sf/validation:v1",
            "evaluation_trace_jsonl_uri": "wandb-artifact://robot-sf/evaluation:v1",
            "trace_source_manifest_uri": "wandb-artifact://robot-sf/manifest:v1",
        },
        # Durable collection destinations are required by the canonical validator
        # (oracle_imitation_launch_packet, #1470); they must be durable artifact URIs and
        # must not depend on the gitignored worktree-local ``output`` directory.
        "collection_roots": {
            "log_root": "wandb-artifact://robot-sf/logs:v1",
            "dataset_output_root": "wandb-artifact://robot-sf/raw-traces:v1",
            "manifest_destination": "wandb-artifact://robot-sf/manifest-dest:v1",
        },
        "checksums": {},
    }
    path = tmp_path / "packet.yaml"
    path.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")
    return path


def _write_training_ready_trace_registry(tmp_path: Path) -> Path:
    """Write trace-URI registry that passes the canonical validator with training_ready=True."""
    registry = {
        "schema_version": "oracle-trace-uri-registry.v1",
        "dataset_id": "unit_test_oracle_imitation",
        "traces": [
            {
                "split": "train",
                "trace_id": "train__unit_test",
                "uri": "wandb-artifact://robot-sf/oracle-imitation/train:v1",
                "sha256": "a" * 64,
                "retrieval_status": "resolvable",
            },
            {
                "split": "validation",
                "trace_id": "validation__unit_test",
                "uri": "wandb-artifact://robot-sf/oracle-imitation/validation:v1",
                "sha256": "b" * 64,
                "retrieval_status": "resolvable",
            },
            {
                "split": "evaluation",
                "trace_id": "evaluation__unit_test",
                "uri": "wandb-artifact://robot-sf/oracle-imitation/evaluation:v1",
                "sha256": "c" * 64,
                "retrieval_status": "resolvable",
            },
        ],
    }
    path = tmp_path / "trace_registry.yaml"
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    return path


def test_real_issue_1496_manifest_is_blocked_on_dataset() -> None:
    """The checked-in #1496 manifest is blocked because #1397 is not training-ready yet."""
    report = check_warm_start_readiness(Path(_REAL_MANIFEST))

    assert report["status"] == "blocked"
    assert report["experiment_id"] == "issue_1496_oracle_imitation_warm_start_v1"
    # Durable dataset launch packet and trace registry are blocked; configs/contract are present.
    assert len(report["blockers"]) == 2
    assert report["blockers"][0].startswith("dataset_launch_packet not training-ready")
    assert report["blockers"][1].startswith("trace_uri_registry not training-ready")
    assert report["prerequisites"]["dataset_launch_packet"]["ready"] is False
    assert report["prerequisites"]["trace_uri_registry"]["ready"] is False
    assert report["prerequisites"]["warm_start_config"]["ready"] is True
    assert report["prerequisites"]["baseline_config"]["ready"] is True
    assert report["prerequisites"]["finetuning_config"]["ready"] is True
    assert report["prerequisites"]["split_contract"]["ready"] is True


def test_ready_manifest_reports_ready(tmp_path: Path) -> None:
    """A manifest whose dataset is training-ready and configs exist reports ready."""
    manifest = _write_manifest(tmp_path, _ready_manifest(tmp_path))

    report = check_warm_start_readiness(manifest)

    assert report["status"] == "ready"
    assert report["blockers"] == []
    assert report["prerequisites"]["dataset_launch_packet"]["training_ready"] is True
    assert report["prerequisites"]["trace_uri_registry"]["training_ready"] is True


def test_missing_config_is_a_blocker_not_an_error(tmp_path: Path) -> None:
    """A missing required config file is a recorded blocker, not a manifest error."""
    manifest_dict = _ready_manifest(tmp_path)
    manifest_dict["baseline_config"] = "configs/does/not/exist.yaml"
    manifest = _write_manifest(tmp_path, manifest_dict)

    report = check_warm_start_readiness(manifest)

    assert report["status"] == "blocked"
    assert any(b.startswith("baseline_config is not an existing file") for b in report["blockers"])


def test_require_ready_fails_closed_when_blocked(tmp_path: Path) -> None:
    """require_ready turns a blocked manifest into a fail-closed PrerequisitesNotReadyError.

    The specific subclass (not just the base WarmStartReadinessError) is what lets the CLI
    map an unmet-prerequisite gate rejection (exit 1) apart from a malformed manifest (exit 2)
    without a fragile error-message string check.
    """
    manifest_dict = _ready_manifest(tmp_path)
    manifest_dict["warm_start_config"] = "configs/missing.yaml"
    manifest = _write_manifest(tmp_path, manifest_dict)

    with pytest.raises(PrerequisitesNotReadyError, match="prerequisites not ready"):
        check_warm_start_readiness(manifest, require_ready=True)
    # A malformed manifest must NOT raise the prerequisite subclass: keep the two error
    # classes distinct so the CLI exit-code split stays principled.
    assert issubclass(PrerequisitesNotReadyError, WarmStartReadinessError)


def test_blank_dataset_reference_is_a_blocker(tmp_path: Path) -> None:
    """A blank dataset reference is reported as a blocker with a clear message."""
    manifest_dict = _ready_manifest(tmp_path)
    manifest_dict["dataset_launch_packet"] = "   "
    manifest = _write_manifest(tmp_path, manifest_dict)

    report = check_warm_start_readiness(manifest)

    assert report["status"] == "blocked"
    assert any("dataset_launch_packet must be a non-empty" in b for b in report["blockers"])


def test_missing_trace_uri_registry_is_a_blocker(tmp_path: Path) -> None:
    """A missing durable trace registry blocks readiness before warm-start training."""
    manifest_dict = _ready_manifest(tmp_path)
    manifest_dict.pop("trace_uri_registry")
    manifest = _write_manifest(tmp_path, manifest_dict)

    report = check_warm_start_readiness(manifest)

    assert report["status"] == "blocked"
    assert any("trace_uri_registry must be non-empty" in b for b in report["blockers"])
    assert report["prerequisites"]["trace_uri_registry"]["ready"] is False


def test_optional_finetuning_config_omitted_is_ok(tmp_path: Path) -> None:
    """Omitting the optional finetuning config leaves it out of the report, still ready."""
    manifest_dict = _ready_manifest(tmp_path)
    manifest_dict.pop("finetuning_config", None)
    manifest = _write_manifest(tmp_path, manifest_dict)

    report = check_warm_start_readiness(manifest)

    assert report["status"] == "ready"
    assert "finetuning_config" not in report["prerequisites"]


def test_blank_optional_finetuning_config_is_a_blocker(tmp_path: Path) -> None:
    """A present-but-blank optional config is surfaced as a manifest mistake."""
    manifest_dict = _ready_manifest(tmp_path)
    manifest_dict["finetuning_config"] = ""
    manifest = _write_manifest(tmp_path, manifest_dict)

    report = check_warm_start_readiness(manifest)

    assert report["status"] == "blocked"
    assert any("finetuning_config must be a non-empty" in b for b in report["blockers"])


def test_wrong_schema_version_raises(tmp_path: Path) -> None:
    """A wrong schema version is a malformed-manifest error, not a blocker."""
    manifest_dict = _ready_manifest(tmp_path)
    manifest_dict["schema_version"] = "wrong.v0"
    manifest = _write_manifest(tmp_path, manifest_dict)

    with pytest.raises(WarmStartReadinessError, match="schema_version must be"):
        check_warm_start_readiness(manifest)


def test_blank_experiment_id_raises(tmp_path: Path) -> None:
    """A blank experiment_id is a malformed-manifest error."""
    manifest_dict = _ready_manifest(tmp_path)
    manifest_dict["experiment_id"] = "  "
    manifest = _write_manifest(tmp_path, manifest_dict)

    with pytest.raises(WarmStartReadinessError, match="experiment_id must be"):
        check_warm_start_readiness(manifest)


def test_load_readiness_manifest_rejects_non_mapping(tmp_path: Path) -> None:
    """A non-mapping YAML document is rejected at load time."""
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(WarmStartReadinessError, match="must be a YAML mapping"):
        load_readiness_manifest(path)


def test_missing_manifest_file_raises(tmp_path: Path) -> None:
    """A missing manifest file raises a clear error."""
    with pytest.raises(WarmStartReadinessError, match="is not a file"):
        check_warm_start_readiness(tmp_path / "nope.yaml")


def test_cli_blocked_returns_exit_1_on_real_manifest(capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI exits 1 (blocked) on the real, dataset-gated #1496 manifest."""
    exit_code = check_cli_main(["--manifest", _REAL_MANIFEST, "--json"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert '"status": "blocked"' in captured.out


def test_cli_malformed_manifest_returns_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI exits 2 when the manifest is structurally malformed."""
    manifest_dict = _ready_manifest(tmp_path)
    manifest_dict["schema_version"] = "wrong.v0"
    manifest = _write_manifest(tmp_path, manifest_dict)

    exit_code = check_cli_main(["--manifest", str(manifest)])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "schema_version must be" in captured.out


def test_cli_require_ready_returns_exit_1_when_blocked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI exits 1 under --require-ready when a prerequisite blocker remains."""
    manifest_dict = _ready_manifest(tmp_path)
    manifest_dict["baseline_config"] = "configs/missing.yaml"
    manifest = _write_manifest(tmp_path, manifest_dict)

    exit_code = check_cli_main(["--manifest", str(manifest), "--require-ready"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "prerequisites not ready" in captured.out


def test_cli_ready_returns_exit_0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI exits 0 and reports ready for a fully-satisfied manifest."""
    manifest = _write_manifest(tmp_path, _ready_manifest(tmp_path))

    exit_code = check_cli_main(["--manifest", str(manifest)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "readiness: ready" in captured.out
