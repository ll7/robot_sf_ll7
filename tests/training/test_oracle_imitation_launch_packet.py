"""Tests for oracle-imitation launch-packet validation."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from robot_sf.training.oracle_imitation_launch_packet import (
    LaunchPacketError,
    load_launch_packet,
    validate_launch_packet,
)
from scripts.validation.validate_oracle_imitation_launch_packet import main as validate_cli_main


def _write_packet(tmp_path: Path, packet: dict[str, object]) -> Path:
    path = tmp_path / "packet.yaml"
    path.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")
    return path


def _valid_packet(tmp_path: Path) -> dict[str, object]:
    fixture = tmp_path / "fixture.json"
    fixture.write_text('{"status": "dry-run"}\n', encoding="utf-8")
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    return {
        "schema_version": "oracle-imitation-launch-packet.v1",
        "dataset_id": "demo_oracle_imitation",
        "source_candidate": "hybrid_rule_v3_static_margin0_waypoint2",
        "source_candidate_config": "configs/policy_search/candidates/"
        "hybrid_rule_v3_static_margin0_waypoint2.yaml",
        "source_report": "docs/context/policy_search/reports/"
        "2026-04-30_best_non_learning_local_policy_report.md",
        "split_contract": "docs/context/policy_search/contracts/oracle_imitation_dataset_split.md",
        "scenario_source": "configs/policy_search/nominal_sanity_matrix.yaml",
        "scenario_ids": ["planner_sanity_simple", "classic_crossing_low"],
        "seed_set_refs": {
            "manifest": "configs/benchmarks/seed_sets_v1.yaml",
            "validation": "dev",
            "evaluation": "eval",
            "train_excludes": "paper_eval_s20",
        },
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
        "hard_slice_assignment": [
            {
                "slice_id": "stress_slice",
                "split": "validation",
                "predeclared_for_evaluation": False,
            }
        ],
        "relabeling_policy": None,
        "exclusion_rules": ["exclude eval seeds from train"],
        "provenance": "unit-test fixture",
        "created_at": "2026-05-24T08:55:00+00:00",
        "generating_commit": "e14e2f8bc2058d9f0e071219629915dd5b5dd5a8",
        "artifact_paths": {
            "dry_run_fixture": str(fixture),
            "future_dataset_npz_uri": "wandb-artifact://robot-sf/demo:pending",
        },
        "checksums": {str(fixture): digest},
        "collection_roots": {
            "log_root": "wandb-artifact://robot-sf/demo-logs:pending",
            "dataset_output_root": "wandb-artifact://robot-sf/demo-traces:pending",
            "manifest_destination": "wandb-artifact://robot-sf/demo-manifest:pending",
        },
    }


def test_issue_1397_launch_packet_validates() -> None:
    """The checked-in #1397 launch packet should pass the fail-closed preflight."""
    report = validate_launch_packet(
        Path("configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml")
    )

    assert report["status"] == "valid"
    assert report["dataset_id"] == "issue_1397_oracle_imitation_v1"
    assert report["source_candidate"] == "hybrid_rule_v3_static_margin0_waypoint2"
    assert report["scenario_count"] == 6
    assert report["training_ready"] is False
    assert set(report["collection_roots"]) == {
        "log_root",
        "dataset_output_root",
        "manifest_destination",
    }


def test_issue_1397_launch_packet_fails_training_ready_gate() -> None:
    """The checked-in #1397 packet must not pass as downstream training input yet."""
    with pytest.raises(LaunchPacketError, match="training-ready oracle-imitation"):
        validate_launch_packet(
            Path("configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml"),
            require_training_ready=True,
        )


def test_validate_launch_packet_rejects_seed_overlap(tmp_path: Path) -> None:
    """Seed overlap across train/validation/evaluation should fail closed."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["seeds_by_split"]["train"] = [101]

    with pytest.raises(LaunchPacketError, match="seed overlap"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_rejects_unpredeclared_evaluation_hard_slice(
    tmp_path: Path,
) -> None:
    """Hard-slice examples cannot enter evaluation unless predeclared."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["hard_slice_assignment"] = [
        {"slice_id": "stress_slice", "split": "evaluation", "predeclared_for_evaluation": False}
    ]

    with pytest.raises(LaunchPacketError, match="assigns evaluation without predeclaration"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_requires_collection_roots(tmp_path: Path) -> None:
    """A launch packet that omits collection_roots must fail closed."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    del broken["collection_roots"]

    with pytest.raises(LaunchPacketError, match="collection_roots must be a mapping"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_requires_each_collection_root(tmp_path: Path) -> None:
    """Each required collection root (incl. the manifest destination) must be present."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    del broken["collection_roots"]["manifest_destination"]

    with pytest.raises(
        LaunchPacketError,
        match="collection_roots.manifest_destination must be a non-empty string",
    ):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_rejects_non_durable_collection_root(tmp_path: Path) -> None:
    """Collection roots must be durable artifact URIs, not bare local paths."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["collection_roots"]["dataset_output_root"] = "data/oracle_imitation/raw_traces"

    with pytest.raises(LaunchPacketError, match="must be a durable artifact URI"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_rejects_output_collection_root(tmp_path: Path) -> None:
    """Collection roots must never point at the worktree-local output directory."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["collection_roots"]["log_root"] = "output/oracle_imitation/logs"

    with pytest.raises(LaunchPacketError, match="must not depend on worktree-local output"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_rejects_output_artifact_paths(tmp_path: Path) -> None:
    """Launch packets must not depend on worktree-local output artifacts."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["artifact_paths"]["dry_run_fixture"] = "output/local-only.json"
    broken["checksums"] = {"output/local-only.json": "0" * 64}

    with pytest.raises(LaunchPacketError, match="worktree-local output"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_accepts_training_ready_trace_artifacts(tmp_path: Path) -> None:
    """Concrete durable trace pointers satisfy the downstream training gate."""
    packet = _valid_packet(tmp_path)
    packet["artifact_paths"].update(
        {
            "train_trace_jsonl_uri": "wandb-artifact://robot-sf/oracle-imitation/train:v1",
            "validation_trace_jsonl_uri": "wandb-artifact://robot-sf/oracle-imitation/val:v1",
            "evaluation_trace_jsonl_uri": "wandb-artifact://robot-sf/oracle-imitation/eval:v1",
            "trace_source_manifest_uri": "wandb-artifact://robot-sf/oracle-imitation/manifest:v1",
        }
    )
    packet["artifact_paths"]["future_dataset_npz_uri"] = "wandb-artifact://robot-sf/demo-dataset:v1"

    report = validate_launch_packet(
        _write_packet(tmp_path, packet),
        require_training_ready=True,
    )

    assert report["training_ready"] is True


def test_validate_launch_packet_rejects_pending_training_artifacts(tmp_path: Path) -> None:
    """Pending aliases are placeholders, not durable training inputs."""
    packet = _valid_packet(tmp_path)
    packet["artifact_paths"].update(
        {
            "train_trace_jsonl_uri": "wandb-artifact://robot-sf/oracle-imitation/train:pending",
            "validation_trace_jsonl_uri": "wandb-artifact://robot-sf/oracle-imitation/val:v1",
            "evaluation_trace_jsonl_uri": "wandb-artifact://robot-sf/oracle-imitation/eval:v1",
            "trace_source_manifest_uri": "wandb-artifact://robot-sf/oracle-imitation/manifest:v1",
        }
    )

    with pytest.raises(LaunchPacketError, match="must not use pending durable aliases") as exc_info:
        validate_launch_packet(
            _write_packet(tmp_path, packet),
            require_training_ready=True,
        )
    assert "must use concrete durable URIs" not in str(exc_info.value)


def test_validate_launch_packet_cli_reports_json() -> None:
    """The CLI should expose the same validation report for automation."""
    exit_code = validate_cli_main(
        [
            "--config",
            "configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml",
            "--json",
        ]
    )

    assert exit_code == 0


def test_validate_launch_packet_cli_training_ready_gate_fails_closed() -> None:
    """Automation can request the stricter downstream training gate."""
    exit_code = validate_cli_main(
        [
            "--config",
            "configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml",
            "--json",
            "--require-training-ready",
        ]
    )

    assert exit_code == 2


def test_seed_manifest_missing_file_fails_closed(tmp_path: Path) -> None:
    """A seed manifest that does not exist must fail closed."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["seed_set_refs"]["manifest"] = "nonexistent/manifest.yaml"

    with pytest.raises(LaunchPacketError, match="not a regular file"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_seed_manifest_directory_fails_closed(tmp_path: Path) -> None:
    """A seed manifest that is a directory must fail closed."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["seed_set_refs"]["manifest"] = str(tmp_path)

    with pytest.raises(LaunchPacketError, match="not a regular file"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_seed_manifest_malformed_yaml_fails_closed(tmp_path: Path) -> None:
    """A malformed seed manifest must become a validation error, not an escaped YAML error."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken_manifest = tmp_path / "bad_manifest.yaml"
    broken_manifest.write_text("dev: [101, 102\n", encoding="utf-8")
    broken["seed_set_refs"]["manifest"] = str(broken_manifest)

    with pytest.raises(LaunchPacketError, match="manifest could not be loaded"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_load_launch_packet_malformed_yaml_raises_launch_packet_error(tmp_path: Path) -> None:
    """Malformed launch-packet YAML should preserve the validator's public exception type."""
    malformed = tmp_path / "packet.yaml"
    malformed.write_text("schema_version: [broken\n", encoding="utf-8")

    with pytest.raises(LaunchPacketError, match="failed to load launch packet YAML"):
        load_launch_packet(malformed)


def test_seed_ref_non_list_fails_closed(tmp_path: Path) -> None:
    """A seed set in the manifest that is not a list must fail closed."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken_manifest = tmp_path / "bad_manifest.yaml"
    broken_manifest.write_text(
        yaml.safe_dump({"dev": 42, "eval": [111, 112, 113]}), encoding="utf-8"
    )
    broken["seed_set_refs"]["manifest"] = str(broken_manifest)

    with pytest.raises(LaunchPacketError, match="must be a list"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_seed_ref_non_integer_entry_fails_closed(tmp_path: Path) -> None:
    """A seed set containing a non-integer entry must fail closed."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken_manifest = tmp_path / "bad_manifest.yaml"
    broken_manifest.write_text(
        yaml.safe_dump({"dev": [101, "bad", 103], "eval": [111, 112, 113]}),
        encoding="utf-8",
    )
    broken["seed_set_refs"]["manifest"] = str(broken_manifest)

    with pytest.raises(LaunchPacketError, match="non-integer entry"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_train_excludes_non_list_fails_closed(tmp_path: Path) -> None:
    """A train_excludes set that is not a list must fail closed."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken_manifest = tmp_path / "bad_manifest.yaml"
    broken_manifest.write_text(
        yaml.safe_dump({"dev": [101, 102, 103], "eval": [111, 112, 113], "paper_eval_s20": "bad"}),
        encoding="utf-8",
    )
    broken["seed_set_refs"]["manifest"] = str(broken_manifest)

    with pytest.raises(LaunchPacketError, match="must be a list"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_output_component_in_path_rejected(tmp_path: Path) -> None:
    """A path with 'output' as a directory component must be rejected."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["artifact_paths"]["dry_run_fixture"] = "output/subdir/local-only.json"
    broken["checksums"] = {"output/subdir/local-only.json": "0" * 64}

    with pytest.raises(LaunchPacketError, match="worktree-local output"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_output_substring_not_in_component_allowed(tmp_path: Path) -> None:
    """A path containing 'output' only as a substring of another name is NOT rejected."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["artifact_paths"]["dry_run_fixture"] = "docs/my_output_config/stub.json"
    broken["checksums"] = {"docs/my_output_config/stub.json": "0" * 64}

    with pytest.raises(LaunchPacketError, match="local artifact is missing"):
        validate_launch_packet(_write_packet(tmp_path, broken))
