"""Tests for issue #4245 offline checkpoint manifest contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from gymnasium import spaces as gym_spaces

from robot_sf.training.offline_pretraining_manifest import (
    build_finetune_manifest,
    build_offline_checkpoint_manifest,
    load_offline_checkpoint_manifest,
    space_fingerprint,
    validate_offline_checkpoint_manifest,
    write_normalizer_state,
)


def test_offline_checkpoint_manifest_validates_required_files(tmp_path: Path) -> None:
    """Complete checkpoint manifests validate all hashed provenance files."""

    manifest = _offline_manifest(tmp_path)
    validate_offline_checkpoint_manifest(manifest)
    manifest_path = tmp_path / "offline_manifest.json"
    _write_json(manifest_path, manifest)
    assert load_offline_checkpoint_manifest(manifest_path)["schema_version"].endswith(".v1")


def test_offline_checkpoint_manifest_rejects_missing_checkpoint_path(tmp_path: Path) -> None:
    """Missing checkpoint bytes fail closed before fine-tune can consume them."""

    manifest = _offline_manifest(tmp_path)
    Path(manifest["checkpoint_path"]).unlink()
    with pytest.raises(ValueError, match="checkpoint_path"):
        validate_offline_checkpoint_manifest(manifest)


def test_offline_checkpoint_manifest_rejects_checksum_mismatch(tmp_path: Path) -> None:
    """Changed checkpoint bytes invalidate the manifest checksum."""

    manifest = _offline_manifest(tmp_path)
    Path(manifest["checkpoint_path"]).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_offline_checkpoint_manifest(manifest)


def test_offline_checkpoint_manifest_requires_dataset_identity(tmp_path: Path) -> None:
    """Dataset identity is required for the checkpoint handoff."""

    manifest = _offline_manifest(tmp_path)
    manifest["dataset"]["dataset_id"] = ""
    with pytest.raises(ValueError, match="dataset_id"):
        validate_offline_checkpoint_manifest(manifest)


def test_finetune_manifest_records_parent_manifest_hash(tmp_path: Path) -> None:
    """Fine-tune manifests chain back to the validated offline manifest."""

    parent = _offline_manifest(tmp_path)
    parent_path = tmp_path / "offline_manifest.json"
    _write_json(parent_path, parent)
    checkpoint = _file(tmp_path / "finetuned.zip", "fine\n")
    normalizer = _normalizer_state(tmp_path / "finetune_normalizer.json", present=False)
    config = _file(tmp_path / "finetune.yaml", "policy_id: fine\n")

    manifest = build_finetune_manifest(
        parent_manifest_path=parent_path,
        parent_manifest=parent,
        checkpoint_path=checkpoint,
        normalizer_path=normalizer,
        training_config_path=config,
        online_timesteps=1,
        seed=4245,
        environment_contract=parent["environment_contract"],
    )

    assert manifest["parent_checkpoint_sha256"] == parent["checkpoint_sha256"]
    assert manifest["parent_normalizer_sha256"] == parent["normalizer_sha256"]
    assert manifest["inherited_dataset"]["dataset_id"] == "issue_4245_unit"
    assert manifest["eligible_for_claim"] is False


def test_finetune_manifest_rejects_unapplied_parent_normalizer(tmp_path: Path) -> None:
    """Fine-tune provenance fails closed when parent requires a normalizer state."""

    parent = _offline_manifest(tmp_path, normalizer_present=True)
    parent_path = tmp_path / "offline_manifest.json"
    _write_json(parent_path, parent)
    checkpoint = _file(tmp_path / "finetuned.zip", "fine\n")
    normalizer = _normalizer_state(tmp_path / "finetune_normalizer.json", present=False)
    config = _file(tmp_path / "finetune.yaml", "policy_id: fine\n")

    with pytest.raises(ValueError, match="matching parent normalizer"):
        build_finetune_manifest(
            parent_manifest_path=parent_path,
            parent_manifest=parent,
            checkpoint_path=checkpoint,
            normalizer_path=normalizer,
            training_config_path=config,
            online_timesteps=1,
            seed=4245,
            environment_contract=parent["environment_contract"],
        )


def _offline_manifest(tmp_path: Path, *, normalizer_present: bool = False) -> dict[str, object]:
    checkpoint = _file(tmp_path / "checkpoint.zip", "checkpoint\n")
    normalizer = _normalizer_state(tmp_path / "normalizer.json", present=normalizer_present)
    config = _file(tmp_path / "pretrain.yaml", "policy_id: unit\n")
    dataset = _file(tmp_path / "dataset.jsonl", "{}\n")
    dataset_manifest = _file(
        tmp_path / "dataset.manifest.json", '{"dataset_id":"issue_4245_unit"}\n'
    )
    return build_offline_checkpoint_manifest(
        checkpoint_path=checkpoint,
        normalizer_path=normalizer,
        training_config_path=config,
        dataset={
            "dataset_path": str(dataset),
            "dataset_manifest_path": str(dataset_manifest),
            "dataset_sha256": _sha(dataset),
            "dataset_manifest_sha256": _sha(dataset_manifest),
            "dataset_schema_version": "RLTrajectoryDataset.v1",
            "dataset_id": "issue_4245_unit",
            "split": "train",
            "episode_count": 1,
            "accepted_transitions": 2,
            "dropped_terminal_transitions": 0,
        },
        offline_training={
            "gradient_steps": 1,
            "batch_size": 2,
            "seed": 4245,
            "observation_contract": "dataset_observation",
            "action_contract": "box_direct",
        },
        environment_contract={
            "scenario_config": "unit.yaml",
            "observation_space_fingerprint": space_fingerprint(
                gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=float)
            ),
            "action_space_fingerprint": space_fingerprint(
                gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=float)
            ),
        },
        policy_type="MlpPolicy",
        created_at_utc="2026-07-03T00:00:00Z",
    )


def _file(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _normalizer_state(path: Path, *, present: bool) -> Path:
    write_normalizer_state(
        path,
        present=present,
        reason="unit test normalizer present" if present else "unit test explicit absence",
    )
    return path


def _sha(path: Path) -> str:
    from robot_sf.benchmark.rl_trajectory_dataset import sha256_file

    return sha256_file(path)


def _write_json(path: Path, payload: object) -> None:
    import json

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
