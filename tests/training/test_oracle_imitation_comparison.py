"""Contract tests for the outcome-free #1496 comparison packet."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.training.oracle_imitation_comparison import (
    ComparisonArtifactBlockedError,
    ComparisonPacketError,
    build_comparison_packet,
    run_startup_canary,
    validate_dataset_consumer_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARISON_CONFIG = (
    REPO_ROOT / "configs/training/ppo_imitation/oracle_imitation_comparison_issue_1496.yaml"
)


def _write_consumer_manifest(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Write a tiny complete consumer manifest with local, verifiable shards."""
    split_rows = {
        "train": {
            "sample_id": "train__sample1",
            "features": [1.0, 0.0],
            "action": [1.0],
        },
        "validation": {
            "sample_id": "validation__sample1",
            "features": [0.0, 1.0],
            "action": [2.0],
        },
        "evaluation": {
            "sample_id": "evaluation__sample1",
            "features": [1.0, 1.0],
            "action": [3.0],
        },
    }
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    manifest: dict[str, Any] = {
        "schema_version": "oracle-imitation-dataset-consumer.v1",
        "dataset_id": "unit_dataset",
        "dataset_version": "unit_v1",
        "dataset_digest": "a" * 64,
        "artifact": {
            "uri": "private-artifact://unit/dataset.npz",
            "sha256": "b" * 64,
            "size_bytes": 1,
            "availability": "private_required",
        },
        "splits": {},
        "feature_contract": {
            "schema_id": "unit.features",
            "dtype": "float32",
            "dimension": 2,
        },
        "action_contract": {
            "schema_id": "unit.actions",
            "dtype": "float32",
            "dimension": 1,
        },
        "normalization": {"policy": "unit_policy"},
        "quality": {
            "admitted_statuses": ["valid"],
            "excluded_statuses": ["invalid", "fallback", "degraded"],
        },
        "sample_inventory": {"status": "complete", "rows": []},
    }
    for split, row in split_rows.items():
        shard_relative = Path("shards") / f"{split}.jsonl"
        shard_path = tmp_path / shard_relative
        shard_path.write_text(f"{row['sample_id']}\n", encoding="utf-8")
        shard_bytes = shard_path.read_bytes()
        manifest["splits"][split] = {
            "sample_ids": [row["sample_id"]],
            "shards": [
                {
                    "path": shard_relative.as_posix(),
                    "sha256": hashlib.sha256(shard_bytes).hexdigest(),
                    "size_bytes": len(shard_bytes),
                    "sample_ids": [row["sample_id"]],
                }
            ],
        }
        manifest["sample_inventory"]["rows"].append(
            {"sample_id": row["sample_id"], "split": split, "status": "valid", **row}
        )
    path = tmp_path / "consumer.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path, manifest


def _rewrite_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Persist one mutated consumer manifest fixture."""
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def _write_comparison_config(tmp_path: Path, consumer_path: Path) -> Path:
    """Copy the checked-in comparison declaration and replace only its dataset pointer."""
    config = yaml.safe_load(COMPARISON_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    config["dataset_consumer_manifest"] = str(consumer_path)
    path = tmp_path / "comparison.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_consumer_manifest_contract_and_strict_artifact_gate_are_deterministic(
    tmp_path: Path,
) -> None:
    """A valid manifest passes both modes and exposes a stable semantic digest."""
    path, _ = _write_consumer_manifest(tmp_path)

    declared = validate_dataset_consumer_manifest(path, repo_root=tmp_path)
    verified = validate_dataset_consumer_manifest(
        path,
        repo_root=tmp_path,
        artifact_root=tmp_path,
        require_artifacts=True,
    )

    assert declared["status"] == "declared"
    assert verified["status"] == "verified"
    assert declared["manifest_digest"] == verified["manifest_digest"]
    assert verified["split_counts"] == {"train": 1, "validation": 1, "evaluation": 1}
    assert verified["sample_inventory"] == {"status": "complete", "row_count": 3}
    assert all(check["status"] == "verified" for check in verified["artifact_checks"])


def test_missing_shard_is_blocked_only_at_the_strict_loader_gate(tmp_path: Path) -> None:
    """Contract metadata remains inspectable while a missing byte artifact fails closed."""
    path, _ = _write_consumer_manifest(tmp_path)
    (tmp_path / "shards/evaluation.jsonl").unlink()

    declared = validate_dataset_consumer_manifest(path, repo_root=tmp_path)

    assert declared["status"] == "declared"
    with pytest.raises(ComparisonArtifactBlockedError, match="evaluation shard"):
        validate_dataset_consumer_manifest(
            path,
            repo_root=tmp_path,
            artifact_root=tmp_path,
            require_artifacts=True,
        )


def test_digest_drift_and_symlink_shards_fail_closed(tmp_path: Path) -> None:
    """Strict loading rejects changed bytes and indirection through symlinks."""
    path, manifest = _write_consumer_manifest(tmp_path)
    evaluation = tmp_path / "shards/evaluation.jsonl"
    evaluation.write_text("evaluation__sampleX\n", encoding="utf-8")

    with pytest.raises(ComparisonPacketError, match="digest mismatch"):
        validate_dataset_consumer_manifest(
            path,
            repo_root=tmp_path,
            artifact_root=tmp_path,
            require_artifacts=True,
        )

    evaluation.write_text("evaluation__sample1\n", encoding="utf-8")
    target = tmp_path / "shards/evaluation-target.jsonl"
    target.write_bytes(evaluation.read_bytes())
    evaluation.unlink()
    evaluation.symlink_to(target)
    manifest["splits"]["evaluation"]["shards"][0]["path"] = "shards/evaluation.jsonl"
    _rewrite_manifest(path, manifest)

    with pytest.raises(ComparisonPacketError, match="must not be a symlink"):
        validate_dataset_consumer_manifest(
            path,
            repo_root=tmp_path,
            artifact_root=tmp_path,
            require_artifacts=True,
        )


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda payload: payload["splits"]["evaluation"]["sample_ids"].__setitem__(
                0, "train__sample1"
            ),
            "holdout contamination",
        ),
        (
            lambda payload: payload["sample_inventory"]["rows"][0].__setitem__(
                "status", "fallback"
            ),
            "non-admissible status",
        ),
        (
            lambda payload: payload["sample_inventory"]["rows"][0]["features"].__setitem__(
                0, float("nan")
            ),
            "must be finite",
        ),
        (
            lambda payload: payload["sample_inventory"]["rows"][0]["action"].append(4.0),
            "action dimension mismatch",
        ),
        (
            lambda payload: payload["sample_inventory"]["rows"].append(
                copy.deepcopy(payload["sample_inventory"]["rows"][0])
            ),
            "duplicate sample identity",
        ),
    ],
)
def test_consumer_manifest_rejects_split_and_row_contract_violations(
    tmp_path: Path,
    mutation: Any,
    error: str,
) -> None:
    """Holdout leakage, degraded rows, non-finite values, and duplicates are inadmissible."""
    path, original = _write_consumer_manifest(tmp_path)
    manifest = copy.deepcopy(original)
    mutation(manifest)
    _rewrite_manifest(path, manifest)

    with pytest.raises(ComparisonPacketError, match=error):
        validate_dataset_consumer_manifest(path, repo_root=tmp_path)


def test_consumer_manifest_rejects_paths_that_escape_the_artifact_root(tmp_path: Path) -> None:
    """Relative shard paths cannot escape the explicitly authorized root."""
    path, manifest = _write_consumer_manifest(tmp_path)
    manifest["splits"]["train"]["shards"][0]["path"] = "../outside.jsonl"
    _rewrite_manifest(path, manifest)

    with pytest.raises(ComparisonPacketError, match="must not be absolute or escape"):
        validate_dataset_consumer_manifest(path, repo_root=tmp_path)


def test_comparison_packet_freezes_matched_arms_and_public_claim_boundary(tmp_path: Path) -> None:
    """The packet links canonical sources and creates deterministic per-seed identities."""
    consumer_path, _ = _write_consumer_manifest(tmp_path)
    config_path = _write_comparison_config(tmp_path, consumer_path)

    packet = build_comparison_packet(config_path, repo_root=REPO_ROOT)

    assert packet["status"] == "ready"
    assert packet["budget"]["environment_steps"] == 15_000_000
    assert packet["arms"]["rl_only"]["seeds"] == packet["arms"]["bc_warm_start"]["seeds"]
    assert len(packet["identities"]) == 10
    assert packet["loader_gate"]["requires_artifacts"] is False
    assert packet["startup_canary"]["benchmark_evidence"] is False
    assert "no nominal training" in packet["claim_boundary"]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (
            lambda config: config["arms"]["bc_warm_start"].__setitem__(
                "environment_steps", 14_000_000
            ),
            "must equal budget",
        ),
        (
            lambda config: config["arms"]["bc_warm_start"]["seeds"].__setitem__(0, 999),
            "matched seeds",
        ),
        (
            lambda config: config["arms"]["rl_only"].__setitem__(
                "initialization", "dagger_refinement"
            ),
            "DAgger",
        ),
        (
            lambda config: config["out_of_scope_actions"].__setitem__("training_execution", True),
            "out_of_scope_actions",
        ),
    ],
)
def test_comparison_packet_rejects_unmatched_or_out_of_scope_changes(
    tmp_path: Path,
    mutation: Any,
    error: str,
) -> None:
    """The packet refuses budget drift, unmatched seeds, DAgger, and execution claims."""
    consumer_path, _ = _write_consumer_manifest(tmp_path)
    config = yaml.safe_load(COMPARISON_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    config["dataset_consumer_manifest"] = str(consumer_path)
    mutation(config)
    config_path = tmp_path / "comparison-invalid.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ComparisonPacketError, match=error):
        build_comparison_packet(config_path, repo_root=REPO_ROOT)


def test_startup_canary_round_trips_checkpoint_and_writes_diagnostic_result(tmp_path: Path) -> None:
    """The bounded canary proves loader/model/checkpoint/evaluation wiring only."""
    manifest_path, _ = _write_consumer_manifest(tmp_path)
    output_dir = tmp_path / "canary"

    result = run_startup_canary(
        manifest_path,
        output_dir,
        repo_root=tmp_path,
        artifact_root=tmp_path,
    )

    assert result["status"] == "passed"
    assert result["num_updates"] == 1
    assert result["checkpoint_round_trip"] == "passed"
    assert result["benchmark_evidence"] is False
    assert result["training_executed"] is True
    assert (output_dir / "startup_canary_checkpoint.npz").is_file()
    assert (output_dir / "startup_canary_result.json").is_file()


def test_startup_canary_requires_complete_inventory(tmp_path: Path) -> None:
    """A private-required inventory cannot be mistaken for a synthetic canary fixture."""
    manifest_path, manifest = _write_consumer_manifest(tmp_path)
    manifest["sample_inventory"] = {"status": "private_required"}
    _rewrite_manifest(manifest_path, manifest)

    with pytest.raises(ComparisonPacketError, match="complete synthetic or disjoint"):
        run_startup_canary(
            manifest_path,
            tmp_path / "canary",
            repo_root=tmp_path,
            artifact_root=tmp_path,
        )
