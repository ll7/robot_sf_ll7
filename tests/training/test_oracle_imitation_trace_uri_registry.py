"""Tests for the durable oracle-imitation trace-URI registry validator (issue #2655)."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from robot_sf.training.oracle_trace_uri_registry import (
    OracleTraceUriRegistryError,
    load_trace_uri_registry,
    validate_trace_uri_registry,
)
from scripts.validation.validate_oracle_trace_uri_registry import main as validate_cli_main

EXAMPLE_REGISTRY = Path("configs/training/ppo_imitation/oracle_trace_uri_registry_example.yaml")


def _write_registry(tmp_path: Path, registry: dict[str, object]) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    return path


def _training_ready_registry() -> dict[str, object]:
    """A complete, training-ready registry: every split concrete, durable, and resolvable."""
    return {
        "schema_version": "oracle-trace-uri-registry.v1",
        "dataset_id": "demo_oracle_imitation_v1",
        "traces": [
            {
                "split": "train",
                "trace_id": "train__demo_v1",
                "uri": "wandb-artifact://robot-sf/oracle-imitation/train:v1",
                "sha256": "a" * 64,
                "retrieval_status": "resolvable",
            },
            {
                "split": "validation",
                "trace_id": "validation__demo_v1",
                "uri": "wandb-artifact://robot-sf/oracle-imitation/val:v1",
                "sha256": "b" * 64,
                "retrieval_status": "resolvable",
            },
            {
                "split": "evaluation",
                "trace_id": "evaluation__demo_v1",
                "uri": "wandb-artifact://robot-sf/oracle-imitation/eval:v1",
                "sha256": "c" * 64,
                "retrieval_status": "resolvable",
            },
        ],
    }


def test_complete_registry_is_training_ready(tmp_path: Path) -> None:
    """A registry with concrete durable resolvable traces for all splits is training-ready."""
    registry = _training_ready_registry()
    report = validate_trace_uri_registry(
        _write_registry(tmp_path, registry),
        require_training_ready=True,
    )

    assert report["status"] == "valid"
    assert report["training_ready"] is True
    assert report["trace_count"] == 3
    assert report["retrieval_status"]["train__demo_v1"] == "resolvable"


def test_missing_uri_fails_closed(tmp_path: Path) -> None:
    """A trace entry without a URI must fail closed."""
    registry = _training_ready_registry()
    del registry["traces"][0]["uri"]

    with pytest.raises(OracleTraceUriRegistryError, match="uri must be a non-empty durable URI"):
        validate_trace_uri_registry(_write_registry(tmp_path, registry))


def test_missing_checksum_for_resolvable_fails_closed(tmp_path: Path) -> None:
    """A resolvable trace without a checksum must fail closed."""
    registry = _training_ready_registry()
    del registry["traces"][1]["sha256"]

    with pytest.raises(
        OracleTraceUriRegistryError,
        match="sha256 is required when retrieval_status is 'resolvable'",
    ):
        validate_trace_uri_registry(_write_registry(tmp_path, registry))


def test_checksum_mismatch_against_local_mirror_fails_closed(tmp_path: Path) -> None:
    """A staged local mirror whose bytes do not match the declared checksum must fail closed."""
    mirror = tmp_path / "train_trace.jsonl"
    mirror.write_text('{"step": 0}\n', encoding="utf-8")
    wrong_digest = "d" * 64
    assert hashlib.sha256(mirror.read_bytes()).hexdigest() != wrong_digest

    registry = _training_ready_registry()
    registry["traces"][0]["sha256"] = wrong_digest
    registry["traces"][0]["local_mirror"] = str(mirror)

    with pytest.raises(OracleTraceUriRegistryError, match="local_mirror checksum mismatch"):
        validate_trace_uri_registry(_write_registry(tmp_path, registry))


def test_local_mirror_matching_checksum_passes(tmp_path: Path) -> None:
    """A staged local mirror with the correct checksum verifies cleanly."""
    mirror = tmp_path / "train_trace.jsonl"
    mirror.write_text('{"step": 0}\n', encoding="utf-8")
    digest = hashlib.sha256(mirror.read_bytes()).hexdigest()

    registry = _training_ready_registry()
    registry["traces"][0]["sha256"] = digest
    registry["traces"][0]["local_mirror"] = str(mirror)

    report = validate_trace_uri_registry(
        _write_registry(tmp_path, registry),
        require_training_ready=True,
    )
    assert report["training_ready"] is True


def test_blocked_retrieval_is_not_training_ready(tmp_path: Path) -> None:
    """An inaccessible (blocked) trace keeps the lane out of training-ready, failing closed."""
    registry = _training_ready_registry()
    registry["traces"][2]["retrieval_status"] = "blocked"
    registry["traces"][2]["uri"] = "wandb-artifact://robot-sf/oracle-imitation/eval:pending"
    registry["traces"][2]["sha256"] = "pending"

    # Base validation passes (a blocked, not-yet-resolvable trace is a legitimate state)...
    base_report = validate_trace_uri_registry(_write_registry(tmp_path, registry))
    assert base_report["training_ready"] is False
    assert base_report["retrieval_status"]["evaluation__demo_v1"] == "blocked"

    # ...but the strict gate fails closed.
    with pytest.raises(OracleTraceUriRegistryError, match="not resolvable: evaluation"):
        validate_trace_uri_registry(
            _write_registry(tmp_path, registry),
            require_training_ready=True,
        )


def test_pending_uri_is_not_concrete_durable(tmp_path: Path) -> None:
    """A `:pending` URI alias is a placeholder, not a concrete durable training input."""
    registry = _training_ready_registry()
    registry["traces"][0]["uri"] = "wandb-artifact://robot-sf/oracle-imitation/train:pending"
    registry["traces"][0]["sha256"] = "pending"
    registry["traces"][0]["retrieval_status"] = "pending"

    with pytest.raises(OracleTraceUriRegistryError, match="not resolvable: train"):
        validate_trace_uri_registry(
            _write_registry(tmp_path, registry),
            require_training_ready=True,
        )


def test_missing_split_fails_training_ready_gate(tmp_path: Path) -> None:
    """A registry missing a required split cannot be training-ready."""
    registry = _training_ready_registry()
    registry["traces"] = registry["traces"][:2]  # drop evaluation

    with pytest.raises(OracleTraceUriRegistryError, match="missing: evaluation"):
        validate_trace_uri_registry(
            _write_registry(tmp_path, registry),
            require_training_ready=True,
        )


def test_registry_rejects_worktree_local_output_uri(tmp_path: Path) -> None:
    """Trace URIs must not depend on worktree-local output paths."""
    registry = _training_ready_registry()
    registry["traces"][0]["uri"] = "output/oracle-imitation/train.jsonl"

    with pytest.raises(OracleTraceUriRegistryError, match="durable URI"):
        validate_trace_uri_registry(_write_registry(tmp_path, registry))


def test_registry_rejects_bad_schema_version(tmp_path: Path) -> None:
    """An unexpected schema version must fail closed."""
    registry = _training_ready_registry()
    registry["schema_version"] = "oracle-trace-uri-registry.v0"

    with pytest.raises(OracleTraceUriRegistryError, match="schema_version must be"):
        validate_trace_uri_registry(_write_registry(tmp_path, registry))


def test_registry_rejects_duplicate_trace_ids(tmp_path: Path) -> None:
    """Duplicate trace ids must fail closed."""
    registry = _training_ready_registry()
    registry["traces"][1]["trace_id"] = registry["traces"][0]["trace_id"]

    with pytest.raises(OracleTraceUriRegistryError, match="trace_id is duplicated"):
        validate_trace_uri_registry(_write_registry(tmp_path, registry))


def test_example_registry_is_valid_but_not_training_ready() -> None:
    """The checked-in example documents the blocked state: valid schema, not training-ready."""
    report = validate_trace_uri_registry(EXAMPLE_REGISTRY)

    assert report["status"] == "valid"
    assert report["dataset_id"] == "issue_1397_oracle_imitation_v1"
    assert report["training_ready"] is False
    assert set(report["splits"]) == {"train", "validation", "evaluation"}


def test_example_registry_fails_training_ready_gate() -> None:
    """The checked-in example must not pass the strict training-ready gate yet."""
    with pytest.raises(OracleTraceUriRegistryError, match="not resolvable"):
        validate_trace_uri_registry(EXAMPLE_REGISTRY, require_training_ready=True)


def test_load_registry_malformed_yaml_raises_registry_error(tmp_path: Path) -> None:
    """Malformed registry YAML preserves the public exception type."""
    malformed = tmp_path / "registry.yaml"
    malformed.write_text("schema_version: [broken\n", encoding="utf-8")

    with pytest.raises(OracleTraceUriRegistryError, match="failed to load trace-URI registry YAML"):
        load_trace_uri_registry(malformed)


def test_cli_reports_json_for_example() -> None:
    """The CLI exposes the same validation report and succeeds on the valid example."""
    exit_code = validate_cli_main(["--config", str(EXAMPLE_REGISTRY), "--json"])
    assert exit_code == 0


def test_cli_require_training_ready_fails_closed_for_example() -> None:
    """Automation can request the stricter training-ready gate, which fails closed."""
    exit_code = validate_cli_main(
        ["--config", str(EXAMPLE_REGISTRY), "--json", "--require-training-ready"]
    )
    assert exit_code == 2


def test_cli_succeeds_for_training_ready_registry(tmp_path: Path) -> None:
    """The CLI passes the strict gate for a complete training-ready registry."""
    registry = _training_ready_registry()
    path = _write_registry(tmp_path, copy.deepcopy(registry))
    exit_code = validate_cli_main(["--config", str(path), "--json", "--require-training-ready"])
    assert exit_code == 0
