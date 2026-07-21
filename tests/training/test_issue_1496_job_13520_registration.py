"""Contract tests for the issue #1496 job 13520 dataset registration."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from robot_sf.training.oracle_trace_uri_registry import validate_trace_uri_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "docs/context/evidence/issue_1496_oracle_trace_collection_job_13520"
REGISTRY = (
    REPO_ROOT
    / "configs/training/ppo_imitation/oracle_trace_uri_registry_issue_1496_collection.yaml"
)
READINESS = REPO_ROOT / "configs/training/ppo_imitation/oracle_warm_start_readiness_issue_1496.yaml"


def _load_json(name: str) -> dict:
    """Load one registered JSON object."""
    payload = json.loads((BUNDLE / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_private_dataset_pointer_and_checksum_are_registered_without_npz() -> None:
    """The public tree records the exact private pointer but never the dataset bytes."""
    registration = _load_json("registration.json")
    dataset = registration["dataset"]

    assert dataset["uri"] == (
        "private-artifact://oracle-imitation/issue1496/oracle-trace-collection/"
        "dataset/expert_traj_v1.npz"
    )
    assert dataset["sha256"] == ("433c797f6e3635f133d7541c9dd9edd3849cdee7b89e23c930b458e463979388")
    assert dataset["size_bytes"] == 363117931
    assert dataset["committed"] is False
    assert not list(BUNDLE.rglob("*.npz"))


def test_collection_split_and_step_scope_is_explicitly_small() -> None:
    """Registration distinguishes materialization readiness from full BC sufficiency."""
    registration = _load_json("registration.json")
    acceptance = _load_json("acceptance.json")

    assert registration["episodes"] == {
        "total": 12,
        "train": 6,
        "validation": 3,
        "evaluation": 3,
    }
    assert registration["steps"] == {
        "total": 1173,
        "train": 616,
        "validation": 201,
        "evaluation": 356,
        "minimum_episode_steps": 1,
        "maximum_episode_steps": 241,
    }
    assert acceptance["dataset_schema_valid"] is True
    assert acceptance["per_step_observations_actions"] is True
    assert acceptance["training_performed"] is False
    assert registration["sufficiency"]["materialization_blocker_resolved"] is True
    assert registration["sufficiency"]["loader_and_smoke_ready"] is True
    assert registration["sufficiency"]["full_bc_comparison_ready"] is False


def test_registry_has_all_splits_and_remains_fail_closed_without_private_root() -> None:
    """Public metadata validates, but byte readiness still needs private-root resolution."""
    report = validate_trace_uri_registry(REGISTRY, repo_root=REPO_ROOT)
    strict_report = _load_json("strict_registry_validation.json")

    assert report["status"] == "valid"
    assert report["trace_count"] == 3
    assert report["training_ready"] is False
    assert {split: len(ids) for split, ids in report["splits"].items()} == {
        "train": 1,
        "validation": 1,
        "evaluation": 1,
    }
    assert strict_report["status"] == "valid"
    assert strict_report["training_ready"] is True


def test_issue_1496_readiness_uses_the_new_collection_registry() -> None:
    """The warm-start preflight now names the BC-compatible job 13520 data plane."""
    readiness = yaml.safe_load(READINESS.read_text(encoding="utf-8"))
    assert readiness["trace_uri_registry"] == (
        "configs/training/ppo_imitation/oracle_trace_uri_registry_issue_1496_collection.yaml"
    )
