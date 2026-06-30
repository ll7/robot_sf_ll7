"""Tests issue #3204 proxy-based checkpoint preflight readiness contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "research"
    / "check_predictive_checkpoint_proxy_readiness.py"
)
_SPEC = importlib.util.spec_from_file_location("_issue_3204_proxy_readiness", SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _write_yaml(path: Path, payload: object) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _write_registry(path: Path, entries: dict[str, object]) -> Path:
    payload = {"models": [{"model_id": model_id, **entry} for model_id, entry in entries.items()]}
    return _write_yaml(path, payload)


def _make_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    hard_seed = repo / "configs" / "benchmarks" / "hard_seed.yaml"
    hard_seed.parent.mkdir(parents=True, exist_ok=True)
    hard_seed.write_text("seed: 101\n")
    checkpoint = repo / "model" / "predictive" / "proxy.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_text("")
    registry = repo / "model" / "registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    return repo, hard_seed, checkpoint, registry


def _build_config(
    *,
    hard_seed: Path,
    min_checkpoints: int | float,
    include_training_summary_gate: bool,
    include_blocked_artifacts: bool = True,
) -> dict:
    payload = {
        "schema_version": "predictive-checkpoint-proxy-readiness.v1",
        "issue": 3204,
        "hard_seed_fixture": hard_seed.as_posix(),
        "checkpoint_selector": {
            "registry_tag": "predictive",
            "min_resolvable_checkpoints": min_checkpoints,
            "training_run_group_field": "proxy_training_run_id",
            "min_resolvable_training_run_checkpoints": 1.0e-9,
        },
        "proxy_summary_contract": {
            "require_enabled": False,
            "min_proxy_epochs": 6,
        },
    }
    if include_blocked_artifacts:
        payload["blocked_artifacts"] = [
            {
                "id": "missing_durable_predictive_checkpoints",
                "artifact_type": "checkpoint_set",
                "status": "blocked",
                "storage_scope": "worktree_local_output",
                "path_pattern": "output/tmp/predictive/**/*.pt",
                "required_metadata": ["local_path", "proxy_training_run_id"],
                "revival_condition": "Resolve at least one durable checkpoint path.",
            }
        ]
    if include_training_summary_gate:
        payload["known_blockers"] = [
            {
                "id": "degenerate_hardcase_proxy_probe_v1",
                "status": "blocked",
                "issue": 3204,
                "evidence": "summary probe inconclusive",
            }
        ]
    return payload


def test_float_checkpoint_thresholds_and_mapping_metadata_are_surface_ready(tmp_path: Path):
    """Float-encoded YAML thresholds and mapping/public-release metadata surface correctly."""
    repo, hard_seed, checkpoint, registry = _make_repo(tmp_path)

    _write_registry(
        registry,
        {
            "predictive_model_v1": {
                "display_name": "Predictive model (test)",
                "local_path": checkpoint.as_posix(),
                "proxy_training_run_id": "run_alpha",
                "tags": ["predictive", "planner"],
                "github_release": {
                    "repo": "ll7/robot_sf_ll7",
                    "tag": "artifact/models-2026-05-registry-v1",
                    "asset_name": "model.pt",
                    "url": "https://example.org/test.zip",
                    "sha256": "abc",
                    "size_bytes": 1234,
                },
            },
            "non_predictive": {
                "display_name": "Other model",
                "local_path": checkpoint.as_posix(),
                "tags": ["baseline"],
            },
        },
    )
    config = tmp_path / "config.yaml"
    _write_yaml(
        config,
        _build_config(
            hard_seed=hard_seed,
            min_checkpoints=1.0,
            include_training_summary_gate=False,
            include_blocked_artifacts=False,
        ),
    )

    report = _MODULE.check_readiness(
        config_path=config,
        registry_path=registry,
        repo_root=repo,
    )

    assert report["status"] == "ready"
    mapping = report["prerequisites"]["checkpoint_artifacts"]["mapping"]
    assert mapping["candidate_count"] == 1
    assert mapping["resolvable_count"] == 1
    assert mapping["candidates"][0]["public_artifact"]["status"] == "declared"
    assert mapping["candidates"][0]["artifact_scope"] == "repo_relative"
    assert report["prerequisites"]["blocked_artifacts"]["status"] == "passed"


def test_missing_checkpoint_lineage_blocks_with_fail_closed_blocker_message(tmp_path: Path):
    """Missing-checkpoint lineage produces a fail-closed blocked status with a clear message."""
    repo, hard_seed, _checkpoint, registry = _make_repo(tmp_path)

    _write_registry(
        registry,
        {
            "predictive_blocked": {
                "display_name": "Predictive model (blocked)",
                "local_path": "output/tmp/predictive/missing.pt",
                "tags": ["predictive", "planner"],
            }
        },
    )
    config = tmp_path / "config.yaml"
    _write_yaml(
        config,
        _build_config(
            hard_seed=hard_seed,
            min_checkpoints=2,
            include_training_summary_gate=False,
        ),
    )

    report = _MODULE.check_readiness(
        config_path=config,
        registry_path=registry,
        repo_root=repo,
    )

    assert report["status"] == "blocked"
    assert any(
        "resolve locally" in message or "no registry entries carry tag" in message
        for message in report["errors"]
    )
    mapping = report["prerequisites"]["checkpoint_artifacts"]["mapping"]
    assert mapping["candidate_count"] == 1
    assert mapping["resolvable_count"] == 0
    assert any(
        "blocked_artifacts" in check and report["prerequisites"][check]["status"] != "passed"
        for check in ("blocked_artifacts", "known_blockers")
    )
