"""Tests for fail-closed release checkpoint receipt admission."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf.benchmark.checkpoint_staging_receipt import (
    CHECKPOINT_STAGING_RECEIPT_SCHEMA,
    CheckpointStagingReceiptError,
    validate_checkpoint_staging_receipt,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file


def _fixture(tmp_path: Path) -> tuple[SimpleNamespace, Path, Path, Path, dict]:
    """Create one config, checkpoint, campaign stub, and valid receipt payload."""
    config = tmp_path / "campaign.yaml"
    config.write_text("name: release\n", encoding="utf-8")
    checkpoint = tmp_path / "model.zip"
    checkpoint.write_bytes(b"checkpoint")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "models": [
                    {
                        "model_id": "ppo_release",
                        "github_release": {"sha256": sha256_file(checkpoint)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    reference = SimpleNamespace(
        planner_key="ppo",
        algo="ppo",
        kind="model_id",
        value="ppo_release",
        implicit=False,
    )
    cfg = SimpleNamespace(references=[reference])
    payload = {
        "schema_version": CHECKPOINT_STAGING_RECEIPT_SCHEMA,
        "status": "ok",
        "mode": "enforced_staged",
        "stage": True,
        "submit_safe": True,
        "generated_at_utc": "2026-08-21T12:00:00Z",
        "campaign_config_sha256": sha256_file(config),
        "checkpoint_registry_sha256": sha256_file(registry),
        "arms": [
            {
                "planner_key": "ppo",
                "algo": "ppo",
                "kind": "model_id",
                "value": "ppo_release",
                "implicit": False,
                "status": "staged",
                "resolved_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "hash_source": "computed_file",
            }
        ],
    }
    receipt = tmp_path / "receipt.json"
    return cfg, config, registry, receipt, payload


def _write_receipt(path: Path, payload: dict) -> None:
    """Write one receipt fixture."""
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_valid_receipt_is_admitted(tmp_path: Path, monkeypatch) -> None:
    """Exact, recent, staged, checksum-valid coverage is accepted."""
    cfg, config, registry, receipt, payload = _fixture(tmp_path)
    _write_receipt(receipt, payload)
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )
    result = validate_checkpoint_staging_receipt(
        cfg,
        receipt,
        campaign_config_path=config,
        registry_path=registry,
        now=datetime(2026, 8, 21, 13, tzinfo=UTC),
    )
    assert result["submit_safe"] is True


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda payload: payload.update(submit_safe=False), "submit_safe=false"),
        (lambda payload: payload.update(campaign_config_sha256="0" * 64), "config hash"),
        (lambda payload: payload["arms"][0].update(value="other"), "identities"),
        (lambda payload: payload["arms"][0].update(status="stageable_remote"), "non-staged"),
    ],
)
def test_receipt_rejects_unsafe_or_mismatched_state(
    tmp_path: Path, monkeypatch, mutation, match: str
) -> None:
    """Unsafe modes, config drift, arm drift, and unstaged rows fail closed."""
    cfg, config, registry, receipt, payload = _fixture(tmp_path)
    mutation(payload)
    _write_receipt(receipt, payload)
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )
    with pytest.raises(CheckpointStagingReceiptError, match=match):
        validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=config,
            registry_path=registry,
            now=datetime(2026, 8, 21, 13, tzinfo=UTC),
        )


def test_receipt_rejects_stale_and_changed_checkpoint(tmp_path: Path, monkeypatch) -> None:
    """Age and post-staging checkpoint mutation are both admission blockers."""
    cfg, config, registry, receipt, payload = _fixture(tmp_path)
    _write_receipt(receipt, payload)
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )
    with pytest.raises(CheckpointStagingReceiptError, match="stale"):
        validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=config,
            registry_path=registry,
            now=datetime(2026, 8, 23, 13, tzinfo=UTC),
        )
    Path(payload["arms"][0]["resolved_path"]).write_bytes(b"changed")
    with pytest.raises(CheckpointStagingReceiptError, match="checksum changed"):
        validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=config,
            registry_path=registry,
            now=datetime(2026, 8, 21, 13, tzinfo=UTC),
        )


def test_receipt_rejects_self_declared_hash_that_disagrees_with_registry(
    tmp_path: Path, monkeypatch
) -> None:
    """A forged receipt cannot replace a registry-pinned checkpoint with self-consistent bytes."""
    cfg, config, registry, receipt, payload = _fixture(tmp_path)
    checkpoint = Path(payload["arms"][0]["resolved_path"])
    checkpoint.write_bytes(b"forged-checkpoint")
    payload["arms"][0]["checkpoint_sha256"] = sha256_file(checkpoint)
    _write_receipt(receipt, payload)
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )
    with pytest.raises(CheckpointStagingReceiptError, match="registry-pinned"):
        validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=config,
            registry_path=registry,
            now=datetime(2026, 8, 21, 13, tzinfo=UTC),
        )


def test_receipt_rejects_registry_drift(tmp_path: Path, monkeypatch) -> None:
    """Registry bytes are part of the staging receipt admission identity."""
    cfg, config, registry, receipt, payload = _fixture(tmp_path)
    _write_receipt(receipt, payload)
    registry.write_text(registry.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )
    with pytest.raises(CheckpointStagingReceiptError, match="registry hash"):
        validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=config,
            registry_path=registry,
            now=datetime(2026, 8, 21, 13, tzinfo=UTC),
        )
