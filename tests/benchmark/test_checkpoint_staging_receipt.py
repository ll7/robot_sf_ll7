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


def test_receipt_accepts_exact_remote_path_map_without_rewriting_payload(
    tmp_path: Path, monkeypatch
) -> None:
    """A mapped local file retains the receipt path and both checksum bindings."""
    cfg, config, registry, receipt, payload = _fixture(tmp_path)
    remote_path = "/hpc/gpfs2/licca/checkpoints/model.zip"
    payload["arms"][0]["resolved_path"] = remote_path
    _write_receipt(receipt, payload)
    receipt_bytes = receipt.read_bytes()
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )
    result = validate_checkpoint_staging_receipt(
        cfg,
        receipt,
        campaign_config_path=config,
        registry_path=registry,
        checkpoint_path_map=[f"{remote_path}=model.zip"],
        repo_root=tmp_path,
        now=datetime(2026, 8, 21, 13, tzinfo=UTC),
    )
    assert result["arms"][0]["resolved_path"] == remote_path
    assert receipt.read_bytes() == receipt_bytes


def test_receipt_remap_preserves_direct_checkpoint_config_identity(
    tmp_path: Path, monkeypatch
) -> None:
    """A direct model path is checked against the receipt source, not the local destination."""
    cfg, config, registry, receipt, payload = _fixture(tmp_path)
    remote_path = "/hpc/gpfs2/licca/checkpoints/model.zip"
    cfg.references[0] = cfg.references[0].__class__(
        planner_key="ppo",
        algo="ppo",
        kind="model_path",
        value=remote_path,
        implicit=False,
    )
    payload["arms"][0].update(kind="model_path", value=remote_path, resolved_path=remote_path)
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
        checkpoint_path_map={remote_path: tmp_path / "model.zip"},
        repo_root=tmp_path,
        now=datetime(2026, 8, 21, 13, tzinfo=UTC),
    )
    assert result["arms"][0]["value"] == remote_path


@pytest.mark.parametrize(
    ("mapping", "match"),
    [
        (["/unknown/source.zip=model.zip"], "unknown or unused"),
        (["/hpc/source.zip=../outside.zip"], "outside the repository root"),
        (["/hpc/source.zip=missing.zip"], "not a regular file"),
        (["/hpc/source.zip="], "destination is malformed"),
        (["=model.zip"], "source is malformed"),
        (["malformed"], "RECEIPT_PATH=LOCAL_PATH"),
        (["/hpc/source.zip=model.zip", "/hpc/source.zip=model.zip"], "duplicated"),
    ],
)
def test_receipt_rejects_invalid_checkpoint_path_maps(
    tmp_path: Path, monkeypatch, mapping: list[str], match: str
) -> None:
    """Explicit mappings fail closed before they can weaken receipt admission."""
    cfg, config, registry, receipt, payload = _fixture(tmp_path)
    payload["arms"][0]["resolved_path"] = "/hpc/source.zip"
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
            checkpoint_path_map=mapping,
            repo_root=tmp_path,
            now=datetime(2026, 8, 21, 13, tzinfo=UTC),
        )


def test_receipt_rejects_remap_without_explicit_repo_root(tmp_path: Path, monkeypatch) -> None:
    """Mapped validation cannot silently select the process working directory."""
    cfg, config, registry, receipt, payload = _fixture(tmp_path)
    remote_path = "/hpc/source.zip"
    payload["arms"][0]["resolved_path"] = remote_path
    _write_receipt(receipt, payload)
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )
    with pytest.raises(CheckpointStagingReceiptError, match="explicit repository root"):
        validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=config,
            registry_path=registry,
            checkpoint_path_map=[f"{remote_path}=model.zip"],
            now=datetime(2026, 8, 21, 13, tzinfo=UTC),
        )


def test_receipt_rejects_mapped_checksum_mismatch(tmp_path: Path, monkeypatch) -> None:
    """The mapped local bytes remain independently bound to the receipt digest."""
    cfg, config, registry, receipt, payload = _fixture(tmp_path)
    remote_path = "/hpc/source.zip"
    payload["arms"][0]["resolved_path"] = remote_path
    (tmp_path / "model.zip").write_bytes(b"changed-local-bytes")
    _write_receipt(receipt, payload)
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )
    with pytest.raises(CheckpointStagingReceiptError, match="checksum changed"):
        validate_checkpoint_staging_receipt(
            cfg,
            receipt,
            campaign_config_path=config,
            registry_path=registry,
            checkpoint_path_map=[f"{remote_path}=model.zip"],
            repo_root=tmp_path,
            now=datetime(2026, 8, 21, 13, tzinfo=UTC),
        )


def test_receipt_rejects_symlink_escape_and_directory_destination(
    tmp_path: Path, monkeypatch
) -> None:
    """Mapped paths cannot escape the selected root or point at a directory."""
    cfg, config, registry, receipt, payload = _fixture(tmp_path)
    remote_path = "/hpc/source.zip"
    payload["arms"][0]["resolved_path"] = remote_path
    _write_receipt(receipt, payload)
    outside = tmp_path.parent / "outside-checkpoint.zip"
    outside.write_bytes(b"checkpoint")
    escaped = tmp_path / "escaped.zip"
    escaped.symlink_to(outside)
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )
    for destination, match in ((escaped, "outside"), (tmp_path, "regular file")):
        with pytest.raises(CheckpointStagingReceiptError, match=match):
            validate_checkpoint_staging_receipt(
                cfg,
                receipt,
                campaign_config_path=config,
                registry_path=registry,
                checkpoint_path_map=[f"{remote_path}={destination}"],
                repo_root=tmp_path,
                now=datetime(2026, 8, 21, 13, tzinfo=UTC),
            )


def test_receipt_rejects_mapped_registry_pin_mismatch(tmp_path: Path, monkeypatch) -> None:
    """The mapped bytes must also match the model-registry pin."""
    cfg, config, registry, receipt, payload = _fixture(tmp_path)
    remote_path = "/hpc/source.zip"
    payload["arms"][0]["resolved_path"] = remote_path
    registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "models": [
                    {
                        "model_id": "ppo_release",
                        "github_release": {"sha256": "0" * 64},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    payload["checkpoint_registry_sha256"] = sha256_file(registry)
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
            checkpoint_path_map=[f"{remote_path}=model.zip"],
            repo_root=tmp_path,
            now=datetime(2026, 8, 21, 13, tzinfo=UTC),
        )


def test_receipt_resolves_default_registry_from_repo_root_regardless_of_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    """The default model registry resolves against repo_root even when CWD is unrelated."""
    repo_fake = tmp_path / "repo"
    repo_fake.mkdir()
    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir()

    cfg, config, registry, receipt, payload = _fixture(repo_fake)
    # Move registry to standard default location: <repo>/model/registry.yaml
    default_registry_dir = repo_fake / "model"
    default_registry_dir.mkdir(parents=True, exist_ok=True)
    default_registry = default_registry_dir / "registry.yaml"
    default_registry.write_text(registry.read_text(encoding="utf-8"), encoding="utf-8")
    payload["checkpoint_registry_sha256"] = sha256_file(default_registry)
    _write_receipt(receipt, payload)

    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )
    monkeypatch.chdir(unrelated_cwd)

    result = validate_checkpoint_staging_receipt(
        cfg,
        receipt,
        campaign_config_path=config,
        repo_root=repo_fake,
        now=datetime(2026, 8, 21, 13, tzinfo=UTC),
    )
    assert result["submit_safe"] is True


def test_receipt_resolves_relative_registry_path_from_repo_root(
    tmp_path: Path, monkeypatch
) -> None:
    """An explicit relative registry path resolves against repo_root when CWD is unrelated."""
    repo_fake = tmp_path / "repo"
    repo_fake.mkdir()
    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir()

    cfg, config, registry, receipt, payload = _fixture(repo_fake)
    custom_reg_dir = repo_fake / "custom_models"
    custom_reg_dir.mkdir(parents=True, exist_ok=True)
    custom_registry = custom_reg_dir / "reg.yaml"
    custom_registry.write_text(registry.read_text(encoding="utf-8"), encoding="utf-8")
    payload["checkpoint_registry_sha256"] = sha256_file(custom_registry)
    _write_receipt(receipt, payload)

    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )
    monkeypatch.chdir(unrelated_cwd)

    result = validate_checkpoint_staging_receipt(
        cfg,
        receipt,
        campaign_config_path=config,
        registry_path=Path("custom_models/reg.yaml"),
        repo_root=repo_fake,
        now=datetime(2026, 8, 21, 13, tzinfo=UTC),
    )
    assert result["submit_safe"] is True


def test_receipt_direct_model_path_relative_resolves_from_repo_root(
    tmp_path: Path, monkeypatch
) -> None:
    """A relative direct model path resolves against repo_root when CWD is unrelated."""
    repo_fake = tmp_path / "repo"
    repo_fake.mkdir()
    unrelated_cwd = tmp_path / "unrelated_cwd"
    unrelated_cwd.mkdir()

    cfg, config, registry, receipt, payload = _fixture(repo_fake)
    model_rel_path = "models/my_model.zip"
    model_abs_path = repo_fake / model_rel_path
    model_abs_path.parent.mkdir(parents=True, exist_ok=True)
    model_abs_path.write_bytes(b"direct-checkpoint-content")

    cfg.references = [
        SimpleNamespace(
            planner_key="ppo",
            algo="ppo",
            kind="model_path",
            value=model_rel_path,
            implicit=False,
        )
    ]
    payload["arms"][0].update(
        planner_key="ppo",
        algo="ppo",
        kind="model_path",
        value=model_rel_path,
        resolved_path=str(model_abs_path),
        checkpoint_sha256=sha256_file(model_abs_path),
    )
    _write_receipt(receipt, payload)

    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )
    monkeypatch.chdir(unrelated_cwd)

    result = validate_checkpoint_staging_receipt(
        cfg,
        receipt,
        campaign_config_path=config,
        registry_path=registry,
        repo_root=repo_fake,
        now=datetime(2026, 8, 21, 13, tzinfo=UTC),
    )
    assert result["submit_safe"] is True


def test_receipt_direct_model_path_relative_resolves_from_cwd_when_repo_root_is_none(
    tmp_path: Path, monkeypatch
) -> None:
    """A relative direct model path falls back to CWD when repo_root is None in arm validation."""
    from robot_sf.benchmark.checkpoint_staging_receipt import _validate_direct_checkpoint_path

    model_rel = "local_model.zip"
    model_abs = (tmp_path / model_rel).resolve()
    model_abs.write_bytes(b"content")

    ref = SimpleNamespace(kind="model_path", value=model_rel)
    monkeypatch.chdir(tmp_path)
    # Should succeed without error
    _validate_direct_checkpoint_path(
        ref,
        planner_key="goal",
        resolved=model_abs,
        receipt_path=str(model_abs),
        repo_root=None,
    )


def test_receipt_resolves_default_registry_when_repo_root_is_none(
    tmp_path: Path, monkeypatch
) -> None:
    """Default registry resolves via get_repository_root when repo_root parameter is None."""
    repo_fake = tmp_path / "repo"
    repo_fake.mkdir()

    cfg, config, registry, receipt, payload = _fixture(repo_fake)
    model_dir = repo_fake / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    default_registry = model_dir / "registry.yaml"
    default_registry.write_text(registry.read_text(encoding="utf-8"), encoding="utf-8")
    payload["checkpoint_registry_sha256"] = sha256_file(default_registry)
    _write_receipt(receipt, payload)

    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.get_repository_root",
        lambda: repo_fake,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: _cfg.references,
    )

    result = validate_checkpoint_staging_receipt(
        cfg,
        receipt,
        campaign_config_path=config,
        repo_root=None,
        now=datetime(2026, 8, 21, 13, tzinfo=UTC),
    )
    assert result["submit_safe"] is True
