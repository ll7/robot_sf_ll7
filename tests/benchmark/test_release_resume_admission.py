"""Tests for infrastructure-only same-campaign release resume admission."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.release_resume_admission import (
    RELEASE_RESUME_RECEIPT_SCHEMA,
    ReleaseResumeAdmissionError,
    validate_release_resume_admission,
)

if TYPE_CHECKING:
    from pathlib import Path


_SOURCE_SHA = "a" * 40
_NOW = datetime(2026, 8, 22, 6, 0, tzinfo=UTC)


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "campaign"
    (root / "runs" / "goal__differential_drive").mkdir(parents=True)
    (root / "runs" / "goal__differential_drive" / "episodes.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    config = tmp_path / "campaign.yaml"
    config.write_text("horizon: 600\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}\n", encoding="utf-8")
    manifest = root / "campaign_manifest.json"
    _write(
        manifest,
        {
            "campaign_id": "fixed-release",
            "git": {"commit": _SOURCE_SHA},
        },
    )
    return root, config, checkpoint, manifest


def _receipt(
    path: Path,
    *,
    config: Path,
    checkpoint: Path,
    manifest: Path,
    **updates: object,
) -> Path:
    payload: dict[str, object] = {
        "schema_version": RELEASE_RESUME_RECEIPT_SCHEMA,
        "status": "approved",
        "resume_same_campaign": True,
        "interruption_class": "infrastructure",
        "interruption_reason": "node_failure",
        "created_at_utc": _NOW.isoformat(),
        "campaign_id": "fixed-release",
        "source_commit": _SOURCE_SHA,
        "campaign_config_sha256": sha256_file(config),
        "checkpoint_receipt_sha256": sha256_file(checkpoint),
        "prior_campaign_manifest_sha256": sha256_file(manifest),
    }
    payload.update(updates)
    _write(path, payload)
    return path


def _validate(
    root: Path,
    config: Path,
    checkpoint: Path,
    receipt: Path | None,
    **kwargs: object,
) -> dict[str, object] | None:
    return validate_release_resume_admission(
        campaign_root=root,
        campaign_id="fixed-release",
        campaign_config_path=config,
        checkpoint_receipt_path=checkpoint,
        current_source_commit=_SOURCE_SHA,
        resume_enabled=True,
        resume_receipt_path=receipt,
        now=_NOW,
        **kwargs,
    )


def test_fresh_campaign_needs_no_resume_receipt(tmp_path: Path) -> None:
    root = tmp_path / "fresh"
    config = tmp_path / "campaign.yaml"
    checkpoint = tmp_path / "checkpoint.json"
    config.write_text("horizon: 600\n", encoding="utf-8")
    checkpoint.write_text("{}\n", encoding="utf-8")
    assert _validate(root, config, checkpoint, None) is None


def test_existing_campaign_requires_receipt(tmp_path: Path) -> None:
    root, config, checkpoint, _manifest = _setup(tmp_path)
    with pytest.raises(ReleaseResumeAdmissionError, match="infrastructure-only"):
        _validate(root, config, checkpoint, None)


def test_valid_infrastructure_receipt_binds_all_inputs(tmp_path: Path) -> None:
    root, config, checkpoint, manifest = _setup(tmp_path)
    receipt = _receipt(
        tmp_path / "resume.json", config=config, checkpoint=checkpoint, manifest=manifest
    )
    result = _validate(root, config, checkpoint, receipt)
    assert result is not None
    assert result["interruption_reason"] == "node_failure"
    assert result["source_commit"] == _SOURCE_SHA
    assert result["sha256"] == sha256_file(receipt)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"interruption_class": "code_defect"}, "infrastructure-only"),
        ({"interruption_reason": "config_fix"}, "reason is not admitted"),
        ({"source_commit": "b" * 40}, "source commit does not match"),
        ({"campaign_id": "other"}, "campaign_id does not match"),
        ({"checkpoint_receipt_sha256": "0" * 64}, "checkpoint hash does not match"),
        ({"campaign_config_sha256": "0" * 64}, "config hash does not match"),
        ({"prior_campaign_manifest_sha256": "0" * 64}, "manifest hash does not match"),
        ({"zenodo_token": "do-not-store"}, "credential-shaped"),
    ],
)
def test_resume_receipt_rejects_non_infrastructure_or_drift(
    tmp_path: Path, updates: dict[str, object], message: str
) -> None:
    root, config, checkpoint, manifest = _setup(tmp_path)
    receipt = _receipt(
        tmp_path / "resume.json",
        config=config,
        checkpoint=checkpoint,
        manifest=manifest,
        **updates,
    )
    with pytest.raises(ReleaseResumeAdmissionError, match=message):
        _validate(root, config, checkpoint, receipt)


def test_resume_receipt_rejects_stale_and_fresh_misuse(tmp_path: Path) -> None:
    root, config, checkpoint, manifest = _setup(tmp_path)
    receipt = _receipt(
        tmp_path / "resume.json",
        config=config,
        checkpoint=checkpoint,
        manifest=manifest,
        created_at_utc=(_NOW - timedelta(hours=25)).isoformat(),
    )
    with pytest.raises(ReleaseResumeAdmissionError, match="stale"):
        _validate(root, config, checkpoint, receipt)

    fresh_root = tmp_path / "fresh"
    with pytest.raises(ReleaseResumeAdmissionError, match="only valid"):
        _validate(fresh_root, config, checkpoint, receipt)


def test_resume_rejects_source_change_even_with_matching_receipt(tmp_path: Path) -> None:
    root, config, checkpoint, manifest = _setup(tmp_path)
    prior = json.loads(manifest.read_text(encoding="utf-8"))
    prior["git"]["commit"] = "b" * 40
    _write(manifest, prior)
    receipt = _receipt(
        tmp_path / "resume.json", config=config, checkpoint=checkpoint, manifest=manifest
    )
    with pytest.raises(ReleaseResumeAdmissionError, match="cannot cross a source commit"):
        _validate(root, config, checkpoint, receipt)
