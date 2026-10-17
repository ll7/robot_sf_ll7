"""Offline release-path regressions; no planner or environment execution."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.camera_ready._preflight import _setup_campaign_directories
from scripts.tools import run_benchmark_release as release


@pytest.mark.parametrize("campaign_id", ["REL008_Round22C", "REL008 Round22C"])
def test_mixed_case_campaign_retains_spawn_reports_and_requires_resume_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, campaign_id: str
) -> None:
    """Spawn report custody and resume lookup follow the actual campaign directory."""
    output_root = tmp_path / "campaigns"
    cfg = SimpleNamespace(resume=True)
    monkeypatch.setattr(
        release,
        "run_manifest_preflight",
        lambda *_args, **_kwargs: {
            "schema_version": "spawn_matrix_preflight.v1",
            "status": "valid",
            "cells": [],
        },
    )
    preflight_root = release._fixed_campaign_root(output_root=output_root, campaign_id=campaign_id)
    summary, _ = release._run_spawn_matrix_preflight(
        manifest=SimpleNamespace(), campaign_root=preflight_root, source_commit=None, workers=1
    )
    actual_id, actual_root, _, _ = _setup_campaign_directories(
        cfg, output_root=output_root, label=None, campaign_id=campaign_id
    )
    assert actual_id == "rel008_round22c"
    assert actual_root == output_root / "rel008_round22c"
    release._assert_spawn_preflight_report_identity(actual_root, summary)
    assert preflight_root == actual_root
    assert not (output_root / campaign_id).exists()

    runs = actual_root / "runs" / "goal__differential_drive"
    runs.mkdir(parents=True)
    (runs / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        campaign_id=campaign_id,
        output_root=output_root,
        resume_receipt=None,
        resume_receipt_max_age_hours=24.0,
    )
    with pytest.raises(
        release.ReleaseResumeAdmissionError, match="infrastructure-only resume receipt"
    ):
        release._admit_release_resume(
            args=args,
            cfg=cfg,
            campaign_config_path=tmp_path / "campaign.yaml",
            checkpoint_receipt_path=tmp_path / "checkpoint.json",
        )

    def check_resume_identity(**kwargs):
        assert kwargs["campaign_root"] == output_root / "rel008_round22c"
        assert kwargs["campaign_id"] == "rel008_round22c"

    monkeypatch.setattr(release, "validate_release_resume_admission", check_resume_identity)
    release._admit_release_resume(
        args=args,
        cfg=cfg,
        campaign_config_path=tmp_path / "campaign.yaml",
        checkpoint_receipt_path=tmp_path / "checkpoint.json",
    )
