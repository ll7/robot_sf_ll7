"""Tests for the benchmark release CLI."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf.benchmark.camera_ready_campaign import CampaignConfig, PlannerSpec, SeedPolicy
from robot_sf.benchmark.orca_preflight import OrcaRvo2PreflightError
from robot_sf.benchmark.release_protocol import load_release_manifest
from scripts.tools import rebuild_campaign_reports_from_rows, run_benchmark_release


def _write_json(path: Path, payload: dict) -> None:
    """Write an indented JSON release fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_campaign_tree(tmp_path: Path) -> Path:
    """Build the minimal campaign artifact tree expected by release checks."""
    campaign_root = tmp_path / "out" / "campaign_release"
    _write_json(campaign_root / "campaign_manifest.json", {"campaign_id": "campaign_release"})
    _write_json(campaign_root / "manifest.json", {"schema_version": "benchmark-run-manifest.v1"})
    _write_json(campaign_root / "run_meta.json", {"runtime_sec": 1.0})
    _write_json(campaign_root / "preflight" / "validate_config.json", {"valid": True})
    _write_json(campaign_root / "preflight" / "preview_scenarios.json", {"scenario_count": 1})
    _write_json(
        campaign_root / "reports" / "campaign_summary.json",
        {
            "campaign": {
                "repository_url": "https://github.com/ll7/robot_sf_ll7",
                "doi": "10.5281/zenodo.<record-id>",
            },
            "benchmark_success": True,
        },
    )
    (campaign_root / "reports" / "campaign_report.md").write_text("# Report\n", encoding="utf-8")
    _write_json(campaign_root / "reports" / "matrix_summary.json", {"rows": []})
    (campaign_root / "reports" / "campaign_table.md").write_text("|planner|\n", encoding="utf-8")
    _write_json(campaign_root / "reports" / "snqi_diagnostics.json", {"contract_status": "pass"})
    return campaign_root


def _manifest_fixture() -> SimpleNamespace:
    """Return a minimal valid release-manifest stub for CLI tests."""
    return SimpleNamespace(
        canonical_campaign_config_path=Path("configs/benchmarks/paper_experiment_matrix_v1.yaml"),
        required_artifact_paths=(
            "campaign_manifest.json",
            "manifest.json",
            "run_meta.json",
            "preflight/validate_config.json",
            "preflight/preview_scenarios.json",
            "reports/campaign_summary.json",
            "reports/campaign_report.md",
            "reports/matrix_summary.json",
            "reports/campaign_table.md",
            "reports/snqi_diagnostics.json",
        ),
        release_tag="paper-benchmark-smoke-v0.1.0",
        planner_keys=(),
        doi="10.5281/zenodo.<record-id>",
        repository_url="https://github.com/ll7/robot_sf_ll7",
    )


def test_required_artifact_check_is_campaign_contained_and_regular(tmp_path: Path) -> None:
    """Runtime artifact admission rejects escapes, links, and non-files."""
    campaign_root = tmp_path / "campaign"
    (campaign_root / "reports").mkdir(parents=True)
    (campaign_root / "reports" / "safe.json").write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (campaign_root / "reports" / "escape.json").symlink_to(outside)

    required = (
        "reports/safe.json",
        "reports/escape.json",
        "../outside.json",
        str(outside),
        "reports",
    )

    expected_missing = [
        "reports/escape.json",
        "../outside.json",
        str(outside),
        "reports",
    ]
    assert run_benchmark_release._required_artifacts_missing(campaign_root, required) == (
        expected_missing
    )
    assert (
        rebuild_campaign_reports_from_rows._required_artifacts_missing(campaign_root, required)
        == expected_missing
    )


@pytest.mark.parametrize(
    "record",
    (
        run_benchmark_release._record_publication_payload,
        rebuild_campaign_reports_from_rows._record_publication_payload,
    ),
)
def test_campaign_summary_record_rejects_symlink_before_merge(tmp_path: Path, record) -> None:
    """Release summary writes do not follow a symlink before reading or merging."""
    campaign_root = _make_campaign_tree(tmp_path)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    outside = tmp_path / "outside-summary.json"
    outside.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
    summary_path.unlink()
    summary_path.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        record(campaign_root, {"bundle_dir": "outside"})


def _admit_checkpoint_receipt(monkeypatch, tmp_path: Path) -> Path:
    """Install a valid receipt stub for runner-focused tests."""
    receipt = tmp_path / "checkpoint_staging_receipt.json"
    _write_json(receipt, {"generated_at_utc": "2026-08-21T00:00:00Z"})
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_checkpoint_staging_receipt",
        lambda *args, **kwargs: {"generated_at_utc": "2026-08-21T00:00:00Z"},
    )
    monkeypatch.setattr(run_benchmark_release, "get_repository_root", lambda: tmp_path)
    return receipt


def _rehearsal_fixture(tmp_path: Path) -> tuple[SimpleNamespace, SimpleNamespace, Path, Path]:
    """Create repository-local inputs for no-campaign rehearsal tests."""
    manifest_path = tmp_path / "configs" / "release.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("schema_version: benchmark-release-manifest.v0.2\n", encoding="utf-8")
    config_path = tmp_path / "configs" / "campaign.yaml"
    config_path.write_text("name: rehearsal\n", encoding="utf-8")
    planner = SimpleNamespace(
        key="goal",
        algo="goal",
        planner_group="core",
        enabled=True,
    )
    cfg = SimpleNamespace(planners=(planner,), kinematics_matrix=("differential_drive",))
    manifest = SimpleNamespace(
        path=manifest_path,
        schema_version="benchmark-release-manifest.v0.2",
        source_sha="a" * 40,
        canonical_campaign_config_path=config_path,
        planner_keys=("goal",),
        planner_groups={"goal": "core"},
        expected_kinematics_matrix=("differential_drive",),
    )
    checkpoint = tmp_path / "receipt.json"
    checkpoint_arm = {
        "planner_key": "goal",
        "algo": "goal",
        "kind": "model_id",
        "value": "goal-model",
        "implicit": False,
        "checkpoint_sha256": "d" * 64,
    }
    _write_json(
        checkpoint,
        {
            "campaign_config_sha256": "f" * 64,
            "arms": [checkpoint_arm],
        },
    )
    runtime_checkpoint = tmp_path / "runtime-receipt.json"
    _write_json(
        runtime_checkpoint,
        {
            "campaign_config_sha256": "r" * 64,
            "arms": [checkpoint_arm],
        },
    )
    smoke = tmp_path / "smoke" / "release_result.json"
    smoke.parent.mkdir()
    _write_json(
        smoke,
        {
            "checkpoint_staging_receipt": {
                "path": runtime_checkpoint.name,
                "sha256": run_benchmark_release.sha256_file(runtime_checkpoint),
            }
        },
    )
    return manifest, cfg, checkpoint, smoke


def _patch_valid_rehearsal_admissions(monkeypatch, tmp_path: Path) -> None:
    """Patch external and release-data admissions while retaining roster logic."""
    monkeypatch.setattr(run_benchmark_release, "get_repository_root", lambda: tmp_path)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: object())
    monkeypatch.setattr(run_benchmark_release, "_current_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(run_benchmark_release, "_current_worktree_clean", lambda: True)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", lambda cfg: None)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda *args, **kwargs: {"status": "valid", "problem_count": 0, "problems": []},
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_checkpoint_staging_receipt",
        lambda *args, **kwargs: {
            "generated_at_utc": "2026-08-27T00:00:00Z",
            "submit_safe": True,
            "arms": run_benchmark_release._read_json(tmp_path / "receipt.json")["arms"],
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_runtime_smoke_result",
        lambda *args, **kwargs: {
            "status": "admitted",
            "result_sha256": "b" * 64,
            "checkpoint_receipt_sha256": run_benchmark_release.sha256_file(
                tmp_path / "runtime-receipt.json"
            ),
            "source_commit": "a" * 40,
            "planner_arms": 1,
            "episode_cells": 1,
            "fallback_or_degraded_rows": 0,
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_resolved_release_manifest",
        lambda *args, **kwargs: {"release_id": "rehearsal"},
    )


def test_canonical_rehearsal_manifest_requires_explicit_source_identity(capsys) -> None:
    """The historical canonical manifest cannot silently admit an unpinned checkout."""
    manifest = load_release_manifest(
        Path("configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml")
    )

    assert manifest.source_sha is None
    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml",
            "--mode",
            "rehearsal",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "source_identity_rejected"
    assert "--source-commit" in payload["status_reason"]
    assert payload["campaign_execution_status"] == "not_started"


def test_rehearsal_rejects_mismatched_explicit_source_identity(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """An explicit source pin must match the clean checked-out source exactly."""
    manifest, _cfg, _checkpoint, _smoke = _rehearsal_fixture(tmp_path)
    manifest.source_sha = None
    _patch_valid_rehearsal_admissions(monkeypatch, tmp_path)
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)

    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "configs/release.yaml",
            "--mode",
            "rehearsal",
            "--source-commit",
            "b" * 40,
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "startup_admission_failed"
    assert "does not match the rehearsal source pin" in payload["status_reason"]
    assert payload["startup_admission"]["source_identity"]["status"] == "rejected"
    assert payload["campaign_execution_status"] == "not_started"


def test_rehearsal_explicit_source_identity_accepts_historical_manifest_pin() -> None:
    """Historical manifests can be rehearsed only with an explicit exact source pin."""
    manifest = load_release_manifest(
        Path("configs/benchmarks/releases/benchmark_data_release_s30_h600.yaml")
    )

    expected, evidence = run_benchmark_release._resolve_rehearsal_source_identity(
        manifest, "a" * 40
    )

    assert expected == "a" * 40
    assert evidence["source"] == "explicit_argument"
    assert evidence["manifest_source_sha"] is None


def test_rehearsal_rejects_explicit_source_identity_drift_from_manifest(tmp_path: Path) -> None:
    """An explicit pin cannot override a manifest-declared source identity."""
    _manifest, _cfg, _checkpoint, _smoke = _rehearsal_fixture(tmp_path)
    manifest = SimpleNamespace(source_sha="a" * 40)

    with pytest.raises(ValueError, match="does not match manifest source_sha"):
        run_benchmark_release._resolve_rehearsal_source_identity(manifest, "b" * 40)


def test_source_commit_option_is_rehearsal_only(capsys, tmp_path: Path) -> None:
    """Run and preflight modes must not silently ignore a rehearsal-only source pin."""
    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            str(tmp_path / "release.yaml"),
            "--source-commit",
            "a" * 40,
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "unsupported_combination"
    assert "only accepted in rehearsal mode" in payload["status_reason"]


def test_release_rehearsal_admits_inputs_without_campaign_side_effects(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """A successful rehearsal reports every gate and never starts campaign execution."""
    manifest, cfg, _checkpoint, _smoke = _rehearsal_fixture(tmp_path)
    manifest.source_sha = None
    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    called = {"campaign": False, "preflight": False}
    _patch_valid_rehearsal_admissions(monkeypatch, tmp_path)
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda *args, **kwargs: called.__setitem__("campaign", True),
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "prepare_campaign_preflight",
        lambda *args, **kwargs: called.__setitem__("preflight", True),
    )

    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "configs/release.yaml",
            "--mode",
            "rehearsal",
            "--source-commit",
            "a" * 40,
            "--checkpoint-receipt",
            "receipt.json",
            "--runtime-smoke-receipt",
            "smoke/release_result.json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "release_rehearsal_passed"
    assert payload["campaign_execution_status"] == "not_started"
    assert payload["campaign_output_created"] is False
    assert payload["checkpoint_staging_admission"]["status"] == "admitted"
    assert payload["runtime_smoke_admission"]["status"] == "admitted"
    assert payload["checkpoint_identity_admission"]["status"] == "admitted"
    assert (
        payload["checkpoint_identity_admission"]["release_receipt_sha256"]
        != payload["checkpoint_identity_admission"]["runtime_smoke_receipt_sha256"]
    )
    assert payload["planner_roster_admission"]["status"] == "valid"
    assert payload["release_inputs"]["manifest_path"] == "configs/release.yaml"
    assert payload["startup_admission"]["source_identity"]["source"] == "explicit_argument"
    assert payload["startup_admission"]["source_identity"]["checked_out_source_commit"] == "a" * 40
    assert called == {"campaign": False, "preflight": False}


def test_release_rehearsal_rejects_checkpoint_identity_drift(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """Runtime smoke must retain the same checkpoint arm and model-byte identities."""
    manifest, cfg, _checkpoint, smoke = _rehearsal_fixture(tmp_path)
    _patch_valid_rehearsal_admissions(monkeypatch, tmp_path)
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    runtime_checkpoint = tmp_path / "runtime-receipt.json"
    runtime_payload = run_benchmark_release._read_json(runtime_checkpoint)
    runtime_payload["arms"][0]["checkpoint_sha256"] = "e" * 64
    _write_json(runtime_checkpoint, runtime_payload)
    smoke_payload = run_benchmark_release._read_json(smoke)
    smoke_payload["checkpoint_staging_receipt"]["sha256"] = run_benchmark_release.sha256_file(
        runtime_checkpoint
    )
    _write_json(smoke, smoke_payload)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_runtime_smoke_result",
        lambda *args, **kwargs: {
            "status": "admitted",
            "result_sha256": "b" * 64,
            "checkpoint_receipt_sha256": run_benchmark_release.sha256_file(runtime_checkpoint),
        },
    )

    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "configs/release.yaml",
            "--mode",
            "rehearsal",
            "--checkpoint-receipt",
            "receipt.json",
            "--runtime-smoke-receipt",
            "smoke/release_result.json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "checkpoint_identity_mismatch"
    assert payload["campaign_execution_status"] == "not_started"
    assert payload["runtime_smoke_admission"]["status"] == "admitted"
    assert payload["checkpoint_identity_admission"]["status"] == "rejected"
    assert "arm identities" in payload["checkpoint_identity_admission"]["blockers"][0]


def test_release_rehearsal_rejects_resume_age_option(monkeypatch, capsys, tmp_path: Path) -> None:
    """Resume-only age tuning is not silently accepted by the no-campaign mode."""
    manifest, _cfg, _checkpoint, _smoke = _rehearsal_fixture(tmp_path)
    monkeypatch.setattr(run_benchmark_release, "get_repository_root", lambda: tmp_path)
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)

    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "configs/release.yaml",
            "--mode",
            "rehearsal",
            "--resume-receipt-max-age-hours",
            "12",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "unsupported_combination"
    assert "resume-receipt-max-age-hours" in payload["status_reason"]


@pytest.mark.parametrize("empty_option", ["--label=", "--campaign-id="])
def test_release_rehearsal_rejects_empty_allocation_options(
    capsys, tmp_path: Path, empty_option: str
) -> None:
    """Empty allocation options are still explicit unsupported rehearsal inputs."""
    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            str(tmp_path / "release.yaml"),
            "--mode",
            "rehearsal",
            empty_option,
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "unsupported_combination"
    assert empty_option.split("=", maxsplit=1)[0] in payload["status_reason"]


def test_release_rehearsal_rejects_stale_planner_roster_before_receipt_admission(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """A planner-roster drift cannot reach checkpoint or campaign execution."""
    manifest, cfg, _checkpoint, _smoke = _rehearsal_fixture(tmp_path)
    cfg.planners = (SimpleNamespace(key="orca", algo="orca", planner_group="core", enabled=True),)
    _patch_valid_rehearsal_admissions(monkeypatch, tmp_path)
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    called = {"checkpoint": False, "runtime": False, "campaign": False}
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_checkpoint_staging_receipt",
        lambda *args, **kwargs: called.__setitem__("checkpoint", True),
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_runtime_smoke_result",
        lambda *args, **kwargs: called.__setitem__("runtime", True),
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda *args, **kwargs: called.__setitem__("campaign", True),
    )

    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "configs/release.yaml",
            "--mode",
            "rehearsal",
            "--checkpoint-receipt",
            "receipt.json",
            "--runtime-smoke-receipt",
            "smoke/release_result.json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "planner_roster_rejected"
    assert payload["planner_roster_admission"]["status"] == "invalid"
    assert called == {"checkpoint": False, "runtime": False, "campaign": False}


def test_release_rehearsal_serializes_manifest_admission_failure(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """Unexpected manifest I/O is still reported as a structured stop receipt."""
    manifest, cfg, _checkpoint, _smoke = _rehearsal_fixture(tmp_path)
    _patch_valid_rehearsal_admissions(monkeypatch, tmp_path)
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("manifest disappeared")),
    )

    exit_code = run_benchmark_release.main(
        ["--manifest", "configs/release.yaml", "--mode", "rehearsal"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "manifest_rejected"
    assert payload["campaign_execution_status"] == "not_started"


def test_release_rehearsal_serializes_malformed_manifest_yaml(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """Malformed YAML manifest returns structured JSON without unhandled traceback."""
    _rehearsal_fixture(tmp_path)
    monkeypatch.setattr(run_benchmark_release, "get_repository_root", lambda: tmp_path)
    bad_manifest = tmp_path / "configs" / "bad_manifest.yaml"
    bad_manifest.write_text("bad: [yaml: invalid\n", encoding="utf-8")

    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "configs/bad_manifest.yaml",
            "--mode",
            "rehearsal",
            "--source-commit",
            "a" * 40,
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "manifest_rejected"
    assert "release manifest could not be admitted" in payload["status_reason"]
    assert payload["campaign_execution_status"] == "not_started"
    assert payload["benchmark_success"] is False
    assert payload["release_exit_code"] == 2


def test_release_rehearsal_serializes_malformed_campaign_config_yaml(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """Malformed campaign config YAML returns structured startup admission failure."""
    manifest, _cfg, _checkpoint, _smoke = _rehearsal_fixture(tmp_path)
    monkeypatch.setattr(run_benchmark_release, "get_repository_root", lambda: tmp_path)
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "_current_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(run_benchmark_release, "_current_worktree_clean", lambda: True)
    bad_config = tmp_path / "configs" / "bad_config.yaml"
    bad_config.write_text("bad: [config: invalid\n", encoding="utf-8")
    manifest.canonical_campaign_config_path = bad_config

    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "configs/release.yaml",
            "--mode",
            "rehearsal",
            "--source-commit",
            "a" * 40,
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "startup_admission_failed"
    assert "release rehearsal startup admission failed" in payload["status_reason"]
    assert payload["campaign_execution_status"] == "not_started"
    assert payload["benchmark_success"] is False
    assert payload["release_exit_code"] == 2


def test_release_rehearsal_admits_inputs_from_unrelated_cwd_with_real_receipt_validation(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """Rehearsal succeeds with real receipt validation when invoked from an unrelated CWD."""
    manifest, cfg, _checkpoint, smoke = _rehearsal_fixture(tmp_path)
    model_file = tmp_path / "model.zip"
    model_file.write_bytes(b"checkpoint-bytes")

    default_registry_dir = tmp_path / "model"
    default_registry_dir.mkdir(parents=True, exist_ok=True)
    default_registry = default_registry_dir / "registry.yaml"
    default_registry.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "models": [
                    {
                        "model_id": "goal-model",
                        "github_release": {"sha256": run_benchmark_release.sha256_file(model_file)},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    receipt_payload = {
        "schema_version": "campaign-checkpoint-staging-receipt.v1",
        "status": "ok",
        "mode": "enforced_staged",
        "stage": True,
        "submit_safe": True,
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "campaign_config_sha256": run_benchmark_release.sha256_file(
            manifest.canonical_campaign_config_path
        ),
        "checkpoint_registry_sha256": run_benchmark_release.sha256_file(default_registry),
        "arms": [
            {
                "planner_key": "goal",
                "algo": "goal",
                "kind": "model_id",
                "value": "goal-model",
                "implicit": False,
                "status": "staged",
                "resolved_path": str(model_file),
                "checkpoint_sha256": run_benchmark_release.sha256_file(model_file),
                "hash_source": "computed_file",
            }
        ],
    }
    _write_json(tmp_path / "receipt.json", receipt_payload)
    runtime_checkpoint = tmp_path / "runtime-receipt.json"
    _write_json(
        runtime_checkpoint,
        {
            "campaign_config_sha256": run_benchmark_release.sha256_file(
                manifest.canonical_campaign_config_path
            ),
            "arms": receipt_payload["arms"],
        },
    )
    _write_json(
        smoke,
        {
            "checkpoint_staging_receipt": {
                "path": runtime_checkpoint.name,
                "sha256": run_benchmark_release.sha256_file(runtime_checkpoint),
            }
        },
    )

    monkeypatch.setattr(run_benchmark_release, "get_repository_root", lambda: tmp_path)
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    monkeypatch.setattr(run_benchmark_release, "_current_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(run_benchmark_release, "_current_worktree_clean", lambda: True)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", lambda cfg: None)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda *args, **kwargs: {"status": "valid", "problem_count": 0, "problems": []},
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_runtime_smoke_result",
        lambda *args, **kwargs: {
            "status": "admitted",
            "result_sha256": "b" * 64,
            "checkpoint_receipt_sha256": run_benchmark_release.sha256_file(
                tmp_path / "receipt.json"
            ),
            "source_commit": "a" * 40,
            "planner_arms": 1,
            "episode_cells": 1,
            "fallback_or_degraded_rows": 0,
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_resolved_release_manifest",
        lambda *args, **kwargs: {"release_id": "rehearsal"},
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.checkpoint_staging_receipt.iter_campaign_arm_checkpoint_references",
        lambda _cfg: [
            SimpleNamespace(
                planner_key="goal",
                algo="goal",
                kind="model_id",
                value="goal-model",
                implicit=False,
            )
        ],
    )

    unrelated_cwd = tmp_path / "unrelated_working_directory"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "configs/release.yaml",
            "--mode",
            "rehearsal",
            "--source-commit",
            "a" * 40,
            "--checkpoint-receipt",
            "receipt.json",
            "--runtime-smoke-receipt",
            "smoke/release_result.json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "release_rehearsal_passed"
    assert payload["checkpoint_staging_admission"]["status"] == "admitted"
    assert payload["campaign_execution_status"] == "not_started"


def test_release_rehearsal_fails_closed_on_checkpoint_receipt(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """Checkpoint admission failure stops the no-campaign path before runtime smoke."""
    manifest, cfg, _checkpoint, _smoke = _rehearsal_fixture(tmp_path)
    _patch_valid_rehearsal_admissions(monkeypatch, tmp_path)
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    called = {"runtime": False, "campaign": False}
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_checkpoint_staging_receipt",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            run_benchmark_release.CheckpointStagingReceiptError("stale receipt")
        ),
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_runtime_smoke_result",
        lambda *args, **kwargs: called.__setitem__("runtime", True),
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda *args, **kwargs: called.__setitem__("campaign", True),
    )

    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "configs/release.yaml",
            "--mode",
            "rehearsal",
            "--checkpoint-receipt",
            "receipt.json",
            "--runtime-smoke-receipt",
            "smoke/release_result.json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "checkpoint_receipt_rejected"
    assert payload["checkpoint_staging_admission"]["status"] == "rejected"
    assert called == {"runtime": False, "campaign": False}


def test_release_input_path_rejects_external_location_without_leaking_it(
    monkeypatch, tmp_path: Path
) -> None:
    """Publication provenance cannot contain an absolute path outside the release worktree."""
    repo = tmp_path / "repo"
    repo.mkdir()
    external = tmp_path / "private" / "receipt.json"
    external.parent.mkdir()
    external.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(run_benchmark_release, "get_repository_root", lambda: repo)

    try:
        run_benchmark_release._required_repo_relative(external)
    except ValueError as exc:
        assert str(external) not in str(exc)
        assert "inside the repository worktree" in str(exc)
    else:
        raise AssertionError("external release input was not rejected")


def test_public_release_invocation_omits_absolute_scheduler_paths(
    monkeypatch, tmp_path: Path
) -> None:
    """Public provenance retains the entrypoint but not private launch paths."""
    repo = tmp_path / "repo"
    manifest = repo / "configs" / "benchmarks" / "release.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("schema_version: benchmark-release-manifest.v0.2\n", encoding="utf-8")
    monkeypatch.setattr(run_benchmark_release, "get_repository_root", lambda: repo)
    monkeypatch.setattr(
        run_benchmark_release.sys,
        "executable",
        "/home/luttkule/private/release-worktree/.venv/bin/python",
    )

    command = run_benchmark_release._public_release_invocation(str(manifest), "run")

    assert command == (
        "python scripts/tools/run_benchmark_release.py "
        "--manifest configs/benchmarks/release.yaml --mode run"
    )
    assert "/home/" not in command
    assert not run_benchmark_release.find_offending_paths({"invoked_release_command": command})


def test_local_stress_run_rejects_dirty_worktree(monkeypatch, capsys, tmp_path: Path) -> None:
    """The runner applies the exact-source clean-worktree gate outside SLURM too."""
    manifest = SimpleNamespace(
        schema_version="benchmark-release-manifest.v0.1",
        release_kind="benchmark-stress-smoke",
        maturity="diagnostic",
        canonical_campaign_config_path=Path("campaign.yaml"),
    )
    observed: dict[str, object] = {}

    def _fake_identity(*args, **kwargs):
        observed.update(kwargs)
        return {
            "status": "invalid",
            "runtime_source_commit": "a" * 40,
            "blockers": ["dirty worktree"],
        }

    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: object())
    monkeypatch.setattr(run_benchmark_release, "_current_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(run_benchmark_release, "_current_worktree_clean", lambda: False)
    monkeypatch.setattr(run_benchmark_release, "_private_stress_launch", lambda: False)
    monkeypatch.setattr(
        run_benchmark_release, "validate_stress_smoke_runtime_identity", _fake_identity
    )

    exit_code = run_benchmark_release.main(["--manifest", "manifest.yaml"])

    assert exit_code == 2
    assert observed["worktree_clean"] is False
    assert observed["require_clean_worktree"] is True
    assert json.loads(capsys.readouterr().out)["status"] == "stress_smoke_source_rejected"


def test_release_run_rejects_historical_campaign_artifact_identity(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """A stale fixed-campaign artifact cannot contaminate a new release bundle."""
    campaign_root = _make_campaign_tree(tmp_path)
    (campaign_root / "reports" / "stale_identity.md").write_text(
        "release_tag: 0.0.3.post1\ndoi: 10.5281/zenodo.19482025\n",
        encoding="utf-8",
    )
    manifest = _manifest_fixture()
    cfg = SimpleNamespace(export_publication_bundle=False)

    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", lambda cfg: None)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda *args, **kwargs: {"status": "valid", "problem_count": 0, "problems": []},
    )
    monkeypatch.setattr(
        run_benchmark_release, "build_resolved_release_manifest", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda *args, **kwargs: {
            "campaign_root": str(campaign_root),
            "benchmark_success": True,
            "campaign_execution_status": "completed",
            "status": "benchmark_success",
            "status_reason": "all rows succeeded",
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_release_provenance",
        lambda *args, **kwargs: {
            "benchmark_protocol_version": "0.1.0",
            "release_id": "smoke",
            "release_tag": manifest.release_tag,
            "manifest_path": "manifest.yaml",
            "manifest_sha256": "a" * 64,
            "canonical_campaign_config": "campaign.yaml",
        },
    )
    receipt = _admit_checkpoint_receipt(monkeypatch, tmp_path)

    exit_code = run_benchmark_release.main(
        ["--manifest", "manifest.yaml", "--checkpoint-receipt", str(receipt)]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "release_identity_rejected"
    assert payload["release_status"] == "release_identity_rejected"
    assert payload["release_benchmark_success"] is False
    assert "historical release identity" in payload["status_reason"]


def test_publication_identity_rejection_does_not_log_campaign_paths(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """Publication identity failures keep filesystem-derived details out of stdout."""
    secret_marker = "secret-release-path-should-not-be-logged"
    campaign_root = _make_campaign_tree(tmp_path / secret_marker)
    bundle_dir = tmp_path / secret_marker / "publication_bundle"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "stale_identity.md").write_text(
        "release_tag: 0.0.3.post1\n",
        encoding="utf-8",
    )
    manifest = _manifest_fixture()
    cfg = SimpleNamespace(export_publication_bundle=True)

    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", lambda cfg: None)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda *args, **kwargs: {"status": "valid", "problem_count": 0, "problems": []},
    )
    monkeypatch.setattr(
        run_benchmark_release, "build_resolved_release_manifest", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda *args, **kwargs: {
            "campaign_root": str(campaign_root),
            "benchmark_success": True,
            "campaign_execution_status": "completed",
            "status": "benchmark_success",
            "status_reason": "all rows succeeded",
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_release_provenance",
        lambda *args, **kwargs: {
            "benchmark_protocol_version": "0.1.0",
            "release_id": "smoke",
            "release_tag": manifest.release_tag,
            "manifest_path": "manifest.yaml",
            "manifest_sha256": "a" * 64,
            "canonical_campaign_config": "campaign.yaml",
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_full_benchmark_release_acceptance",
        lambda *args, **kwargs: {"status": "valid", "benchmark_success": True, "blockers": []},
    )
    publication_payload = {
        "bundle_dir": str(bundle_dir),
        "archive_path": str(bundle_dir.with_suffix(".tar.gz")),
        "checksums_path": str(bundle_dir / "checksums.sha256"),
        "manifest_path": str(bundle_dir / "publication_manifest.json"),
        "file_count": 1,
        "total_bytes": 32,
    }
    monkeypatch.setattr(
        run_benchmark_release,
        "_build_publication_payload",
        lambda **kwargs: publication_payload,
    )
    receipt = _admit_checkpoint_receipt(monkeypatch, tmp_path)

    exit_code = run_benchmark_release.main(
        ["--manifest", "manifest.yaml", "--checkpoint-receipt", str(receipt)]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    persisted = json.loads(
        (campaign_root / "release" / "release_result.json").read_text(encoding="utf-8")
    )
    assert exit_code == 2
    assert payload["release_status"] == "publication_identity_rejected"
    assert payload["release_benchmark_success"] is False
    assert secret_marker not in stdout
    assert secret_marker not in json.dumps(persisted)
    assert "campaign_root" not in persisted


def test_release_preflight_uses_camera_ready_preflight(monkeypatch, capsys, tmp_path: Path) -> None:
    """Preflight mode should validate the manifest and emit preflight artifact paths."""
    manifest = SimpleNamespace(
        canonical_campaign_config_path=Path("configs/benchmarks/paper_experiment_matrix_v1.yaml")
    )
    sentinel_cfg = object()
    called = {"orca_preflight": False, "campaign_id": None}

    def _fake_orca_preflight(cfg) -> None:
        """Record that release preflight applies the ORCA runtime guard."""
        assert cfg is sentinel_cfg
        called["orca_preflight"] = True

    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: sentinel_cfg)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", _fake_orca_preflight)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda manifest, campaign_config=None: {
            "status": "valid",
            "problem_count": 0,
            "problems": [],
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_resolved_release_manifest",
        lambda manifest, campaign_config=None: {"release_id": "rid"},
    )

    def _fake_prepare_campaign_preflight(cfg, **kwargs):
        assert cfg is sentinel_cfg
        called["campaign_id"] = kwargs["campaign_id"]
        return {
            "campaign_id": "cid",
            "campaign_root": tmp_path / "out" / "cid",
            "validate_config_path": tmp_path / "out" / "cid" / "preflight" / "validate_config.json",
            "preview_scenarios_path": tmp_path
            / "out"
            / "cid"
            / "preflight"
            / "preview_scenarios.json",
            "matrix_summary_json_path": tmp_path
            / "out"
            / "cid"
            / "reports"
            / "matrix_summary.json",
            "matrix_summary_csv_path": tmp_path / "out" / "cid" / "reports" / "matrix_summary.csv",
        }

    monkeypatch.setattr(
        run_benchmark_release,
        "prepare_campaign_preflight",
        _fake_prepare_campaign_preflight,
    )

    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "manifest.yaml",
            "--mode",
            "preflight",
            "--campaign-id",
            "fixed-preflight",
        ],
    )

    assert exit_code == 0
    assert called["orca_preflight"] is True
    assert called["campaign_id"] == "fixed-preflight"
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_validation"]["status"] == "valid"
    assert payload["campaign_id"] == "cid"


def test_release_run_fails_closed_on_invalid_manifest(monkeypatch, capsys) -> None:
    """Invalid release manifests must stop before campaign execution."""
    manifest = SimpleNamespace(
        canonical_campaign_config_path=Path("configs/benchmarks/paper_experiment_matrix_v1.yaml")
    )
    sentinel_cfg = object()
    called = {"run": False}

    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: sentinel_cfg)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", lambda cfg: None)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda manifest, campaign_config=None: {
            "status": "invalid",
            "problem_count": 1,
            "problems": ["bad"],
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_resolved_release_manifest",
        lambda manifest, campaign_config=None: {"release_id": "rid"},
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda *args, **kwargs: called.__setitem__("run", True),
    )

    exit_code = run_benchmark_release.main(["--manifest", "manifest.yaml"])

    assert exit_code == 2
    assert called["run"] is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "invalid_manifest"
    assert payload["benchmark_success"] is False
    assert payload["campaign_execution_status"] == "failed"
    assert payload["evidence_status"] == "invalid"
    assert payload["row_status_summary"] == {
        "successful_evidence_rows": 0,
        "accepted_unavailable_rows": 0,
        "unexpected_failed_rows": 0,
        "fallback_or_degraded_rows": 0,
    }


def test_release_run_reports_orca_preflight_failure_as_structured_json(
    monkeypatch,
    capsys,
) -> None:
    """ORCA runtime failures should keep the release CLI's structured exit contract."""
    manifest = SimpleNamespace(
        canonical_campaign_config_path=Path("configs/benchmarks/paper_experiment_matrix_v1.yaml")
    )
    sentinel_cfg = object()
    called = {"validate": False, "run": False}

    def _raise_orca_preflight(_cfg) -> None:
        """Simulate the real missing-rvo2 preflight path."""
        raise OrcaRvo2PreflightError("The required optional dependency 'rvo2' is not importable.")

    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: sentinel_cfg)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", _raise_orca_preflight)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda *args, **kwargs: called.__setitem__("validate", True),
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda *args, **kwargs: called.__setitem__("run", True),
    )

    exit_code = run_benchmark_release.main(["--manifest", "manifest.yaml"])

    assert exit_code == 2
    assert called == {"validate": False, "run": False}
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "run"
    assert payload["status"] == "orca_preflight_failed"
    assert payload["status_reason"] == payload["release_status_reason"]
    assert payload["benchmark_success"] is False
    assert payload["exit_code"] == 2
    assert payload["campaign_execution_status"] == "failed"
    assert payload["evidence_status"] == "blocked"
    assert payload["row_status_summary"] == {
        "successful_evidence_rows": 0,
        "accepted_unavailable_rows": 0,
        "unexpected_failed_rows": 0,
        "fallback_or_degraded_rows": 0,
    }
    assert payload["release_status"] == "orca_preflight_failed"
    assert payload["release_exit_code"] == 2
    assert "rvo2" in payload["release_status_reason"]


def test_release_run_exports_publication_only_after_benchmark_success(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """Successful releases should export publication bundles after artifact checks pass."""
    campaign_root = _make_campaign_tree(tmp_path)
    manifest = _manifest_fixture()
    sentinel_cfg = object()
    called = {"orca_preflight": False}
    export_release_result_states: list[bool] = []
    publication_preflight_called = {"value": False}

    def _fake_orca_preflight(cfg) -> None:
        """Release runs should fail fast before campaign execution when ORCA is unavailable."""
        assert cfg is sentinel_cfg
        called["orca_preflight"] = True

    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: sentinel_cfg)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", _fake_orca_preflight)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda manifest, campaign_config=None: {
            "status": "valid",
            "problem_count": 0,
            "problems": [],
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_resolved_release_manifest",
        lambda manifest, campaign_config=None: {"release_id": "rid"},
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda cfg, **kwargs: {
            "campaign_id": "campaign_release",
            "campaign_root": str(campaign_root),
            "benchmark_success": True,
            "status": "benchmark_success",
            "campaign_execution_status": "completed",
            "evidence_status": "valid",
            "row_status_summary": {
                "successful_evidence_rows": 1,
                "accepted_unavailable_rows": 0,
                "unexpected_failed_rows": 0,
                "fallback_or_degraded_rows": 0,
            },
            "status_reason": "all planner rows were benchmark-success",
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_release_provenance",
        lambda manifest, campaign_root, invoked_command: {
            "benchmark_protocol_version": "0.1.0",
            "release_id": "rid",
            "release_tag": manifest.release_tag,
            "manifest_path": "configs/benchmarks/releases/smoke.yaml",
            "manifest_sha256": "abc",
            "canonical_campaign_config": "configs/benchmarks/paper_experiment_matrix_v1.yaml",
        },
    )

    def _fake_build_publication_payload(**kwargs):
        """Record that each refreshed export sees the final release result."""
        del kwargs
        export_release_result_states.append(
            (campaign_root / "release" / "release_result.json").is_file()
        )
        return {
            "bundle_dir": "output/benchmarks/publication/bundle",
            "archive_path": "output/benchmarks/publication/bundle.tar.gz",
            "checksums_path": "output/benchmarks/publication/bundle/checksums.sha256",
            "manifest_path": "output/benchmarks/publication/bundle/publication_manifest.json",
            "file_count": 3,
            "total_bytes": 123,
        }

    def _fake_publication_preflight(bundle_dir: Path) -> None:
        """The final preflight must run after the release result has been written."""
        del bundle_dir
        publication_preflight_called["value"] = True
        assert (campaign_root / "release" / "release_result.json").is_file()

    monkeypatch.setattr(
        run_benchmark_release, "_build_publication_payload", _fake_build_publication_payload
    )
    monkeypatch.setattr(
        run_benchmark_release, "_run_publication_preflight", _fake_publication_preflight
    )
    receipt = _admit_checkpoint_receipt(monkeypatch, tmp_path)

    exit_code = run_benchmark_release.main(
        ["--manifest", "manifest.yaml", "--checkpoint-receipt", str(receipt)]
    )

    assert exit_code == 0
    assert called["orca_preflight"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "benchmark_success"
    assert payload["benchmark_success"] is True
    assert payload["campaign_execution_status"] == "completed"
    assert payload["evidence_status"] == "valid"
    assert payload["exit_code"] == 0
    assert payload["release_status"] == "ok"
    assert payload["release_benchmark_success"] is True
    assert payload["release_exit_code"] == 0
    assert payload["release_status_reason"] == (
        "release artifacts validated and benchmark campaign was benchmark-success"
    )
    assert payload["benchmark_success"] is True
    assert payload["publication_bundle"]["archive_path"].endswith("bundle.tar.gz")
    assert (campaign_root / "release" / "release_result.json").exists()
    assert (campaign_root / "release" / "release_manifest.resolved.json").exists()
    assert export_release_result_states[0] is False
    assert all(export_release_result_states[1:])
    assert publication_preflight_called["value"] is True


def test_release_run_preserves_campaign_status_for_accepted_unavailable_only(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """Accepted-unavailable campaigns should keep campaign semantics in release_result.json."""
    campaign_root = _make_campaign_tree(tmp_path)
    manifest = _manifest_fixture()
    sentinel_cfg = object()
    publication_called = {"value": False}

    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: sentinel_cfg)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", lambda cfg: None)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda manifest, campaign_config=None: {
            "status": "valid",
            "problem_count": 0,
            "problems": [],
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_resolved_release_manifest",
        lambda manifest, campaign_config=None: {"release_id": "rid"},
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda cfg, **kwargs: {
            "campaign_id": "campaign_release",
            "campaign_root": str(campaign_root),
            "benchmark_success": False,
            "status": "accepted_unavailable_only",
            "campaign_execution_status": "completed",
            "evidence_status": "partial",
            "row_status_summary": {
                "successful_evidence_rows": 1,
                "accepted_unavailable_rows": 1,
                "unexpected_failed_rows": 0,
                "fallback_or_degraded_rows": 1,
            },
            "status_reason": (
                "campaign contains accepted unavailable/excluded rows and no unexpected failed rows"
            ),
            "exit_code": 3,
            "successful_runs": 1,
            "accepted_unavailable_runs": 1,
            "unexpected_failed_runs": 0,
            "non_success_runs": 1,
            "total_runs": 2,
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_release_provenance",
        lambda manifest, campaign_root, invoked_command: {
            "benchmark_protocol_version": "0.1.0",
            "release_id": "rid",
            "release_tag": manifest.release_tag,
            "manifest_path": "configs/benchmarks/releases/smoke.yaml",
            "manifest_sha256": "abc",
            "canonical_campaign_config": "configs/benchmarks/paper_experiment_matrix_v1.yaml",
        },
    )

    def _unexpected_publication(**kwargs) -> dict:
        publication_called["value"] = True
        raise AssertionError(
            "publication bundle must not export for accepted-unavailable campaigns"
        )

    monkeypatch.setattr(
        run_benchmark_release, "_build_publication_payload", _unexpected_publication
    )
    receipt = _admit_checkpoint_receipt(monkeypatch, tmp_path)

    exit_code = run_benchmark_release.main(
        ["--manifest", "manifest.yaml", "--checkpoint-receipt", str(receipt)]
    )

    assert exit_code == 3
    payload = json.loads(capsys.readouterr().out)
    release_result = json.loads(
        (campaign_root / "release" / "release_result.json").read_text(encoding="utf-8")
    )
    assert publication_called["value"] is False
    assert payload["status"] == "accepted_unavailable_only"
    assert payload["status_reason"] == (
        "campaign contains accepted unavailable/excluded rows and no unexpected failed rows"
    )
    assert payload["benchmark_success"] is False
    assert payload["campaign_execution_status"] == "completed"
    assert payload["evidence_status"] == "partial"
    assert payload["exit_code"] == 3
    assert payload["release_status"] == "accepted_unavailable_only"
    assert payload["release_status_reason"] == (
        "campaign contains accepted unavailable/excluded rows and no unexpected failed rows"
    )
    assert payload["release_benchmark_success"] is False
    assert payload["release_exit_code"] == 3
    assert release_result["status"] == "accepted_unavailable_only"
    assert release_result["exit_code"] == 3
    assert release_result["release_status"] == "accepted_unavailable_only"
    assert release_result["release_exit_code"] == 3


def test_runtime_smoke_skips_publication_when_config_disables_export(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """A release-protocol runtime smoke must not be forced through publication export."""
    campaign_root = _make_campaign_tree(tmp_path)
    manifest = _manifest_fixture()
    cfg = SimpleNamespace(export_publication_bundle=False)
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", lambda cfg: None)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda *args, **kwargs: {"status": "valid", "problem_count": 0, "problems": []},
    )
    monkeypatch.setattr(
        run_benchmark_release, "build_resolved_release_manifest", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda *args, **kwargs: {
            "campaign_root": str(campaign_root),
            "benchmark_success": True,
            "status": "benchmark_success",
            "status_reason": "all rows succeeded",
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_release_provenance",
        lambda *args, **kwargs: {
            "benchmark_protocol_version": "0.1.0",
            "release_id": "smoke",
            "release_tag": manifest.release_tag,
            "manifest_path": "smoke.yaml",
            "manifest_sha256": "a" * 64,
            "canonical_campaign_config": "smoke-config.yaml",
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "_build_publication_payload",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("publication must be skipped")),
    )
    receipt = _admit_checkpoint_receipt(monkeypatch, tmp_path)
    exit_code = run_benchmark_release.main(
        ["--manifest", "manifest.yaml", "--checkpoint-receipt", str(receipt)]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["release_benchmark_success"] is True
    assert payload["publication_requested"] is False
    assert payload["publication_preflight_status"] == "not_requested"


def test_diagnostic_stress_success_is_never_release_success(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """A valid stress smoke uses diagnostic status fields and never release success."""
    campaign_root = _make_campaign_tree(tmp_path)
    base_manifest = _manifest_fixture()
    manifest = SimpleNamespace(
        **base_manifest.__dict__,
        schema_version="benchmark-release-manifest.v0.1",
        release_kind="benchmark-stress-smoke",
        maturity="diagnostic",
        stress_smoke_review_base_commit="b" * 40,
        stress_smoke_source_policy="exact-immutable-worktree-sha-required",
    )
    cfg = SimpleNamespace(export_publication_bundle=False)
    runtime_commit = "a" * 40

    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", lambda cfg: None)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda *args, **kwargs: {"status": "valid", "problem_count": 0, "problems": []},
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_resolved_release_manifest",
        lambda *args, source_commit=None, **kwargs: {
            "provenance": {"source_commit": source_commit}
        },
    )
    monkeypatch.setattr(run_benchmark_release, "_current_source_commit", lambda: runtime_commit)
    monkeypatch.setattr(run_benchmark_release, "_current_worktree_clean", lambda: True)
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda *args, **kwargs: {
            "campaign_root": str(campaign_root),
            "benchmark_success": True,
            "status": "benchmark_success",
            "status_reason": "all rows succeeded",
            "campaign_execution_status": "completed",
            "evidence_status": "valid",
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_release_provenance",
        lambda *args, source_commit=None, **kwargs: {
            "benchmark_protocol_version": "0.1.0",
            "release_id": "stress",
            "release_tag": manifest.release_tag,
            "manifest_path": "manifest.yaml",
            "manifest_sha256": "a" * 64,
            "canonical_campaign_config": "campaign.yaml",
            "source_commit": source_commit,
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_diagnostic_stress_smoke_acceptance",
        lambda *args, **kwargs: {
            "status": "valid",
            "diagnostic_success": True,
            "blockers": [],
            "source_provenance": {"status": "valid"},
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_full_benchmark_release_acceptance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("diagnostic stress must not use full-release acceptance")
        ),
    )
    receipt = _admit_checkpoint_receipt(monkeypatch, tmp_path)

    exit_code = run_benchmark_release.main(
        ["--manifest", "manifest.yaml", "--checkpoint-receipt", str(receipt)]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["campaign_benchmark_success"] is True
    assert payload["benchmark_success"] is False
    assert payload["diagnostic_success"] is True
    assert payload["release_benchmark_success"] is False
    assert payload["release_status"] == "diagnostic_stress_smoke_passed"
    assert payload["release_status"] != "ok"
    assert payload["status"] == "diagnostic_stress_smoke_passed"
    assert payload["publication_bundle"] is None


def test_diagnostic_stress_rejects_launch_source_mismatch(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """A private launch pin that differs from checked-out HEAD blocks execution."""
    manifest = SimpleNamespace(
        **_manifest_fixture().__dict__,
        schema_version="benchmark-release-manifest.v0.1",
        release_kind="benchmark-stress-smoke",
        maturity="diagnostic",
        stress_smoke_review_base_commit="c" * 40,
        stress_smoke_source_policy="exact-immutable-worktree-sha-required",
    )
    cfg = SimpleNamespace()
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    monkeypatch.setattr(run_benchmark_release, "_current_source_commit", lambda: "a" * 40)
    monkeypatch.setenv("SLURM_EXPECTED_PUBLIC_COMMIT", "b" * 40)
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("source mismatch must block campaign execution")
        ),
    )

    exit_code = run_benchmark_release.main(["--manifest", "manifest.yaml"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["status"] == "stress_smoke_source_rejected"
    assert payload["release_benchmark_success"] is False
    assert payload["diagnostic_success"] is False
    assert payload["stress_smoke_runtime_identity"]["status"] == "invalid"


def test_future_release_rejects_checkout_drift_from_manifest_source(monkeypatch, capsys) -> None:
    """A future release cannot execute from a checkout different from source_sha."""
    manifest = SimpleNamespace(
        canonical_campaign_config_path=Path("campaign.yaml"),
        source_sha="b" * 40,
    )
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: object())
    monkeypatch.setattr(run_benchmark_release, "_current_source_commit", lambda: "a" * 40)

    exit_code = run_benchmark_release.main(["--manifest", "manifest.yaml"])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "release_source_rejected"
    assert "does not match manifest source_sha" in payload["status_reason"]
    assert payload["release_benchmark_success"] is False


def test_full_release_acceptance_failure_blocks_publication(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """A permissive campaign result cannot publish when the full gate fails."""
    campaign_root = _make_campaign_tree(tmp_path)
    manifest = SimpleNamespace(
        **_manifest_fixture().__dict__,
        schema_version="benchmark-release-manifest.v0.2",
    )
    cfg = SimpleNamespace(export_publication_bundle=True)
    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    monkeypatch.setattr(run_benchmark_release, "check_orca_rvo2_preflight", lambda cfg: None)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda *args, **kwargs: {"status": "valid", "problem_count": 0, "problems": []},
    )
    monkeypatch.setattr(
        run_benchmark_release, "build_resolved_release_manifest", lambda *a, **k: {}
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "run_campaign",
        lambda *args, **kwargs: {
            "campaign_root": str(campaign_root),
            "benchmark_success": True,
            "status": "benchmark_success",
            "status_reason": "core rows passed",
            "exit_code": 0,
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_release_provenance",
        lambda *args, **kwargs: {
            "benchmark_protocol_version": "0.1.0",
            "release_id": "full",
            "release_tag": manifest.release_tag,
            "manifest_path": "manifest.yaml",
            "manifest_sha256": "a" * 64,
            "canonical_campaign_config": "campaign.yaml",
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_full_benchmark_release_acceptance",
        lambda *args, **kwargs: {
            "status": "invalid",
            "benchmark_success": False,
            "blockers": ["trusted root /home/example/private-source is unavailable"],
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "_build_publication_payload",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("publication must be blocked by full acceptance")
        ),
    )
    receipt = _admit_checkpoint_receipt(monkeypatch, tmp_path)
    smoke_receipt = tmp_path / "runtime_smoke_result.json"
    _write_json(smoke_receipt, {})
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_runtime_smoke_result",
        lambda *args, **kwargs: {
            "schema_version": "benchmark-runtime-smoke-admission.v1",
            "status": "admitted",
        },
    )
    monkeypatch.setattr(run_benchmark_release, "_current_source_commit", lambda: "a" * 40)

    exit_code = run_benchmark_release.main(
        [
            "--manifest",
            "manifest.yaml",
            "--checkpoint-receipt",
            str(receipt),
            "--runtime-smoke-receipt",
            str(smoke_receipt),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    persisted = json.loads(
        (campaign_root / "release" / "release_result.json").read_text(encoding="utf-8")
    )
    assert exit_code == 2
    assert payload["campaign_benchmark_success"] is True
    assert payload["benchmark_success"] is False
    assert payload["release_benchmark_success"] is False
    assert payload["release_status"] == "full_release_acceptance_failed"
    assert payload["release_exit_code"] == 2
    assert payload["publication_bundle"] is None
    assert "/home/example/private-source" not in json.dumps(persisted)
    assert persisted["release_acceptance"]["blockers"] == [
        "release acceptance diagnostics contained non-public fields"
    ]


def test_release_preflight_fails_closed_when_orca_rvo2_missing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """Preflight mode should emit fail-closed JSON when enabled ORCA planners lack rvo2."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    cfg = CampaignConfig(
        name="orca_release_guard",
        scenario_matrix_path=scenario_path,
        planners=(PlannerSpec(key="orca", algo="orca"),),
        seed_policy=SeedPolicy(),
    )
    manifest = SimpleNamespace(
        canonical_campaign_config_path=Path("configs/benchmarks/paper_experiment_matrix_v1.yaml")
    )

    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: cfg)
    monkeypatch.setattr(
        run_benchmark_release,
        "validate_release_manifest",
        lambda manifest, campaign_config=None: {
            "status": "valid",
            "problem_count": 0,
            "problems": [],
        },
    )
    monkeypatch.setattr(
        run_benchmark_release,
        "build_resolved_release_manifest",
        lambda manifest, campaign_config=None: {"release_id": "rid"},
    )
    monkeypatch.setitem(sys.modules, "rvo2", None)

    exit_code = run_benchmark_release.main(["--manifest", "manifest.yaml", "--mode", "preflight"])

    assert exit_code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "preflight"
    assert payload["status"] == "orca_preflight_failed"
    assert payload["status_reason"] == payload["release_status_reason"]
    assert payload["benchmark_success"] is False
    assert payload["campaign_execution_status"] == "failed"
    assert payload["evidence_status"] == "blocked"
    assert payload["exit_code"] == 2
    assert payload["release_status"] == "orca_preflight_failed"
    assert payload["release_exit_code"] == 2
    assert "uv sync --extra orca" in payload["release_status_reason"]


def test_release_resume_receipt_requires_fixed_campaign_id(tmp_path: Path) -> None:
    """A resume ruling cannot be applied to a fresh timestamped campaign."""
    args = SimpleNamespace(
        campaign_id=None,
        output_root=tmp_path,
        resume_receipt=tmp_path / "resume.json",
        resume_receipt_max_age_hours=24.0,
    )
    cfg = SimpleNamespace(resume=True)
    config = tmp_path / "campaign.yaml"
    checkpoint = tmp_path / "checkpoint.json"
    config.write_text("horizon: 600\n", encoding="utf-8")
    checkpoint.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        run_benchmark_release.ReleaseResumeAdmissionError,
        match="explicit fixed campaign_id",
    ):
        run_benchmark_release._admit_release_resume(
            args=args,
            cfg=cfg,
            campaign_config_path=config,
            checkpoint_receipt_path=checkpoint,
        )


def test_release_existing_fixed_campaign_requires_resume_receipt(tmp_path: Path) -> None:
    """Prior planner output cannot resume from only a fixed campaign id."""
    campaign_id = "fixed-release"
    runs = tmp_path / campaign_id / "runs" / "goal__differential_drive"
    runs.mkdir(parents=True)
    (runs / "episodes.jsonl").write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        campaign_id=campaign_id,
        output_root=tmp_path,
        resume_receipt=None,
        resume_receipt_max_age_hours=24.0,
    )
    cfg = SimpleNamespace(resume=True)
    config = tmp_path / "campaign.yaml"
    checkpoint = tmp_path / "checkpoint.json"
    config.write_text("horizon: 600\n", encoding="utf-8")
    checkpoint.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        run_benchmark_release.ReleaseResumeAdmissionError,
        match="infrastructure-only resume receipt",
    ):
        run_benchmark_release._admit_release_resume(
            args=args,
            cfg=cfg,
            campaign_config_path=config,
            checkpoint_receipt_path=checkpoint,
        )


def test_public_campaign_result_rejects_unexpected_fields() -> None:
    """New runner fields must be classified before entering a public release result."""
    with pytest.raises(
        run_benchmark_release.ReleaseResultPrivacyError,
        match="unsupported result fields",
    ):
        run_benchmark_release._public_campaign_result(
            {"status": "benchmark_success", "new_artifact_location": "/srv/private/result"}
        )


def test_public_campaign_result_rejects_nested_private_paths() -> None:
    """Free-form public fields cannot smuggle a machine-local path into logs or JSON."""
    with pytest.raises(
        run_benchmark_release.ReleaseResultPrivacyError,
        match="contains private filesystem data",
    ):
        run_benchmark_release._public_campaign_result(
            {
                "status": "benchmark_success",
                "warnings": ["diagnostic file: /home/example/private/result.json"],
            }
        )
