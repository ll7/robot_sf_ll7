"""Tests for the benchmark release CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark.camera_ready_campaign import CampaignConfig, PlannerSpec, SeedPolicy
from robot_sf.benchmark.orca_preflight import OrcaRvo2PreflightError
from scripts.tools import run_benchmark_release


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
    assert secret_marker in persisted["campaign_root"]


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
            "blockers": ["fallback row present"],
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
    assert exit_code == 2
    assert payload["campaign_benchmark_success"] is True
    assert payload["benchmark_success"] is False
    assert payload["release_benchmark_success"] is False
    assert payload["release_status"] == "full_release_acceptance_failed"
    assert payload["release_exit_code"] == 2
    assert payload["publication_bundle"] is None


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
