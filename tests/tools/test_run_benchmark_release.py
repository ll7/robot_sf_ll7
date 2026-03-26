"""Tests for the benchmark release CLI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.tools import run_benchmark_release


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _make_campaign_tree(tmp_path: Path) -> Path:
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


def test_release_preflight_uses_camera_ready_preflight(monkeypatch, capsys, tmp_path: Path) -> None:
    """Preflight mode should validate the manifest and emit preflight artifact paths."""
    manifest = SimpleNamespace(
        canonical_campaign_config_path=Path("configs/benchmarks/paper_experiment_matrix_v1.yaml")
    )
    sentinel_cfg = object()

    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: sentinel_cfg)
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
        "prepare_campaign_preflight",
        lambda cfg, **kwargs: {
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
        },
    )

    exit_code = run_benchmark_release.main(
        ["--manifest", "manifest.yaml", "--mode", "preflight"],
    )

    assert exit_code == 0
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


def test_release_run_exports_publication_only_after_benchmark_success(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """Successful releases should export publication bundles after artifact checks pass."""
    campaign_root = _make_campaign_tree(tmp_path)
    manifest = SimpleNamespace(
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
        doi="10.5281/zenodo.<record-id>",
        repository_url="https://github.com/ll7/robot_sf_ll7",
    )
    sentinel_cfg = object()

    monkeypatch.setattr(run_benchmark_release, "load_release_manifest", lambda path: manifest)
    monkeypatch.setattr(run_benchmark_release, "load_campaign_config", lambda path: sentinel_cfg)
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
            "status": "ok",
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
    monkeypatch.setattr(
        run_benchmark_release,
        "_build_publication_payload",
        lambda **kwargs: {
            "archive_path": "output/benchmarks/publication/bundle.tar.gz",
            "checksums_path": "output/benchmarks/publication/bundle/checksums.sha256",
            "manifest_path": "output/benchmarks/publication/bundle/publication_manifest.json",
        },
    )

    exit_code = run_benchmark_release.main(["--manifest", "manifest.yaml"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["benchmark_success"] is True
    assert payload["publication_bundle"]["archive_path"].endswith("bundle.tar.gz")
    assert (campaign_root / "release" / "release_result.json").exists()
    assert (campaign_root / "release" / "release_manifest.resolved.json").exists()
