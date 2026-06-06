"""Tests for benchmark publication bundle CLI helper."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from scripts.tools import benchmark_publication_bundle

_EVIDENCE_BUNDLE_SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "evidence_bundle.v1.json"
)


def _write(path: Path, payload: str) -> None:
    """Write UTF-8 payload to disk for test fixtures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _make_run(run_dir: Path) -> None:
    """Create a minimal run directory fixture."""
    _write(run_dir / "manifest.json", '{"git_hash":"abc123","scenario_matrix_hash":"m1"}')
    _write(run_dir / "run_meta.json", '{"repo":{"remote":"git@github.com:ll7/robot_sf_ll7.git"}}')
    _write(run_dir / "episodes" / "episodes.jsonl", '{"episode_id":"ep-1"}\n')
    _write(run_dir / "aggregates" / "summary.json", '{"success_rate": 1.0}\n')
    _write(run_dir / "reports" / "report.md", "# report\n")


def test_size_report_command_writes_json(tmp_path: Path, capsys) -> None:
    """CLI size-report should emit and optionally persist a JSON payload."""
    root = tmp_path / "benchmarks"
    _make_run(root / "run_1")
    output_json = tmp_path / "size_report.json"

    exit_code = benchmark_publication_bundle.main(
        [
            "size-report",
            "--benchmarks-root",
            str(root),
            "--output-json",
            str(output_json),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["run_count"] == 1
    stdout_payload = json.loads(captured.out)
    assert stdout_payload["schema_version"] == payload["schema_version"]


def test_export_command_creates_bundle_artifacts(tmp_path: Path, capsys) -> None:
    """CLI export should produce publication bundle directory and archive."""
    run_dir = tmp_path / "benchmarks" / "run_2"
    _make_run(run_dir)
    out_dir = tmp_path / "publication"

    exit_code = benchmark_publication_bundle.main(
        [
            "export",
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            "run_2_bundle",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert Path(payload["bundle_dir"]).exists()
    assert Path(payload["manifest_path"]).exists()
    assert Path(payload["checksums_path"]).exists()
    assert Path(payload["archive_path"]).exists()
    assert payload["file_count"] > 0


def _make_evidence_source(source_root: Path) -> None:
    """Create a compact evidence source fixture."""
    _write(source_root / "summary.json", '{"result_classification":"diagnostic-only"}\n')
    _write(source_root / "metric_table.csv", "metric,value\nsuccess_rate,0.5\n")
    _write(source_root / "trace_manifest.yaml", "trace_count: 1\n")
    _write(source_root / "claim_boundary.md", "# Claim Boundary\n\nDiagnostic only.\n")


def test_evidence_bundle_command_creates_manifest_and_checksums(tmp_path: Path, capsys) -> None:
    """CLI evidence-bundle should copy selected compact files with provenance metadata."""
    source_root = tmp_path / "evidence_source"
    _make_evidence_source(source_root)
    out_dir = tmp_path / "bundles"

    exit_code = benchmark_publication_bundle.main(
        [
            "evidence-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out_dir),
            "--bundle-name",
            "issue_2460_example",
            "--file",
            "summary.json",
            "--file",
            "metric_table.csv",
            "--file",
            "trace_manifest.yaml",
            "--file",
            "claim_boundary.md",
            "--command",
            "uv run python scripts/example.py --config tracked.yaml",
            "--commit",
            "abc123",
            "--claim-boundary",
            "diagnostic_only_not_benchmark_evidence",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    bundle_dir = Path(payload["bundle_dir"])
    manifest_path = Path(payload["manifest_path"])
    checksums_path = Path(payload["checksums_path"])
    assert bundle_dir.exists()
    assert manifest_path.exists()
    assert checksums_path.exists()
    assert payload["file_count"] == 4
    assert (bundle_dir / "payload" / "summary.json").exists()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(_EVIDENCE_BUNDLE_SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(instance=manifest, schema=schema)
    assert manifest["schema_version"] == "evidence_bundle.v1"
    assert manifest["command"] == "uv run python scripts/example.py --config tracked.yaml"
    assert manifest["commit"] == "abc123"
    assert manifest["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"
    assert manifest["policy"]["large_raw_artifacts"] == "excluded"
    assert [entry["path"] for entry in manifest["files"]] == [
        "claim_boundary.md",
        "metric_table.csv",
        "summary.json",
        "trace_manifest.yaml",
    ]
    checksums = checksums_path.read_text(encoding="utf-8")
    assert "summary.json" in checksums
    assert "metric_table.csv" in checksums


def test_evidence_bundle_command_fails_closed_for_missing_file(tmp_path: Path) -> None:
    """Evidence bundles should fail before writing manifests when a requested file is missing."""
    source_root = tmp_path / "evidence_source"
    _make_evidence_source(source_root)

    with pytest.raises(FileNotFoundError):
        benchmark_publication_bundle.main(
            [
                "evidence-bundle",
                "--source-root",
                str(source_root),
                "--out-dir",
                str(tmp_path / "bundles"),
                "--bundle-name",
                "missing_file",
                "--file",
                "summary.json",
                "--file",
                "missing.json",
                "--command",
                "uv run python scripts/example.py",
                "--commit",
                "abc123",
                "--claim-boundary",
                "diagnostic_only",
            ]
        )
