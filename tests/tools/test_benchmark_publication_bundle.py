"""Tests for benchmark publication bundle CLI helper."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.tools import benchmark_publication_bundle


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
