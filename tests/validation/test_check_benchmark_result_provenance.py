"""Smoke tests for the benchmark result provenance validation CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.result_provenance import (
    build_result_provenance_manifest,
    write_result_provenance_manifest,
)
from scripts.validation import check_benchmark_result_provenance


def _write_valid_manifest(tmp_path: Path) -> Path:
    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text('{"episode_id":"a--1","scenario_id":"a","seed":1}\n', encoding="utf-8")
    manifest = build_result_provenance_manifest(
        out_path=jsonl_path,
        episode_records=[
            {
                "episode_id": "a--1",
                "scenario_id": "a",
                "seed": 1,
                "config_hash": "abc",
                "git_hash": "def",
            },
        ],
        schema_path="schema.json",
        scenario_path=Path("scenarios.yaml"),
        scenarios=[{"name": "a"}],
        algo="goal",
        algo_config_path=None,
        benchmark_profile="baseline-safe",
        suite_key="test",
        total_jobs=1,
        written=1,
        horizon=100,
        dt=0.1,
        record_forces=False,
        active_observation_mode="lidar",
        active_observation_level="full",
    )
    manifest_path = tmp_path / "episodes.jsonl.provenance.json"
    write_result_provenance_manifest(manifest_path, manifest)
    return manifest_path


def test_cli_checker_accepts_valid_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The validation CLI exits 0 for a known-good manifest."""
    manifest_path = _write_valid_manifest(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["check_benchmark_result_provenance.py", "--manifest", str(manifest_path)],
    )

    with pytest.raises(SystemExit) as exc_info:
        check_benchmark_result_provenance.main()

    assert exc_info.value.code == 0
    assert "OK:" in capsys.readouterr().err


def test_cli_checker_fails_closed_on_invalid_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The validation CLI exits 2 when required provenance fields are missing."""
    manifest_path = _write_valid_manifest(tmp_path)
    manifest_path.write_text('{"schema_version":"benchmark_result_provenance.v1"}\n', encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["check_benchmark_result_provenance.py", "--manifest", str(manifest_path)],
    )

    with pytest.raises(SystemExit) as exc_info:
        check_benchmark_result_provenance.main()

    assert exc_info.value.code == 2
    assert "FAIL:" in capsys.readouterr().err
