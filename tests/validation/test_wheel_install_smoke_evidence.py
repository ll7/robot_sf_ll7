"""Tests for invocation-local wheel-smoke evidence paths (issue #8917)."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from scripts.validation.wheel_install_smoke_evidence import (
    LOG_ROOT_NAME,
    RUN_ID_ENV,
    enrich_extras_rows,
    extra_log_paths,
    log_dir_for_report,
    run_id_from_env,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_logs(log_dir: Path, extra: str, *, install: str, probe: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    install_log, probe_log = extra_log_paths(log_dir, extra)
    install_log.write_text(install, encoding="utf-8")
    probe_log.write_text(probe, encoding="utf-8")


def test_distinct_reports_get_distinct_log_dirs(tmp_path: Path) -> None:
    """Distinct report destinations must not share a log directory."""
    first = log_dir_for_report(tmp_path / "a" / "report.json", "run-a")
    second = log_dir_for_report(tmp_path / "b" / "report.json", "run-b")

    assert first != second
    assert first.parent.name == LOG_ROOT_NAME
    assert first.name == "run-a"


def test_same_report_and_run_id_are_deterministic(tmp_path: Path) -> None:
    """Repeated derivation for the same report and run id is stable."""
    report = tmp_path / "report.json"

    assert log_dir_for_report(report, "run-a") == log_dir_for_report(report, "run-a")


def test_run_id_env_override_and_fallback() -> None:
    """An explicit run id wins; the fallback stays path-safe."""
    assert run_id_from_env({RUN_ID_ENV: " explicit-run "}) == "explicit-run"

    fallback = run_id_from_env({})

    assert fallback
    assert "/" not in fallback


def test_invalid_run_id_is_rejected(tmp_path: Path) -> None:
    """A run id that could escape the log root is refused."""
    with pytest.raises(ValueError):
        log_dir_for_report(tmp_path / "report.json", "../escape")


def test_enrich_binds_paths_and_digests(tmp_path: Path) -> None:
    """Rows gain their own log paths, digests, and version evidence."""
    log_dir = log_dir_for_report(tmp_path / "report.json", "run-a")
    _write_logs(log_dir, "progress", install="install progress\n", probe="")
    rows = [{"extra": "progress", "status": "passed", "distribution_versions": {"tqdm": "4"}}]

    enriched = enrich_extras_rows(rows, log_dir)

    install_log, probe_log = extra_log_paths(log_dir, "progress")
    assert enriched[0]["install_log"] == str(install_log)
    assert enriched[0]["probe_log"] == str(probe_log)
    assert enriched[0]["install_log_sha256"] == hashlib.sha256(b"install progress\n").hexdigest()
    assert enriched[0]["probe_log_sha256"] == hashlib.sha256(b"").hexdigest()
    assert enriched[0]["distribution_versions"] == {"tqdm": "4"}


def test_enrich_fails_closed_on_missing_log(tmp_path: Path) -> None:
    """A report cannot claim evidence whose log is missing."""
    log_dir = log_dir_for_report(tmp_path / "report.json", "run-a")
    _write_logs(log_dir, "progress", install="x\n", probe="")
    rows = [
        {"extra": "progress", "status": "passed"},
        {"extra": "analytics", "status": "passed"},
    ]

    with pytest.raises(FileNotFoundError, match="analytics"):
        enrich_extras_rows(rows, log_dir)


def test_concurrent_enrichment_keeps_distinct_evidence(tmp_path: Path) -> None:
    """Concurrent reports keep distinct complete logs and digests."""
    jobs = []
    for index in range(8):
        log_dir = log_dir_for_report(tmp_path / f"report-{index}.json", f"run-{index}")
        _write_logs(log_dir, "viz", install=f"install-{index}\n", probe=f"probe-{index}\n")
        jobs.append((log_dir, index))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda job: enrich_extras_rows(
                    [{"extra": "viz", "status": "passed", "run": job[1]}], job[0]
                )[0],
                jobs,
            )
        )

    for result, (log_dir, index) in zip(results, jobs, strict=True):
        assert result["log_dir"] == str(log_dir)
        assert (
            result["install_log_sha256"]
            == hashlib.sha256(f"install-{index}\n".encode()).hexdigest()
        )
        assert result["probe_log_sha256"] == hashlib.sha256(f"probe-{index}\n".encode()).hexdigest()
        assert result["run"] == index
