"""Tests for policy analysis CLI helper behavior."""

from __future__ import annotations

import argparse
import subprocess
from typing import TYPE_CHECKING

import pytest
from loguru import logger

from scripts.tools import policy_analysis_run

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_policies_rejects_invalid_names():
    """Reject invalid policy names in sweeps to prevent silent typos."""
    args = argparse.Namespace(
        policy="ppo",
        policy_sweep=True,
        policies="fast_pysf_planner,typo",
    )
    with pytest.raises(ValueError, match="Invalid policies"):
        policy_analysis_run._resolve_policies(args)


def test_run_frame_extraction_logs_timeout(monkeypatch, tmp_path: Path):
    """Log and return cleanly when frame extraction hits a timeout."""
    report_json = tmp_path / "report.json"
    report_json.write_text("{}", encoding="utf-8")

    def _boom(*_args, **_kwargs):
        exc = subprocess.TimeoutExpired(
            cmd="extract_failure_frames.py",
            timeout=60,
            output="stdout message",
            stderr="stderr message",
        )
        if not hasattr(exc, "stdout"):
            try:
                exc.stdout = exc.output
            except AttributeError:
                pass
        raise exc

    monkeypatch.setattr(policy_analysis_run.subprocess, "run", _boom)
    captured: list[str] = []
    handle = logger.add(lambda message: captured.append(str(message)), level="WARNING")
    try:
        policy_analysis_run._run_frame_extraction(report_json, output_root=tmp_path)
    finally:
        logger.remove(handle)

    joined = "\n".join(captured)
    assert "timed out" in joined.lower()
    assert str(report_json) in joined


def test_resolve_termination_reason_filters_rejects_overlap() -> None:
    """Overlapping include/exclude reasons should raise a ValueError."""
    args = argparse.Namespace(
        termination_reason=["collision", "success"],
        exclude_termination_reason=["collision"],
    )
    with pytest.raises(ValueError, match="both include and exclude"):
        policy_analysis_run._resolve_termination_reason_filters(args)


def test_record_matches_termination_reason_filter() -> None:
    """Termination-reason filtering should enforce include/exclude constraints."""
    record = {"termination_reason": "collision"}
    assert policy_analysis_run._record_matches_termination_reason_filter(
        record,
        include=set(),
        exclude=set(),
    )
    assert policy_analysis_run._record_matches_termination_reason_filter(
        record,
        include={"collision"},
        exclude=set(),
    )
    assert not policy_analysis_run._record_matches_termination_reason_filter(
        record,
        include={"success"},
        exclude=set(),
    )
    assert not policy_analysis_run._record_matches_termination_reason_filter(
        record,
        include=set(),
        exclude={"collision"},
    )


def test_summarize_records_includes_reason_counts() -> None:
    """Summary payload should include per-reason count and rate fields."""
    records = [
        {
            "status": "success",
            "termination_reason": "success",
            "metrics": {"success": 1, "collisions": 0},
        },
        {
            "status": "collision",
            "termination_reason": "collision",
            "metrics": {"success": 0, "collisions": 1},
        },
        {
            "status": "failure",
            "termination_reason": "max_steps",
            "metrics": {"success": 0, "collisions": 0},
        },
    ]
    summary = policy_analysis_run._summarize_records(records)
    reason_counts = summary["termination_reason_counts"]
    reason_rates = summary["termination_reason_rates"]

    assert reason_counts["success"] == 1
    assert reason_counts["collision"] == 1
    assert reason_counts["max_steps"] == 1
    assert reason_rates["success"] == pytest.approx(1 / 3)
    assert reason_rates["collision"] == pytest.approx(1 / 3)
