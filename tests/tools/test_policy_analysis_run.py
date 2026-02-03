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
