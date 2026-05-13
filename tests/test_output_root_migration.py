"""Regression tests for remaining user-facing output-root migration helpers.

These tests verify that the bounded #1186 cleanup keeps actively used scripts on
the canonical `output/benchmarks` artifact tree. They matter because the issue
updates documented defaults and helper-backed script paths rather than changing
artifact semantics.
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from scripts import run_social_navigation_benchmark
from scripts.perf import baseline_factory_creation

if TYPE_CHECKING:
    from pathlib import Path


class _StubEnv:
    """Minimal environment stub used to exercise factory timing script output wiring."""

    def reset(self) -> None:
        """Simulate a no-op reset."""

    def close(self) -> None:
        """Simulate a no-op close."""


def test_factory_baseline_script_writes_under_benchmarks_artifact_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the factory baseline script writes to `output/benchmarks` via artifact helpers."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        baseline_factory_creation,
        "make_robot_env",
        lambda config, debug=False: _StubEnv(),
    )
    monkeypatch.setattr(
        baseline_factory_creation,
        "make_image_robot_env",
        lambda config, debug=False: _StubEnv(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["baseline_factory_creation.py", "--iterations", "1"],
    )

    baseline_factory_creation.main()

    output_path = tmp_path / "benchmarks" / "factory_perf_baseline.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert payload["iterations"] == 1
    assert set(payload["results"]) == {"make_robot_env", "make_image_robot_env"}


def test_social_navigation_benchmark_output_root_uses_benchmarks_category(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify the benchmark runner resolves its timestamped root under `output/benchmarks`."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))

    resolved = run_social_navigation_benchmark._resolve_output_root("20260513_000000")

    assert resolved == tmp_path / "benchmarks" / "social_nav_benchmark_20260513_000000"
