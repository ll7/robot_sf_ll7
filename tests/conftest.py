"""Minimal pytest configuration: headless graphics + slow test timing.

Rewritten to purge legacy pytest_sessionfinish hook with invalid signature.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="session", autouse=True)
def headless_pygame_environment() -> Generator[None, None, None]:
    originals: dict[str, str | None] = {
        "DISPLAY": os.environ.get("DISPLAY"),
        "SDL_VIDEODRIVER": os.environ.get("SDL_VIDEODRIVER"),
        "MPLBACKEND": os.environ.get("MPLBACKEND"),
        "SDL_AUDIODRIVER": os.environ.get("SDL_AUDIODRIVER"),
        "PYGAME_HIDE_SUPPORT_PROMPT": os.environ.get("PYGAME_HIDE_SUPPORT_PROMPT"),
    }
    os.environ.update(
        {
            "DISPLAY": "",
            "SDL_VIDEODRIVER": "dummy",
            "MPLBACKEND": "Agg",
            "SDL_AUDIODRIVER": "dummy",
            "PYGAME_HIDE_SUPPORT_PROMPT": "hide",
        },
    )
    yield
    for k, v in originals.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="session")
def perf_policy():  # type: ignore[missing-return-type-doc]
    try:
        from tests.perf_utils.policy import PerformanceBudgetPolicy

        return PerformanceBudgetPolicy()
    except Exception:  # pragma: no cover

        class _Fallback:
            soft_threshold_seconds = 20.0
            hard_timeout_seconds = 60.0
            report_count = 10
            relax_env_var = "ROBOT_SF_PERF_RELAX"

            def classify(self, duration_seconds: float):
                if duration_seconds >= self.hard_timeout_seconds:
                    return "hard"
                if duration_seconds >= self.soft_threshold_seconds:
                    return "soft"
                return "none"

        return _Fallback()


_SLOW_SAMPLES: list[tuple[str, float]] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):  # type: ignore[missing-type-doc]
    start = time.perf_counter()
    try:
        outcome = yield
    finally:
        # Always record duration, even if the test raised/failed (ensures slow report completeness).
        _SLOW_SAMPLES.append((item.nodeid, time.perf_counter() - start))
    return outcome


@pytest.hookimpl(trylast=True)
def pytest_terminal_summary(terminalreporter, exitstatus, config):  # type: ignore[missing-type-doc]
    """Emit slow test report and optionally enforce soft breaches.

    Enforcement logic (feature 124):
      * If ROBOT_SF_PERF_ENFORCE=1 and any soft breach occurs, convert to failure.
      * Hard breaches are expected to be handled by individual test timeouts / assertions.
    """
    if not _SLOW_SAMPLES:
        return
    try:
        from tests.perf_utils.policy import PerformanceBudgetPolicy
        from tests.perf_utils.reporting import SlowTestSample, format_report, generate_report
    except Exception:  # pragma: no cover
        return
    policy = PerformanceBudgetPolicy()
    # Optional environment overrides (not part of public API; used for enforcement test)
    try:
        soft_override = os.environ.get("ROBOT_SF_PERF_SOFT")
        hard_override = os.environ.get("ROBOT_SF_PERF_HARD")
        if soft_override:
            policy.soft_threshold_seconds = float(soft_override)  # type: ignore[misc]
        if hard_override:
            policy.hard_timeout_seconds = float(hard_override)  # type: ignore[misc]
    except Exception:  # pragma: no cover
        pass
    samples = [SlowTestSample(test_identifier=n, duration_seconds=d) for n, d in _SLOW_SAMPLES]
    records = generate_report(samples, policy)
    if not records:
        return
    relax = os.environ.get(policy.relax_env_var) == "1"
    enforce = os.environ.get(getattr(policy, "enforce_env_var", "ROBOT_SF_PERF_ENFORCE")) == "1"
    terminalreporter.write_line(
        "\n"
        + format_report(records, policy)
        + ("\n(relax mode active)" if relax else "")
        + ("\n(enforce mode)" if enforce else "")
        + "\n",
    )
    if enforce and not relax:
        # Treat any soft or hard breach as failure. Hard breaches ideally already handled
        # by per-test timeout markers, but we enforce here for determinism in minimal runs.
        if any(r.breach_type in {"soft", "hard"} for r in records):
            msg = "Performance breach (soft/hard) detected under enforce mode"
            terminalreporter.write_line(msg)
            # Use pytest's exit API to set return code cleanly (avoids raw SystemExit trace noise)
            pytest.exit(msg, returncode=pytest.ExitCode.TESTS_FAILED)


# Coverage-specific fixtures for test isolation


@pytest.fixture
def sample_coverage_data():
    """
    Provide sample coverage data for testing coverage tools.

    Returns a minimal but complete coverage.json structure
    for unit testing formatters, analyzers, and comparators.
    """
    return {
        "meta": {
            "version": "7.0.0",
            "timestamp": "2025-10-23T12:00:00",
            "branch_coverage": False,
            "show_contexts": False,
        },
        "files": {
            "robot_sf/gym_env/environment.py": {
                "executed_lines": [1, 2, 3, 10, 11, 12, 20],
                "summary": {
                    "covered_lines": 7,
                    "num_statements": 10,
                    "percent_covered": 70.0,
                    "missing_lines": 3,
                    "excluded_lines": 0,
                },
                "missing_lines": [4, 5, 6],
            },
            "robot_sf/sim/simulator.py": {
                "executed_lines": [1, 2, 3, 4, 5],
                "summary": {
                    "covered_lines": 5,
                    "num_statements": 8,
                    "percent_covered": 62.5,
                    "missing_lines": 3,
                    "excluded_lines": 0,
                },
                "missing_lines": [10, 11, 12],
            },
        },
        "totals": {
            "covered_lines": 12,
            "num_statements": 18,
            "percent_covered": 66.67,
            "missing_lines": 6,
            "excluded_lines": 0,
        },
    }


@pytest.fixture
def sample_gap_data():
    """Provide sample gap analysis data for testing."""
    return {
        "gaps": [
            {
                "file": "robot_sf/gym_env/environment.py",
                "coverage_percent": 70.0,
                "uncovered_lines": 3,
                "priority_score": 4.5,
                "recommendation": "Add integration tests for reset() method",
            },
            {
                "file": "robot_sf/sim/simulator.py",
                "coverage_percent": 62.5,
                "uncovered_lines": 3,
                "priority_score": 4.5,
                "recommendation": "Add unit tests for step() method",
            },
        ],
        "summary": {
            "total_gaps": 2,
            "total_uncovered_lines": 6,
            "average_coverage": 66.25,
        },
    }


@pytest.fixture
def sample_trend_data():
    """Provide sample trend analysis data for testing."""
    return {
        "direction": "improving",
        "rate_per_week": 0.5,
        "current_coverage": 66.67,
        "oldest_coverage": 60.0,
        "snapshot_count": 10,
        "date_range": {
            "start": "2025-10-01",
            "end": "2025-10-23",
        },
    }


@pytest.fixture
def sample_baseline_data():
    """Provide sample baseline comparison data for testing."""
    return {
        "current_coverage": 66.67,
        "baseline_coverage": 70.0,
        "delta": -3.33,
        "threshold": 1.0,
        "changed_files": [
            {
                "file": "robot_sf/gym_env/environment.py",
                "current": 70.0,
                "baseline": 75.0,
                "delta": -5.0,
            },
        ],
    }
