"""
Pytest configuration to ensure Pygame runs headless during the test session.

This prevents any real OS window from opening when tests create a SimulationView
or otherwise initialize Pygame displays. It mirrors the headless env vars used
in docs and CI, but enforces them automatically for local test runs as well.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Generator, Optional

import pytest


@pytest.fixture(scope="session", autouse=True)
def headless_pygame_environment() -> Generator[None, None, None]:
    """Force headless graphics for the whole pytest session.

    Sets common environment variables so that Pygame uses the dummy video driver
    and no real window is created. This applies to all tests regardless of
    individual test configuration and ensures running tests locally won't pop up
    a window.
    """
    # Save original values to restore after the session
    originals: Dict[str, Optional[str]] = {
        "DISPLAY": os.environ.get("DISPLAY"),
        "SDL_VIDEODRIVER": os.environ.get("SDL_VIDEODRIVER"),
        "MPLBACKEND": os.environ.get("MPLBACKEND"),
        "SDL_AUDIODRIVER": os.environ.get("SDL_AUDIODRIVER"),
        "PYGAME_HIDE_SUPPORT_PROMPT": os.environ.get("PYGAME_HIDE_SUPPORT_PROMPT"),
    }

    # Enforce headless
    os.environ["DISPLAY"] = ""  # Treat as no display
    os.environ["SDL_VIDEODRIVER"] = "dummy"  # Pygame dummy video driver
    os.environ["MPLBACKEND"] = "Agg"  # Non-GUI matplotlib backend
    os.environ["SDL_AUDIODRIVER"] = "dummy"  # Avoid audio device errors
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"  # Cleaner test logs

    # Yield control to the test session
    yield

    # Restore original environment after session completes
    for key, value in originals.items():
        if value is None:
            if key in os.environ:
                del os.environ[key]
        else:
            os.environ[key] = value


@pytest.fixture(scope="session")
def perf_policy():  # type: ignore[missing-return-type-doc]
    """Return default performance budget policy for tests.

    Centralizes soft/hard thresholds so helper fixtures/plugins can reuse.
    """
    try:
        from tests.perf_utils.policy import PerformanceBudgetPolicy
    except Exception:  # pragma: no cover - defensive import
        # Fallback to simple namespace if policy not available (should not happen post-implementation)
        class _Fallback:  # noqa: D401 - internal simple fallback
            soft_threshold_seconds = 20.0
            hard_timeout_seconds = 60.0
            report_count = 10
            relax_env_var = "ROBOT_SF_PERF_RELAX"

            def classify(self, duration_seconds: float):  # noqa: D401
                if duration_seconds >= self.hard_timeout_seconds:
                    return "hard"
                if duration_seconds >= self.soft_threshold_seconds:
                    return "soft"
                return "none"

        return _Fallback()
    return PerformanceBudgetPolicy()


# ---------------- Slow Test Collector -----------------
_SLOW_SAMPLES: list[tuple[str, float]] = []


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):  # type: ignore[missing-type-doc]
    start = time.perf_counter()
    outcome = yield
    duration = time.perf_counter() - start
    _SLOW_SAMPLES.append((item.nodeid, duration))
    return outcome


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, _exitstatus):  # type: ignore[missing-type-doc]
    # Avoid report if policy fixture not yet constructed or no samples.
    if not _SLOW_SAMPLES:
        return
    # Lazy import to avoid import cost if unused.
    try:
        from tests.perf_utils.reporting import SlowTestSample, format_report, generate_report
    except Exception:  # pragma: no cover
        return
    # Construct a fresh policy instance (cheap) instead of introspecting fixture internals.
    try:
        from tests.perf_utils.policy import PerformanceBudgetPolicy

        policy = PerformanceBudgetPolicy()
    except Exception:  # pragma: no cover
        return
    # Build samples and generate report
    samples = [SlowTestSample(test_identifier=n, duration_seconds=d) for n, d in _SLOW_SAMPLES]
    report_records = generate_report(samples, policy)
    if not report_records:
        return
    relax = os.environ.get(policy.relax_env_var) == "1"
    text = format_report(report_records, policy)
    session.config.pluginmanager.get_plugin("terminalreporter").write_line(  # type: ignore[no-untyped-call]
        f"\n{text}\n(relax mode active)\n" if relax else f"\n{text}\n"
    )
