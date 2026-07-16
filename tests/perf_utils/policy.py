"""Performance budget policy utilities for tests.

Defines the data structure representing the per-test performance budget
(soft and hard thresholds) and classification helpers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

BreachType = Literal["none", "soft", "hard"]

# pytest-xdist sets PYTEST_XDIST_WORKER to the worker id (e.g. "gw0") in every
# worker subprocess. The controller process leaves it unset. Detecting this lets
# performance-sensitive tests relax their soft runtime envelope when the machine
# is contended by sibling workers rather than failing on unrelated CPU/I/O load.
XDIST_WORKER_ENV = "PYTEST_XDIST_WORKER"


@dataclass(slots=True)
class PerformanceBudgetPolicy:
    """Represents performance budget thresholds for tests.

    Attributes:
        soft_threshold_seconds: Advisory runtime goal (< this considered OK).
        hard_timeout_seconds: Hard fail threshold (>= leads to timeout/failure logic elsewhere).
        report_count: Number of slow tests to surface in a report.
        relax_env_var: Environment variable name that, if set ("1"), relaxes soft enforcement.
        enforce_env_var: Environment variable name that, if set ("1"), escalates soft breaches to failures.
    """

    soft_threshold_seconds: float = 20.0
    hard_timeout_seconds: float = 60.0
    report_count: int = 10
    relax_env_var: str = "ROBOT_SF_PERF_RELAX"
    enforce_env_var: str = "ROBOT_SF_PERF_ENFORCE"
    # Multiplier applied to the soft envelope when the test runs under pytest-xdist
    # contention. The hard boundary is never scaled (a genuine regression must still
    # fail), only the advisory soft budget that trips when unrelated workers starve
    # this one of CPU/I/O (issue #5836).
    xdist_contention_multiplier: float = 3.0

    def __post_init__(self) -> None:  # lightweight construction validation
        """Validate configuration invariants.

        Raises:
            ValueError: If soft/hard thresholds have invalid ordering or report_count < 1.
        """
        if not (
            self.soft_threshold_seconds > 0
            and self.soft_threshold_seconds < self.hard_timeout_seconds
        ):
            raise ValueError("soft_threshold_seconds must be > 0 and < hard_timeout_seconds")
        if self.report_count < 1:
            raise ValueError("report_count must be >= 1")
        if self.xdist_contention_multiplier <= 0:
            raise ValueError("xdist_contention_multiplier must be > 0")

    def is_under_xdist(self) -> bool:
        """Return True when running inside a pytest-xdist worker subprocess.

        pytest-xdist exports ``PYTEST_XDIST_WORKER`` (the worker id, e.g. ``gw0``)
        in each worker; the controller and standalone runs leave it unset. This is
        the canonical, dependency-free way to detect full-suite contention.
        """
        worker = os.environ.get(XDIST_WORKER_ENV)
        return bool(worker) and worker.strip() != ""

    def effective_soft_threshold(self, *, ci: bool = False) -> float:
        """Return the soft runtime envelope to enforce for this invocation.

        Under pytest-xdist the envelope is multiplied by ``xdist_contention_multiplier``
        so a saturated machine (sibling workers consuming CPU/I/O) no longer trips the
        advisory budget. The hard boundary is intentionally never scaled: a genuine
        benchmark runtime regression must still fail even under contention (issue #5836).
        The widened soft value is clamped to stay strictly below the hard boundary so it
        remains advisory rather than colliding with the hard wall.
        """
        base = self.soft_threshold_seconds if ci else (self.soft_threshold_seconds / 2.0)
        if self.is_under_xdist():
            widened = base * self.xdist_contention_multiplier
            return min(widened, self.hard_timeout_seconds * 0.9)
        return base

    def classify(self, duration_seconds: float) -> BreachType:
        """Classify a test duration.

        Returns:
            "hard" if >= hard timeout,
            "soft" if >= soft but < hard,
            "none" otherwise.
        """
        if duration_seconds >= self.hard_timeout_seconds:
            return "hard"
        if duration_seconds >= self.soft_threshold_seconds:
            return "soft"
        return "none"
