"""Performance budget policy utilities for tests.

Defines the data structure representing the per-test performance budget
(soft and hard thresholds) and classification helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BreachType = Literal["none", "soft", "hard"]


@dataclass(slots=True)
class PerformanceBudgetPolicy:
    """Represents performance budget thresholds for tests.

    Attributes:
        soft_threshold_seconds: Advisory runtime goal (< this considered OK).
        hard_timeout_seconds: Hard fail threshold (>= leads to timeout/failure logic elsewhere).
        report_count: Number of slow tests to surface in a report.
        relax_env_var: Environment variable name that, if set ("1"), relaxes soft enforcement.
    """

    soft_threshold_seconds: float = 20.0
    hard_timeout_seconds: float = 60.0
    report_count: int = 10
    relax_env_var: str = "ROBOT_SF_PERF_RELAX"

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
