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
        enforce_env_var: Environment variable name that, if set ("1"), escalates soft breaches to failures.
    """

    soft_threshold_seconds: float = 20.0
    hard_timeout_seconds: float = 60.0
    report_count: int = 10
    relax_env_var: str = "ROBOT_SF_PERF_RELAX"
    enforce_env_var: str = "ROBOT_SF_PERF_ENFORCE"

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
