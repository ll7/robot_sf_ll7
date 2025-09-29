"""Slow test reporting utilities.

Provides helpers to transform raw runtime samples into a ranked report
with breach classification and guidance suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .guidance import default_guidance, format_guidance_lines

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .policy import PerformanceBudgetPolicy


@dataclass(slots=True)
class SlowTestSample:
    test_identifier: str
    duration_seconds: float


@dataclass(slots=True)
class SlowTestRecord:
    test_identifier: str
    duration_seconds: float
    breach_type: str
    guidance: list[str]

    def format_block(self) -> str:
        if not self.guidance:
            return f"{self.test_identifier}  {self.duration_seconds:.2f}s"
        return (
            f"{self.test_identifier}  {self.duration_seconds:.2f}s  {self.breach_type.upper()}\n"
            f"{format_guidance_lines(self.guidance)}"
        )


def generate_report(
    samples: Iterable[SlowTestSample],
    policy: PerformanceBudgetPolicy,
) -> list[SlowTestRecord]:
    ordered = sorted(samples, key=lambda s: s.duration_seconds, reverse=True)
    top = ordered[: policy.report_count]
    records: list[SlowTestRecord] = []
    for s in top:
        breach = policy.classify(s.duration_seconds)
        guidance = default_guidance(s.duration_seconds, breach)
        records.append(
            SlowTestRecord(
                test_identifier=s.test_identifier,
                duration_seconds=s.duration_seconds,
                breach_type=breach,
                guidance=guidance,
            ),
        )
    return records


def format_report(records: Iterable[SlowTestRecord], policy: PerformanceBudgetPolicy) -> str:
    lines = [
        f"Slow Test Report (soft<{policy.soft_threshold_seconds:.0f}s hard={policy.hard_timeout_seconds:.0f}s, top {policy.report_count})",
    ]
    for idx, r in enumerate(records, 1):
        prefix = f"{idx}) "
        block = r.format_block().replace("\n", "\n   ")
        lines.append(prefix + block)
    return "\n".join(lines)
