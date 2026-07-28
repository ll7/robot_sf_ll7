"""Slow test reporting utilities.

Provides helpers to transform raw runtime samples into a ranked report
with breach classification and guidance suggestions.

A narrow, deliberately scoped set of *diagnostic contract* nodes (tests whose
runtime is dominated by a deliberate subprocess/gate contract that carries its
own budget) are classified as ``"diagnostic"`` instead of the generic
``"soft"``/``"hard"`` envelope. They still appear in the report (nothing is
hidden), but they are labelled and explained rather than emitted as unexplained
``SOFT`` breaches whose generic "reduce episode count" guidance does not apply.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .guidance import default_guidance, format_guidance_lines

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .policy import PerformanceBudgetPolicy


# Scoped diagnostic-contract nodes. Each entry maps an exact pytest node id to a
# short explanation. A node listed here is reported as ``"diagnostic"`` for an
# expected ``"soft"`` breach, rather than as an unexplained ``"soft"`` breach.
# The ``"hard"`` envelope is never exempted: a registered node at or above the
# hard timeout remains a hard breach and therefore remains enforceable. This set
# is intentionally narrow: only tests that own a separate, explicit gate-specific
# budget (for example an inner ``assert elapsed < N`` cap) belong here. Generic test
# slowness must keep flowing through the normal envelope. See issue #6320.
DIAGNOSTIC_NODES: dict[str, str] = {
    "tests/dev/test_base_sensitive_gate_contract.py::TestGateScript::test_subset_run_under_two_minutes": (
        "Base-sensitive gate subset subprocess contract. The outer test walls a "
        "`pytest -m base_sensitive` subprocess whose runtime is dominated by full "
        "suite collection (~tens of seconds, by design); the test carries its own "
        "120s hard cap (`assert elapsed < 120`). Its expected soft-threshold breach "
        "is therefore diagnostic, while the report's configured hard threshold "
        "still applies."
    ),
}


def _diagnostic_note(test_identifier: str) -> str | None:
    """Return the explanatory note if ``test_identifier`` is a diagnostic node.

    Matching is robust to absolute-vs-relative paths and OS path separators:
    pytest emits repo-relative node ids with forward slashes, but the check also
    accepts a path that ends with the registered relative key.

    Returns:
        The explanatory note for the node, or ``None`` when it is not a
        registered diagnostic contract.
    """
    if not test_identifier:
        return None
    norm = test_identifier.replace("\\", "/")
    for key, note in DIAGNOSTIC_NODES.items():
        if norm == key or norm.endswith("/" + key):
            return note
    return None


@dataclass(slots=True)
class SlowTestSample:
    """TODO docstring. Document this class."""

    test_identifier: str
    duration_seconds: float


@dataclass(slots=True)
class SlowTestRecord:
    """TODO docstring. Document this class."""

    test_identifier: str
    duration_seconds: float
    breach_type: str
    guidance: list[str]

    def format_block(self) -> str:
        """TODO docstring. Document this function.


        Returns:
            TODO docstring.
        """
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
    """Rank runtime samples and classify their performance-budget status.

    Args:
        samples: Test identifiers paired with their measured call durations.
        policy: Thresholds and report-size limit used for classification.

    Returns:
        Slowest samples, ordered by duration, with guidance for their normal
        soft/hard status or scoped diagnostic-contract explanation.
    """
    ordered = sorted(samples, key=lambda s: s.duration_seconds, reverse=True)
    top = ordered[: policy.report_count]
    records: list[SlowTestRecord] = []
    for s in top:
        breach = policy.classify(s.duration_seconds)
        note = _diagnostic_note(s.test_identifier)
        if note is not None and breach == "soft":
            # Accepted, explicitly bounded diagnostic contract: report it transparently
            # as ``"diagnostic"`` with an explanation instead of a generic soft
            # breach whose episode/horizon guidance does not apply (issue #6320).
            # Do not bypass the hard boundary: that must remain enforceable.
            guidance = [
                f"Accepted diagnostic contract: {note}",
                (
                    "Configured performance envelope remains active: "
                    f"soft<{policy.soft_threshold_seconds:g}s, "
                    f"hard={policy.hard_timeout_seconds:g}s."
                ),
            ]
            records.append(
                SlowTestRecord(
                    test_identifier=s.test_identifier,
                    duration_seconds=s.duration_seconds,
                    breach_type="diagnostic",
                    guidance=guidance,
                ),
            )
            continue
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
    """TODO docstring. Document this function.

    Args:
        records: TODO docstring.
        policy: TODO docstring.

    Returns:
        TODO docstring.
    """
    lines = [
        f"Slow Test Report (soft<{policy.soft_threshold_seconds:.0f}s hard={policy.hard_timeout_seconds:.0f}s, top {policy.report_count})",
    ]
    for idx, r in enumerate(records, 1):
        prefix = f"{idx}) "
        block = r.format_block().replace("\n", "\n   ")
        lines.append(prefix + block)
    return "\n".join(lines)
