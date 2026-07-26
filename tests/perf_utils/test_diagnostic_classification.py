"""Regression tests for the diagnostic-contract classification in reporting.

Pins the contract from issue #6320: the base-sensitive gate subset node
``test_subset_run_under_two_minutes`` is reported as an accepted ``"diagnostic"``
contract (with an explanation) instead of an unexplained ``"soft"`` breach,
because that test walls a ``pytest -m base_sensitive`` subprocess whose runtime
is dominated by full-suite collection and carries its own 120s hard cap.

Generic slow nodes must keep flowing through the normal soft/hard envelope.
"""

from __future__ import annotations

from .policy import PerformanceBudgetPolicy
from .reporting import DIAGNOSTIC_NODES, SlowTestSample, format_report, generate_report

GATE_NODE = (
    "tests/dev/test_base_sensitive_gate_contract.py::TestGateScript::"
    "test_subset_run_under_two_minutes"
)


def test_gate_node_is_registered_diagnostic() -> None:
    """The gate-contract node must be in the scoped diagnostic set with a note."""
    assert GATE_NODE in DIAGNOSTIC_NODES
    assert DIAGNOSTIC_NODES[GATE_NODE], "diagnostic note must be non-empty"


def test_diagnostic_set_is_scoped_to_gate_node_only() -> None:
    """Guardrail: the diagnostic set stays narrow (only the gate node today).

    The issue #6320 contract scopes the classification to this single test. If a
    future PR registers another diagnostic contract, update this assertion
    deliberately so the scope change is visible at review.
    """
    assert set(DIAGNOSTIC_NODES) == {GATE_NODE}


def test_gate_node_classified_as_diagnostic_not_soft() -> None:
    """A slow sample for the gate node reports as diagnostic, never soft."""
    policy = PerformanceBudgetPolicy()  # soft=20.0, hard=60.0
    samples = [SlowTestSample(test_identifier=GATE_NODE, duration_seconds=38.0)]
    records = generate_report(samples, policy)
    assert len(records) == 1
    rec = records[0]
    assert rec.breach_type == "diagnostic"
    # Explanatory note present; generic episode/horizon guidance must NOT appear.
    joined = " ".join(rec.guidance).lower()
    assert "diagnostic" in joined
    assert "episode" not in joined
    assert "horizon" not in joined


def test_generic_slow_node_still_classified_soft() -> None:
    """Non-diagnostic slow nodes keep the normal soft classification and guidance."""
    policy = PerformanceBudgetPolicy()
    samples = [SlowTestSample(test_identifier="dummy::test_slow_example", duration_seconds=38.0)]
    records = generate_report(samples, policy)
    assert len(records) == 1
    rec = records[0]
    assert rec.breach_type == "soft"
    # Generic guidance still applies to non-diagnostic nodes.
    joined = " ".join(rec.guidance).lower()
    assert "episode" in joined


def test_diagnostic_node_does_not_shadow_slower_generic_node() -> None:
    """A slower generic node ranks above the diagnostic node and keeps its breach."""
    policy = PerformanceBudgetPolicy()  # soft=20.0, hard=60.0
    samples = [
        SlowTestSample(test_identifier=GATE_NODE, duration_seconds=38.0),
        SlowTestSample(test_identifier="dummy::test_slower", duration_seconds=70.0),
    ]
    records = generate_report(samples, policy)
    by_id = {r.test_identifier: r for r in records}
    assert by_id[GATE_NODE].breach_type == "diagnostic"
    # 70s >= hard_timeout (60s) so the generic node is a hard breach, unaffected by
    # the diagnostic classification scoped to the gate node.
    assert by_id["dummy::test_slower"].breach_type == "hard"


def test_format_report_labels_diagnostic_clearly() -> None:
    """The rendered report shows a DIAGNOSTIC label, not a SOFT breach, for the gate node."""
    policy = PerformanceBudgetPolicy()
    samples = [SlowTestSample(test_identifier=GATE_NODE, duration_seconds=38.0)]
    records = generate_report(samples, policy)
    rendered = format_report(records, policy)
    assert "DIAGNOSTIC" in rendered
    assert "SOFT" not in rendered
    assert GATE_NODE in rendered
    assert "Accepted diagnostic contract" in rendered


def test_diagnostic_matcher_is_path_separator_robust() -> None:
    """The matcher accepts backslash-separated variants of the registered node."""
    from .reporting import _diagnostic_note

    backslash_node = GATE_NODE.replace("/", "\\")
    assert _diagnostic_note(backslash_node) is not None
    # Unrelated nodes are never matched.
    assert _diagnostic_note("dummy::test_example") is None
    assert _diagnostic_note("") is None
