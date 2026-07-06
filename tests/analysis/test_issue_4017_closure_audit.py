"""Tests the issue #4017 closure-audit evidence artifact."""

from __future__ import annotations

from pathlib import Path

AUDIT_PATH = Path("docs/context/evidence/issue_4017_closure_audit_2026-07-05.md")


def test_issue_4017_closure_audit_maps_required_pr_evidence() -> None:
    """Closure audit must cite the merged PRs that satisfy the CPU-side DoD."""
    text = AUDIT_PATH.read_text(encoding="utf-8")

    assert "Live issue reviewed: <https://github.com/ll7/robot_sf_ll7/issues/4017>" in text
    for pr_number in ("#4155", "#4214", "#4259", "#4477"):
        assert f"https://github.com/ll7/robot_sf_ll7/pull/{pr_number[1:]}" in text

    required_criteria = (
        "Constrained PPO training runs through a smoke scenario without crashing",
        "Constraint costs and multipliers are logged",
        "At least one constraint budget violation causes multiplier update",
        "Unconstrained and constrained policies are evaluated on the same scenario set and paired seeds",
        "Comparison report is diagnostic-only and includes caveats",
        "Existing metric semantics are not redefined",
        "Safety-wrapper and uncertainty-buffer functionality are not mixed into the training claim",
    )
    for criterion in required_criteria:
        assert criterion in text


def test_issue_4017_closure_audit_preserves_claim_boundary() -> None:
    """The audit must not promote benchmark or paper-facing safety claims."""
    text = AUDIT_PATH.read_text(encoding="utf-8")

    assert "does not claim benchmark-strength" in text
    assert "paper-grade" in text
    assert "dissertation safety evidence" in text
    assert "only remaining item" in text
    assert "broader empirical campaign" in text
    assert "no Slurm submission" in text
    assert "no GPU/full benchmark campaign" in text
    assert "no paper/dissertation claim edit" in text
