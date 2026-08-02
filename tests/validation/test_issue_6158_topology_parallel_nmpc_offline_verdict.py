"""Regression checks for the issue #6158 offline-verdict validator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.validation.check_issue_6158_topology_parallel_nmpc_offline_verdict import (
    GateResult,
    _assess_pairwise_distinctness,
    _derive_verdict,
    _nmpc_config_from_file,
    _run_gate,
    _synth_diag,
    gate_1_k1_legacy_parity,
)


def test_pairwise_distinctness_requires_rollouts_for_every_feasible_pair() -> None:
    """A missing feasible rollout cannot turn partial measurements into proof."""
    diagnostics = [
        _synth_diag("pass_left", feasible=True, objective=1.0),
        _synth_diag("yield_straight", feasible=True, objective=2.0),
        _synth_diag("pass_right", feasible=True, objective=3.0),
    ]
    states = {
        "pass_left": np.array([[0.0, 0.0], [0.0, 0.0]]),
        "yield_straight": np.array([[0.01, 0.0], [0.01, 0.0]]),
    }

    feasible, pairwise, missing_pairs, min_sep, proves_distinctness = _assess_pairwise_distinctness(
        diagnostics, states
    )

    assert feasible == ["pass_left", "yield_straight", "pass_right"]
    assert pairwise == [{"pair": ["pass_left", "yield_straight"], "separation_m": 0.01}]
    assert missing_pairs == [["pass_left", "pass_right"], ["yield_straight", "pass_right"]]
    assert min_sep == 0.01
    assert not proves_distinctness


def test_k1_parity_covers_social_conflict_fixture() -> None:
    """Gate 1 must expose the current social-state parity regression."""
    repo_root = Path(__file__).resolve().parents[2]
    config = repo_root / "configs" / "algos" / "issue_5310_topology_parallel_nmpc.yaml"

    result = gate_1_k1_legacy_parity(_nmpc_config_from_file(config))

    assert not result.passed
    assert [case["fixture"] for case in result.evidence["fixtures"]] == [
        "open_space",
        "pedestrian_conflict",
    ]
    assert result.evidence["fixtures"][0]["passed"]
    assert not result.evidence["fixtures"][1]["passed"]


def test_gate_execution_error_records_incomplete_before_other_verdict_precedence() -> None:
    """A failed diagnostic must record `incomplete`, not a derived planner verdict."""
    names = (
        "gate_1_k1_legacy_parity",
        "gate_2_material_distinctness",
        "gate_3_objective_invariance",
        "gate_4_selection_and_hysteresis",
        "gate_5_fail_closed",
        "gate_6_registration_smoke",
        "gate_7_latency",
        "gate_8_pr_audit",
    )
    gates = [GateResult(name=name, passed=True, detail="ok") for name in names]
    gates[0] = GateResult(
        name="gate_1_k1_legacy_parity",
        passed=False,
        detail="execution failed",
        evidence={"execution_error": "RuntimeError: fixture unavailable"},
    )

    verdict, rationale = _derive_verdict(gates)

    assert verdict == "incomplete"
    assert "gate_1_k1_legacy_parity" in rationale


def test_gate_runner_preserves_an_unexpected_error_as_evidence() -> None:
    """A gate exception must be available to the durable incomplete verdict."""

    def raise_fixture_error() -> GateResult:
        raise RuntimeError("fixture unavailable")

    result = _run_gate("gate_7_latency", raise_fixture_error)

    assert not result.passed
    assert result.evidence == {"execution_error": "RuntimeError: fixture unavailable"}
