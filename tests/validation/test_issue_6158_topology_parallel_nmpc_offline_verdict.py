"""Regression checks for the issue #6158 offline-verdict validator."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scripts.validation.check_issue_6158_topology_parallel_nmpc_offline_verdict as validator
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


def test_evidence_provenance_separates_audited_commit_from_generation_head(
    monkeypatch, tmp_path: Path
) -> None:
    """A generated document must not claim to validate its own future commit."""
    evidence_doc = tmp_path / "issue_6158_verdict.md"
    monkeypatch.setattr(validator, "EVIDENCE_DIR", tmp_path)
    monkeypatch.setattr(validator, "EVIDENCE_DOC", evidence_doc)
    gates = [GateResult(name=f"gate_{index}", passed=True, detail="ok") for index in range(1, 9)]
    gates[6] = GateResult(
        name="gate_7_latency",
        passed=True,
        detail="latency recorded",
        evidence={
            "latency_exceeds_100ms": False,
            "worst_hypothesis_p95_ms": 12.0,
            "per_hypothesis_solver_runtime_ms": {
                "default": {"p50_ms": 10.0, "p95_ms": 12.0, "max_ms": 13.0, "n": 1}
            },
            "plan_wall_clock_ms_measurement_safe_deadline": {},
            "plan_wall_clock_ms_real_2s_deadline": {},
            "real_deadline_fires_out_of_8": 0,
            "measurement_note": "descriptive",
        },
    )
    gates[7] = GateResult(
        name="gate_8_pr_audit",
        passed=True,
        detail="audit recorded",
        evidence={
            "files": [],
            "head_post_merge_note": "prototype preserved",
        },
    )

    generation_head = "a" * 40
    summary = validator._write_evidence_doc(
        verdict="invalid_regression",
        rationale="parity failed",
        gates=gates,
        generation_head=generation_head,
        config_rel="configs/algos/issue_5310_topology_parallel_nmpc.yaml",
        hardware={},
        branch="test",
    )

    contents = evidence_doc.read_text()
    assert summary["audited_prototype_commit"] == validator.SOURCE_MERGE_COMMIT
    assert summary["evidence_generation_head"] == generation_head
    assert "validated_commit" not in summary
    assert f"Audited prototype commit: `{validator.SOURCE_MERGE_COMMIT}`" in contents
    assert f"Evidence-generation Git HEAD: `{generation_head}`" in contents
    assert "Validated commit (`git rev-parse HEAD`)" not in contents
