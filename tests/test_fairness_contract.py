"""Tests for the matched-capability fairness contract (issue #5353)."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.fairness_contract import (
    FairnessReport,
    build_capability_entry,
    build_capability_matrix,
    detect_mismatches,
    emit_mismatch_flags,
    fair_comparison_subset,
    ranking_claim_gate,
)

# ──────────────────────────────────────────────────────────────────────
# Capability matrix generation
# ──────────────────────────────────────────────────────────────────────


def test_build_capability_entry_goal():
    """Goal planner has native execution, goal_state observation, realtime budget."""
    entry = build_capability_entry("goal")
    assert entry.canonical_name == "goal"
    assert entry.baseline_category == "classical"
    assert entry.readiness_tier == "baseline-ready"
    assert entry.observation_mode == "goal_state"
    assert entry.default_execution_mode == "native"
    assert entry.adapter_active is False
    assert entry.has_privileged_inputs is False
    assert entry.diagnostic_reference_only is False


def test_build_capability_entry_orca():
    """ORCA planner uses adapter execution with socnav_state."""
    entry = build_capability_entry("orca")
    assert entry.canonical_name == "orca"
    assert entry.default_execution_mode == "adapter"
    assert entry.adapter_active is True
    assert entry.default_adapter_name == "ORCAPlannerAdapter"
    assert entry.observation_mode == "socnav_state"
    assert entry.has_privileged_inputs is True


def test_build_capability_entry_ppo():
    """PPO planner uses mixed execution with sensor_fusion_state."""
    entry = build_capability_entry("ppo")
    assert entry.canonical_name == "ppo"
    assert entry.baseline_category == "learning"
    assert entry.default_execution_mode == "mixed"
    assert entry.adapter_active is True
    assert entry.observation_mode == "sensor_fusion_state"
    assert entry.compute_budget_class == "learned_inference"


def test_build_capability_entry_with_overrides():
    """Entry respects observation_mode and tuning overrides."""
    entry = build_capability_entry(
        "orca",
        observation_mode="socnav_state",
        tuning_budget_runs=50,
        tuning_source="declared",
    )
    assert entry.observation_mode == "socnav_state"
    assert entry.tuning_budget_runs == 50
    assert entry.tuning_source == "declared"


def test_build_capability_matrix_from_configs():
    """Matrix is built from campaign-style planner config dicts."""
    configs = [
        {"algo": "goal"},
        {"algo": "social_force"},
        {"algo": "orca"},
    ]
    matrix = build_capability_matrix(configs)
    assert len(matrix.entries) == 3
    assert "goal" in matrix.entries
    assert "social_force" in matrix.entries
    assert "orca" in matrix.entries


def test_build_capability_matrix_deduplicates():
    """Duplicate algo keys produce a single matrix entry."""
    configs = [{"algo": "orca"}, {"algo": "orca"}]
    matrix = build_capability_matrix(configs)
    assert len(matrix.entries) == 1


def test_build_capability_matrix_tolerates_malformed_tuning() -> None:
    """Non-mapping tuning metadata fails safely to unknown provenance."""
    matrix = build_capability_matrix([{"algo": "orca", "tuning": "invalid"}])
    assert matrix.entries["orca"].tuning_budget_runs is None
    assert matrix.entries["orca"].tuning_source == "unknown"


def test_capability_entry_to_dict():
    """Entry serialization produces all expected keys."""
    entry = build_capability_entry("orca")
    d = entry.to_dict()
    assert "canonical_name" in d
    assert "observation_privilege_level" in d
    assert "adapter_active" in d
    assert "tuning_budget_runs" in d


# ──────────────────────────────────────────────────────────────────────
# Mismatch detection
# ──────────────────────────────────────────────────────────────────────


def test_no_mismatch_between_matched_adapters():
    """social_force and ORCA share the same privilege tier; only soft adapter name diff."""
    configs = [{"algo": "social_force"}, {"algo": "orca"}]
    matrix = build_capability_matrix(configs)
    mismatches = detect_mismatches(matrix)
    hard = [m for m in mismatches if m.severity == "hard"]
    assert len(hard) == 0, f"Unexpected hard mismatches: {hard}"


def test_adapter_mismatch_native_vs_adapter():
    """goal (native) vs orca (adapter) produces a hard adapter mismatch."""
    configs = [{"algo": "goal"}, {"algo": "orca"}]
    matrix = build_capability_matrix(configs)
    mismatches = detect_mismatches(matrix)
    adapter_hard = [m for m in mismatches if m.dimension == "adapter" and m.severity == "hard"]
    assert len(adapter_hard) == 1
    assert adapter_hard[0].planner_a == "goal"
    assert adapter_hard[0].planner_b == "orca"


def test_observation_privilege_mismatch():
    """goal (goal_state, tier 1) vs ppo (sensor_fusion_state, tier 3) is a hard mismatch."""
    configs = [{"algo": "goal"}, {"algo": "ppo"}]
    matrix = build_capability_matrix(configs)
    mismatches = detect_mismatches(matrix)
    obs_hard = [
        m for m in mismatches if m.dimension == "observation_privilege" and m.severity == "hard"
    ]
    assert len(obs_hard) == 1
    assert "goal" in obs_hard[0].description
    assert "ppo" in obs_hard[0].description


def test_tuning_asymmetry_detected():
    """Different tuning sources produce a soft tuning_asymmetry mismatch."""
    configs = [
        {"algo": "orca", "tuning": {"budget_runs": 100, "source": "declared"}},
        {
            "algo": "social_force",
            "tuning": {"budget_runs": 50, "source": "backfilled"},
        },
    ]
    matrix = build_capability_matrix(configs)
    mismatches = detect_mismatches(matrix)
    tuning_soft = [
        m for m in mismatches if m.dimension == "tuning_asymmetry" and m.severity == "soft"
    ]
    assert len(tuning_soft) >= 1


def test_no_tuning_mismatch_when_unknown():
    """Unknown tuning sources do not trigger a tuning source mismatch."""
    configs = [
        {"algo": "orca", "tuning": {"source": "unknown"}},
        {"algo": "social_force", "tuning": {"source": "unknown"}},
    ]
    matrix = build_capability_matrix(configs)
    mismatches = detect_mismatches(matrix)
    tuning = [m for m in mismatches if m.dimension == "tuning_asymmetry"]
    assert len(tuning) == 0


# ──────────────────────────────────────────────────────────────────────
# Fair-comparison subset
# ──────────────────────────────────────────────────────────────────────


def test_fair_subset_all_matched():
    """All matched planners appear in the fair subset."""
    configs = [{"algo": "social_force"}, {"algo": "orca"}]
    matrix = build_capability_matrix(configs)
    mismatches = detect_mismatches(matrix)
    fair, excluded = fair_comparison_subset(matrix, mismatches)
    assert set(fair) == {"social_force", "orca"}
    assert len(excluded) == 0


def test_fair_subset_excludes_hard_mismatch():
    """Planners with hard mismatches are excluded from the fair subset."""
    configs = [{"algo": "goal"}, {"algo": "orca"}]
    matrix = build_capability_matrix(configs)
    mismatches = detect_mismatches(matrix)
    _fair, excluded = fair_comparison_subset(matrix, mismatches)
    # Both get excluded because the hard mismatch involves both
    assert "goal" in excluded or "orca" in excluded


def test_fair_subset_three_planners_mixed():
    """Three planners with mixed privilege levels: only matched pair stays fair."""
    configs = [{"algo": "social_force"}, {"algo": "orca"}, {"algo": "ppo"}]
    matrix = build_capability_matrix(configs)
    mismatches = detect_mismatches(matrix)
    _fair, excluded = fair_comparison_subset(matrix, mismatches)
    # ppo has different privilege tier -> excluded
    assert "ppo" in excluded


# ──────────────────────────────────────────────────────────────────────
# Ranking-claim gate
# ──────────────────────────────────────────────────────────────────────


def test_ranking_gate_allows_matched_planners():
    """Ranking claim is allowed when all planners have matched capabilities."""
    configs = [{"algo": "social_force"}, {"algo": "orca"}]
    matrix = build_capability_matrix(configs)
    verdict = ranking_claim_gate(matrix)
    assert verdict.allowed is True
    assert verdict.hard_mismatch_count == 0


def test_ranking_gate_blocks_mismatched_planners():
    """Ranking claim is blocked when hard mismatches exist."""
    configs = [{"algo": "goal"}, {"algo": "ppo"}]
    matrix = build_capability_matrix(configs)
    verdict = ranking_claim_gate(matrix)
    assert verdict.allowed is False
    assert verdict.hard_mismatch_count >= 1
    assert "blocked" in verdict.reason.lower()


def test_ranking_gate_soft_mismatch_does_not_block():
    """Soft mismatches (adapter name, tuning budget) do not block ranking claims."""
    configs = [{"algo": "social_force"}, {"algo": "orca"}]
    matrix = build_capability_matrix(configs)
    verdict = ranking_claim_gate(matrix)
    assert verdict.allowed is True
    if verdict.soft_mismatch_count > 0:
        assert "soft" in verdict.reason.lower() or "caveat" in verdict.reason.lower()


def test_ranking_gate_verdict_serialization():
    """Verdict serialization includes all expected fields."""
    configs = [{"algo": "social_force"}, {"algo": "orca"}]
    matrix = build_capability_matrix(configs)
    verdict = ranking_claim_gate(matrix)
    d = verdict.to_dict()
    assert "allowed" in d
    assert "reason" in d
    assert "fair_subset" in d
    assert "excluded" in d
    assert "hard_mismatch_count" in d
    assert "soft_mismatch_count" in d


def test_ranking_gate_single_planner():
    """A single planner trivially passes the ranking gate."""
    configs = [{"algo": "orca"}]
    matrix = build_capability_matrix(configs)
    verdict = ranking_claim_gate(matrix)
    assert verdict.allowed is True
    assert verdict.hard_mismatch_count == 0


# ──────────────────────────────────────────────────────────────────────
# Report integration
# ──────────────────────────────────────────────────────────────────────


def test_emit_mismatch_flags_adds_keys():
    """emit_mismatch_flags augments a report row with fairness keys."""
    configs = [{"algo": "social_force"}, {"algo": "orca"}]
    matrix = build_capability_matrix(configs)
    row: dict[str, object] = {"success_mean": 0.8}
    augmented = emit_mismatch_flags(matrix, row, "orca")
    assert "fairness_mismatch_flags" in augmented
    assert "fairness_in_ranking_subset" in augmented
    assert isinstance(augmented["fairness_mismatch_flags"], list)


def test_emit_mismatch_flags_marks_excluded():
    """Excluded planners get fairness_in_ranking_subset=False."""
    configs = [{"algo": "goal"}, {"algo": "ppo"}]
    matrix = build_capability_matrix(configs)
    row: dict[str, object] = {}
    augmented = emit_mismatch_flags(matrix, row, "goal")
    assert augmented["fairness_in_ranking_subset"] is False


def test_emit_mismatch_flags_normalizes_and_validates_planner_name() -> None:
    """Report rows use canonical planner identity and reject unknown planners."""
    matrix = build_capability_matrix([{"algo": "orca"}])
    assert emit_mismatch_flags(matrix, {}, "ORCA")["fairness_in_ranking_subset"] is True
    with pytest.raises(KeyError, match="not in the capability matrix"):
        emit_mismatch_flags(matrix, {}, "unknown_planner")


def test_fairness_report_serialization():
    """FairnessReport serialization round-trips through to_dict."""
    configs = [{"algo": "social_force"}, {"algo": "orca"}]
    matrix = build_capability_matrix(configs)
    mismatches = detect_mismatches(matrix)
    fair, excluded = fair_comparison_subset(matrix, mismatches)
    report = FairnessReport(
        matrix=matrix,
        mismatches=mismatches,
        fair_subset=fair,
        excluded_planners=excluded,
        ranking_claim_allowed=True,
    )
    d = report.to_dict()
    assert "matrix" in d
    assert "mismatches" in d
    assert "fair_subset" in d
    assert "ranking_claim_allowed" in d


# ──────────────────────────────────────────────────────────────────────
# Mismatch row cannot pass ranking gate
# ──────────────────────────────────────────────────────────────────────


def test_mismatched_row_cannot_pass_ranking_gate():
    """A cross-planner table with mismatched rows is rejected by the ranking gate.

    This is the core acceptance criterion: when planners have different
    observation privilege or execution mode, the ranking gate blocks
    algorithm-ranking claims.
    """
    configs = [
        {"algo": "goal"},
        {"algo": "social_force"},
        {"algo": "ppo"},
    ]
    matrix = build_capability_matrix(configs)
    verdict = ranking_claim_gate(matrix)
    assert verdict.allowed is False
    assert verdict.hard_mismatch_count >= 1
    assert len(verdict.fair_subset) < len(configs)


def test_matched_rows_pass_ranking_gate():
    """A cross-planner table with all matched rows passes the ranking gate."""
    configs = [
        {"algo": "social_force"},
        {"algo": "orca"},
        {"algo": "socnav_orca_nonholonomic"},
    ]
    matrix = build_capability_matrix(configs)
    verdict = ranking_claim_gate(matrix)
    # socnav_orca_nonholonomic is experimental vs baseline-ready but
    # readiness tier is not a fairness dimension; observation/adapter match
    assert verdict.hard_mismatch_count == 0


# ──────────────────────────────────────────────────────────────────────
# Fairness report builder
# ──────────────────────────────────────────────────────────────────────


def test_build_fairness_report_all_matched():
    """build_fairness_report returns a complete report for matched planners."""
    from robot_sf.benchmark.fairness_contract import build_fairness_report

    configs = [
        {"algo": "social_force"},
        {"algo": "orca"},
    ]
    report = build_fairness_report(configs)
    assert report.ranking_claim_allowed is True
    assert len(report.mismatches) >= 1  # soft adapter name mismatch expected
    assert len(report.fair_subset) == 2
    assert len(report.excluded_planners) == 0


def test_build_fairness_report_mismatched():
    """build_fairness_report blocks ranking claims for mismatched planners."""
    from robot_sf.benchmark.fairness_contract import build_fairness_report

    configs = [
        {"algo": "goal"},
        {"algo": "ppo"},
    ]
    report = build_fairness_report(configs)
    assert report.ranking_claim_allowed is False
    assert len(report.excluded_planners) >= 1


def test_build_fairness_report_serialization():
    """Fairness report serialization round-trips through to_dict."""
    from robot_sf.benchmark.fairness_contract import build_fairness_report

    configs = [
        {"algo": "social_force"},
        {"algo": "orca"},
    ]
    report = build_fairness_report(configs)
    d = report.to_dict()
    assert "matrix" in d
    assert "mismatches" in d
    assert "fair_subset" in d
    assert "ranking_claim_allowed" in d
    assert isinstance(d["mismatches"], list)


def test_emit_fairness_annotations():
    """emit_fairness_annotations mutates rows in place with fairness keys."""
    from robot_sf.benchmark.fairness_contract import (
        build_fairness_report,
        emit_fairness_annotations,
    )

    configs = [
        {"algo": "social_force"},
        {"algo": "orca"},
    ]
    report = build_fairness_report(configs)
    rows = [
        {"planner_key": "social_force", "algo": "social_force", "success_mean": 0.8},
        {"planner_key": "orca", "algo": "orca", "success_mean": 0.7},
    ]
    emit_fairness_annotations(report, rows)
    for row in rows:
        assert "fairness_mismatch_flags" in row
        assert "fairness_in_ranking_subset" in row
        assert isinstance(row["fairness_mismatch_flags"], list)


def test_emit_fairness_annotations_excluded_planners():
    """Excluded planners get fairness_in_ranking_subset=False."""
    from robot_sf.benchmark.fairness_contract import (
        build_fairness_report,
        emit_fairness_annotations,
    )

    configs = [
        {"algo": "goal"},
        {"algo": "ppo"},
    ]
    report = build_fairness_report(configs)
    rows = [
        {"planner_key": "goal", "algo": "goal"},
        {"planner_key": "ppo", "algo": "ppo"},
    ]
    emit_fairness_annotations(report, rows)
    for row in rows:
        assert row["fairness_in_ranking_subset"] is False


def test_emit_fairness_annotations_handles_unknown_planner():
    """emit_fairness_annotations handles rows with unknown planners gracefully."""
    from robot_sf.benchmark.fairness_contract import (
        build_fairness_report,
        emit_fairness_annotations,
    )

    configs = [{"algo": "orca"}]
    report = build_fairness_report(configs)
    rows = [{"planner_key": "unknown_planner", "algo": "unknown"}]
    emit_fairness_annotations(report, rows)
    assert rows[0]["fairness_mismatch_flags"] == []
    assert rows[0]["fairness_in_ranking_subset"] is False
