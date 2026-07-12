"""End-to-end artifact-writer regression for fairness contract report integration (issue #5365).

Validates that the canonical campaign report writer emits fairness mismatch
annotations and gates ranking claims when planners have hard capability
mismatches.  Report-layer only — no metric-semantics changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from robot_sf.benchmark.camera_ready._reporting import write_campaign_report
from robot_sf.benchmark.fairness_contract import (
    build_capability_matrix,
    detect_mismatches,
    emit_mismatch_flags,
    fair_comparison_subset,
    ranking_claim_gate,
)

if TYPE_CHECKING:
    from pathlib import Path


def _build_fairness_payload(planner_configs: list[dict[str, object]]) -> dict[str, object]:
    """Build a fairness payload mirroring the campaign orchestrator logic."""
    matrix = build_capability_matrix(planner_configs)
    mismatches = detect_mismatches(matrix)
    fair_subset, excluded = fair_comparison_subset(matrix, mismatches)
    verdict = ranking_claim_gate(matrix)
    return {
        "capability_matrix": matrix.to_dict(),
        "ranking_claim_verdict": verdict.to_dict(),
        "fair_subset": list(fair_subset),
        "excluded_planners": list(excluded),
        "mismatches": [
            {
                "dimension": m.dimension,
                "planner_a": m.planner_a,
                "planner_b": m.planner_b,
                "value_a": m.value_a,
                "value_b": m.value_b,
                "severity": m.severity,
                "description": m.description,
            }
            for m in mismatches
        ],
    }


def _make_planner_row(
    algo: str,
    *,
    success: str = "0.5",
    snqi: str = "0.5",
) -> dict[str, object]:
    """Build a minimal planner row matching the report writer contract."""
    return {
        "planner_key": f"planner_{algo}",
        "algo": algo,
        "planner_group": "experimental",
        "kinematics": "holonomic",
        "status": "ok",
        "started_at_utc": "2026-07-12T00:00:00Z",
        "runtime_sec": 10.0,
        "episodes": 100,
        "episodes_per_second": 10.0,
        "success_mean": success,
        "collisions_mean": "0.0",
        "snqi_mean": snqi,
        "projection_rate": "0.0",
        "infeasible_rate": "0.0",
        "execution_mode": "native",
        "execution_detail": "direct_holonomic_world_velocity",
        "planner_command_space": "holonomic_vxy_world",
        "benchmark_command_space": "holonomic_vxy_world",
        "projection_policy": "world_velocity_passthrough",
        "readiness_status": "ok",
        "readiness_tier": "baseline-ready",
        "preflight_status": "ok",
        "learned_policy_contract_status": "not_applicable",
        "socnav_prereq_policy": "fail-fast",
    }


def test_mismatched_planners_report_emits_fairness_section(tmp_path: Path) -> None:
    """Report with mismatched planners contains fairness contract section."""
    planner_configs = [{"algo": "goal"}, {"algo": "ppo"}]
    rows = [_make_planner_row("goal"), _make_planner_row("ppo")]
    matrix = build_capability_matrix(planner_configs)
    for row in rows:
        emit_mismatch_flags(matrix, row, str(row["algo"]))

    fairness = _build_fairness_payload(planner_configs)
    report_path = tmp_path / "campaign_report.md"
    payload = {
        "campaign": {"campaign_id": "test_fairness"},
        "planner_rows": rows,
        "warnings": [],
        "fairness": fairness,
    }
    write_campaign_report(report_path, payload)
    report_text = report_path.read_text(encoding="utf-8")

    assert "## Fairness Contract" in report_text
    assert "Ranking claim allowed: **False**" in report_text
    assert "Hard mismatch count:" in report_text
    assert "Excluded from ranking" in report_text
    assert "## Pairwise Mismatches" in report_text


def test_matched_planners_report_emits_fair_subset(tmp_path: Path) -> None:
    """Report with matched planners shows fair subset and allows ranking."""
    planner_configs = [{"algo": "social_force"}, {"algo": "orca"}]
    rows = [_make_planner_row("social_force"), _make_planner_row("orca")]
    matrix = build_capability_matrix(planner_configs)
    for row in rows:
        emit_mismatch_flags(matrix, row, str(row["algo"]))

    fairness = _build_fairness_payload(planner_configs)
    report_path = tmp_path / "campaign_report.md"
    payload = {
        "campaign": {"campaign_id": "test_fairness"},
        "planner_rows": rows,
        "warnings": [],
        "fairness": fairness,
    }
    write_campaign_report(report_path, payload)
    report_text = report_path.read_text(encoding="utf-8")

    assert "## Fairness Contract" in report_text
    assert "Ranking claim allowed: **True**" in report_text
    assert "Fair comparison subset:" in report_text


def test_planner_rows_annotated_with_fairness_flags() -> None:
    """Planner rows receive fairness_mismatch_flags after emit_mismatch_flags."""
    planner_configs = [{"algo": "goal"}, {"algo": "ppo"}, {"algo": "social_force"}]
    rows = [
        _make_planner_row("goal"),
        _make_planner_row("ppo"),
        _make_planner_row("social_force"),
    ]
    matrix = build_capability_matrix(planner_configs)
    for row in rows:
        emit_mismatch_flags(matrix, row, str(row["algo"]))

    assert "fairness_mismatch_flags" in rows[0]
    assert "fairness_in_ranking_subset" in rows[0]
    assert isinstance(rows[0]["fairness_mismatch_flags"], list)
    assert isinstance(rows[0]["fairness_in_ranking_subset"], bool)
    # social_force is in a different privilege tier than goal and ppo
    assert rows[2]["fairness_in_ranking_subset"] is False


def test_ranking_gate_blocks_when_hard_mismatches_exist() -> None:
    """Ranking claim gate returns allowed=False when hard mismatches exist."""
    planner_configs = [{"algo": "goal"}, {"algo": "ppo"}]
    matrix = build_capability_matrix(planner_configs)
    verdict = ranking_claim_gate(matrix)
    assert verdict.allowed is False
    assert verdict.hard_mismatch_count >= 1
    assert len(verdict.fair_subset) < len(planner_configs)


def test_report_without_fairness_section_is_backward_compatible(tmp_path: Path) -> None:
    """Reports without fairness payload render normally (backward compat)."""
    rows = [_make_planner_row("goal")]
    report_path = tmp_path / "campaign_report.md"
    payload = {
        "campaign": {"campaign_id": "no_fairness"},
        "planner_rows": rows,
        "warnings": [],
    }
    write_campaign_report(report_path, payload)
    report_text = report_path.read_text(encoding="utf-8")
    assert "## Fairness Contract" not in report_text
    assert "## Planner Summary" in report_text


def test_fairness_annotated_rows_in_report_table(tmp_path: Path) -> None:
    """Report planner table includes rows with fairness annotations."""
    planner_configs = [{"algo": "social_force"}, {"algo": "orca"}]
    rows = [_make_planner_row("social_force"), _make_planner_row("orca")]
    matrix = build_capability_matrix(planner_configs)
    for row in rows:
        emit_mismatch_flags(matrix, row, str(row["algo"]))

    fairness = _build_fairness_payload(planner_configs)
    report_path = tmp_path / "campaign_report.md"
    payload = {
        "campaign": {"campaign_id": "test_table"},
        "planner_rows": rows,
        "warnings": [],
        "fairness": fairness,
    }
    write_campaign_report(report_path, payload)
    report_text = report_path.read_text(encoding="utf-8")

    assert "planner_social_force" in report_text
    assert "planner_orca" in report_text
    assert "## Fairness Contract" in report_text
    # Both are in the fair subset
    assert "Fair comparison subset:" in report_text


def test_mismatched_row_cannot_pass_ranking_gate() -> None:
    """Core acceptance: mismatched rows are rejected by the ranking gate.

    This is the core acceptance criterion from issue #5365: when planners have
    different observation privilege or execution mode, the ranking gate blocks
    algorithm-ranking claims.  Mismatched rows remain case-study evidence.
    """
    planner_configs = [
        {"algo": "goal"},
        {"algo": "social_force"},
        {"algo": "ppo"},
    ]
    matrix = build_capability_matrix(planner_configs)
    verdict = ranking_claim_gate(matrix)
    assert verdict.allowed is False
    assert verdict.hard_mismatch_count >= 1
    assert len(verdict.fair_subset) < len(planner_configs)
    assert len(verdict.excluded) >= 1
