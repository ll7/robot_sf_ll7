"""Tests combined SNQI governance preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validation.check_snqi_governance import build_governance_report, main

REPO_ROOT = Path(__file__).parents[2]


def test_governance_report_marks_current_blockers_secondary_diagnostic() -> None:  # noqa: PLR0915
    """Current unresolved SNQI blockers are explicit without changing scoring."""
    report = build_governance_report(repo_root=REPO_ROOT)

    assert report["status"] == "failed"
    assert "secondary_diagnostic_only" in report["claim_boundary"]
    assert "primary safety ranking" in report["claim_boundary"]
    assert {blocker["issue"] for blocker in report["blockers"]} == {3723}
    assert report["weights"]["has_blocking_conflict"] is True

    legacy = report["legacy_normalization"]
    assert legacy["score_version"] == "SNQI-v0"
    assert legacy["mixed_scale"] is True
    assert set(legacy["raw_penalty_terms"]) == {"time", "comfort"}

    active = report["normalization"]
    assert active["score_version"] == "SNQI-v1"
    assert active["mixed_scale"] is False
    assert active["is_consistent"] is True
    assert active["score_version_contract"]["status"] == "bounded_baseline_relative_diagnostic_only"

    blocker_3723 = next(
        blocker for blocker in report["blockers"] if blocker["kind"] == "weight_provenance_conflict"
    )
    assert blocker_3723["registered_sources"] == [
        "code_default",
        "camera_ready_v1",
        "camera_ready_v2",
        "camera_ready_v3",
        "model_canonical_v1",
    ]
    assert blocker_3723["discovered_shipped_sources"] == [
        "camera_ready_v1",
        "camera_ready_v2",
        "camera_ready_v3",
        "model_canonical_v1",
    ]
    expected_source_names = [
        "code_default",
        "camera_ready_v1",
        "camera_ready_v2",
        "camera_ready_v3",
        "model_canonical_v1",
    ]
    discovered_sources = blocker_3723["discovered_weight_sources"]
    assert [source["name"] for source in discovered_sources] == expected_source_names
    assert len(discovered_sources) == len(expected_source_names)
    discovered_by_name = {source["name"]: source for source in discovered_sources}
    assert discovered_by_name["code_default"]["versioned_id"] == "snqi_weights_code_default_v1"
    assert discovered_by_name["code_default"]["declares_canonical"] is True
    assert discovered_by_name["code_default"]["dominant_term"] == "w_collisions"
    assert discovered_by_name["code_default"]["scale_class"] == "raw"
    assert discovered_by_name["model_canonical_v1"]["versioned_id"] == (
        "snqi_weights_model_canonical_v1"
    )
    assert discovered_by_name["model_canonical_v1"]["declares_canonical"] is True
    assert discovered_by_name["model_canonical_v1"]["dominant_term"] == "w_jerk"
    assert discovered_by_name["camera_ready_v2"]["scale_class"] == "normalized_simplex"
    assert discovered_by_name["camera_ready_v3"]["dominant_term"] == "w_near"
    assert len(blocker_3723["blocking_conflicts"]) == 1
    conflict = blocker_3723["blocking_conflicts"][0]
    assert conflict["kind"] == "canonical_direction_conflict"
    assert conflict["severity"] == "error"
    assert conflict["sources"] == ["code_default", "model_canonical_v1"]
    assert "dominant=w_collisions" in conflict["detail"]
    assert "dominant=w_jerk" in conflict["detail"]
    comparisons_by_source = {
        comparison["source"]: comparison
        for comparison in blocker_3723["code_default_shipped_direction_comparisons"]
    }
    assert set(comparisons_by_source) == {
        "camera_ready_v1",
        "camera_ready_v2",
        "camera_ready_v3",
        "model_canonical_v1",
    }
    assert comparisons_by_source["model_canonical_v1"]["relationship"] == "different_direction"
    assert comparisons_by_source["model_canonical_v1"]["source_dominant_term"] == "w_jerk"
    assert comparisons_by_source["camera_ready_v3"]["relationship"] == "different_direction"
    assert comparisons_by_source["camera_ready_v3"]["source_dominant_term"] == "w_near"

    legacy_contributions = report["legacy_normalization_contributions"]
    assert legacy_contributions["score_version_contract"]["score_version"] == "SNQI-v0"
    assert legacy_contributions["score_version_contract"]["score_semantics_changed"] is False
    assert legacy_contributions["mixed_basis"] is True
    assert legacy_contributions["normalization_contract"]["decision_issue"] == 3699
    assert legacy_contributions["normalization_contract"]["successor_issue"] == 3978
    assert legacy_contributions["normalization_contract"]["weights_comparable"] is False
    assert {term["term"] for term in legacy_contributions["weight_bound_exceedances"]} == {
        "time",
        "comfort",
    }

    active_contributions = report["normalization_contributions"]
    assert active_contributions["schema_version"] == "snqi_normalization_contributions.v1"
    assert active_contributions["score_version_contract"]["score_version"] == "SNQI-v1"
    assert active_contributions["score_version_contract"]["score_semantics_changed"] is True
    assert active_contributions["mixed_basis"] is False
    assert active_contributions["raw_penalty_terms_dominate"] is False
    assert active_contributions["has_weight_bound_exceedance"] is False
    assert active_contributions["weight_bound_exceedances"] == []
    assert active_contributions["normalization_contract"]["status"] == "comparable_weight_basis"
    assert active_contributions["normalization_contract"]["weights_comparable"] is True
    assert active_contributions["score_version_contract"]["decision_issue"] == 3978


def test_governance_main_fails_closed_but_allows_inspection(tmp_path: Path) -> None:
    """Default command fails closed; inspection mode emits the same report."""
    out = tmp_path / "report.json"

    assert main(["--repo-root", str(REPO_ROOT), "--json-out", str(out)]) == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"

    assert (
        main(
            [
                "--repo-root",
                str(REPO_ROOT),
                "--json",
                "--allow-current-blockers",
            ]
        )
        == 0
    )


def test_governance_text_lists_per_term_normalization_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Human preflight output lists legacy and active normalization statuses."""
    assert main(["--repo-root", str(REPO_ROOT), "--allow-current-blockers"]) == 0

    out = capsys.readouterr().out
    assert "Canonical SNQI weight decision: unresolved_decision_required" in out
    assert "selected_versioned_id=None" in out
    assert "ambiguous_label=canonical" in out
    assert "Weight sources:" in out
    assert "code_default (code_default, <code default>): canonical=True" in out
    assert "versioned_id=snqi_weights_code_default_v1" in out
    assert "model_canonical_v1 (shipped_json, model/snqi_canonical_weights_v1.json)" in out
    assert "versioned_id=snqi_weights_model_canonical_v1" in out
    assert (
        "camera_ready_v3 (shipped_json, configs/benchmarks/snqi_weights_camera_ready_v3.json)"
        in out
    )
    assert "versioned_id=snqi_weights_camera_ready_v3" in out
    assert "dominant=w_collisions; scale=raw; sha256=" in out
    assert "dominant=w_near; scale=normalized_simplex; sha256=" in out
    assert "Code-default vs shipped JSON directions:" in out
    assert (
        "model_canonical_v1 (model/snqi_canonical_weights_v1.json): "
        "different_direction; source_dominant=w_jerk; source_scale=raw" in out
    )
    assert "Weight provenance conflicts:" in out
    assert "canonical_direction_conflict" in out
    assert "(code_default, model_canonical_v1)" in out
    assert "warning code_default_shipped_direction_mismatch (code_default, camera_ready_v1)" in out
    assert "warning code_default_shipped_direction_mismatch (code_default, camera_ready_v3)" in out
    assert "Legacy normalization basis: score_version=SNQI-v0" in out
    assert "raw_penalty_terms=time, comfort" in out
    assert "Term normalization status:" in out
    assert (
        "Score version contract: SNQI-v1; "
        "status=bounded_baseline_relative_diagnostic_only; diagnostic_only=True" in out
    )
    assert "time (time_to_goal_norm, w_time): baseline_normalized_bounded" in out
    assert "basis=baseline-relative median/p95 clamped value" in out
    assert "Contribution diagnostic:" in out
    assert "raw_penalty_share=0.000" in out
    assert "raw_penalty_terms_dominate=False" in out
    assert "has_weight_bound_exceedance=False" in out
    assert "Normalization checker: issue=3978 legacy_issue=3699" in out
    assert "score_version=SNQI-v1" in out
    assert "status=bounded_baseline_relative_diagnostic_only" in out


def test_governance_report_checks_optional_baseline_coverage(tmp_path: Path) -> None:
    """When baseline stats are supplied, missing SNQI-v1 normalized terms block."""
    baseline_stats = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
    }
    path = tmp_path / "baseline_stats.json"
    path.write_text(json.dumps(baseline_stats), encoding="utf-8")

    assert main(["--repo-root", str(REPO_ROOT), "--baseline-stats", str(path)]) == 2

    report = build_governance_report(repo_root=REPO_ROOT, baseline_stats=baseline_stats)
    missing = [b for b in report["blockers"] if b["kind"] == "missing_snqi_v1_baseline_coverage"]
    assert missing
    assert set(missing[0]["metrics"]) == {
        "time_to_goal_norm",
        "comfort_exposure",
        "force_exceed_events",
        "jerk_mean",
    }
    assert missing[0]["issue"] == 3978


def test_governance_checker_payload_is_referenced_in_report() -> None:
    """Normalization checker packet is included in the machine-parseable report."""
    report = build_governance_report(repo_root=REPO_ROOT)
    checker = report["normalization_checker"]
    assert checker["issue"] == 3978
    assert checker["legacy_issue"] == 3699
    assert checker["score_version"] == "SNQI-v1"
    assert checker["status"] == "bounded_baseline_relative_diagnostic_only"
    assert checker["diagnostic_only"] is True
    assert checker["decision_recorded"] is True
    assert checker["score_semantics_changed"] is True
    assert checker["assumption"] == (
        "SNQI-v1 is an opt-in bounded baseline-relative diagnostic. SNQI-v0 remains "
        "available for historical comparability and is not numerically comparable to SNQI-v1."
    )
    assert checker["mixed_scale"] is False
    assert checker["weights_comparable"] is True
    assert checker["normalization_contract_status"] == "comparable_weight_basis"
    assert checker["raw_penalty_terms_dominate"] is False
    assert checker["has_weight_bound_exceedance"] is False
    contributions = report["normalization_contributions"]
    assert checker["raw_penalty_absolute_share"] == pytest.approx(
        contributions["raw_penalty_absolute_share"]
    )
    assert checker["baseline_normalized_penalty_absolute_share"] == pytest.approx(
        contributions["baseline_normalized_penalty_absolute_share"]
    )


def test_governance_report_json_exports_normalization_checker(tmp_path: Path) -> None:
    """JSON export includes checker packet for external consumers."""
    out = tmp_path / "snqi_governance.json"
    exit_code = main(
        [
            "--repo-root",
            str(REPO_ROOT),
            "--json",
            "--json-out",
            str(out),
            "--allow-current-blockers",
        ]
    )

    assert exit_code == 0
    assert out.is_file()
    payload = json.loads(out.read_text(encoding="utf-8"))
    checker = payload["normalization_checker"]
    assert isinstance(checker["assumption"], str)
    assert "SNQI-v1 is an opt-in bounded baseline-relative diagnostic" in checker["assumption"]
    assert checker["legacy_issue"] == 3699
    assert checker["score_version"] == "SNQI-v1"
    assert checker["status"] == "bounded_baseline_relative_diagnostic_only"
    assert checker["diagnostic_only"] is True
    assert checker["decision_recorded"] is True
    assert checker["score_semantics_changed"] is True
