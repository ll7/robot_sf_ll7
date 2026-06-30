"""Tests for the combined SNQI governance preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validation.check_snqi_governance import build_governance_report, main

REPO_ROOT = Path(__file__).parents[2]


def test_governance_report_marks_current_blockers_secondary_diagnostic() -> None:
    """Current unresolved SNQI blockers are explicit without changing scoring."""
    report = build_governance_report(repo_root=REPO_ROOT)

    assert report["status"] == "failed"
    assert "secondary_diagnostic_only" in report["claim_boundary"]
    assert "primary safety ranking" in report["claim_boundary"]
    assert {blocker["issue"] for blocker in report["blockers"]} == {3699, 3723}
    assert report["weights"]["has_blocking_conflict"] is True
    assert report["normalization"]["mixed_scale"] is True
    assert set(report["normalization"]["raw_penalty_terms"]) == {"time", "comfort"}
    blocker_3699 = next(
        blocker for blocker in report["blockers"] if blocker["kind"] == "mixed_normalization_basis"
    )
    status_by_term = {
        entry["term"]: entry["normalization_status"] for entry in blocker_3699["mixed_inputs"]
    }
    assert status_by_term == {
        "time": "raw_unbounded",
        "collisions": "baseline_normalized_bounded",
        "near": "baseline_normalized_bounded",
        "comfort": "raw_unbounded",
        "force_exceed": "baseline_normalized_bounded",
        "jerk": "baseline_normalized_bounded",
    }
    contributions = report["normalization_contributions"]
    assert contributions["schema_version"] == "snqi_normalization_contributions.v1"
    assert contributions["diagnostic_only"] is True
    assert contributions["mixed_basis"] is True
    assert contributions["raw_penalty_terms_dominate"] is True
    assert contributions["has_weight_bound_exceedance"] is True
    assert {term["term"] for term in contributions["weight_bound_exceedances"]} == {
        "time",
        "comfort",
    }
    assert (
        contributions["raw_penalty_absolute_share"]
        > contributions["baseline_normalized_penalty_absolute_share"]
    )


def test_governance_main_fails_closed_but_allows_inspection(tmp_path: Path) -> None:
    """Default command fails closed; inspection mode emits the same report and exits 0."""
    json_out = tmp_path / "snqi_governance.json"

    assert main(["--repo-root", str(REPO_ROOT), "--json-out", str(json_out)]) == 2
    payload = json.loads(json_out.read_text(encoding="utf-8"))
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
    """Human preflight output lists raw and baseline-normalized term statuses."""
    assert main(["--repo-root", str(REPO_ROOT), "--allow-current-blockers"]) == 0

    out = capsys.readouterr().out
    assert "Weight sources:" in out
    assert "code_default (code_default, <code default>): canonical=True" in out
    assert "model_canonical_v1 (shipped_json, model/snqi_canonical_weights_v1.json)" in out
    assert (
        "camera_ready_v3 (shipped_json, configs/benchmarks/snqi_weights_camera_ready_v3.json)"
    ) in out
    assert "dominant=w_collisions; scale=raw; sha256=" in out
    assert "dominant=w_near; scale=normalized_simplex; sha256=" in out
    assert "Weight provenance conflicts:" in out
    assert "error canonical_direction_conflict (code_default, model_canonical_v1)" in out
    assert "Term normalization status:" in out
    assert "time (time_to_goal_norm, w_time): raw_unbounded" in out
    assert "basis=raw time-to-goal ratio" in out
    assert "comfort (comfort_exposure, w_comfort): raw_unbounded" in out
    assert "basis=raw accumulated comfort-exposure value" in out
    assert "collisions (collisions, w_collisions): baseline_normalized_bounded" in out
    assert "basis=baseline-relative median/p95 clamped value" in out
    assert "jerk (jerk_mean, w_jerk): baseline_normalized_bounded" in out
    assert "Contribution diagnostic:" in out
    assert "raw_penalty_terms_dominate=True" in out
    assert "has_weight_bound_exceedance=True" in out


def test_governance_report_checks_optional_baseline_coverage(tmp_path: Path) -> None:
    """When baseline stats are supplied, missing normalized terms are blockers."""
    baseline_stats = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
    }
    path = tmp_path / "baseline_stats.json"
    path.write_text(json.dumps(baseline_stats), encoding="utf-8")

    assert main(["--repo-root", str(REPO_ROOT), "--baseline-stats", str(path)]) == 2

    report = build_governance_report(repo_root=REPO_ROOT, baseline_stats=baseline_stats)
    missing = [b for b in report["blockers"] if b["kind"] == "missing_baseline_coverage"]
    assert missing
    assert set(missing[0]["metrics"]) == {"force_exceed_events", "jerk_mean"}
