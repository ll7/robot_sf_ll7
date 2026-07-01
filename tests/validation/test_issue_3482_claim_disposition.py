"""Regression tests for issue #3482 release 0.0.2 claim disposition."""

from __future__ import annotations

import json
from pathlib import Path

DISPOSITION = Path(
    "docs/context/evidence/issue_3482_release_0_0_2_claim_disposition_2026_07_01/manifest.json"
)
BOUNDARY = Path(
    "docs/context/evidence/issue_3482_release_0_0_2_collision_count_boundary/manifest.json"
)
TABLE_BUNDLE_MANIFEST = Path(
    "docs/context/evidence/issue_2686_release_0_0_2_table_bundle/artifact_manifest.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_issue_3482_disposition_withdraws_release_collision_count_claims() -> None:
    """The terminal disposition must withdraw, not validate, collision-count claims."""
    payload = _load(DISPOSITION)

    assert payload["schema_version"] == "issue_3482_release_0_0_2_claim_disposition.v1"
    assert payload["release_tag"] == "0.0.2"
    assert payload["release_target_commit"] == "f7ebdcae2375d085e925213197a75a386e26a79c"
    assert payload["claim_disposition"]["release_0_0_2_total_collision_count"]["status"] == (
        "withdrawn_exact_event_provenance_unavailable"
    )
    assert payload["closure_path"]["github_closure_reason"] == "not_planned"
    assert payload["closure_path"]["auto_close_pr_keyword_allowed"] is False


def test_issue_3482_boundary_links_terminal_disposition() -> None:
    """The older boundary manifest must point to the terminal disposition packet."""
    payload = _load(BOUNDARY)

    assert payload["claim_boundaries"]["collision_count_metric_status"] == (
        "withdrawn_exact_event_provenance_unavailable"
    )
    assert payload["claim_boundaries"]["exact_collision_outcome_status"] == (
        "bounded_diagnostic_only"
    )
    assert payload["claim_boundaries"]["open_gates"] == []
    assert payload["source"]["claim_disposition"] == str(DISPOSITION)


def test_release_table_bundle_marks_affected_collision_fields_withdrawn() -> None:
    """Every release 0.0.2 table artifact must carry the issue #3482 caveat."""
    payload = _load(TABLE_BUNDLE_MANIFEST)

    affected_ids = {
        "tab_release_failure_count_slices",
        "tab_results_overview",
        "tab_robot_sf_release_planner_results",
    }
    artifacts = {artifact["artifact_id"]: artifact for artifact in payload["artifacts"]}

    assert affected_ids <= set(artifacts)
    for artifact_id in affected_ids:
        metadata = artifacts[artifact_id]["metadata"]["issue_3482_collision_count_disposition"]
        assert metadata["status"] == "withdrawn_exact_event_provenance_unavailable"
        assert metadata["manifest"] == str(DISPOSITION)
        assert "total_collision_count" in metadata["affected_fields"]
