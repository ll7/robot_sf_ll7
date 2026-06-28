"""Tests for the combined SNQI governance preflight."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.validation.check_snqi_governance import build_governance_report, main


def test_governance_report_marks_current_blockers_secondary_diagnostic() -> None:
    """Current unresolved SNQI blockers are explicit without changing scoring."""
    report = build_governance_report(repo_root=Path.cwd())

    assert report["status"] == "failed"
    assert "secondary_diagnostic_only" in report["claim_boundary"]
    assert "primary safety ranking" in report["claim_boundary"]
    assert {blocker["issue"] for blocker in report["blockers"]} == {3699, 3723}
    assert report["weights"]["has_blocking_conflict"] is True
    assert report["normalization"]["mixed_scale"] is True
    assert set(report["normalization"]["raw_penalty_terms"]) == {"time", "comfort"}


def test_governance_main_fails_closed_but_allows_inspection(tmp_path: Path) -> None:
    """Default command fails closed; inspection mode emits the same report and exits 0."""
    json_out = tmp_path / "snqi_governance.json"

    assert main(["--repo-root", str(Path.cwd()), "--json-out", str(json_out)]) == 2
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"

    assert (
        main(
            [
                "--repo-root",
                str(Path.cwd()),
                "--json",
                "--allow-current-blockers",
            ]
        )
        == 0
    )


def test_governance_report_checks_optional_baseline_coverage(tmp_path: Path) -> None:
    """When baseline stats are supplied, missing normalized terms are blockers."""
    baseline_stats = {
        "collisions": {"med": 0.0, "p95": 1.0},
        "near_misses": {"med": 0.0, "p95": 1.0},
    }
    path = tmp_path / "baseline_stats.json"
    path.write_text(json.dumps(baseline_stats), encoding="utf-8")

    assert main(["--repo-root", str(Path.cwd()), "--baseline-stats", str(path)]) == 2

    report = build_governance_report(repo_root=Path.cwd(), baseline_stats=baseline_stats)
    missing = [b for b in report["blockers"] if b["kind"] == "missing_baseline_coverage"]
    assert missing
    assert set(missing[0]["metrics"]) == {"force_exceed_events", "jerk_mean"}
