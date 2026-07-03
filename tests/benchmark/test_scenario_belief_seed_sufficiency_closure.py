"""Tests for the issue #3556 ScenarioBelief seed-sufficiency closure resolver.

Covers the pure closure-packet builder and the script-level durable-root probe /
fail-closed behavior. These tests use only synthetic fixtures; they never launch a
benchmark campaign.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.scenario_belief_screening import (
    REQUIRED_SEED_SUFFICIENCY_REPORTS,
    SEED_SUFFICIENCY_CLOSURE_LABELS,
    build_seed_sufficiency_closure_packet,
)
from scripts.validation import close_issue_3556_seed_sufficiency as closer

if TYPE_CHECKING:
    from pathlib import Path


def _write_seed_reports(campaign_root: Path, *, include_episode_rows: bool = True) -> None:
    """Create a minimal campaign root that the analyzer discovery would accept."""
    reports = campaign_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "seed_variability_by_scenario.json").write_text(
        json.dumps({"rows": []}), encoding="utf-8"
    )
    if include_episode_rows:
        (reports / "seed_episode_rows.csv").write_text("seed,scenario\n", encoding="utf-8")


# --- pure packet builder -------------------------------------------------------


def test_packet_blocked_when_no_resolved_root() -> None:
    """No resolved root yields a fail-closed blocker packet, not fabricated evidence."""
    packet = build_seed_sufficiency_closure_packet(
        searched_roots=[
            {"search_root": "docs/context/evidence", "exists": True, "campaign_roots_found": []}
        ],
        resolved_campaign_root=None,
        analyzer_command=["uv", "run", "python", "scripts/tools/analyze_seed_sufficiency.py"],
    )
    assert packet["evidence_status"] == "blocked"
    assert packet["decision_label"] == "blocked_missing_retained_campaign_outputs"
    assert packet["decision_label"] in SEED_SUFFICIENCY_CLOSURE_LABELS
    assert packet["resolved_campaign_root"] is None
    assert packet["analyzer_summary"] is None
    assert packet["required_report_files"] == list(REQUIRED_SEED_SUFFICIENCY_REPORTS)
    # Fail-closed packets must never silently imply campaign/Slurm/claim actions ran.
    assert packet["forbidden_actions_confirmed"]["full_benchmark_campaign_run"] is False
    assert packet["forbidden_actions_confirmed"]["slurm_or_gpu_submission"] is False


def test_packet_promoted_when_root_resolved() -> None:
    """A resolved root promotes the packet and carries the analyzer summary."""
    packet = build_seed_sufficiency_closure_packet(
        searched_roots=[
            {
                "search_root": "docs/context/evidence",
                "exists": True,
                "campaign_roots_found": ["docs/context/evidence/issue_3556_x"],
                "usable_campaign_roots": ["docs/context/evidence/issue_3556_x"],
            }
        ],
        resolved_campaign_root="docs/context/evidence/issue_3556_x",
        analyzer_command=["uv", "run", "python", "x"],
        analyzer_output_dir="output/issue_3556_seed_sufficiency",
        analyzer_summary={"headline_claim_status": "blocked", "campaign_count": 1},
    )
    assert packet["evidence_status"] == "promoted"
    assert packet["decision_label"] == "resolved_retained_campaign"
    assert packet["resolved_campaign_root"] == "docs/context/evidence/issue_3556_x"
    assert packet["analyzer_summary"]["campaign_count"] == 1


# --- script probe / resolve ----------------------------------------------------


def test_probe_reports_missing_episode_rows(tmp_path: Path) -> None:
    """A root with only the JSON report is discovered but not usable."""
    container = tmp_path / "evidence"
    campaign = container / "issue_3556_partial"
    _write_seed_reports(campaign, include_episode_rows=False)

    probe = closer._probe_search_root(container, "issue_3556")
    assert probe["exists"] is True
    assert probe["campaign_roots_found"], "json-only root should still be discovered"
    assert probe["usable_campaign_roots"] == []
    # The missing file must be named explicitly for a reproducible blocker.
    missing = next(iter(probe["missing_report_files"].values()))
    assert "seed_episode_rows.csv" in missing


def test_probe_missing_root_is_empty(tmp_path: Path) -> None:
    """A non-existent search root probes cleanly as empty, not an error."""
    probe = closer._probe_search_root(tmp_path / "does_not_exist", "issue_3556")
    assert probe["exists"] is False
    assert probe["campaign_roots_found"] == []
    assert probe["usable_campaign_roots"] == []


def test_run_closure_fails_closed_without_retained_reports(tmp_path: Path) -> None:
    """With no retained reports, the closure writes a blocked packet and skips analysis."""
    evidence_dir = tmp_path / "closure"
    packet = closer.run_closure(
        search_roots=[tmp_path / "empty_a", tmp_path / "empty_b"],
        campaign_id="issue_3556",
        analyzer_output_dir=tmp_path / "analyzer_out",
        evidence_dir=evidence_dir,
    )
    assert packet["evidence_status"] == "blocked"
    assert packet["decision_label"] == "blocked_missing_retained_campaign_outputs"
    # Durable artifacts must be written even in the blocked case.
    assert (evidence_dir / "summary.json").is_file()
    assert (evidence_dir / "README.md").is_file()
    on_disk = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "issue_3556_seed_sufficiency_closure.v1"
    # The analyzer output dir must not exist because analysis never ran.
    assert not (tmp_path / "analyzer_out").exists()


def test_run_closure_resolves_and_runs_analyzer(tmp_path: Path) -> None:
    """A usable retained root resolves and the analyzer produces artifacts."""
    container = tmp_path / "evidence"
    campaign = container / "issue_3556_full"
    _write_seed_reports(campaign, include_episode_rows=True)
    evidence_dir = tmp_path / "closure"
    analyzer_out = tmp_path / "analyzer_out"

    packet = closer.run_closure(
        search_roots=[container],
        campaign_id="issue_3556",
        analyzer_output_dir=analyzer_out,
        evidence_dir=evidence_dir,
    )
    assert packet["evidence_status"] == "promoted"
    assert packet["decision_label"] == "resolved_retained_campaign"
    assert packet["resolved_campaign_root"].endswith("issue_3556_full")
    # The analyzer ran and wrote its canonical JSON artifact.
    assert (analyzer_out / "seed_sufficiency_analysis.json").is_file()
    assert packet["analyzer_summary"] is not None


def test_main_blocked_exit_code(tmp_path: Path) -> None:
    """The CLI returns the blocked exit code and still writes the packet."""
    evidence_dir = tmp_path / "closure"
    code = closer.main(
        [
            "--search-root",
            str(tmp_path / "nothing"),
            "--analyzer-output-dir",
            str(tmp_path / "out"),
            "--evidence-dir",
            str(evidence_dir),
        ]
    )
    assert code == closer.BLOCKED_EXIT_CODE
    assert (evidence_dir / "summary.json").is_file()


def test_main_exit_zero_on_blocked_flag(tmp_path: Path) -> None:
    """The opt-in flag makes a blocked closure exit 0 for CI-safe packet generation."""
    code = closer.main(
        [
            "--search-root",
            str(tmp_path / "nothing"),
            "--analyzer-output-dir",
            str(tmp_path / "out"),
            "--evidence-dir",
            str(tmp_path / "closure"),
            "--exit-zero-on-blocked",
        ]
    )
    assert code == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
