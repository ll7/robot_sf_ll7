"""Tests for the issue #4328 named-candidate seed-sufficiency closure evaluation.

Covers the pure per-candidate compatibility logic, the composed candidate closure
packet, and the script-level probe / fail-closed behavior. All tests use synthetic
fixtures; they never launch a benchmark campaign or touch the real named roots.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.scenario_belief_screening import (
    REQUIRED_SEED_SUFFICIENCY_REPORTS,
    SEED_SUFFICIENCY_CANDIDATE_LABELS,
    build_h600_candidate_closure_packet,
    evaluate_candidate_root_provenance,
    evaluate_seed_sufficiency_candidate,
)
from scripts.validation import (
    evaluate_issue_4328_h600_seed_sufficiency_candidates as evaluator,
)

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


# --- provenance check ----------------------------------------------------------


def test_provenance_incompatible_for_foreign_h600_lineage() -> None:
    """A named h600 roster campaign has no ScenarioBelief marker and is incompatible."""
    result = evaluate_candidate_root_provenance(
        name="issue3810-h600-extroster-run",
        lineage="issue #3810 h600 extended-roster run",
    )
    assert result["provenance_compatible"] is False
    assert result["matched_markers"] == []
    assert "different question" in result["reason"]


def test_provenance_compatible_for_scenario_belief_lineage() -> None:
    """A ScenarioBelief #3556 lineage matches at least one marker and is compatible."""
    result = evaluate_candidate_root_provenance(
        name="issue_3556_belief_mode_campaign",
        lineage="ScenarioBelief drop-vs-retain",
    )
    assert result["provenance_compatible"] is True
    assert result["matched_markers"]


# --- per-candidate compatibility ----------------------------------------------


def test_candidate_absent_root_records_all_blockers() -> None:
    """An absent foreign root fails every gate and is not compatible."""
    record = evaluate_seed_sufficiency_candidate(
        name="issue4230-h600-hybrid-roster-run",
        root="output/issue4230-h600-hybrid-roster-run/13282",
        exists_on_host=False,
        missing_report_files=list(REQUIRED_SEED_SUFFICIENCY_REPORTS),
        lineage="issue #4230 h600 hybrid-roster run",
    )
    assert record["compatible"] is False
    assert "root_absent_on_host" in record["blockers"]
    assert "missing_required_reports" in record["blockers"]
    assert "provenance_incompatible_with_3556" in record["blockers"]
    assert record["required_reports_present"] is False


def test_candidate_present_but_provenance_only_blocker() -> None:
    """A present, complete, but foreign root is blocked solely on provenance."""
    record = evaluate_seed_sufficiency_candidate(
        name="issue3810-h600-longhorizon-confirm-run",
        root="output/issue3810-h600-longhorizon-confirm-run/13268",
        exists_on_host=True,
        missing_report_files=[],
        lineage="issue #3810 h600 long-horizon confirmation run",
    )
    assert record["compatible"] is False
    assert record["blockers"] == ["provenance_incompatible_with_3556"]
    assert record["required_reports_present"] is True


def test_candidate_fully_compatible() -> None:
    """A present, complete, ScenarioBelief-lineage root passes every gate."""
    record = evaluate_seed_sufficiency_candidate(
        name="issue_3556_belief_mode_campaign",
        root="output/issue_3556_belief_mode_campaign/1",
        exists_on_host=True,
        missing_report_files=[],
        lineage="ScenarioBelief drop-vs-retain h600",
    )
    assert record["compatible"] is True
    assert record["blockers"] == []


# --- composed packet -----------------------------------------------------------


def test_packet_blocked_when_no_compatible_candidate() -> None:
    """No compatible candidate yields a fail-closed blocker packet, not evidence."""
    records = [
        evaluate_seed_sufficiency_candidate(
            name="issue3810-h600-extroster-run",
            root="output/issue3810-h600-extroster-run/13273",
            exists_on_host=False,
            missing_report_files=list(REQUIRED_SEED_SUFFICIENCY_REPORTS),
            lineage="issue #3810 h600 extended-roster run",
        )
    ]
    packet = build_h600_candidate_closure_packet(
        candidates=records,
        analyzer_command=["uv", "run", "python", "scripts/tools/analyze_seed_sufficiency.py"],
        resolved_candidate=None,
        queue_row_request={"kind": "scenario_belief_seed_sufficiency_campaign", "issue": 3556},
    )
    assert packet["schema_version"] == "issue_4328_h600_candidate_closure.v1"
    assert packet["evidence_status"] == "blocked"
    assert packet["decision_label"] == "blocked_no_compatible_candidate"
    assert packet["decision_label"] in SEED_SUFFICIENCY_CANDIDATE_LABELS
    assert packet["closure_target_issue"] == 3556
    assert packet["closure_attempt_issue"] == 4328
    assert packet["compatible_candidates"] == []
    assert packet["resolved_candidate"] is None
    assert packet["queue_row_request"]["issue"] == 3556
    # Fail-closed packets must never imply campaign/Slurm/claim actions ran.
    assert packet["forbidden_actions_confirmed"]["full_benchmark_campaign_run"] is False
    assert packet["forbidden_actions_confirmed"]["slurm_or_gpu_submission"] is False


def test_packet_promoted_when_candidate_resolves() -> None:
    """A resolved compatible candidate promotes the packet and carries the summary."""
    record = evaluate_seed_sufficiency_candidate(
        name="issue_3556_belief_mode_campaign",
        root="output/issue_3556_belief_mode_campaign/1",
        exists_on_host=True,
        missing_report_files=[],
        lineage="ScenarioBelief drop-vs-retain",
    )
    packet = build_h600_candidate_closure_packet(
        candidates=[record],
        analyzer_command=["uv", "run", "python", "x"],
        resolved_candidate=record,
        analyzer_output_dir="output/issue_4328_h600_seed_sufficiency",
        analyzer_summary={"headline_claim_status": "blocked", "campaign_count": 1},
    )
    assert packet["evidence_status"] == "promoted"
    assert packet["decision_label"] == "resolved_compatible_candidate"
    assert packet["resolved_candidate"]["name"] == "issue_3556_belief_mode_campaign"
    assert packet["queue_row_request"] is None


# --- script probe / run --------------------------------------------------------


def test_run_candidate_closure_fails_closed_for_named_roots(tmp_path: Path) -> None:
    """The default named h600 roots fail closed on a host without them; packet written."""
    evidence_dir = tmp_path / "closure"
    packet = evaluator.run_candidate_closure(
        candidates=[dict(candidate) for candidate in evaluator.DEFAULT_CANDIDATES],
        analyzer_output_dir=tmp_path / "analyzer_out",
        evidence_dir=evidence_dir,
    )
    assert packet["evidence_status"] == "blocked"
    assert packet["decision_label"] == "blocked_no_compatible_candidate"
    # Every named candidate is at least provenance-incompatible with #3556.
    for candidate in packet["candidates"]:
        assert "provenance_incompatible_with_3556" in candidate["blockers"]
    # Durable artifacts must be written even in the blocked case.
    assert (evidence_dir / "summary.json").is_file()
    assert (evidence_dir / "README.md").is_file()
    on_disk = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["schema_version"] == "issue_4328_h600_candidate_closure.v1"
    # The analyzer must not have run because no candidate qualified.
    assert not (tmp_path / "analyzer_out").exists()


def test_run_candidate_closure_resolves_and_runs_analyzer(tmp_path: Path) -> None:
    """A present ScenarioBelief-lineage candidate resolves and the analyzer runs."""
    campaign = tmp_path / "issue_3556_belief_mode_campaign"
    _write_seed_reports(campaign, include_episode_rows=True)
    analyzer_out = tmp_path / "analyzer_out"
    evidence_dir = tmp_path / "closure"

    packet = evaluator.run_candidate_closure(
        candidates=[
            {
                "name": "issue_3556_belief_mode_campaign",
                "root": str(campaign),
                "lineage": "ScenarioBelief drop-vs-retain h600",
            }
        ],
        analyzer_output_dir=analyzer_out,
        evidence_dir=evidence_dir,
    )
    assert packet["evidence_status"] == "promoted"
    assert packet["decision_label"] == "resolved_compatible_candidate"
    assert (analyzer_out / "seed_sufficiency_analysis.json").is_file()
    assert packet["analyzer_summary"] is not None


def test_main_blocked_exit_code(tmp_path: Path) -> None:
    """The CLI returns the blocked exit code and still writes the packet for named roots."""
    evidence_dir = tmp_path / "closure"
    code = evaluator.main(
        [
            "--analyzer-output-dir",
            str(tmp_path / "out"),
            "--evidence-dir",
            str(evidence_dir),
        ]
    )
    assert code == evaluator.BLOCKED_EXIT_CODE
    assert (evidence_dir / "summary.json").is_file()


def test_main_exit_zero_on_blocked_flag(tmp_path: Path) -> None:
    """The opt-in flag makes a blocked closure exit 0 for CI-safe packet generation."""
    code = evaluator.main(
        [
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
