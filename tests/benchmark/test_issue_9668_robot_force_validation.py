"""Release force validation must retain arm identity and predecessor custody."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.analysis.issue_9668_robot_force_validation import (
    BASELINE_ARCHIVE_SHA256,
    BASELINE_SOURCE_SHA,
    build_report,
)

if TYPE_CHECKING:
    from pathlib import Path

SOURCE = "a" * 40


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _gate(path: Path, *, status: str = "pass") -> None:
    _write_json(
        path,
        {
            "status": status,
            "baseline_archive_sha256": BASELINE_ARCHIVE_SHA256,
            "baseline_source_sha": BASELINE_SOURCE_SHA,
            "candidate_source_sha": SOURCE,
            "expected_rows": 2,
            "candidate_rows": 2,
            "baseline_rows": 2,
            "mismatch_episodes": 0,
            "scientific_manifest_differences": [],
            "robot_force_metrics": {"status": "pass", "checked_rows": 2},
        },
    )


def _episode(impulse: float, distance: float) -> dict:
    return {
        "scenario_id": "doorway",
        "seed": 111,
        "algo": "hybrid_rule_local_planner",
        "steps": 10,
        "git_hash": SOURCE,
        "result_provenance": {"repo_commit": SOURCE},
        "scenario_params": {"metadata": {"archetype": "bottleneck"}},
        "metrics": {
            "robot_force_impulse_total": impulse,
            "robot_force_peak": impulse,
            "robot_force_time_above_ref_s": 0.0,
            "robot_force_exposed_ped_count": 1,
            "robot_force_pp_equiv_impulse_total": impulse,
            "robot_force_pp_equiv_peak": impulse,
            "robot_force_pp_equiv_time_above_ref_s": 0.0,
            "min_distance": distance,
            "near_misses": 1,
            "human_discomfort_exposure_m_s": 0.1,
        },
    }


def _campaign(tmp_path: Path) -> tuple[Path, Path]:
    campaign = tmp_path / "campaign"
    for arm, row in (("hybrid_a", _episode(1.0, 0.2)), ("hybrid_b", _episode(2.0, 0.1))):
        _write_json(campaign / "runs" / f"{arm}__differential_drive" / "episodes.jsonl", row)
    gate = tmp_path / "equivalence.json"
    _gate(gate)
    return campaign, gate


def test_release_force_report_keeps_shared_algorithm_arms_separate(tmp_path: Path) -> None:
    """Two planner arms with one algorithm name remain distinct release cells."""
    campaign, gate = _campaign(tmp_path)
    report = build_report(
        campaign,
        expected_source=SOURCE,
        expected_rows=2,
        equivalence_report=gate,
    )

    assert report["classification"] == "release_robot_force_validation"
    assert report["episodes"] == 2
    assert len(report["sources"]) == 2
    cohorts = {row["cohort"] for row in report["correlations"]}
    assert {"planner:hybrid_a", "planner:hybrid_b"} <= cohorts
    assert "family:bottleneck" in cohorts
    assert {row["arm"] for row in report["largest_rank_disagreements"]} == {
        "hybrid_a",
        "hybrid_b",
    }


def test_release_force_report_rejects_failed_equivalence_gate(tmp_path: Path) -> None:
    """Correlation reporting never upgrades a predecessor mismatch into release evidence."""
    campaign, gate = _campaign(tmp_path)
    _gate(gate, status="mismatch")
    with pytest.raises(ValueError, match="complete passing equivalence gate"):
        build_report(
            campaign,
            expected_source=SOURCE,
            expected_rows=2,
            equivalence_report=gate,
        )


def test_release_force_report_requires_frozen_007_archive(tmp_path: Path) -> None:
    """A passing comparison against some other archive is insufficient custody."""
    campaign, gate = _campaign(tmp_path)
    payload = json.loads(gate.read_text(encoding="utf-8"))
    payload["baseline_archive_sha256"] = "b" * 64
    _write_json(gate, payload)
    with pytest.raises(ValueError, match="complete passing equivalence gate"):
        build_report(
            campaign,
            expected_source=SOURCE,
            expected_rows=2,
            equivalence_report=gate,
        )


def test_release_force_report_rejects_source_mismatch(tmp_path: Path) -> None:
    """A changed row source cannot borrow a passing gate from another commit."""
    campaign, gate = _campaign(tmp_path)
    path = campaign / "runs" / "hybrid_b__differential_drive" / "episodes.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["git_hash"] = "b" * 40
    row["result_provenance"]["repo_commit"] = "b" * 40
    _write_json(path, row)
    with pytest.raises(ValueError, match="source commit differs"):
        build_report(
            campaign,
            expected_source=SOURCE,
            expected_rows=2,
            equivalence_report=gate,
        )
