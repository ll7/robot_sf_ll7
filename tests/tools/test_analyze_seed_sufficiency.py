"""Tests for seed-sufficiency and ranking-stability analysis."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools.analyze_seed_sufficiency import (
    analyze_seed_sufficiency,
    main,
    resolve_campaign_roots,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    """Write an indented JSON fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_seed_rows(path: Path, rows: list[dict]) -> None:
    """Write seed episode rows CSV fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_id",
                "scenario_id",
                "scenario_family",
                "planner_key",
                "seed",
                "success",
                "collision",
                "snqi",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _campaign(
    root: Path,
    *,
    seed_count: int,
    goal_snqi: float,
    orca_snqi: float,
    ci_half_width: float,
    goal_successes: int,
    orca_successes: int,
    orca_status: str = "valid",
) -> Path:
    """Create a compact campaign fixture with two planners and two scenario families."""

    reports = root / "reports"
    _write_json(
        reports / "seed_variability_by_scenario.json",
        {
            "schema_version": "benchmark-seed-variability-by-scenario.v1",
            "metrics": ["success", "collisions", "snqi"],
            "rows": [
                {
                    "scenario_id": "family_a_s1",
                    "scenario_family": "family_a",
                    "planner_key": "goal",
                    "kinematics": "differential_drive",
                    "seed_count": seed_count,
                    "summary": {
                        "success": {"mean": goal_successes / seed_count, "ci_half_width": 0.1},
                        "collisions": {"mean": 0.0, "ci_half_width": 0.0},
                        "snqi": {"mean": goal_snqi, "ci_half_width": ci_half_width},
                    },
                },
                {
                    "scenario_id": "family_a_s1",
                    "scenario_family": "family_a",
                    "planner_key": "orca",
                    "kinematics": "differential_drive",
                    "row_status": orca_status,
                    "seed_count": seed_count,
                    "summary": {
                        "success": {"mean": orca_successes / seed_count, "ci_half_width": 0.1},
                        "collisions": {"mean": 0.0, "ci_half_width": 0.0},
                        "snqi": {"mean": orca_snqi, "ci_half_width": ci_half_width},
                    },
                },
                {
                    "scenario_id": "family_b_s1",
                    "scenario_family": "family_b",
                    "planner_key": "goal",
                    "kinematics": "differential_drive",
                    "seed_count": seed_count,
                    "summary": {
                        "success": {"mean": 1.0, "ci_half_width": 0.1},
                        "collisions": {"mean": 0.0, "ci_half_width": 0.0},
                        "snqi": {"mean": goal_snqi + 0.1, "ci_half_width": ci_half_width},
                    },
                },
                {
                    "scenario_id": "family_b_s1",
                    "scenario_family": "family_b",
                    "planner_key": "orca",
                    "kinematics": "differential_drive",
                    "row_status": orca_status,
                    "seed_count": seed_count,
                    "summary": {
                        "success": {"mean": 0.0, "ci_half_width": 0.1},
                        "collisions": {"mean": 1.0, "ci_half_width": 0.0},
                        "snqi": {"mean": orca_snqi - 0.1, "ci_half_width": ci_half_width},
                    },
                },
            ],
        },
    )
    seed_rows = []
    for planner, successes in [("goal", goal_successes), ("orca", orca_successes)]:
        for seed_index in range(seed_count):
            seed_rows.append(
                {
                    "episode_id": f"{planner}-{seed_index}",
                    "scenario_id": "family_a_s1",
                    "planner_key": planner,
                    "seed": 111 + seed_index,
                    "success": "1" if seed_index < successes else "0",
                    "collision": "0" if seed_index < successes else "1",
                    "snqi": goal_snqi if planner == "goal" else orca_snqi,
                }
            )
    _write_seed_rows(reports / "seed_episode_rows.csv", seed_rows)
    _write_json(reports / "statistical_sufficiency.json", {"bootstrap": {"seed": 123}})
    return root


def test_analyze_seed_sufficiency_reports_stable_case(tmp_path: Path) -> None:
    """Stable rankings should be reported without hiding raw counts."""

    small = _campaign(
        tmp_path / "s3",
        seed_count=3,
        goal_snqi=0.7,
        orca_snqi=0.4,
        ci_half_width=0.2,
        goal_successes=2,
        orca_successes=1,
    )
    extended = _campaign(
        tmp_path / "s5",
        seed_count=5,
        goal_snqi=0.72,
        orca_snqi=0.41,
        ci_half_width=0.1,
        goal_successes=4,
        orca_successes=2,
    )

    payload = analyze_seed_sufficiency([small, extended], tmp_path / "out")

    assert payload["schema_version"] == "seed_sufficiency_analysis.v1"
    assert payload["summary"]["ranking_instability_rows"] == 0
    assert payload["summary"]["advisory_campaigns"] == []
    assert payload["headline_rank_stability_contract"]["label"] == "blocked_pending_s20_s30"
    assert any(
        row["scenario_family"] == "family_a"
        and row["scenario_id"] == "family_a_s1"
        and row["success_count"] == 4
        for row in payload["outcome_counts"]
    )
    assert (tmp_path / "out" / "seed_sufficiency_analysis.json").exists()
    assert (tmp_path / "out" / "headline_rank_stability_contract.json").exists()
    assert (tmp_path / "out" / "headline_rank_stability_pairwise.csv").exists()
    assert (tmp_path / "out" / "planner_rank_stability.csv").exists()
    assert (tmp_path / "out" / "seed_sufficiency_summary.md").exists()
    assert (tmp_path / "out" / "fig_seed_interval_width.png").exists()


def test_analyze_seed_sufficiency_flags_unstable_and_underpowered(tmp_path: Path) -> None:
    """Single-seed and rank-flip surfaces should be caveated."""

    single = _campaign(
        tmp_path / "s1",
        seed_count=1,
        goal_snqi=0.8,
        orca_snqi=0.2,
        ci_half_width=0.4,
        goal_successes=1,
        orca_successes=0,
    )
    extended = _campaign(
        tmp_path / "s5",
        seed_count=5,
        goal_snqi=0.2,
        orca_snqi=0.8,
        ci_half_width=0.1,
        goal_successes=2,
        orca_successes=4,
    )

    payload = analyze_seed_sufficiency([single, extended], tmp_path / "out")

    assert payload["summary"]["ranking_instability_rows"] > 0
    assert payload["summary"]["scenario_family_winner_changes"] > 0
    assert payload["summary"]["advisory_campaigns"] == ["s1"]
    assert "single-seed" in (tmp_path / "out" / "seed_sufficiency_summary.md").read_text(
        encoding="utf-8"
    )


def test_headline_contract_blocks_non_promotable_rows(tmp_path: Path) -> None:
    """Fallback/degraded/not-available rows should be excluded from headline promotion."""

    durable = tmp_path / "durable_s20"
    durable.mkdir()
    (durable / "manifest.json").write_text("{}", encoding="utf-8")
    campaign = _campaign(
        tmp_path / "s20",
        seed_count=20,
        goal_snqi=0.7,
        orca_snqi=0.8,
        ci_half_width=0.03,
        goal_successes=18,
        orca_successes=19,
        orca_status="fallback",
    )

    payload = analyze_seed_sufficiency(
        [campaign],
        tmp_path / "out",
        headline_required_durable_roots=(durable,),
    )

    contract = payload["headline_rank_stability_contract"]
    assert contract["label"] == "row_status_exclusions_present"
    assert contract["promotion_allowed"] is False
    assert contract["row_status_exclusions"][0]["row_status"] == "fallback"

    partial_failure_campaign = _campaign(
        tmp_path / "partial_failure_s20",
        seed_count=20,
        goal_snqi=0.7,
        orca_snqi=0.8,
        ci_half_width=0.03,
        goal_successes=18,
        orca_successes=19,
        orca_status="partial_failure",
    )
    partial_failure_payload = analyze_seed_sufficiency(
        [partial_failure_campaign],
        tmp_path / "partial_failure_out",
        headline_required_durable_roots=(durable,),
    )
    assert (
        partial_failure_payload["headline_rank_stability_contract"]["label"]
        == "row_status_exclusions_present"
    )


def test_headline_contract_s20_stable_and_rank_flip_labels(tmp_path: Path) -> None:
    """Synthetic S20-style roots should classify pairwise rank stability deterministically."""

    durable = tmp_path / "durable_s20"
    durable.mkdir()
    (durable / "manifest.json").write_text("{}", encoding="utf-8")
    s20 = _campaign(
        tmp_path / "s20",
        seed_count=20,
        goal_snqi=0.7,
        orca_snqi=0.4,
        ci_half_width=0.03,
        goal_successes=18,
        orca_successes=10,
    )
    s30 = _campaign(
        tmp_path / "s30",
        seed_count=30,
        goal_snqi=0.72,
        orca_snqi=0.41,
        ci_half_width=0.02,
        goal_successes=27,
        orca_successes=14,
    )
    flip_s30 = _campaign(
        tmp_path / "s30_flip",
        seed_count=30,
        goal_snqi=0.2,
        orca_snqi=0.8,
        ci_half_width=0.02,
        goal_successes=12,
        orca_successes=27,
    )

    stable_payload = analyze_seed_sufficiency(
        [s20, s30],
        tmp_path / "stable_out",
        headline_required_durable_roots=(durable,),
    )
    flip_payload = analyze_seed_sufficiency(
        [s20, flip_s30],
        tmp_path / "flip_out",
        headline_required_durable_roots=(durable,),
    )

    stable_contract = stable_payload["headline_rank_stability_contract"]
    flip_contract = flip_payload["headline_rank_stability_contract"]
    assert stable_contract["label"] == "stable"
    assert stable_contract["claim_status"] == "paper_grade"
    assert stable_contract["promotion_allowed"] is True
    assert stable_contract["pairwise"][0]["rank_label"] == "stable"
    assert flip_contract["label"] == "rank_flip_detected"
    assert flip_contract["claim_status"] == "not_statistically_distinguishable_budget"
    assert flip_contract["promotion_allowed"] is False
    assert flip_contract["pairwise"][0]["rank_label"] == "rank_flip"
    assert flip_contract["pairwise"][0]["kendall_tau"] == -1.0


def test_headline_contract_blocks_missing_durable_roots(tmp_path: Path) -> None:
    """Absent required S20/S30 roots should fail closed even with enough synthetic seeds."""

    campaign = _campaign(
        tmp_path / "s20",
        seed_count=20,
        goal_snqi=0.7,
        orca_snqi=0.4,
        ci_half_width=0.03,
        goal_successes=18,
        orca_successes=10,
    )

    payload = analyze_seed_sufficiency(
        [campaign],
        tmp_path / "out",
        headline_required_durable_roots=(tmp_path / "missing_s20",),
    )

    contract = payload["headline_rank_stability_contract"]
    assert contract["label"] == "blocked_pending_s20_s30"
    assert contract["claim_status"] == "blocked_missing_increased_seed_rows"
    assert contract["missing_durable_roots"] == [str(tmp_path / "missing_s20")]


def test_resolve_campaign_roots_discovers_slurm_output_container(tmp_path: Path) -> None:
    """CLI resolver accepts a Slurm output container and campaign-id filter."""

    container = tmp_path / "2026-06-issue1554-s20-h500-l40s-mem180"
    ignored = _campaign(
        container / "old_s10_baseline",
        seed_count=10,
        goal_snqi=0.7,
        orca_snqi=0.4,
        ci_half_width=0.04,
        goal_successes=8,
        orca_successes=5,
    )
    target = _campaign(
        container / "issue1554_s20_h500",
        seed_count=20,
        goal_snqi=0.75,
        orca_snqi=0.45,
        ci_half_width=0.03,
        goal_successes=18,
        orca_successes=10,
    )

    roots = resolve_campaign_roots(
        campaign_output_roots=[container],
        campaign_ids=["issue1554_s20_h500"],
    )

    assert roots == [target]
    assert ignored not in roots


def test_main_accepts_campaign_output_root(tmp_path: Path) -> None:
    """CLI feeds discovered S20/H500 roots through existing analysis path."""

    container = tmp_path / "slurm-output"
    campaign = _campaign(
        container / "issue1554_s20_h500",
        seed_count=20,
        goal_snqi=0.75,
        orca_snqi=0.45,
        ci_half_width=0.03,
        goal_successes=18,
        orca_successes=10,
    )
    exit_code = main(
        [
            "--campaign-output-root",
            str(container),
            "--campaign-id",
            "issue1554_s20_h500",
            "--headline-required-durable-root",
            str(campaign),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    payload = json.loads((tmp_path / "out" / "seed_sufficiency_analysis.json").read_text())
    assert payload["campaigns"][0]["root"] == str(campaign)


def test_resolve_campaign_roots_fails_closed_on_empty_output_root(tmp_path: Path) -> None:
    """Empty Slurm containers fail closed instead of yielding zero-campaign evidence."""

    container = tmp_path / "empty-slurm-output"
    container.mkdir()

    with pytest.raises(FileNotFoundError, match="No seed-sufficiency campaign reports"):
        resolve_campaign_roots(campaign_output_roots=[container])


def test_main_writes_requested_outputs(tmp_path: Path) -> None:
    """CLI should write JSON, CSV, Markdown, and figure outputs."""

    campaign = _campaign(
        tmp_path / "s3",
        seed_count=3,
        goal_snqi=0.7,
        orca_snqi=0.4,
        ci_half_width=0.2,
        goal_successes=2,
        orca_successes=1,
    )

    exit_code = main(["--campaign-root", str(campaign), "--output-dir", str(tmp_path / "out")])

    assert exit_code == 0
    assert (tmp_path / "out" / "seed_sufficiency_analysis.json").exists()
