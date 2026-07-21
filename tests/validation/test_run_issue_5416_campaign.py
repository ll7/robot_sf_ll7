"""Contract tests for the issue #5416 campaign runner (issue #6082)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.validation.run_issue_5416_campaign as campaign
from scripts.analysis.analyze_issue_5416_sipp_four_geometry import build_analysis
from scripts.validation.check_issue_5416_sipp_four_geometry_packet import load_packet
from scripts.validation.run_issue_5416_campaign import (
    CampaignError,
    CampaignRow,
    enumerate_rows,
    episode_is_checker_valid,
    plan_output_path,
    run_campaign,
)

CONFIG_PATH = Path("configs/benchmarks/issue_5416_sipp_four_geometry_preregistration.yaml")
PLANNERS = ("sipp_lattice", "hybrid_rule_v0_minimal", "teb", "nmpc_social", "dwa")
SCENARIOS = (
    "classic_head_on_corridor_low",
    "classic_doorway_low",
    "classic_station_platform_medium",
    "classic_merging_low",
)
SEEDS = (111, 112, 113, 114, 115)


def _eligible_episode(row: CampaignRow) -> dict[str, Any]:
    """Return a synthetic episode that the paired-outcome analyzer accepts."""
    return {
        "version": "v1",
        "horizon": 500,
        "planner_id": row.planner_id,
        "scenario_id": row.scenario_id,
        "seed": row.seed,
        "termination_reason": "success",
        "outcome": {
            "route_complete": True,
            "collision_event": False,
            "timeout_event": False,
        },
        "integrity": {"contradictions": []},
        "result_provenance": {
            "schema_version": "benchmark_row_provenance.v1",
            "scenario_id": row.scenario_id,
            "seed": row.seed,
            "config_hash": "synthetic-config-hash",
            "repo_commit": "synthetic-repo-commit",
            "simulator_settings": {"horizon": 500, "dt": 0.1},
            "postprocessing": [],
        },
        "metrics": {
            "deadlock": False,
            "ped_collision_count": 0.0,
            "obstacle_collision_count": 0.0,
            "time_to_goal_norm": 1.0,
            "path_efficiency": 0.9,
        },
        "algorithm_metadata": {
            "status": "ok",
            "fallback_or_degraded": False,
            "algorithm": row.planner_id,
            "config": {"planner_variant": row.planner_id},
            "planner_kinematics": {"execution_mode": "native"},
            "planner_diagnostics": {
                "expansion_limit_hits": 0,
                "runtime_bound_exits": 0,
                "fallback_count": 0,
                "commitment_invalidations": 0,
                "planner_step_runtime_seconds": [0.01, 0.02],
            },
        },
    }


def test_enumerate_rows_is_the_frozen_100_row_matrix() -> None:
    """The matrix enumerates exactly 5x4x5=100 rows in the frozen roster order."""
    rows = enumerate_rows(load_packet(CONFIG_PATH))
    assert len(rows) == 100
    # Planner blocks are contiguous and in roster order.
    planner_blocks = [rows[i * 20].planner_id for i in range(5)]
    assert tuple(planner_blocks) == PLANNERS
    # Indexing is 1-based and dense.
    assert [row.index for row in rows] == list(range(1, 101))
    # Each planner covers every scenario and every seed.
    first_planner = [row for row in rows if row.planner_id == "sipp_lattice"]
    assert [row.scenario_id for row in first_planner[::5]] == list(SCENARIOS)
    assert [row.seed for row in first_planner[:5]] == list(SEEDS)


def test_dry_run_lists_100_rows_without_executing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A dry run plans 100 rows and never touches the execution path."""

    def fail_if_called(**_: Any) -> Any:
        raise AssertionError("dry run must not execute any row")

    monkeypatch.setattr(campaign, "execute_row", fail_if_called)
    summary = run_campaign(
        config_path=CONFIG_PATH,
        output_root=tmp_path,
        dry_run=True,
    )
    assert summary["dry_run"] is True
    assert summary["planned_rows"] == 100
    assert summary["selected_rows"] == 100
    assert len(summary["rows"]) == 100
    assert summary["executed_rows"] == 0
    # Nothing is written during a dry run.
    assert not any(tmp_path.rglob("episodes.jsonl"))


def test_rows_slice_is_one_based_and_inclusive() -> None:
    """--rows N-M selects an inclusive 1-based slice and rejects out-of-range."""
    rows = enumerate_rows(load_packet(CONFIG_PATH))
    sliced = campaign.select_rows(rows, "3-4")
    assert [row.index for row in sliced] == [3, 4]
    single = campaign.select_rows(rows, "100")
    assert [row.index for row in single] == [100]
    with pytest.raises(CampaignError):
        campaign.select_rows(rows, "0")
    with pytest.raises(CampaignError):
        campaign.select_rows(rows, "100-101")
    with pytest.raises(CampaignError):
        campaign.select_rows(rows, "5-2")


def test_execute_resume_skip_and_checker_valid_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 2-row micro-matrix executes once, skips on resume, and stays eligible."""
    rows = enumerate_rows(load_packet(CONFIG_PATH))
    micro = rows[:2]
    executed_paths: list[Path] = []

    def fake_run_map_batch(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        """Write one synthetic eligible episode in place of the real native run."""
        episodes_path = Path(kwargs["out_path"] if "out_path" in kwargs else _args[1])
        row = next(row for row in micro if plan_output_path(tmp_path, row) == episodes_path)
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_path.write_text(json.dumps(_eligible_episode(row)) + "\n", encoding="utf-8")
        executed_paths.append(episodes_path)
        return {"total_jobs": 1, "written": 1}

    monkeypatch.setattr(campaign, "run_map_batch", fake_run_map_batch)

    summary = run_campaign(
        config_path=CONFIG_PATH,
        output_root=tmp_path,
        rows_spec="1-2",
    )
    assert summary["executed_rows"] == 2
    assert summary["skipped_rows"] == 0
    assert summary["failed_rows"] == 0
    assert len(executed_paths) == 2
    # Each executed row carries the per-row deadlock/stall metric.
    assert all(entry.get("deadlock") is False for entry in summary["rows"])

    # The freshly written episodes are checker-valid (analyzer-eligible).
    episode_paths = [plan_output_path(tmp_path, row) for row in micro]
    report = build_analysis(
        episode_paths=episode_paths,
        output_dir=tmp_path / "analyzer",
        packet_path=CONFIG_PATH,
    )
    assert report["matrix"]["eligible_rows"] == 2
    assert report["matrix"]["excluded_rows"] == 0

    # Resume-skip: re-running must not re-execute already-valid rows.
    summary_repeat = run_campaign(
        config_path=CONFIG_PATH,
        output_root=tmp_path,
        rows_spec="1-2",
    )
    assert summary_repeat["executed_rows"] == 0
    assert summary_repeat["skipped_rows"] == 2
    assert summary_repeat["failed_rows"] == 0
    # run_map_batch was not called again during resume.
    assert len(executed_paths) == 2


def test_resume_re_runs_a_row_that_is_not_checker_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A partial/invalid existing output is re-run rather than trusted."""
    rows = enumerate_rows(load_packet(CONFIG_PATH))
    row = rows[0]
    path = plan_output_path(tmp_path, row)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A corrupt/non-native episode must not count as a completed row.
    path.write_text(json.dumps({"version": "v1"}) + "\n", encoding="utf-8")

    valid, reasons = episode_is_checker_valid(
        json.loads(path.read_text(encoding="utf-8")), row, planners=PLANNERS
    )
    assert valid is False
    assert reasons

    def fake_run_map_batch(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        episodes_path = Path(kwargs.get("out_path", _args[1]))
        episodes_path.write_text(json.dumps(_eligible_episode(row)) + "\n", encoding="utf-8")
        return {"total_jobs": 1, "written": 1}

    monkeypatch.setattr(campaign, "run_map_batch", fake_run_map_batch)
    summary = run_campaign(
        config_path=CONFIG_PATH,
        output_root=tmp_path,
        rows_spec="1",
    )
    assert summary["executed_rows"] == 1
    assert summary["skipped_rows"] == 0


def test_output_layout_keys_each_row_uniquely(tmp_path: Path) -> None:
    """Each campaign cell maps to a distinct planner/scenario/seed JSONL path."""
    rows = enumerate_rows(load_packet(CONFIG_PATH))
    paths = {plan_output_path(tmp_path, row) for row in rows}
    assert len(paths) == 100
    sample = plan_output_path(
        tmp_path,
        CampaignRow(
            index=1,
            planner_id="sipp_lattice",
            scenario_id="classic_head_on_corridor_low",
            seed=111,
            native_config_path=Path("configs/algos/sipp_lattice_native_command.yaml"),
        ),
    )
    assert (
        sample
        == tmp_path
        / "sipp_lattice"
        / "classic_head_on_corridor_low"
        / "seed_111"
        / "episodes.jsonl"
    )
