"""Focused tests for the issue #4882 S30 interpretation packet."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType

SCRIPT = (
    Path(__file__).parents[2] / "scripts" / "analysis" / "build_issue_4882_s30_interpretation.py"
)


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("issue_4882_builder", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def builder() -> ModuleType:
    """Load the issue-specific builder as a test module."""

    return _module()


def _episode(
    planner: str,
    scenario: str,
    seed: int,
    commit: str,
    *,
    success: bool,
    collision: bool,
) -> dict[str, object]:
    return {
        "scenario_id": scenario,
        "seed": seed,
        "git_hash": commit,
        "horizon": 600,
        "metrics": {
            "success": success,
            "near_misses": 1 if collision else 0,
            "time_to_goal_norm": 0.4 if success else 1.0,
            "path_efficiency": 0.9 if success else 0.5,
            "snqi": 0.1 if success else -0.5,
        },
        "outcome": {"collision_event": collision},
        "pedestrian_model": {"fallback_degraded_status": "native"},
        "wall_time_sec": 1.0,
        "scenario_params": {
            "algo": planner,
            "robot_config": {"type": "differential_drive"},
        },
    }


def _write_root(
    root: Path,
    planners: list[str],
    commit: str,
    *,
    target_separates: bool,
) -> None:
    root.mkdir(parents=True)
    (root / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "scenario_matrix_hash": "fixture-matrix",
                "planners": [
                    {
                        "key": planner,
                        "planner_group": "experimental" if planner != "orca" else "core",
                    }
                    for planner in planners
                ],
            }
        ),
        encoding="utf-8",
    )
    for planner in planners:
        run = root / "runs" / f"{planner}__differential_drive"
        run.mkdir(parents=True)
        rows = []
        for scenario in ("scenario_a", "scenario_b"):
            for seed in (111, 112, 113):
                if planner == "orca":
                    success, collision = False, True
                elif planner == "hybrid_rule_v3_fast_progress_static_escape_continuous":
                    success, collision = (True, False) if target_separates else (False, True)
                else:
                    success, collision = True, False
                rows.append(
                    _episode(
                        planner,
                        scenario,
                        seed,
                        commit,
                        success=success,
                        collision=collision,
                    )
                )
        (run / "episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def _fixture_arms(
    builder: ModuleType, tmp_path: Path, *, target_separates: bool
) -> tuple[list[object], Path, Path]:
    five_root = tmp_path / "five"
    ppo_root = tmp_path / "ppo"
    _write_root(
        five_root,
        list(builder.FIVE_ARM_KEYS),
        "five-commit",
        target_separates=target_separates,
    )
    _write_root(ppo_root, ["ppo"], "ppo-commit", target_separates=target_separates)
    arms = [
        builder.load_arm(
            five_root,
            planner,
            source_job="13376",
            source_commit="five-commit",
        )
        for planner in builder.FIVE_ARM_KEYS
    ]
    arms.append(
        builder.load_arm(
            ppo_root,
            "ppo",
            source_job="13388",
            source_commit="ppo-commit",
        )
    )
    arms.sort(key=lambda arm: builder.ALL_ARM_KEYS.index(arm.planner_key))
    return arms, five_root, ppo_root


def test_fixture_campaign_produces_branch_a_separation(builder: ModuleType, tmp_path: Path) -> None:
    """A positive paired interval selects the separation branch."""

    arms, _five_root, _ppo_root = _fixture_arms(builder, tmp_path, target_separates=True)
    builder.validate_grid(arms, expected_scenarios=2, expected_seeds=[111, 112, 113])
    verdict = builder.classify_branch(arms, [111, 112, 113], samples=200, bootstrap_seed=123)
    assert verdict["verdict"] == "branch_a_separation"
    assert verdict["collision_field"] == "outcome.collision_event"


def test_fixture_campaign_produces_branch_b_boundary(builder: ModuleType, tmp_path: Path) -> None:
    """An interval without positive separation selects the boundary branch."""

    arms, _five_root, _ppo_root = _fixture_arms(builder, tmp_path, target_separates=False)
    verdict = builder.classify_branch(arms, [111, 112, 113], samples=200, bootstrap_seed=123)
    assert verdict["verdict"] == "branch_b_boundary"


def test_missing_row_blocks_grid(builder: ModuleType, tmp_path: Path) -> None:
    """Incomplete planner grids fail closed."""

    arms, _five_root, _ppo_root = _fixture_arms(builder, tmp_path, target_separates=True)
    arms[0].rows.pop()
    with pytest.raises(ValueError, match="selected rows"):
        builder.validate_grid(arms, expected_scenarios=2, expected_seeds=[111, 112, 113])


def test_failed_job_without_supplemental_arm_routes_to_failure_triage(builder: ModuleType) -> None:
    """A failed campaign without a replacement arm emits failure triage."""

    assert builder.determine_mode("FAILED", supplemental_ppo_available=False) == "failure_triage"
    assert builder.determine_mode("FAILED", supplemental_ppo_available=True) == "interpretation"


def test_rank_stability_handles_ties_deterministically(builder: ModuleType, tmp_path: Path) -> None:
    """Tie ranks use stable key ordering and average rank positions."""

    arms, _five_root, _ppo_root = _fixture_arms(builder, tmp_path, target_separates=False)
    report = builder.rank_stability(arms, [111, 112, 113])
    success = report["metrics"]["success"]
    assert success["s20_order"] == success["s30_order"]
    assert success["kendall_tau"] == 1.0
    tied_orca = next(row for row in success["planners"] if row["planner_key"] == "orca")
    tied_target = next(
        row
        for row in success["planners"]
        if row["planner_key"] == "hybrid_rule_v3_fast_progress_static_escape_continuous"
    )
    assert tied_orca["s30_rank"] == tied_target["s30_rank"]


def test_end_to_end_packet_contains_claim_boundary(builder: ModuleType, tmp_path: Path) -> None:
    """The generated Markdown and JSON carry the non-promotion boundary."""

    _arms, five_root, ppo_root = _fixture_arms(builder, tmp_path, target_separates=True)
    output = tmp_path / "packet"
    args = argparse.Namespace(
        five_arm_root=five_root,
        five_arm_commit="five-commit",
        five_arm_job_state="FAILED",
        ppo_root=ppo_root,
        ppo_commit="ppo-commit",
        job13376_analysis_json=None,
        job13378_analysis_json=None,
        output_dir=output,
        expected_scenarios=2,
        expected_seeds=3,
        expected_horizon=600,
        seed_start=111,
        bootstrap_samples=200,
        bootstrap_seed=123,
    )
    result = builder.build(args)
    assert result["verdict"] == "branch_a_separation"
    assert builder.CLAIM_BOUNDARY in (output / "README.md").read_text(encoding="utf-8")
    assert (
        builder.CLAIM_BOUNDARY
        == json.loads((output / "branch_verdict.json").read_text(encoding="utf-8"))[
            "claim_boundary"
        ]
    )
    checksum_names = {
        line.split("  ", 1)[1]
        for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert "README.md" in checksum_names
    assert "branch_verdict.json" in checksum_names
