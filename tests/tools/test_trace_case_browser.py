"""Synthetic-fixture tests for the edge-case trace browser CLI."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools.trace_case_browser import main

if TYPE_CHECKING:
    from pathlib import Path


def _trace(*, collision: bool = False) -> dict[str, object]:
    """Build a compact ``simulation-step-trace.v1`` fixture."""
    robot_x = [0.0, 1.0, 2.0, 3.0, 4.0] if collision else [0.0, 1.0, 2.0]
    steps: list[dict[str, object]] = []
    for index, x_position in enumerate(robot_x):
        steps.append(
            {
                "step": index,
                "time_s": (index + 1) * 0.1,
                "robot": {
                    "position": [x_position, 0.0],
                    "velocity": [0.0 if index == len(robot_x) - 1 else 1.0, 0.0],
                },
                "pedestrians": [
                    {
                        "id": 0,
                        "position": [4.5, 0.0],
                        "velocity": [0.0, 0.0],
                    }
                ],
                "rl": {
                    "terminated": collision and index == len(robot_x) - 1,
                    "truncated": False,
                },
            }
        )
    return {
        "schema_version": "simulation-step-trace.v1",
        "dt": 0.1,
        "steps": steps,
    }


def _episode(
    arm: str,
    seed: int,
    outcome: str,
    *,
    near_misses: int,
    min_clearance: float,
    include_trace: bool = True,
) -> dict[str, object]:
    """Build one representative benchmark episode row."""
    metadata: dict[str, object] = {"algorithm": arm}
    if include_trace:
        metadata["simulation_step_trace"] = _trace(collision=outcome == "collision")
    return {
        "algo": arm,
        "algorithm_metadata": metadata,
        "episode_id": f"doorway--{arm}--{seed}",
        "scenario_id": "classic_doorway_medium",
        "seed": seed,
        "status": outcome,
        "termination_reason": outcome,
        "steps": 5 if outcome == "collision" else 3,
        "outcome": {
            "route_complete": outcome == "success",
            "collision_event": outcome == "collision",
            "timeout_event": outcome == "timeout",
        },
        "metrics": {
            "success": outcome == "success",
            "collisions": int(outcome == "collision"),
            "near_misses": near_misses,
            "min_clearance": min_clearance,
        },
    }


@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    """Write two planner-arm files under the supported ``runs/`` layout."""
    runs = tmp_path / "runs"
    alpha = runs / "alpha__differential_drive"
    beta = runs / "beta__differential_drive"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    alpha_rows = [
        _episode("alpha", 113, "success", near_misses=2, min_clearance=0.25),
        _episode("alpha", 114, "collision", near_misses=9, min_clearance=-0.05),
        _episode(
            "alpha",
            115,
            "timeout",
            near_misses=1,
            min_clearance=0.4,
            include_trace=False,
        ),
    ]
    beta_rows = [
        _episode("beta", 113, "collision", near_misses=6, min_clearance=0.0),
    ]
    alpha.joinpath("episodes.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in alpha_rows),
        encoding="utf-8",
    )
    beta.joinpath("episodes.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in beta_rows),
        encoding="utf-8",
    )
    return runs


def test_list_filters_seed_ranges_sorts_and_emits_json(
    runs_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """List should combine files, apply filters, and keep JSON machine-readable."""
    result = main(
        [
            "list",
            str(runs_dir),
            "--filter",
            "outcome=collision",
            "--filter",
            "near>=5",
            "--filter",
            "seed=113-114",
            "--sort=-near",
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["count"] == 2
    assert [row["seed"] for row in payload["episodes"]] == [114, 113]
    assert payload["episodes"][0]["has_trace"] is True


def test_list_emits_a_text_table(
    runs_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Human output should expose the requested columns."""
    result = main(["list", str(runs_dir), "--filter", "outcome=success"])

    assert result == 0
    output = capsys.readouterr().out
    assert "ARM" in output
    assert "MIN PED CLEARANCE" in output
    assert "alpha" in output
    assert "113" in output


def test_summary_reports_arm_scenario_cells(
    runs_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Summary should aggregate outcome and safety fields by arm and scenario."""
    result = main(["summary", str(runs_dir), "--json"])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    alpha = next(cell for cell in payload["cells"] if cell["arm"] == "alpha")
    assert alpha == {
        "arm": "alpha",
        "scenario": "classic_doorway_medium",
        "n": 3,
        "success": 1,
        "collision": 1,
        "timeout": 1,
        "near_miss_mean": 4.0,
        "near_miss_max": 9,
        "min_clearance_min": -0.05,
    }


def test_pairs_finds_seed_flip_and_planner_upset_with_commands(
    runs_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pair mining should expose both requested match types and runnable hints."""
    result = main(["pairs", str(runs_dir), "--json"])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    seed_flip = payload["seed_flips"][0]
    assert {seed_flip["episode_a"]["seed"], seed_flip["episode_b"]["seed"]} == {113, 114}
    planner_upset = payload["planner_upsets"][0]
    assert planner_upset["episode_a"]["seed"] == planner_upset["episode_b"]["seed"] == 113
    assert {planner_upset["episode_a"]["arm"], planner_upset["episode_b"]["arm"]} == {
        "alpha",
        "beta",
    }
    for pair in (seed_flip, planner_upset):
        commands = "\n".join(pair["command_hint"]["commands"])
        assert "scripts/repro/butterfly_reexport_to_trace_series.py" in commands
        assert "scripts/repro/butterfly_hinge_figure_proto.py" in commands


def test_critical_reports_terminal_collision_and_trace_windows(
    runs_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Critical should rank the selected collision's final-step window without errors."""
    result = main(
        [
            "critical",
            str(runs_dir),
            "--arm",
            "alpha",
            "--seed",
            "114",
            "--top-k",
            "4",
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["episode"]["outcome"] == "collision"
    assert payload["trace_step_count"] == 5
    assert payload["windows"][0]["anchor"] == "collision_termination"
    assert payload["windows"][0]["anchor_step"] == 4
    assert payload["windows"][0]["end_step_exclusive"] == 5
    assert any(window["anchor"] == "closest_approach" for window in payload["windows"])
