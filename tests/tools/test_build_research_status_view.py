"""Tests for the research-status coverage view generator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from scripts.tools.build_research_status_view import (
    build_research_status_view,
    main,
    render_markdown,
)
from scripts.tools.campaign_result_store import read_parquet_frame, write_result_store

if TYPE_CHECKING:
    from pathlib import Path


def _write_suite_config(path: Path) -> None:
    """Write a compact suite fixture with two scenario families and two pilot seeds."""
    payload = {
        "name": "fixture_research_suite",
        "seed_policy": {"pilot_set": {"name": "S2", "seeds": [1, 2]}},
        "scenario_families": [
            {
                "family_id": "static_obstacle_detour",
                "scenario_ids": ["single_obstacle_circle", "line_wall_detour"],
            },
            {
                "family_id": "paired_pedestrian_interactions",
                "scenario_ids": ["single_ped_crossing_orthogonal"],
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_planner_matrix(path: Path) -> None:
    """Write a compact known-planner fixture."""
    payload = {
        "schema_version": "planner-readiness-matrix.v1",
        "rows": [
            {"planner_id": "orca"},
            {"planner_id": "social_force"},
            {"planner_id": "prediction_planner"},
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _episode(
    *,
    planner: str,
    scenario_family: str,
    scenario_id: str,
    seed: int,
    row_status: str = "native",
    run_id: str = "research-2026-06-18-run",
) -> dict[str, Any]:
    """Return one canonical result-store episode row."""
    return {
        "run_id": run_id,
        "episode_id": f"{run_id}-{planner}-{scenario_id}-{seed}",
        "planner": planner,
        "scenario_id": scenario_id,
        "scenario_family": scenario_family,
        "seed": seed,
        "row_status": row_status,
        "artifact_uri": f"wandb://robot-sf/{run_id}/{planner}/{scenario_id}/{seed}.jsonl",
        "artifact_sha256": "a" * 64,
    }


def test_view_aggregates_complete_valid_coverage_with_seed_budget(tmp_path: Path) -> None:
    """Native/adapter rows across all suite scenario/seed pairs should count as coverage."""
    suite_config = tmp_path / "suite.yaml"
    planner_matrix = tmp_path / "planner_matrix.yaml"
    result_store = tmp_path / "result-store"
    _write_suite_config(suite_config)
    _write_planner_matrix(planner_matrix)
    rows = [
        _episode(
            planner="orca",
            scenario_family="static_obstacle_detour",
            scenario_id=scenario_id,
            seed=seed,
            row_status="adapter",
        )
        for scenario_id in ["single_obstacle_circle", "line_wall_detour"]
        for seed in [1, 2]
    ]
    write_result_store(result_store, rows, study_id="fixture", command="fixture command")

    payload = build_research_status_view(
        result_store,
        suite_config=suite_config,
        planner_matrix=planner_matrix,
    )

    cell = payload["coverage_matrix"]["orca"]["static_obstacle_detour"]
    assert cell["valid_coverage"] is True
    assert cell["coverage_status"] == "covered"
    assert cell["evidence_grade"] == "nominal_benchmark_candidate"
    assert cell["seed_budget"] == {
        "expected_seed_count": 2,
        "valid_seed_count": 2,
        "expected_seeds": [1, 2],
        "valid_seeds": [1, 2],
        "missing_seeds": [],
        "extra_seeds": [],
    }
    assert cell["valid_episode_count"] == 4
    assert cell["expected_episode_count"] == 4
    assert cell["latest_run_date"] == "2026-06-18"
    assert cell["row_status_breakdown"] == {"adapter": 4}
    assert not [
        gap
        for gap in payload["gaps"]
        if gap["planner"] == "orca" and gap["scenario_family"] == "static_obstacle_detour"
    ]


def test_view_marks_fallback_degraded_and_invalid_denominators_fail_closed(
    tmp_path: Path,
) -> None:
    """Fallback/degraded rows and incomplete denominators should not become valid coverage."""
    suite_config = tmp_path / "suite.yaml"
    planner_matrix = tmp_path / "planner_matrix.yaml"
    result_store = tmp_path / "result-store"
    _write_suite_config(suite_config)
    _write_planner_matrix(planner_matrix)
    rows = [
        _episode(
            planner="orca",
            scenario_family="static_obstacle_detour",
            scenario_id="single_obstacle_circle",
            seed=1,
            row_status="native",
        ),
        _episode(
            planner="orca",
            scenario_family="static_obstacle_detour",
            scenario_id="line_wall_detour",
            seed=1,
            row_status="fallback",
        ),
        _episode(
            planner="orca",
            scenario_family="paired_pedestrian_interactions",
            scenario_id="single_ped_crossing_orthogonal",
            seed=1,
            row_status="degraded",
        ),
    ]
    write_result_store(result_store, rows, study_id="fixture", command="fixture command")

    payload = build_research_status_view(
        result_store,
        suite_config=suite_config,
        planner_matrix=planner_matrix,
    )

    fallback_cell = payload["coverage_matrix"]["orca"]["static_obstacle_detour"]
    assert fallback_cell["valid_coverage"] is False
    assert fallback_cell["coverage_status"] == "fail_closed_fallback_or_degraded"
    assert fallback_cell["evidence_grade"] == "fail_closed"
    assert fallback_cell["seed_budget"]["missing_seeds"] == [2]
    assert fallback_cell["row_status_breakdown"] == {"fallback": 1, "native": 1}
    assert fallback_cell["fail_closed_reasons"] == [
        "fallback_or_degraded_rows_present",
        "denominator_invalid",
    ]

    degraded_cell = payload["coverage_matrix"]["orca"]["paired_pedestrian_interactions"]
    assert degraded_cell["coverage_status"] == "fail_closed_fallback_or_degraded"
    assert degraded_cell["row_status_breakdown"] == {"degraded": 1}


def test_view_enumerates_missing_planner_family_gaps(tmp_path: Path) -> None:
    """Known planners without suite-family rows should be visible as missing gaps."""
    suite_config = tmp_path / "suite.yaml"
    planner_matrix = tmp_path / "planner_matrix.yaml"
    result_store = tmp_path / "result-store"
    _write_suite_config(suite_config)
    _write_planner_matrix(planner_matrix)
    rows = [
        _episode(
            planner="orca",
            scenario_family="static_obstacle_detour",
            scenario_id=scenario_id,
            seed=seed,
        )
        for scenario_id in ["single_obstacle_circle", "line_wall_detour"]
        for seed in [1, 2]
    ]
    write_result_store(result_store, rows, study_id="fixture", command="fixture command")

    payload = build_research_status_view(
        result_store,
        suite_config=suite_config,
        planner_matrix=planner_matrix,
    )
    markdown = render_markdown(payload)

    missing_gap = {
        "planner": "prediction_planner",
        "scenario_family": "paired_pedestrian_interactions",
        "gap_type": "missing_cell",
        "reason": "no result-store rows for known planner and suite scenario family",
    }
    assert missing_gap in payload["gaps"]
    assert (
        payload["coverage_matrix"]["prediction_planner"]["paired_pedestrian_interactions"][
            "coverage_status"
        ]
        == "missing"
    )
    assert "| prediction_planner | paired_pedestrian_interactions | missing | missing |" in markdown


def test_cli_accepts_documented_markdown_alias(tmp_path: Path) -> None:
    """The issue-documented --markdown flag should emit the Markdown view."""
    suite_config = tmp_path / "suite.yaml"
    planner_matrix = tmp_path / "planner_matrix.yaml"
    result_store = tmp_path / "result-store"
    output_json = tmp_path / "view.json"
    output_md = tmp_path / "view.md"
    _write_suite_config(suite_config)
    _write_planner_matrix(planner_matrix)
    write_result_store(
        result_store,
        [
            _episode(
                planner="orca",
                scenario_family="static_obstacle_detour",
                scenario_id=scenario_id,
                seed=seed,
            )
            for scenario_id in ["single_obstacle_circle", "line_wall_detour"]
            for seed in [1, 2]
        ],
        study_id="fixture",
        command="fixture command",
    )

    exit_code = main(
        [
            "--result-store",
            str(result_store),
            "--suite-config",
            str(suite_config),
            "--planner-matrix",
            str(planner_matrix),
            "--json-output",
            str(output_json),
            "--markdown",
            str(output_md),
        ]
    )

    assert exit_code == 0
    assert output_json.exists()
    assert "# Research Status Coverage View" in output_md.read_text(encoding="utf-8")


def test_view_ignores_null_planner_ids_and_bad_seed_pairs(tmp_path: Path) -> None:
    """Null metadata should not become literal coverage identifiers."""
    suite_config = tmp_path / "suite.yaml"
    planner_matrix = tmp_path / "planner_matrix.yaml"
    result_store = tmp_path / "result-store"
    _write_suite_config(suite_config)
    planner_matrix.write_text(
        yaml.safe_dump(
            {
                "schema_version": "planner-readiness-matrix.v1",
                "rows": [{"planner_id": None}, {"planner_id": "orca"}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    rows = [
        _episode(
            planner="orca",
            scenario_family="static_obstacle_detour",
            scenario_id="single_obstacle_circle",
            seed=1,
        ),
        {
            **_episode(
                planner="orca",
                scenario_family="static_obstacle_detour",
                scenario_id="line_wall_detour",
                seed=2,
            ),
        },
    ]
    write_result_store(result_store, rows, study_id="fixture", command="fixture command")
    episode_frame = read_parquet_frame(result_store / "episodes.parquet")
    episode_frame.loc[episode_frame["scenario_id"] == "line_wall_detour", "seed"] = float("nan")
    episode_frame.to_parquet(result_store / "episodes.parquet", index=False)

    payload = build_research_status_view(
        result_store,
        suite_config=suite_config,
        planner_matrix=planner_matrix,
    )

    assert "None" not in payload["planners"]
    cell = payload["coverage_matrix"]["orca"]["static_obstacle_detour"]
    assert cell["valid_pair_count"] == 1
    assert cell["coverage_status"] == "fail_closed_denominator_invalid"
