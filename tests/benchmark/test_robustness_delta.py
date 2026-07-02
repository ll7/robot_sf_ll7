"""Tests for observation robustness delta reports."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

from robot_sf.benchmark.robustness_delta import (
    build_robustness_delta_report,
    format_report_markdown,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "build_observation_robustness_delta_issue_3952.py"
)


def _row(
    *,
    planner_key: str = "goal",
    algo: str = "goal",
    scenario_id: str = "planner_sanity_simple",
    seed: int = 0,
    route_complete: bool = True,
    collision: bool = False,
    noise: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "planner_key": planner_key,
        "algo": algo,
        "scenario_id": scenario_id,
        "seed": seed,
        "kinematics": "differential_drive",
        "outcome": {
            "route_complete": route_complete,
            "collision": collision,
        },
        "metrics": {
            "success": route_complete,
            "collisions": int(collision),
        },
    }
    if noise:
        row.update(
            {
                "observation_noise": {
                    "enabled": True,
                    "profile": "issue_3952_robustness_smoke_v1",
                },
                "observation_noise_hash": "abc123def456",
                "observation_noise_stats": {
                    "steps_with_noise": 5,
                    "pedestrians_removed": 1,
                    "pedestrians_added": 2,
                },
            }
        )
    return row


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_paired_row_delta_reports_success_and_collision_incidence(tmp_path: Path) -> None:
    """Same planner/scenario/seed rows produce nominal-vs-perturbed deltas."""
    nominal = tmp_path / "nominal.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    _write_jsonl(nominal, [_row(route_complete=True, collision=False)])
    _write_jsonl(perturbed, [_row(route_complete=False, collision=True, noise=True)])

    report = build_robustness_delta_report(nominal_jsonl=nominal, perturbed_jsonl=perturbed)

    assert report["pairing"]["paired_rows"] == 1
    row = report["planner_rows"][0]
    assert row["paired_episodes"] == 1
    assert row["nominal_success_incidence"] == 1.0
    assert row["perturbed_success_incidence"] == 0.0
    assert row["success_delta"] == -1.0
    assert row["nominal_collision_incidence"] == 0.0
    assert row["perturbed_collision_incidence"] == 1.0
    assert row["collision_delta"] == 1.0


def test_unmatched_rows_are_reported_without_contaminating_aggregate(tmp_path: Path) -> None:
    """Unpaired rows stay visible but do not affect paired incidence aggregates."""
    nominal = tmp_path / "nominal.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    _write_jsonl(
        nominal,
        [
            _row(route_complete=True, collision=False),
            _row(scenario_id="extra", route_complete=False, collision=True),
        ],
    )
    _write_jsonl(perturbed, [_row(route_complete=True, collision=False, noise=True)])

    report = build_robustness_delta_report(nominal_jsonl=nominal, perturbed_jsonl=perturbed)

    assert report["pairing"]["paired_rows"] == 1
    assert report["pairing"]["unmatched_nominal_rows"] == 1
    assert report["pairing"]["unmatched_perturbed_rows"] == 0
    row = report["planner_rows"][0]
    assert row["nominal_success_incidence"] == 1.0
    assert row["nominal_collision_incidence"] == 0.0


def test_noise_metadata_is_carried_from_perturbed_rows(tmp_path: Path) -> None:
    """Perturbation profile, hash, and counters are aggregated per planner."""
    nominal = tmp_path / "nominal.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    _write_jsonl(nominal, [_row()])
    _write_jsonl(perturbed, [_row(noise=True)])

    report = build_robustness_delta_report(nominal_jsonl=nominal, perturbed_jsonl=perturbed)

    row = report["planner_rows"][0]
    assert row["perturbation_profiles"] == ["issue_3952_robustness_smoke_v1"]
    assert row["perturbation_hashes"] == ["abc123def456"]
    assert row["noise_stats_sum"] == {
        "pedestrians_added": 2,
        "pedestrians_removed": 1,
        "steps_with_noise": 5,
    }


def test_cli_writes_json_csv_and_markdown(tmp_path: Path) -> None:
    """CLI wrapper writes all durable report artifact formats."""
    nominal = tmp_path / "nominal.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    output_json = tmp_path / "summary.json"
    output_csv = tmp_path / "robustness_delta.csv"
    output_md = tmp_path / "README.md"
    _write_jsonl(nominal, [_row()])
    _write_jsonl(perturbed, [_row(noise=True)])

    main = runpy.run_path(SCRIPT_PATH)["main"]
    exit_code = main(
        [
            "--nominal-jsonl",
            str(nominal),
            "--perturbed-jsonl",
            str(perturbed),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--output-md",
            str(output_md),
        ]
    )

    assert exit_code == 0
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["schema_version"] == "observation_robustness_delta.v1"
    assert report["planner_rows"][0]["planner_key"] == "goal"
    assert "success_delta" in output_csv.read_text(encoding="utf-8")
    markdown = output_md.read_text(encoding="utf-8")
    assert "Issue #3952 Observation Robustness Smoke" in markdown
    assert "goal" in markdown


def test_markdown_includes_diagnostic_caveats(tmp_path: Path) -> None:
    """Markdown report keeps the claim boundary near the top."""
    nominal = tmp_path / "nominal.jsonl"
    perturbed = tmp_path / "perturbed.jsonl"
    _write_jsonl(nominal, [_row()])
    _write_jsonl(perturbed, [_row(noise=True)])

    report = build_robustness_delta_report(nominal_jsonl=nominal, perturbed_jsonl=perturbed)
    markdown = format_report_markdown(report)

    assert "not paper-facing benchmark evidence" in markdown
    assert "Nominal metric definitions unchanged." in markdown
