"""Tests for config-only scenario coverage entropy reports."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from robot_sf.benchmark.runner import load_scenario_matrix
from robot_sf.benchmark.scenario_coverage import (
    build_scenario_coverage_report,
    scenario_coverage_report_markdown,
    write_scenario_coverage_report,
)


def _scenario(
    name: str,
    *,
    archetype: str = "crossing",
    density: str = "low",
    ped_density: float = 0.02,
    flow: str = "bi",
    single_pedestrians: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Return a compact scenario fixture for coverage-analysis tests."""
    return {
        "name": name,
        "map_file": "maps/svg_maps/classic_station_platform.svg",
        "simulation_config": {
            "max_episode_steps": 650,
            "ped_density": ped_density,
        },
        "single_pedestrians": single_pedestrians or [],
        "metadata": {
            "archetype": archetype,
            "density": density,
            "flow": flow,
            "evaluation_scope": "exploratory",
        },
        "seeds": [1, 2, 3],
    }


def test_build_scenario_coverage_report_flags_redundant_and_novel_rows() -> None:
    """Coverage report should rank scenarios by transparent config-derived novelty."""
    scenarios = [
        _scenario("route_flow_low"),
        _scenario("route_flow_low_duplicate"),
        _scenario(
            "waiting_passengers_medium",
            density="medium",
            ped_density=0.05,
            flow="bi-with-dwell",
            single_pedestrians=[
                {
                    "id": "p1",
                    "goal": None,
                    "trajectory": [[0.0, 0.0], [1.0, 1.0]],
                    "wait_at": [{"waypoint_index": 1, "wait_s": 4.0}],
                }
            ],
        ),
    ]

    report = build_scenario_coverage_report(scenarios, source="fixture.yaml")

    assert report["schema_version"] == "scenario_coverage_entropy.v1"
    assert report["source"] == "fixture.yaml"
    assert report["summary"]["scenario_count"] == 3
    assert 0.0 <= report["summary"]["coverage_entropy"] <= 1.0
    assert report["feature_contract"]["mode"] == "config_only"
    assert report["feature_contract"]["metrics_are_benchmark_claims"] is False

    rows = {row["scenario_id"]: row for row in report["scenario_rows"]}
    assert rows["route_flow_low_duplicate"]["nearest_neighbor"] == "route_flow_low"
    assert rows["route_flow_low_duplicate"]["recommendation"] == "merge_or_drop"
    assert rows["waiting_passengers_medium"]["recommendation"] == "retain_or_investigate"
    assert (
        rows["waiting_passengers_medium"]["novelty_score"]
        > rows["route_flow_low_duplicate"]["novelty_score"]
    )
    assert "wait_behavior" in rows["waiting_passengers_medium"]["distinct_features"]


def test_station_platform_pack_report_uses_existing_distinct_probe_metadata() -> None:
    """The issue #736 candidate pack should produce deterministic coverage rows."""
    matrix = Path("configs/scenarios/sets/station_platform_candidate_pack_issue736.yaml")
    scenarios = load_scenario_matrix(matrix)

    report = build_scenario_coverage_report(scenarios, source=str(matrix))

    assert report["summary"]["scenario_count"] == 4
    assert report["summary"]["feature_count"] >= 6
    assert report["summary"]["coverage_entropy"] > 0.0

    rows = {row["scenario_id"]: row for row in report["scenario_rows"]}
    waiting = rows["station_platform_waiting_passengers_medium"]
    assert waiting["metadata"]["distinct_coverage_probe"]["compare_against"]
    assert waiting["recommendation"] == "retain_or_investigate"
    assert "wait_behavior" in waiting["distinct_features"]


def test_build_scenario_coverage_report_handles_minimal_legacy_rows() -> None:
    """Minimal legacy scenario rows should still produce an explicit diagnostic report."""
    report = build_scenario_coverage_report(
        [
            {
                "map": "legacy_map.svg",
                "simulation_config": {},
                "single_pedestrians": "not-a-list",
                "seeds": "not-a-list",
            }
        ],
        source="legacy.yaml",
    )

    row = report["scenario_rows"][0]

    assert row["scenario_id"] == "scenario_000"
    assert row["nearest_neighbor"] is None
    assert row["feature_tokens"]["archetype"] == "unknown"
    assert row["feature_tokens"]["ped_density_bin"] == "unknown"
    assert row["feature_tokens"]["single_pedestrians"] == "none"
    assert row["feature_tokens"]["seed_count"] == "none"
    assert row["metadata"] == {}


def test_build_scenario_coverage_report_rejects_empty_and_duplicate_inputs() -> None:
    """Invalid inputs should fail before emitting misleading novelty rows."""
    with pytest.raises(ValueError, match="requires at least one scenario"):
        build_scenario_coverage_report([], source="empty.yaml")

    with pytest.raises(ValueError, match="duplicate scenario ids"):
        build_scenario_coverage_report(
            [_scenario("same"), _scenario("same")],
            source="dupes.yaml",
        )


def test_report_markdown_and_writers_tolerate_partial_payloads(tmp_path: Path) -> None:
    """Markdown rendering and artifact writing should tolerate missing optional sections."""
    markdown = scenario_coverage_report_markdown(
        {"source": "partial.yaml", "scenario_rows": [None]}
    )
    assert "partial.yaml" in markdown
    assert "## Summary" in markdown

    report = build_scenario_coverage_report([_scenario("single")], source="single.yaml")
    json_path = tmp_path / "nested" / "coverage.json"
    md_path = tmp_path / "nested" / "coverage.md"

    write_scenario_coverage_report(report, json_path=json_path)
    write_scenario_coverage_report(report, markdown_path=md_path)

    assert json.loads(json_path.read_text(encoding="utf-8"))["source"] == "single.yaml"
    assert "single" in md_path.read_text(encoding="utf-8")


def test_scenario_coverage_entropy_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    """CLI should emit reviewable JSON and Markdown without running episodes."""
    json_path = tmp_path / "coverage.json"
    md_path = tmp_path / "coverage.md"

    result = subprocess.run(
        [
            "python",
            "scripts/tools/scenario_coverage_entropy.py",
            "configs/scenarios/sets/station_platform_candidate_pack_issue736.yaml",
            "--output-json",
            str(json_path),
            "--output-markdown",
            str(md_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")

    assert "scenario_coverage_entropy.v1" in result.stdout
    assert payload["summary"]["scenario_count"] == 4
    assert "## Scenario Coverage Entropy" in markdown
    assert "station_platform_waiting_passengers_medium" in markdown
    assert "not a benchmark-success or safety metric" in markdown
