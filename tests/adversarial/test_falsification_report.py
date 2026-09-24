"""Fixture-backed contracts for the falsification convergence report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.adversarial.falsification_report import (
    REPORT_SCHEMA,
    build_convergence_report,
    write_convergence_report,
)
from scripts.tools.report_falsification_search import main

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "adversarial" / "search_report"
REPO_ROOT = Path(__file__).parents[2]


def _report() -> dict[str, object]:
    return build_convergence_report(
        FIXTURE_ROOT / "comparison.json",
        repo_root=REPO_ROOT,
    )


def test_report_derives_candidate_accounting_and_preserves_legacy_summary() -> None:
    report = _report()
    assert report["schema_version"] == REPORT_SCHEMA
    runs = report["runs"]
    random_1101 = next(run for run in runs if run["sampler"] == "random" and run["seed"] == 1101)

    assert random_1101["num_candidates"] == 4
    assert random_1101["num_invalid_candidates"] == 1
    assert random_1101["num_failed_evaluations"] == 1
    assert random_1101["num_valid_candidates"] == 2
    assert random_1101["legacy_summary"]["num_valid_candidates"] == 3
    assert any(
        "legacy manifest summary num_valid_candidates" in warning
        for warning in random_1101["warnings"]
    )
    assert [item["status"] for item in random_1101["evaluations"]] == [
        "scored",
        "invalid",
        "failed",
        "scored",
    ]
    assert [item["best_so_far_objective"] for item in random_1101["evaluations"]] == [
        0.4,
        0.4,
        0.4,
        4.3,
    ]
    assert random_1101["evaluations"][0]["execution_mode"] == "native"
    assert random_1101["evaluations"][0]["readiness_status"] == "native"
    assert random_1101["evaluations"][0]["availability_status"] == "available"
    assert random_1101["execution_mode_counts"] == {"native": 1, "unknown": 3}
    assert random_1101["availability_status_counts"] == {"available": 1, "unknown": 3}


def test_report_keeps_missing_scoreless_duplicate_and_degraded_attempts_visible() -> None:
    report = _report()
    tpe_1101 = next(
        run for run in report["runs"] if run["sampler"] == "optuna" and run["seed"] == 1101
    )
    random_2202 = next(
        run for run in report["runs"] if run["sampler"] == "random" and run["seed"] == 2202
    )
    missing_run = next(run for run in report["runs"] if run["seed"] == 3303)

    assert [item["status"] for item in tpe_1101["evaluations"]] == [
        "scoreless",
        "scored",
        "missing",
        "missing",
    ]
    assert tpe_1101["num_missing_evaluations"] == 2
    assert tpe_1101["evaluations"][0]["best_so_far_objective"] is None
    assert tpe_1101["evaluations"][1]["best_so_far_objective"] == pytest.approx(4.4)
    assert tpe_1101["runtime_seconds"] == pytest.approx(4.25)
    assert random_2202["num_duplicate_candidates"] == 1
    assert random_2202["num_effective_scenario_duplicates"] == 1
    assert random_2202["num_scoreless_valid_candidates"] == 1
    assert random_2202["degraded_candidate_count"] == 1
    assert random_2202["first_critical_evaluation"] == 3
    assert missing_run["artifact_status"] == "missing"
    assert missing_run["num_missing_evaluations"] == 2
    assert missing_run["best_objective_value"] is None


def test_random_tpe_comparison_is_seed_matched_and_descriptive_only() -> None:
    report = _report()
    comparison = next(
        item
        for item in report["random_vs_tpe"]
        if item["objective"] == "constraints_first_lexicographic_v1" and item["budget"] == 4
    )

    assert comparison["matched_seed_count"] == 2
    assert [item["seed"] for item in comparison["pairs"]] == [1101, 2202]
    assert comparison["inference_status"] == "not_performed"
    assert comparison["tpe_minus_random_median"] == pytest.approx(0.25)
    assert comparison["tpe_minus_random_min"] == pytest.approx(0.1)
    assert comparison["tpe_minus_random_max"] == pytest.approx(0.4)
    assert report["provenance"]["source_revision"]["status"] == "partial"
    assert report["provenance"]["source_revision"]["exact_source_revision"] is None


def test_same_inputs_write_byte_stable_json_markdown_and_figure(tmp_path: Path) -> None:
    report = _report()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_outputs = write_convergence_report(report, first)
    second_outputs = write_convergence_report(report, second)

    for key in ("json", "markdown"):
        first_path = Path(first_outputs[key])
        second_path = Path(second_outputs[key])
        assert first_path.read_bytes() == second_path.read_bytes()
    first_figure = Path(first_outputs["figures"][0])
    second_figure = Path(second_outputs["figures"][0])
    assert first_figure.read_bytes() == second_figure.read_bytes()


def test_cli_writes_machine_readable_summary_table_and_figure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "report"
    exit_code = main(
        [
            "--comparison",
            str(FIXTURE_ROOT / "comparison.json"),
            "--output-dir",
            str(output),
            "--repo-root",
            str(REPO_ROOT),
        ]
    )

    assert exit_code == 0
    console = json.loads(capsys.readouterr().out)
    assert console["run_count"] == 5
    assert (output / "falsification_report.json").is_file()
    markdown = (output / "falsification_report.md").read_text(encoding="utf-8")
    assert "Invalid, failed, scoreless and missing" in markdown
    assert "native: 1, unknown: 3" in markdown
    assert "available: 1, unknown: 3" in markdown
    assert "Not performed; descriptive only" in markdown
    assert (output / "convergence_constraints_first_lexicographic_v1.png").is_file()


def test_unsupported_comparison_schema_fails_closed(tmp_path: Path) -> None:
    source = json.loads((FIXTURE_ROOT / "comparison.json").read_text(encoding="utf-8"))
    source["schema_version"] = "unsupported.v9"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(ValueError, match="comparison input schema"):
        build_convergence_report(path, repo_root=REPO_ROOT)


def test_fixture_files_are_checked_in_and_paths_resolve_from_repo_root() -> None:
    assert (FIXTURE_ROOT / "comparison.json").is_file()
    report = _report()
    assert all(
        run["artifact_status"] == "available"
        for run in report["runs"]
        if run["manifest_path"].endswith("/manifest.json")
    )
