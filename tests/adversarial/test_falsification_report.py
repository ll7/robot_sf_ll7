"""Fixture-backed contracts for the falsification convergence report."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.adversarial.falsification_report import (
    REPORT_SCHEMA,
    build_convergence_report,
    render_figures,
    render_markdown,
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


def _report_with_manifest(
    tmp_path: Path,
    *,
    row_index: int,
    mutate: Any,
    execution_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((FIXTURE_ROOT / "comparison.json").read_text(encoding="utf-8"))
    row = comparison["rows"][row_index]
    manifest_path = REPO_ROOT / row["manifest_path"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    temporary_manifest = tmp_path / f"manifest-{row_index}.json"
    temporary_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    if execution_context is not None:
        (tmp_path / "execution_context.txt").write_text(
            json.dumps(execution_context), encoding="utf-8"
        )
    row["manifest_path"] = str(temporary_manifest)
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    return build_convergence_report(comparison_path, repo_root=REPO_ROOT)


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
    assert report["provenance"]["source_revision"]["status"] == "unknown"
    assert report["provenance"]["source_revision"]["exact_source_revision"] is None
    assert report["provenance"]["source_revision"]["unverified_commit_identifiers"] == ["abc123"]


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
    assert "Observed best and critical counts retain raw candidate evidence" in markdown
    assert "analysis_eligibility.eligible=true" in markdown
    assert "native: 1, unknown: 3" in markdown
    assert "available: 1, unknown: 3" in markdown
    assert "Not performed; descriptive only" in markdown
    assert (output / "convergence_constraints_first_lexicographic_v1.png").is_file()


def test_figure_labels_budget_with_no_scored_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    original_subplots = plt.subplots
    captured_figures = []

    def capture_subplots(*args: Any, **kwargs: Any) -> Any:
        figure, axes = original_subplots(*args, **kwargs)
        captured_figures.append(figure)
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_subplots)
    render_figures(_report(), tmp_path)

    assert any(
        "No scored observations" in text.get_text()
        for figure in captured_figures
        for axis in figure.axes
        for text in axis.texts
    )


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


def test_over_budget_attempt_is_auditable_but_cannot_change_budgeted_best_or_pair(
    tmp_path: Path,
) -> None:
    def append_over_budget_candidate(manifest: dict[str, Any]) -> None:
        candidate = copy.deepcopy(manifest["candidates"][0])
        candidate["candidate"]["start"]["x"] = 9.0
        candidate["effective_scenario_hash"] = "over-budget-critical"
        candidate["objective_value"] = 99.0
        candidate["failure_attribution"]["primary_failure"] = "collision"
        manifest["candidates"].append(candidate)

    report = _report_with_manifest(tmp_path, row_index=0, mutate=append_over_budget_candidate)
    random_1101 = next(
        run for run in report["runs"] if run["sampler"] == "random" and run["seed"] == 1101
    )
    comparison = next(item for item in report["random_vs_tpe"] if item["budget"] == 4)

    assert random_1101["num_candidates"] == 5
    assert random_1101["num_budgeted_candidates"] == 4
    assert random_1101["num_over_budget_candidates"] == 1
    assert random_1101["num_over_budget_critical_candidates"] == 1
    assert random_1101["best_objective_value"] == pytest.approx(4.3)
    assert random_1101["best_observed_objective_value"] == pytest.approx(99.0)
    assert random_1101["evaluations"][4]["within_budget"] is False
    assert random_1101["evaluations"][4]["best_so_far_objective"] == pytest.approx(4.3)
    assert random_1101["evaluations"][4]["best_so_far_observed_objective"] == pytest.approx(99.0)
    assert comparison["matched_seed_count"] == 1
    assert comparison["ineligible_matched_seeds"][0]["seed"] == 1101
    assert (
        "over_budget_candidates_present"
        in comparison["ineligible_matched_seeds"][0]["random_reason_codes"]
    )


def test_pairs_fail_closed_when_shared_search_configuration_differs(tmp_path: Path) -> None:
    def change_search_space(manifest: dict[str, Any]) -> None:
        manifest["config"]["search_space"]["variables"]["start_x"]["max"] = 2.0

    report = _report_with_manifest(tmp_path, row_index=1, mutate=change_search_space)
    comparison = next(item for item in report["random_vs_tpe"] if item["budget"] == 4)

    assert comparison["matched_seed_count"] == 1
    excluded = next(item for item in comparison["ineligible_matched_seeds"] if item["seed"] == 1101)
    assert excluded["reason_codes"] == ["shared_configuration_mismatch"]


def test_pairs_fail_closed_when_planner_policy_differs(tmp_path: Path) -> None:
    def change_planner_policy(manifest: dict[str, Any]) -> None:
        manifest["config"]["policy"] = "different_planner"

    report = _report_with_manifest(tmp_path, row_index=1, mutate=change_planner_policy)
    comparison = next(item for item in report["random_vs_tpe"] if item["budget"] == 4)
    excluded = next(item for item in comparison["ineligible_matched_seeds"] if item["seed"] == 1101)

    assert comparison["matched_seed_count"] == 1
    assert excluded["reason_codes"] == ["shared_configuration_mismatch"]


def test_pairs_fail_closed_when_scenario_template_content_differs(tmp_path: Path) -> None:
    def change_scenario_template(manifest: dict[str, Any]) -> None:
        template = tmp_path / "different-scenario.yaml"
        template.write_text("scenarios: []\n", encoding="utf-8")
        manifest["config"]["scenario_template"] = str(template)

    report = _report_with_manifest(tmp_path, row_index=1, mutate=change_scenario_template)
    comparison = next(item for item in report["random_vs_tpe"] if item["budget"] == 4)
    excluded = next(item for item in comparison["ineligible_matched_seeds"] if item["seed"] == 1101)

    assert comparison["matched_seed_count"] == 1
    assert excluded["reason_codes"] == ["shared_configuration_mismatch"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("budget", 5, "manifest_budget_mismatch"),
        ("seed", 9999, "manifest_seed_mismatch"),
        ("objective", "different_objective", "manifest_objective_mismatch"),
    ],
)
def test_pairs_fail_closed_when_index_and_manifest_identity_disagree(
    tmp_path: Path, field: str, value: Any, reason: str
) -> None:
    def change_manifest_identity(manifest: dict[str, Any]) -> None:
        manifest["config"][field] = value

    report = _report_with_manifest(tmp_path, row_index=1, mutate=change_manifest_identity)
    comparison = next(item for item in report["random_vs_tpe"] if item["budget"] == 4)
    changed_run = next(
        run for run in report["runs"] if run["sampler"] == "optuna" and run["seed"] == 1101
    )
    excluded = next(item for item in comparison["ineligible_matched_seeds"] if item["seed"] == 1101)

    assert changed_run["index_identity_valid"] is False
    assert reason in changed_run["index_identity_reason_codes"]
    assert comparison["matched_seed_count"] == 1
    assert reason in excluded["tpe_reason_codes"]


def test_missing_rows_beyond_indexed_budget_are_not_counted_as_budgeted_missing(
    tmp_path: Path,
) -> None:
    def increase_manifest_budget(manifest: dict[str, Any]) -> None:
        manifest["config"]["budget"] = 5

    report = _report_with_manifest(tmp_path, row_index=0, mutate=increase_manifest_budget)
    random_1101 = next(
        run for run in report["runs"] if run["sampler"] == "random" and run["seed"] == 1101
    )
    assert random_1101["expected_evaluations"] == 5
    assert random_1101["num_missing_evaluations"] == 1
    assert random_1101["num_missing_budgeted_evaluations"] == 0
    assert random_1101["evaluations"][4]["status"] == "missing"
    assert random_1101["evaluations"][4]["within_budget"] is False
    assert "manifest_budget_mismatch" in random_1101["comparison_ineligibility_reason_codes"]

    random_aggregate = next(
        aggregate
        for aggregate in report["aggregates"]
        if aggregate["sampler"] == "random" and aggregate["budget"] == 4
    )
    assert random_aggregate["candidate_accounting"]["missing_within_budget"] == 0
    assert random_aggregate["candidate_accounting"]["missing_all_expected_slots"] == 1
    accounting_row = next(
        line for line in render_markdown(report).splitlines() if "| Random | 1101 |" in line
    )
    assert "| 2 / 1 / 1 / 0 / 0 / 1 |" in accounting_row


@pytest.mark.parametrize("eligibility_reason", ["ineligible", "degraded"])
def test_ineligible_degraded_score_and_criticality_remain_observed_but_not_eligible(
    tmp_path: Path, eligibility_reason: str
) -> None:
    def raise_noneligible_score(manifest: dict[str, Any]) -> None:
        candidate = manifest["candidates"][3]
        candidate["objective_value"] = 99.0
        candidate["failure_attribution"]["primary_failure"] = "incomplete"
        if eligibility_reason == "ineligible":
            candidate["analysis_eligibility"]["eligible"] = False
            candidate["failure_attribution"]["details"]["execution_mode"] = "native"
        else:
            candidate["analysis_eligibility"]["eligible"] = True

    report = _report_with_manifest(tmp_path, row_index=2, mutate=raise_noneligible_score)
    random_2202 = next(
        run for run in report["runs"] if run["sampler"] == "random" and run["seed"] == 2202
    )
    comparison = next(item for item in report["random_vs_tpe"] if item["budget"] == 4)
    pair_2202 = next(pair for pair in comparison["pairs"] if pair["seed"] == 2202)
    degraded = random_2202["evaluations"][3]

    assert random_2202["best_observed_objective_value"] == pytest.approx(99.0)
    assert random_2202["best_analysis_eligible_objective_value"] == pytest.approx(4.1)
    assert random_2202["num_observed_critical_candidates"] == 2
    assert random_2202["num_critical_candidates"] == 1
    assert degraded["observed_critical"] is True
    assert degraded["critical"] is False
    assert degraded["analysis_evidence_eligible"] is False
    assert degraded["analysis_eligible"] is (eligibility_reason == "degraded")
    assert degraded["execution_mode"] == (
        "degraded" if eligibility_reason == "degraded" else "native"
    )
    assert pair_2202["random_final_best"] == pytest.approx(4.1)
    assert pair_2202["tpe_minus_random"] == pytest.approx(0.4)


def test_exact_source_revision_requires_full_sha_and_supported_context_schema(
    tmp_path: Path,
) -> None:
    full_sha = "a" * 40
    context = {
        "schema_version": "adversarial_execution_context.v1",
        "commit_sha": full_sha,
    }
    report = _report_with_manifest(
        tmp_path,
        row_index=0,
        mutate=lambda _manifest: None,
        execution_context=context,
    )
    random_1101 = next(
        run for run in report["runs"] if run["sampler"] == "random" and run["seed"] == 1101
    )
    assert random_1101["source_revision"]["exact_source_revision"] == full_sha
    assert "abc123" in random_1101["source_revision"]["unverified_commit_identifiers"]

    bad_schema_context = {"schema_version": "unexpected.v9", "commit_sha": full_sha}
    report = _report_with_manifest(
        tmp_path / "bad-schema",
        row_index=0,
        mutate=lambda _manifest: None,
        execution_context=bad_schema_context,
    )
    random_1101 = next(
        run for run in report["runs"] if run["sampler"] == "random" and run["seed"] == 1101
    )
    assert random_1101["source_revision"]["exact_source_revision"] is None
    assert random_1101["source_revision"]["status"] == "unknown"
    assert full_sha in random_1101["source_revision"]["unverified_commit_identifiers"]
