"""Tests for campaign result-store comparison reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.build_campaign_comparison_report import (
    build_markdown,
    build_report,
    main,
)
from scripts.tools.campaign_result_store import write_result_store


def _write_fixture_store(path: Path, *, analysis: dict | None = None) -> None:
    """Create a small result store with valid and limited rows."""
    fixture = Path("tests/fixtures/campaign_result_store/issue_3063_episode_rows.json")
    rows = json.loads(fixture.read_text(encoding="utf-8"))
    write_result_store(
        path,
        rows,
        study_id="issue-3063-fixture",
        command="uv run python scripts/tools/build_campaign_comparison_report.py ...",
        source_commit="abc123",
        analysis=analysis,
    )


def _write_native_seed_gate_store(
    path: Path,
    *,
    analysis: dict,
) -> None:
    """Create a canonical result store with only benchmark-valid native rows."""
    rows = [
        {
            "run_id": f"run-{seed}",
            "episode_id": f"run-{seed}-001",
            "planner": "orca",
            "scenario_id": "crossing",
            "scenario_family": "crossing",
            "seed": seed,
            "row_status": "native",
            "artifact_uri": f"wandb://robot-sf/run-{seed}/episodes/run-{seed}-001.jsonl",
            "artifact_sha256": str(seed) * 64,
            "success": seed % 2 == 0,
            "collision": False,
            "snqi": 0.5,
        }
        for seed in (5, 6, 7, 8, 9)
    ]
    write_result_store(
        path,
        rows,
        study_id="issue-3160-seed-gate-fixture",
        command="uv run python scripts/tools/build_campaign_comparison_report.py ...",
        source_commit="abc123",
        analysis=analysis,
    )


def test_build_report_surfaces_uncertainty_denominators_and_caveats(tmp_path: Path) -> None:
    """Report payload should expose metrics, denominators, and invalid-row caveats."""
    result_store = tmp_path / "result-store"
    _write_fixture_store(result_store)

    payload = build_report(
        result_store,
        input_label="tests/fixtures/campaign_result_store/issue_3063_episode_rows.json",
        min_sample=3,
    )

    assert payload["schema_version"] == "campaign-comparison-report.v1"
    assert payload["report_status"] == "analysis_only"
    assert (
        payload["input"]["durable_input_label"]
        == "tests/fixtures/campaign_result_store/issue_3063_episode_rows.json"
    )
    assert payload["input"]["result_store"] == "transient_local_result_store"
    assert payload["seed_sufficiency_gate"] is None
    assert payload["row_status"]["benchmark_valid_episode_count"] == 2
    assert payload["row_status"]["excluded_or_limited_episode_count"] == 2
    caveats = {row["row_status"]: row["interpretation"] for row in payload["row_status"]["caveats"]}
    assert caveats["fallback"] == "excluded_or_limited"
    assert caveats["degraded"] == "excluded_or_limited"
    planner_rows = {row["planner"]: row for row in payload["planner_summaries"]}
    assert planner_rows["goal"]["metrics"]["success"]["denominator"] == 2
    assert planner_rows["goal"]["metrics"]["success"]["mean"] == 0.5
    assert planner_rows["orca"]["metrics"]["snqi"]["denominator"] == 2
    assert any(row["metric"] == "snqi" for row in payload["metric_visual_summaries"])
    assert any(hook["sample_gate"] == "underpowered" for hook in payload["statistical_hooks"])


def test_build_markdown_includes_visual_summaries_and_statistical_hooks(tmp_path: Path) -> None:
    """Markdown should make caveats and descriptive hooks visible."""
    result_store = tmp_path / "result-store"
    _write_fixture_store(result_store)

    markdown = build_markdown(build_report(result_store, min_sample=1))

    assert "## Row Status Caveats" in markdown
    assert "| fallback | 1 | excluded_or_limited |" in markdown
    assert "## Metric Visual Summaries" in markdown
    assert "social_compliance" in markdown
    assert "## Statistical Hooks" in markdown
    assert "descriptive_only_formal_test_not_run" in markdown


def test_main_writes_json_and_markdown_outputs(
    tmp_path: Path,
) -> None:
    """CLI should write both report artifacts from a valid result store."""
    result_store = tmp_path / "result-store"
    _write_fixture_store(result_store)
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    assert (
        main(
            [
                "--result-store",
                str(result_store),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
                "--input-label",
                "tests/fixtures/campaign_result_store/issue_3063_episode_rows.json",
                "--min-sample",
                "1",
            ]
        )
        == 0
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["input"]["study_id"] == "issue-3063-fixture"
    assert (
        payload["input"]["durable_input_label"]
        == "tests/fixtures/campaign_result_store/issue_3063_episode_rows.json"
    )
    assert "Campaign Comparison Report" in output_md.read_text(encoding="utf-8")


def test_build_report_embeds_seed_gate_escalation_from_result_store(tmp_path: Path) -> None:
    """Scheduling consumers should derive escalation from canonical result-store inputs."""
    result_store = tmp_path / "result-store"
    _write_native_seed_gate_store(
        result_store,
        analysis={
            "seed_sufficiency_gate": {
                "schedule": "s5",
                "ci_half_width": 0.2,
                "target_ci_half_width": 0.1,
            }
        },
    )

    payload = build_report(result_store, min_sample=1)

    seed_gate = payload["seed_sufficiency_gate"]
    assert seed_gate["source"] == "campaign_result_store.analysis_json"
    assert seed_gate["input"]["invalid_row_count"] == 0
    assert seed_gate["decision"]["decision"] == "escalate"
    assert seed_gate["decision"]["next_schedule"] == "s10"


def test_build_report_embeds_seed_gate_stop_from_result_store(tmp_path: Path) -> None:
    """Stable result-store inputs should let scheduling stop without manual JSON."""
    result_store = tmp_path / "result-store"
    _write_native_seed_gate_store(
        result_store,
        analysis={
            "seed_sufficiency_gate": {
                "schedule": "s10",
                "ci_half_width": 0.05,
                "target_ci_half_width": 0.1,
            }
        },
    )

    payload = build_report(result_store, min_sample=1)

    seed_gate = payload["seed_sufficiency_gate"]
    assert seed_gate["decision"]["decision"] == "stop_confirmed"
    assert seed_gate["decision"]["next_schedule"] is None


def test_build_report_keeps_limited_rows_diagnostic_only_for_seed_gate(
    tmp_path: Path,
) -> None:
    """Excluded or limited result-store rows should prevent escalation claim strength."""
    result_store = tmp_path / "result-store"
    _write_fixture_store(
        result_store,
        analysis={
            "seed_sufficiency_gate": {
                "schedule": "s5",
                "ci_half_width": 0.05,
                "target_ci_half_width": 0.1,
            }
        },
    )

    payload = build_report(result_store, min_sample=1)

    seed_gate = payload["seed_sufficiency_gate"]
    assert seed_gate["input"]["invalid_row_count"] == 2
    assert seed_gate["decision"]["decision"] == "diagnostic_only"
    assert seed_gate["decision"]["next_schedule"] is None


def test_main_writes_seed_gate_decision_output(tmp_path: Path) -> None:
    """CLI should record a durable seed-gate decision artifact when requested."""
    result_store = tmp_path / "result-store"
    _write_native_seed_gate_store(
        result_store,
        analysis={
            "seed_sufficiency_gate": {
                "schedule": "s5",
                "ci_half_width": 0.2,
                "target_ci_half_width": 0.1,
            }
        },
    )
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"
    seed_gate_json = tmp_path / "seed-gate-decision.json"

    assert (
        main(
            [
                "--result-store",
                str(result_store),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
                "--seed-gate-output-json",
                str(seed_gate_json),
            ]
        )
        == 0
    )

    seed_gate = json.loads(seed_gate_json.read_text(encoding="utf-8"))
    assert seed_gate["decision"]["decision"] == "escalate"
    assert "## Seed Sufficiency Gate" in output_md.read_text(encoding="utf-8")


def test_main_fails_when_seed_gate_output_requested_without_store_config(
    tmp_path: Path,
) -> None:
    """Seed-gate scheduling output should not fall back to ad hoc empty decisions."""
    result_store = tmp_path / "result-store"
    _write_fixture_store(result_store)

    assert (
        main(
            [
                "--result-store",
                str(result_store),
                "--output-json",
                str(tmp_path / "report.json"),
                "--output-md",
                str(tmp_path / "report.md"),
                "--seed-gate-output-json",
                str(tmp_path / "seed-gate-decision.json"),
            ]
        )
        == 1
    )


def test_main_fails_closed_for_incomplete_result_store(tmp_path: Path) -> None:
    """Invalid result-store inputs should not produce ad hoc reports."""
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    assert (
        main(
            [
                "--result-store",
                str(incomplete),
                "--output-json",
                str(output_json),
                "--output-md",
                str(output_md),
            ]
        )
        == 1
    )

    assert not output_json.exists()
    assert not output_md.exists()


def test_main_rejects_non_positive_min_sample(tmp_path: Path) -> None:
    """Pairwise hook sample gates should reject nonsensical thresholds."""
    result_store = tmp_path / "result-store"
    _write_fixture_store(result_store)

    with pytest.raises(SystemExit):
        main(
            [
                "--result-store",
                str(result_store),
                "--output-json",
                str(tmp_path / "report.json"),
                "--output-md",
                str(tmp_path / "report.md"),
                "--min-sample",
                "0",
            ]
        )


def test_build_report_redacts_external_absolute_result_store_path(tmp_path: Path) -> None:
    """Reports should not persist host-specific absolute paths."""
    result_store = tmp_path / "result-store"
    _write_fixture_store(result_store)

    payload = build_report(result_store, min_sample=1)

    assert payload["input"]["result_store"] == "absolute_result_store_path_redacted"
