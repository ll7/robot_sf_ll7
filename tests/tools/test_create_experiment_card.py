"""Tests for the experiment card scaffold CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.tools.create_experiment_card import (
    REQUIRED_RECORD_FIELDS,
    TEMPLATE_NAMES,
    _build_record,
    _find_todo_fields,
    _resolve_issue_url,
    main,
)
from scripts.tools.validate_experiment_registry import VALID_EVIDENCE_GRADES, VALID_PAPER_RELEVANCE


def test_resolve_issue_url_default() -> None:
    """Default issue URL should be derived from the issue number."""
    url = _resolve_issue_url("2103", override="")
    assert url == "https://github.com/ll7/robot_sf_ll7/issues/2103"


def test_resolve_issue_url_override() -> None:
    """Explicit --issue-url should override the default."""
    url = _resolve_issue_url("2103", override="https://custom.example.com/2103")
    assert url == "https://custom.example.com/2103"


def test_each_template_produces_experiment_record_v1_schema() -> None:
    """Every built record should declare experiment-record.v1 schema version."""
    for template_name in TEMPLATE_NAMES:
        record = _build_record(
            experiment_id=f"test_{template_name.replace('-', '_')}",
            issue="9999",
            issue_url="https://example.com/9999",
            template_name=template_name,
            output_root=Path("output/experiments"),
        )
        assert record["schema_version"] == "experiment-record.v1"
        assert record["experiment_id"].startswith("test_")


def test_each_template_covers_required_fields(tmp_path: Path) -> None:
    """Every template should populate each required registry field."""
    for template_name in TEMPLATE_NAMES:
        record = _build_record(
            experiment_id=f"test_{template_name.replace('-', '_')}",
            issue="9999",
            issue_url="https://example.com/9999",
            template_name=template_name,
            output_root=Path("output/experiments"),
        )
        assert record["schema_version"] == "experiment-record.v1"

        for req_field in REQUIRED_RECORD_FIELDS:
            assert req_field in record, (
                f"template {template_name!r} missing required field {req_field!r}"
            )
            assert record[req_field] is not None, (
                f"template {template_name!r} required field {req_field!r} must be populated"
            )
            if isinstance(record[req_field], str):
                assert record[req_field].strip(), (
                    f"template {template_name!r} required field {req_field!r} must not be blank"
                )


def test_each_template_marks_todo_in_question_and_hypothesis() -> None:
    """Question and hypothesis should start with TODO so users know to fill them."""
    for template_name in TEMPLATE_NAMES:
        record = _build_record(
            experiment_id=f"test_{template_name.replace('-', '_')}",
            issue="9999",
            issue_url="https://example.com/9999",
            template_name=template_name,
            output_root=Path("output/experiments"),
        )
        todo = _find_todo_fields(record)
        assert "question" in todo, (
            f"template {template_name!r} question should contain TODO placeholder"
        )
        assert "hypothesis" in todo, (
            f"template {template_name!r} hypothesis should contain TODO placeholder"
        )


def test_valid_evidence_grade_and_paper_relevance() -> None:
    """Evidence grade and paper relevance should come from the allowed vocabulary."""
    for template_name in TEMPLATE_NAMES:
        record = _build_record(
            experiment_id=f"test_{template_name.replace('-', '_')}",
            issue="9999",
            issue_url="https://example.com/9999",
            template_name=template_name,
            output_root=Path("output/experiments"),
        )
        assert record["evidence_grade"] in VALID_EVIDENCE_GRADES
        assert record["paper_relevance"] in VALID_PAPER_RELEVANCE


def test_generated_yaml_is_loadable(tmp_path: Path) -> None:
    """YAML output from _build_record should round-trip through yaml.safe_load."""
    for template_name in TEMPLATE_NAMES:
        record = _build_record(
            experiment_id=f"test_{template_name.replace('-', '_')}",
            issue="9999",
            issue_url="https://example.com/9999",
            template_name=template_name,
            output_root=Path("output/experiments"),
        )
        yaml_text = yaml.dump(record, default_flow_style=False, sort_keys=False)
        reloaded = yaml.safe_load(yaml_text)
        assert isinstance(reloaded, dict)
        assert reloaded["experiment_id"] == record["experiment_id"]
        assert reloaded["schema_version"] == "experiment-record.v1"


def test_todo_detects_todo_in_command() -> None:
    """Command field should always contain TODO placeholders."""
    record = _build_record(
        experiment_id="test_todo_cmd",
        issue="9999",
        issue_url="https://example.com/9999",
        template_name="benchmark-analysis",
        output_root=Path("output/experiments"),
    )
    todo = _find_todo_fields(record)
    assert "command" in todo


def test_output_and_expected_artifacts_have_artifact_ids() -> None:
    """Every expected artifact should carry a valid artifact_id."""
    for template_name in TEMPLATE_NAMES:
        record = _build_record(
            experiment_id=f"test_{template_name.replace('-', '_')}",
            issue="9999",
            issue_url="https://example.com/9999",
            template_name=template_name,
            output_root=Path("output/experiments"),
        )
        for artifact in record["expected_artifacts"]:
            assert "artifact_id" in artifact, (
                f"template {template_name!r} expected_artifact missing artifact_id"
            )
            artifact_id = artifact["artifact_id"]
            assert artifact_id[0].islower(), (
                f"template {template_name!r} artifact_id {artifact_id!r} must start with lowercase"
            )
            assert " " not in artifact_id, (
                f"template {template_name!r} artifact_id {artifact_id!r} must not contain spaces"
            )


def test_draft_templates_are_proposal_state_until_placeholders_are_filled() -> None:
    """Generated draft cards should stay non-actionable until TODO fields are replaced."""
    for template_name in TEMPLATE_NAMES:
        record = _build_record(
            experiment_id=f"test_{template_name.replace('-', '_')}",
            issue="9999",
            issue_url="https://example.com/9999",
            template_name=template_name,
            output_root=Path("output/experiments"),
        )
        assert record["evidence_grade"] == "proposal"
        assert record["status"] == "proposal"


def test_training_templates_include_slurm_early_stop_criteria() -> None:
    """Training-oriented experiment cards should predeclare Slurm stop rules."""
    expected_fields = {
        "metric",
        "threshold",
        "check_cadence",
        "minimum_runtime_or_timesteps",
        "cancel_condition",
        "diagnostic_preservation_action",
    }
    training_templates = {"benchmark-analysis", "planner-ablation"}
    for template_name in training_templates:
        record = _build_record(
            experiment_id=f"test_{template_name.replace('-', '_')}",
            issue="9999",
            issue_url="https://example.com/9999",
            template_name=template_name,
            output_root=Path("output/experiments"),
        )
        assert set(record["early_stop_criteria"]) == expected_fields
        assert all(
            str(value).startswith("TODO:") for value in record["early_stop_criteria"].values()
        )
        assert "early_stop_criteria" in _find_todo_fields(record)


def test_figure_table_pack_omits_slurm_early_stop_criteria() -> None:
    """Figure/table rendering from existing records does not launch long Slurm training."""
    record = _build_record(
        experiment_id="test_figure_table_pack",
        issue="9999",
        issue_url="https://example.com/9999",
        template_name="figure-table-pack",
        output_root=Path("output/experiments"),
    )

    assert record["early_stop_criteria"] == {}
    assert "early_stop_criteria" not in _find_todo_fields(record)


def test_invalid_template_name_raises_key_error() -> None:
    """Passing an unrecognised template name should raise KeyError."""
    with pytest.raises(KeyError):
        _build_record(
            experiment_id="test_invalid",
            issue="9999",
            issue_url="https://example.com/9999",
            template_name="nonexistent-template",
            output_root=Path("output/experiments"),
        )


def test_cli_writes_to_requested_output_root(tmp_path: Path) -> None:
    """CLI should treat --output-root as the experiment directory itself."""
    output_root = tmp_path / "output" / "experiments" / "issue_2103_smoke"

    exit_code = main(
        [
            "--issue",
            "2103",
            "--experiment-id",
            "issue_2103_smoke",
            "--template",
            "benchmark-analysis",
            "--output-root",
            str(output_root),
        ]
    )

    assert exit_code == 0
    assert (output_root / "issue_2103_smoke.yaml").is_file()
    assert (output_root / "CHECKLIST.md").is_file()
    assert not (output_root / "issue_2103_smoke" / "issue_2103_smoke.yaml").exists()

    checklist = (output_root / "CHECKLIST.md").read_text(encoding="utf-8")
    assert "Slurm Early-Stop Criteria" in checklist
    assert "diagnostic preservation action" in checklist
