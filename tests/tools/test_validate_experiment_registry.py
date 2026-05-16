"""Tests for the question-first experiment registry validator."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from scripts.tools.validate_experiment_registry import validate_registry


def _write_yaml(path: Path, text: str) -> None:
    """Write a YAML fixture with normalized leading whitespace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")


def test_checked_in_experiment_registry_validates() -> None:
    """The tracked experiment registry should satisfy the workflow contract."""
    errors = validate_registry(Path("experiments/registry.yaml"))

    assert errors == []


def test_validator_reports_missing_required_fields(tmp_path: Path) -> None:
    """Experiment records must state the research question before execution."""
    registry = tmp_path / "registry.yaml"
    record = tmp_path / "missing.yaml"
    _write_yaml(
        registry,
        """
        schema_version: experiment-registry.v1
        records:
          - missing.yaml
        """,
    )
    _write_yaml(
        record,
        """
        schema_version: experiment-record.v1
        experiment_id: missing-question
        question: What does this run answer?
        """,
    )

    errors = validate_registry(registry)

    assert "missing.yaml: missing required field 'hypothesis'" in errors
    assert "missing.yaml: missing required field 'command'" in errors
    assert "missing.yaml: missing required field 'expected_artifacts'" in errors


def test_validator_rejects_paper_facing_local_only_outputs(tmp_path: Path) -> None:
    """Paper-facing records need durable references for output/ artifacts."""
    registry = tmp_path / "registry.yaml"
    record = tmp_path / "paper.yaml"
    _write_yaml(
        registry,
        """
        schema_version: experiment-registry.v1
        records:
          - paper.yaml
        """,
    )
    _write_yaml(
        record,
        """
        schema_version: experiment-record.v1
        experiment_id: paper-local-only
        issue: 1
        issue_url: https://github.com/ll7/robot_sf_ll7/issues/1
        question: Does this result support a paper claim?
        hypothesis: It might, if durable evidence exists.
        config: configs/example.yaml
        command: uv run python scripts/example.py
        inputs:
          - path: configs/example.yaml
        outputs:
          - path: output/benchmarks/local_only
        expected_artifacts:
          - name: report
            path: output/benchmarks/local_only/report.json
        evidence_grade: observed
        paper_relevance: paper_facing
        status: planned
        """,
    )

    errors = validate_registry(registry)

    assert (
        "paper.yaml: paper-facing record references local-only output/ artifact "
        "without durable_reference: output/benchmarks/local_only"
    ) in errors
    assert (
        "paper.yaml: paper-facing record references local-only output/ artifact "
        "without durable_reference: output/benchmarks/local_only/report.json"
    ) in errors
