"""Tests for the question-first experiment registry validator."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from scripts.tools.validate_experiment_registry import (
    _load_issue_state_snapshot,
    build_control_plane_report,
    main,
    validate_registry,
)


def _write_yaml(path: Path, text: str) -> None:
    """Write a YAML fixture with normalized leading whitespace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).strip() + "\n", encoding="utf-8")


def test_checked_in_experiment_registry_validates() -> None:
    """The tracked experiment registry should satisfy the workflow contract."""
    repo_root = Path(__file__).parents[2]
    errors = validate_registry(repo_root / "experiments/registry.yaml")

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


def test_validator_accepts_experiment_record_v2_state(tmp_path: Path) -> None:
    """New records may use the v2 authoritative state field instead of legacy status."""
    registry = tmp_path / "experiments" / "registry.yaml"
    record = registry.parent / "v2.yaml"
    _write_yaml(
        registry,
        """
        schema_version: experiment-registry.v1
        records:
          - v2.yaml
        """,
    )
    _write_yaml(
        record,
        """
        schema_version: experiment-record.v2
        experiment_id: v2-protocol
        issue: 3073
        issue_url: https://github.com/ll7/robot_sf_ll7/issues/3073
        question: What state owns this protocol?
        hypothesis: A single state field prevents drift.
        config:
          - experiments/registry.yaml
        command: uv run python scripts/tools/validate_experiment_registry.py
        inputs:
          - path: experiments/registry.yaml
        outputs:
          - path: output/experiments/v2-protocol
        expected_artifacts:
          - name: report
            path: output/experiments/v2-protocol/report.json
        evidence_grade: proposal
        paper_relevance: exploratory
        state: protocol_frozen
        """,
    )

    assert validate_registry(registry) == []


def test_validator_rejects_unknown_experiment_record_v2_state(tmp_path: Path) -> None:
    """The control plane should reject ad hoc state tokens."""
    registry = tmp_path / "experiments" / "registry.yaml"
    record = registry.parent / "bad.yaml"
    _write_yaml(
        registry,
        """
        schema_version: experiment-registry.v1
        records:
          - bad.yaml
        """,
    )
    _write_yaml(
        record,
        """
        schema_version: experiment-record.v2
        experiment_id: bad-state
        issue: 3073
        issue_url: https://github.com/ll7/robot_sf_ll7/issues/3073
        question: Bad state?
        hypothesis: It should fail.
        config:
          - experiments/registry.yaml
        command: uv run true
        inputs:
          - path: experiments/registry.yaml
        outputs:
          - path: output/experiments/bad-state
        expected_artifacts:
          - name: report
            path: output/experiments/bad-state/report.json
        evidence_grade: proposal
        paper_relevance: exploratory
        state: kind_of_running
        """,
    )

    errors = validate_registry(registry)

    assert "bad.yaml: state must be one of" in errors[0]


def test_control_plane_report_detects_research_state_drift(tmp_path: Path) -> None:
    """The dry-run report should catch the sprint's known drift classes."""
    registry = tmp_path / "experiments" / "registry.yaml"
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "exists.yaml").write_text("ok: true\n", encoding="utf-8")
    _write_yaml(
        registry,
        """
        schema_version: experiment-registry.v1
        records:
          - closed_nonterminal.yaml
          - blocked_on_closed.yaml
          - label_disagreement.yaml
        """,
    )
    _write_yaml(
        registry.parent / "closed_nonterminal.yaml",
        """
        schema_version: experiment-record.v2
        experiment_id: closed-nonterminal
        issue: 1236
        issue_url: https://github.com/ll7/robot_sf_ll7/issues/1236
        question: Is a closed issue still active?
        hypothesis: It should be terminal or superseded.
        config:
          - configs/exists.yaml
        command: uv run true
        inputs:
          - path: configs/exists.yaml
        outputs:
          - path: output/experiments/closed
        expected_artifacts:
          - name: report
            path: output/experiments/closed/report.json
        evidence_grade: proposal
        paper_relevance: exploratory
        state: idea
        """,
    )
    _write_yaml(
        registry.parent / "blocked_on_closed.yaml",
        """
        schema_version: experiment-record.v1
        experiment_id: blocked-on-closed
        issue: 1151
        issue_url: https://github.com/ll7/robot_sf_ll7/issues/1151
        question: Is the blocker still active?
        hypothesis: Closed blockers should not keep cards blocked.
        config:
          - configs/exists.yaml
        command: uv run true
        inputs:
          - path: configs/exists.yaml
        outputs:
          - path: output/experiments/blocked
        expected_artifacts:
          - name: report
            path: output/experiments/blocked/report.json
        evidence_grade: proposal
        paper_relevance: exploratory
        status: blocked_on_issue_1219
        """,
    )
    _write_yaml(
        registry.parent / "label_disagreement.yaml",
        """
        schema_version: experiment-record.v2
        experiment_id: label-disagreement
        issue: 1475
        issue_url: https://github.com/ll7/robot_sf_ll7/issues/1475
        question: Do labels agree with the card?
        hypothesis: They should derive from the card.
        config:
          - configs/missing.yaml
        command: uv run true
        inputs:
          - path: configs/exists.yaml
        outputs:
          - path: output/experiments/label
        expected_artifacts:
          - name: pending_policy
            path: wandb-artifact://pending/issue-1475/policy:smoke
            durable_reference_required: true
        evidence_grade: proposal
        paper_relevance: exploratory
        state: implementation_ready
        """,
    )

    report = build_control_plane_report(
        registry,
        issue_states={1236: "CLOSED", 1219: "CLOSED", 1475: "OPEN"},
        issue_labels={1475: ["state:running"]},
    )

    kinds = {finding["kind"] for finding in report["findings"]}
    assert "closed_issue_with_nonterminal_record" in kinds
    assert "closed_blocker_with_blocked_record" in kinds
    assert "derived_issue_label_update" in kinds
    assert "missing_config_or_input_path" in kinds
    assert "pending_artifact_alias" in kinds
    assert "missing_required_durable_reference" in kinds
    derived_updates = [
        finding for finding in report["findings"] if finding["kind"] == "derived_issue_label_update"
    ]
    assert report["derived_update_count"] == len(derived_updates)
    assert derived_updates == [
        {
            "kind": "derived_issue_label_update",
            "severity": "warning",
            "record": "label_disagreement.yaml",
            "issue": 1475,
            "message": (
                "issue #1475 state labels ['state:running'] should derive from "
                "record state 'implementation_ready' as ['state:ready']"
            ),
            "expected_state_label": "state:ready",
            "current_state_labels": ["state:running"],
            "labels_to_add": ["state:ready"],
            "labels_to_remove": ["state:running"],
            "dry_run_only": True,
        }
    ]


def test_control_plane_report_derives_terminal_state_label_removal(tmp_path: Path) -> None:
    """Released records should suggest removing state labels rather than mutating GitHub."""
    registry = tmp_path / "experiments" / "registry.yaml"
    _write_yaml(
        registry,
        """
        schema_version: experiment-registry.v1
        records:
          - released.yaml
        """,
    )
    _write_yaml(
        registry.parent / "released.yaml",
        """
        schema_version: experiment-record.v2
        experiment_id: released-record
        issue: 2001
        issue_url: https://github.com/ll7/robot_sf_ll7/issues/2001
        question: Is this release state authoritative?
        hypothesis: The dry-run report should remove live state labels.
        config:
          - configs/exists.yaml
        command: uv run true
        inputs:
          - path: configs/exists.yaml
        outputs:
          - path: output/experiments/released
        expected_artifacts:
          - name: report
            path: output/experiments/released/report.json
        evidence_grade: proposal
        paper_relevance: exploratory
        state: released
        """,
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "exists.yaml").write_text("ok: true\n", encoding="utf-8")

    report = build_control_plane_report(
        registry,
        issue_states={2001: "OPEN"},
        issue_labels={2001: ["state:running", "type:analysis"]},
    )

    assert report["derived_update_count"] == 1
    update = report["findings"][0]
    assert update["kind"] == "derived_issue_label_update"
    assert update["expected_state_label"] is None
    assert update["current_state_labels"] == ["state:running"]
    assert update["labels_to_add"] == []
    assert update["labels_to_remove"] == ["state:running"]
    assert update["dry_run_only"] is True


def test_validator_checks_scalar_config_paths(tmp_path: Path) -> None:
    """Scalar config paths should be validated like list-valued paths."""
    registry = tmp_path / "registry.yaml"
    _write_yaml(
        registry,
        """
        schema_version: experiment-registry.v1
        records:
          - scalar_config.yaml
        """,
    )
    _write_yaml(
        registry.parent / "scalar_config.yaml",
        """
        schema_version: experiment-record.v1
        experiment_id: scalar-config
        issue: 1475
        issue_url: https://github.com/ll7/robot_sf_ll7/issues/1475
        question: Does scalar config validation catch missing files?
        hypothesis: It should catch missing files.
        config: configs/missing.yaml
        command: uv run true
        inputs:
          - path: configs/exists.yaml
        outputs:
          - path: output/experiments/scalar
        expected_artifacts:
          - name: report
            path: output/experiments/scalar/report.json
        evidence_grade: proposal
        paper_relevance: exploratory
        status: implementation_ready
        """,
    )
    (registry.parent / "configs").mkdir()
    (registry.parent / "configs" / "exists.yaml").write_text("ok: true\n", encoding="utf-8")

    report = build_control_plane_report(registry)
    messages = {finding["message"] for finding in report["findings"]}

    assert "referenced path does not exist: configs/missing.yaml" in messages


def test_issue_state_snapshot_filters_missing_label_names(tmp_path: Path) -> None:
    """Issue snapshot labels should ignore null or name-less values."""
    snapshot = tmp_path / "issues.json"
    snapshot.write_text(
        """
        {
          "issues": [
            {
              "number": 1475,
              "state": "OPEN",
              "labels": [
                {"name": "state:running"},
                {"color": "ffffff"},
                null,
                "type:analysis"
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    states, labels = _load_issue_state_snapshot(snapshot)

    assert states == {1475: "OPEN"}
    assert labels == {1475: ["state:running", "type:analysis"]}


def test_issue_state_snapshot_errors_are_reported_by_cli(tmp_path: Path) -> None:
    """Invalid issue-state snapshots should fail closed instead of crashing."""
    registry = tmp_path / "registry.yaml"
    report = tmp_path / "report.json"
    issue_state = tmp_path / "issue_state.json"
    _write_yaml(
        registry,
        """
        schema_version: experiment-registry.v1
        records: []
        """,
    )
    issue_state.write_text("{not json", encoding="utf-8")

    exit_code = main(
        [
            str(registry),
            "--issue-state-json",
            str(issue_state),
            "--control-plane-report-json",
            str(report),
        ]
    )

    assert exit_code == 1
    assert report.is_file()
