"""Tests for the issue archetype sync dry-run reporter and apply mode."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from scripts.tools.issue_archetype_sync import (
    apply_labels,
    build_sync_report,
    check_rate_limit,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


def _body(archetype: str = "workflow", evidence_tier: str = "idea") -> str:
    """Build a minimal issue body with canonical archetype metadata."""
    return f"""## Goal / Problem

Context.

## Archetype Metadata

```yaml
archetype: {archetype}
evidence_tier: {evidence_tier}
linked_policy:
  - docs/context/issue_1512_issue_archetypes.md
```
"""


def _skip_reasons(report: object) -> list[str | None]:
    """Return skip reasons from a report object."""
    return [candidate.reason for candidate in report.skipped_label_candidates]


def test_report_proposes_safe_typed_label_without_mutation_plan() -> None:
    """Safe archetype mirrors should be proposed without any write plan."""
    report = build_sync_report(
        issue_number=1557,
        issue_body=_body("workflow"),
        existing_labels=["workflow"],
        available_labels={"type:workflow"},
    )

    assert report.body_metadata == {
        "archetype": "workflow",
        "evidence_tier": "idea",
        "linked_policy": ["docs/context/issue_1512_issue_archetypes.md"],
    }
    assert report.existing_labels == ["workflow"]
    assert report.proposed_label_additions == ["type:workflow"]
    assert report.mutation_plan == []
    assert report.project_sync_mode == "report-only"
    assert "not_low_risk" in _skip_reasons(report)


def test_report_keeps_evidence_tier_body_only_by_default() -> None:
    """Evidence tiers should not be mirrored to labels by the default report."""
    report = build_sync_report(
        issue_number=1,
        issue_body=_body("analysis", evidence_tier="smoke"),
        existing_labels=[],
        available_labels={"type:analysis", "evidence:smoke"},
    )

    assert report.proposed_label_additions == ["type:analysis"]
    evidence_skips = [
        candidate
        for candidate in report.skipped_label_candidates
        if candidate.source_field == "evidence_tier"
    ]
    assert len(evidence_skips) == 1
    assert evidence_skips[0].reason == "not_low_risk"
    assert "evidence:smoke" not in report.proposed_label_additions


def test_report_fails_closed_on_incomplete_metadata_block() -> None:
    """Incomplete metadata blocks should block all typed-label mirrors.

    This matters because the full issue-body metadata block is authoritative,
    so a missing required key must prevent archetype label mirroring even when
    the parsed archetype and evidence tier look individually valid.
    """

    body = """## Goal / Problem

Context.

## Archetype Metadata

```yaml
archetype: workflow
evidence_tier: smoke
```
"""
    report = build_sync_report(
        issue_number=1564,
        issue_body=body,
        existing_labels=[],
        available_labels={"type:workflow", "evidence:smoke"},
    )

    assert report.metadata_findings == ["Missing archetype metadata keys: linked_policy"]
    assert report.proposed_label_additions == []
    assert any(
        candidate.source_field == "archetype" and candidate.reason == "schema_missing"
        for candidate in report.skipped_label_candidates
    )
    evidence_skips = [
        candidate
        for candidate in report.skipped_label_candidates
        if candidate.source_field == "evidence_tier"
    ]
    assert len(evidence_skips) == 1
    assert evidence_skips[0].reason == "not_low_risk"
    assert "evidence:smoke" not in report.proposed_label_additions


def test_report_fails_closed_on_invalid_metadata_values() -> None:
    """Parsed blocks with invalid canonical values should not propose labels."""
    body = """## Archetype Metadata

```yaml
archetype: workflow-ish
evidence_tier: maybe
linked_policy:
  - docs/context/issue_1512_issue_archetypes.md
```
"""
    report = build_sync_report(
        issue_number=1564,
        issue_body=body,
        existing_labels=[],
        available_labels={"type:workflow", "evidence:proposal"},
    )

    assert any("Invalid 'archetype' value" in finding for finding in report.metadata_findings)
    assert any("Invalid 'evidence_tier' value" in finding for finding in report.metadata_findings)
    assert report.proposed_label_additions == []
    assert any(
        candidate.source_field == "archetype" and candidate.reason == "schema_missing"
        for candidate in report.skipped_label_candidates
    )
    assert any(
        candidate.source_field == "evidence_tier" and candidate.reason == "not_low_risk"
        for candidate in report.skipped_label_candidates
    )


def test_report_skips_unmapped_archetype_as_not_low_risk() -> None:
    """Archetypes without exact typed-label mirrors should be skipped."""
    report = build_sync_report(
        issue_number=1,
        issue_body=_body("preflight"),
        existing_labels=[],
        available_labels={"type:workflow"},
    )

    assert report.proposed_label_additions == []
    assert any(
        candidate.source_field == "archetype" and candidate.reason == "not_low_risk"
        for candidate in report.skipped_label_candidates
    )


def test_report_skips_missing_repo_label() -> None:
    """The dry run should not propose labels that are not present in the repo."""
    report = build_sync_report(
        issue_number=1,
        issue_body=_body("docs"),
        existing_labels=[],
        available_labels={"type:workflow"},
    )

    assert report.proposed_label_additions == []
    assert any(candidate.reason == "label_missing" for candidate in report.skipped_label_candidates)


def test_report_skips_ambiguous_existing_typed_label() -> None:
    """Existing conflicting type labels should be reported instead of overwritten."""
    report = build_sync_report(
        issue_number=1,
        issue_body=_body("workflow"),
        existing_labels=["type:docs"],
        available_labels={"type:workflow", "type:docs"},
    )

    assert report.proposed_label_additions == []
    assert any(candidate.reason == "ambiguous" for candidate in report.skipped_label_candidates)


@pytest.mark.parametrize(
    ("body", "expected_reason"),
    [
        ("## Goal / Problem\n\nNo metadata.\n", "schema_missing"),
        (
            "## Archetype Metadata\n\n```yaml\narchetype: [workflow\n```\n",
            "malformed",
        ),
    ],
)
def test_report_surfaces_missing_and_malformed_metadata(
    body: str,
    expected_reason: str,
) -> None:
    """Invalid metadata blocks should become explicit skipped candidates."""
    report = build_sync_report(
        issue_number=1,
        issue_body=body,
        existing_labels=[],
        available_labels={"type:workflow"},
    )

    assert report.body_metadata is None
    assert report.metadata_findings
    assert any(candidate.reason == expected_reason for candidate in report.skipped_label_candidates)
    assert report.mutation_plan == []


@patch("scripts.tools.issue_archetype_sync.fetch_repo_label_names")
@patch("scripts.tools.issue_archetype_sync.fetch_issue_payload")
def test_cli_report_uses_read_only_fetches_and_no_mutating_calls(
    fetch_issue_payload,
    fetch_repo_label_names,
    capsys,
) -> None:
    """The default CLI path should only fetch data and should return an empty mutation plan."""
    fetch_issue_payload.return_value = {
        "body": _body("benchmark-campaign"),
        "labels": [{"name": "benchmark"}],
    }
    fetch_repo_label_names.return_value = {"type:benchmark"}

    exit_code = main(["report", "--issue-number", "1557", "--dry-run"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    fetch_issue_payload.assert_called_once_with("ll7/robot_sf_ll7", 1557)
    fetch_repo_label_names.assert_called_once_with("ll7/robot_sf_ll7")
    assert payload["proposed_label_additions"] == ["type:benchmark"]
    assert payload["mutation_plan"] == []
    assert payload["project_sync_mode"] == "report-only"


def test_cli_report_accepts_offline_body_file(tmp_path: Path, capsys) -> None:
    """Offline body-file mode keeps tests and manual inspection independent from GitHub."""
    body_path = tmp_path / "issue.md"
    labels_path = tmp_path / "labels.json"
    body_path.write_text(_body("training-campaign"), encoding="utf-8")
    labels_path.write_text(json.dumps(["training"]), encoding="utf-8")

    exit_code = main(
        [
            "report",
            "--issue-number",
            "1",
            "--body-file",
            str(body_path),
            "--labels-json",
            str(labels_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["proposed_label_additions"] == ["type:training"]


# -- apply-mode tests ---------------------------------------------------------


def _apply_body_file_args(
    tmp_path: Path,
    issue_number: int = 1588,
    archetype: str = "workflow",
    evidence_tier: str = "idea",
    existing: list[str] | None = None,
    confirm: bool = True,
) -> tuple[list[str], str]:
    """Write body + labels files in tmp_path and return apply CLI argv."""
    if existing is None:
        existing = []
    body_path = tmp_path / "issue.md"
    labels_path = tmp_path / "labels.json"
    body_path.write_text(_body(archetype, evidence_tier=evidence_tier), encoding="utf-8")
    labels_path.write_text(json.dumps(existing), encoding="utf-8")
    argv = [
        "apply",
        "--issue-number",
        str(issue_number),
        "--body-file",
        str(body_path),
        "--labels-json",
        str(labels_path),
    ]
    if confirm:
        argv.append("--confirm-apply-labels")
    return argv, str(body_path)


def _apply_github_args(issue_number: int = 1588, confirm: bool = True) -> list[str]:
    """Return apply CLI argv for the live GitHub metadata path."""
    argv = ["apply", "--issue-number", str(issue_number)]
    if confirm:
        argv.append("--confirm-apply-labels")
    return argv


def _report(
    *,
    issue_number: int = 1588,
    archetype: str = "workflow",
    evidence_tier: str = "idea",
    existing_labels: list[str] | None = None,
) -> object:
    """Build an in-memory report for mocked apply-mode tests."""
    return build_sync_report(
        issue_number=issue_number,
        issue_body=_body(archetype, evidence_tier=evidence_tier),
        existing_labels=existing_labels or [],
        available_labels={"type:workflow", "type:analysis", "evidence:smoke"},
    )


@patch("scripts.tools.issue_archetype_sync.apply_labels")
@patch("scripts.tools.issue_archetype_sync.check_rate_limit")
@patch("scripts.tools.issue_archetype_sync.build_report_from_github")
def test_apply_without_confirm_refuses_mutation(
    mock_build_report_from_github,
    mock_check_rate_limit,
    mock_apply_labels,
    capsys,
) -> None:
    """apply without --confirm-apply-labels prints report but never mutates."""
    mock_build_report_from_github.return_value = _report()

    exit_code = main(_apply_github_args(confirm=False))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Re-run with --confirm-apply-labels" in captured.out
    mock_check_rate_limit.assert_not_called()
    mock_apply_labels.assert_not_called()


@patch("scripts.tools.issue_archetype_sync.apply_labels")
@patch("scripts.tools.issue_archetype_sync.check_rate_limit")
@patch("scripts.tools.issue_archetype_sync.build_report_from_github")
def test_confirmed_apply_calls_mutation_with_proposed_labels(
    mock_build_report_from_github,
    mock_check_rate_limit,
    mock_apply_labels,
    capsys,
) -> None:
    """Confirmed apply should call check_rate_limit then apply_labels with the proposed label."""
    mock_build_report_from_github.return_value = _report()

    exit_code = main(_apply_github_args(confirm=True))
    captured = capsys.readouterr()

    assert exit_code == 0
    mock_check_rate_limit.assert_called_once_with()
    mock_apply_labels.assert_called_once_with("ll7/robot_sf_ll7", 1588, ["type:workflow"])
    payload_text = captured.out[: captured.out.index("\nApplied labels")]
    payload = json.loads(payload_text)
    assert payload["dry_run"] is False
    assert "label writes" in payload["rate_limit_guidance"]
    assert payload["mutation_plan"] == [
        {
            "api": "POST repos/ll7/robot_sf_ll7/issues/1588/labels",
            "issue_number": 1588,
            "labels": ["type:workflow"],
            "operation": "add_labels",
            "repo": "ll7/robot_sf_ll7",
        }
    ]
    assert "Applied labels to issue" in captured.out


@patch("scripts.tools.issue_archetype_sync.apply_labels")
@patch("scripts.tools.issue_archetype_sync.check_rate_limit")
@patch("scripts.tools.issue_archetype_sync.build_report_from_github")
def test_apply_noop_with_no_proposed_labels_skips_mutation(
    mock_build_report_from_github,
    mock_check_rate_limit,
    mock_apply_labels,
    capsys,
) -> None:
    """When proposed_label_additions is empty even after confirm, skip mutation."""
    mock_build_report_from_github.return_value = _report(existing_labels=["type:workflow"])

    exit_code = main(_apply_github_args(confirm=True))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No proposed label additions" in captured.out
    payload_text = captured.out[: captured.out.index("\nNo proposed label additions")]
    payload = json.loads(payload_text)
    assert payload["dry_run"] is False
    assert payload["mutation_plan"] == []
    mock_check_rate_limit.assert_not_called()
    mock_apply_labels.assert_not_called()


@patch("scripts.tools.issue_archetype_sync.apply_labels")
@patch("scripts.tools.issue_archetype_sync.check_rate_limit")
@patch("scripts.tools.issue_archetype_sync.build_report_from_github")
def test_confirmed_apply_noops_for_incomplete_metadata(
    mock_build_report_from_github,
    mock_check_rate_limit,
    mock_apply_labels,
    capsys,
) -> None:
    """Confirmed apply should not mutate when metadata findings fail closed."""
    body = """## Archetype Metadata

```yaml
archetype: workflow
evidence_tier: smoke
```
"""
    mock_build_report_from_github.return_value = build_sync_report(
        issue_number=1564,
        issue_body=body,
        existing_labels=[],
        available_labels={"type:workflow"},
    )

    exit_code = main(_apply_github_args(issue_number=1564, confirm=True))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "No proposed label additions" in captured.out
    payload_text = captured.out[: captured.out.index("\nNo proposed label additions")]
    payload = json.loads(payload_text)
    assert payload["metadata_findings"] == ["Missing archetype metadata keys: linked_policy"]
    assert payload["proposed_label_additions"] == []
    assert payload["mutation_plan"] == []
    mock_check_rate_limit.assert_not_called()
    mock_apply_labels.assert_not_called()


@patch("scripts.tools.issue_archetype_sync.apply_labels")
@patch("scripts.tools.issue_archetype_sync.check_rate_limit")
@patch("scripts.tools.issue_archetype_sync.build_report_from_github")
def test_apply_with_evidence_tier_never_includes_tier_label(
    mock_build_report_from_github,
    mock_check_rate_limit,
    mock_apply_labels,
) -> None:
    """Evidence tier is always skipped; confirmed apply must never propose or apply it."""
    mock_build_report_from_github.return_value = _report(evidence_tier="smoke")

    exit_code = main(_apply_github_args(confirm=True))

    assert exit_code == 0
    mock_apply_labels.assert_called_once()
    called_labels = mock_apply_labels.call_args[0][2]
    assert "type:workflow" in called_labels
    for label in called_labels:
        assert "evidence" not in label.lower()


@patch("scripts.tools.issue_archetype_sync.apply_labels")
@patch("scripts.tools.issue_archetype_sync.check_rate_limit")
@patch("scripts.tools.issue_archetype_sync.build_report_from_github")
def test_low_rate_limit_fails_closed(
    mock_build_report_from_github,
    mock_check_rate_limit,
    mock_apply_labels,
) -> None:
    """When core rate-limit remaining is too low, mutation should fail with RuntimeError."""
    mock_build_report_from_github.return_value = _report()
    mock_check_rate_limit.side_effect = RuntimeError("core remaining too low")

    with pytest.raises(RuntimeError, match="core remaining too low"):
        main(_apply_github_args(confirm=True))

    mock_apply_labels.assert_not_called()


def test_apply_rejects_offline_body_file_arguments(tmp_path: Path) -> None:
    """Apply mode should not accept local-only issue metadata arguments."""
    argv, _ = _apply_body_file_args(tmp_path, confirm=True)

    with pytest.raises(SystemExit) as exc_info:
        main(argv)
    assert exc_info.value.code == 2


def test_report_subcommand_does_not_mutate(tmp_path, capsys) -> None:
    """The report subcommand must never trigger mutation even in apply-like scenarios."""
    body_path = tmp_path / "issue.md"
    labels_path = tmp_path / "labels.json"
    body_path.write_text(_body("workflow"), encoding="utf-8")
    labels_path.write_text(json.dumps([]), encoding="utf-8")

    exit_code = main(
        [
            "report",
            "--issue-number",
            "1",
            "--body-file",
            str(body_path),
            "--labels-json",
            str(labels_path),
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["proposed_label_additions"] == ["type:workflow"]
    assert payload["mutation_plan"] == []
    assert payload["project_sync_mode"] == "report-only"
    assert "Re-run with --confirm-apply-labels" not in captured.out


def test_check_rate_limit_fails_on_low_remaining() -> None:
    """check_rate_limit should raise when core.remaining is below the floor."""
    stub_payload = {"resources": {"core": {"limit": 5000, "remaining": 2, "reset": 9999999999}}}
    with patch("scripts.tools.issue_archetype_sync._run_gh_json", return_value=stub_payload):
        with pytest.raises(RuntimeError, match="below the safe floor"):
            check_rate_limit()


def test_apply_labels_constructs_correct_rest_post() -> None:
    """apply_labels must POST the expected labels JSON to the issue labels endpoint."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = json.dumps([{"name": "type:docs"}])
        apply_labels("ll7/robot_sf_ll7", 1, ["type:docs"])

    mock_run.assert_called_once()
    call_args, call_kwargs = mock_run.call_args
    assert "POST" in call_args[0]
    assert "repos/ll7/robot_sf_ll7/issues/1/labels" in call_args[0][4]
    assert call_kwargs["input"] == json.dumps({"labels": ["type:docs"]})
