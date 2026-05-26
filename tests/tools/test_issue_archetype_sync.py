"""Tests for the issue archetype sync dry-run reporter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from scripts.tools.issue_archetype_sync import (
    build_sync_report,
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
