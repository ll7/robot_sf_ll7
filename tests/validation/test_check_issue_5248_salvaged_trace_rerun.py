"""Tests for the issue #5248 salvaged trace-capable h600 registration checker."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.validation.check_issue_5248_salvaged_trace_rerun import (
    BLOCKED_STATUS,
    READY_STATUS,
    _load_trace_contract,
    build_registration_receipt,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION_CONFIG = (
    REPO_ROOT / "configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml"
)


def _write_campaign(
    root: Path,
    *,
    episode_count: int = 3,
    completed: bool = True,
    trace_labeled_rows: int = 2,
    omit_field: str | None = None,
) -> Path:
    """Write the smallest camera-ready campaign fixture accepted by the checker."""

    reports = root / "reports"
    reports.mkdir(parents=True)
    (reports / "campaign_summary.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "total_episodes": episode_count,
                    "campaign_execution_status": "completed" if completed else "failed",
                    "status": "accepted_unavailable_only",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fields = [
        "mechanism_schema_version",
        "mechanism_label",
        "mechanism_confidence",
        "mechanism_evidence_mode",
        "mechanism_evidence_uri",
        "mechanism_case_id",
        "mechanism_caveat",
    ]
    if omit_field is not None:
        fields.remove(omit_field)
    with (reports / "seed_episode_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(episode_count):
            trace_labeled = index < trace_labeled_rows
            row = {
                "mechanism_schema_version": "failure_mechanism_taxonomy.v1",
                "mechanism_label": "static_deadlock_or_local_minimum"
                if trace_labeled
                else "not_derivable",
                "mechanism_confidence": "observed_mechanism" if trace_labeled else "unknown",
                "mechanism_evidence_mode": "paired_trace" if trace_labeled else "unknown",
                "mechanism_evidence_uri": f"trace://fixture/{index}" if trace_labeled else "",
                "mechanism_case_id": f"fixture-{index}",
                "mechanism_caveat": "",
            }
            writer.writerow({field: row[field] for field in fields})
    return root


def _receipt(root: Path, **kwargs: object) -> dict:
    """Build a deterministic fixture receipt."""

    return build_registration_receipt(
        campaign_root=root,
        job_id="fixture-job",
        expected_total_episodes=3,
        preregistration_config=PREREGISTRATION_CONFIG,
        generated_at="2026-07-11T00:00:00+00:00",
        **kwargs,
    )


def test_complete_campaign_with_trace_rows_is_ready(tmp_path: Path) -> None:
    """A completed campaign above the registered trace-label floor is ready."""

    receipt = _receipt(_write_campaign(tmp_path / "campaign"))

    assert receipt["status"] == READY_STATUS
    assert receipt["blockers"] == []
    assert receipt["campaign"]["episode_row_count"] == 3
    assert receipt["trace_labels"]["trace_labeled_rows"] == 2
    assert receipt["trace_labels"]["trace_labeled_fraction"] == 2 / 3
    assert set(receipt["source_files"]) == {
        "reports/campaign_summary.json",
        "reports/seed_episode_rows.csv",
    }


def test_incomplete_campaign_is_blocked_without_promoting_status(tmp_path: Path) -> None:
    """A non-completed execution state cannot be salvaged by trace rows alone."""

    receipt = _receipt(_write_campaign(tmp_path / "campaign", completed=False))

    assert receipt["status"] == BLOCKED_STATUS
    assert any("campaign_execution_status" in blocker for blocker in receipt["blockers"])


def test_episode_total_mismatch_is_blocked(tmp_path: Path) -> None:
    """A completed campaign cannot register when summary and requested totals disagree."""

    receipt = _receipt(_write_campaign(tmp_path / "campaign", episode_count=2))

    assert receipt["status"] == BLOCKED_STATUS
    assert any("campaign.total_episodes" in blocker for blocker in receipt["blockers"])
    assert any("seed_episode_rows.csv" in blocker for blocker in receipt["blockers"])


def test_insufficient_trace_labels_are_blocked(tmp_path: Path) -> None:
    """Rows below the preregistered trace-label threshold fail closed."""

    receipt = _receipt(_write_campaign(tmp_path / "campaign", trace_labeled_rows=1))

    assert receipt["status"] == BLOCKED_STATUS
    assert any("trace-verified labeled fraction" in blocker for blocker in receipt["blockers"])


def test_missing_mechanism_field_is_blocked(tmp_path: Path) -> None:
    """The registration cannot accept an episode table missing taxonomy fields."""

    receipt = _receipt(_write_campaign(tmp_path / "campaign", omit_field="mechanism_evidence_uri"))

    assert receipt["status"] == BLOCKED_STATUS
    assert any("mechanism_evidence_uri" in blocker for blocker in receipt["blockers"])


def test_trace_contract_rejects_boolean_labeled_fraction(tmp_path: Path) -> None:
    """A boolean is not a numeric preregistration coverage threshold."""
    config = tmp_path / "preregistration.yaml"
    config.write_text(
        PREREGISTRATION_CONFIG.read_text(encoding="utf-8").replace(
            "min_trace_verified_labeled_fraction: 0.5",
            "min_trace_verified_labeled_fraction: true",
        ),
        encoding="utf-8",
    )

    try:
        _load_trace_contract(config)
    except ValueError as exc:
        assert "minimum trace-labeled fraction" in str(exc)
    else:
        raise AssertionError("expected boolean threshold to fail closed")


def test_cli_writes_blocked_receipt_and_nonzero_exit(tmp_path: Path, capsys) -> None:
    """The CLI leaves a reviewable receipt even when source registration is blocked."""

    campaign = _write_campaign(tmp_path / "campaign", trace_labeled_rows=0)
    output_dir = tmp_path / "receipt"

    exit_code = main(
        [
            "--campaign-root",
            str(campaign),
            "--job-id",
            "fixture-job",
            "--expected-total-episodes",
            "3",
            "--preregistration-config",
            str(PREREGISTRATION_CONFIG),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-07-11T00:00:00+00:00",
        ]
    )

    assert exit_code == 2
    assert "status: blocked_campaign_registration" in capsys.readouterr().out
    payload = json.loads((output_dir / "registration.json").read_text(encoding="utf-8"))
    assert payload["status"] == BLOCKED_STATUS
    assert (output_dir / "registration.md").is_file()
