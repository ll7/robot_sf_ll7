"""Tests for frozen-trace before/after event-ledger reconciliation reports."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from robot_sf.benchmark.frozen_trace_reconciliation import (
    FROZEN_TRACE_RECONCILIATION_SCHEMA,
    build_frozen_trace_reconciliation_report,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "compare_frozen_trace_event_ledgers.py"
)
_SPEC = importlib.util.spec_from_file_location("compare_frozen_trace_event_ledgers", SCRIPT_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def _ledger_row(
    episode_id: str,
    *,
    collision: bool = False,
    near_miss: bool = False,
    planner: str = "demo",
) -> dict:
    """Build a frozen episode row carrying an already-materialized event ledger."""

    return {
        "episode_id": episode_id,
        "scenario_id": "scenario-a",
        "concrete_case_id": episode_id,
        "seed": 7,
        "planner": planner,
        "event_ledger": {
            "schema_version": "EpisodeEventLedger.v1",
            "scenario_id": "scenario-a",
            "concrete_case_id": episode_id,
            "seed": 7,
            "planner": planner,
            "software_commit": "abc123",
            "exact_events": {
                "collision": collision,
                "goal_reached": not collision,
                "timeout": False,
                "invalid_run": False,
            },
            "surrogate_events": {
                "near_miss": near_miss,
                "clearance_breach": False,
                "ttc_breach": False,
                "oscillation": False,
                "late_evasive": False,
                "occlusion_near_miss": False,
            },
            "metric_definitions": {},
            "reconciliation": {"audit_result": "pass"},
            "provenance": {"source": "fixture"},
        },
    }


def test_report_counts_event_deltas_and_affected_artifact_status() -> None:
    """Report old/new event counts, row deltas, and claim/table/figure impact status."""

    old_rows = [
        _ledger_row("episode-1", collision=False, near_miss=True),
        _ledger_row("episode-2", collision=False, near_miss=False),
    ]
    new_rows = [
        _ledger_row("episode-1", collision=True, near_miss=True),
        _ledger_row("episode-2", collision=False, near_miss=False),
    ]
    artifact_manifest = [
        {
            "artifact_id": "claim:collision-free",
            "artifact_type": "claim",
            "consumes_event_fields": ["exact_events.collision"],
        },
        {
            "artifact_id": "table:safety-summary",
            "artifact_type": "table",
            "consumes_event_fields": ["exact_events.collision", "surrogate_events.near_miss"],
        },
        {
            "artifact_id": "figure:near-miss",
            "artifact_type": "figure",
            "consumes_event_fields": ["surrogate_events.near_miss"],
        },
    ]

    report = build_frozen_trace_reconciliation_report(
        old_rows,
        new_rows,
        artifact_manifest=artifact_manifest,
        old_label="0.0.2-before",
        new_label="0.0.2-after",
    )

    assert report["schema_version"] == FROZEN_TRACE_RECONCILIATION_SCHEMA
    assert report["summary"] == {
        "old_label": "0.0.2-before",
        "new_label": "0.0.2-after",
        "old_row_count": 2,
        "new_row_count": 2,
        "changed_row_count": 1,
        "unchanged_row_count": 1,
        "added_row_count": 0,
        "removed_row_count": 0,
    }
    assert report["old_event_counts"]["exact_events.collision"] == 0
    assert report["new_event_counts"]["exact_events.collision"] == 1
    assert report["old_event_counts"]["surrogate_events.near_miss"] == 1
    assert report["new_event_counts"]["surrogate_events.near_miss"] == 1

    assert report["changed_rows"] == [
        {
            "episode_key": "scenario-a|episode-1|7|demo",
            "changed_event_fields": ["exact_events.collision", "exact_events.goal_reached"],
            "old_events": {
                "exact_events.collision": False,
                "exact_events.goal_reached": True,
            },
            "new_events": {
                "exact_events.collision": True,
                "exact_events.goal_reached": False,
            },
        }
    ]
    assert report["unchanged_rows"] == [{"episode_key": "scenario-a|episode-2|7|demo"}]

    statuses = {row["artifact_id"]: row for row in report["affected_artifacts"]}
    assert statuses["claim:collision-free"]["status"] == "affected_reconciliation_required"
    assert statuses["table:safety-summary"]["status"] == "affected_reconciliation_required"
    assert statuses["figure:near-miss"]["status"] == "unchanged_by_event_delta"


def test_cli_writes_frozen_reconciliation_report(tmp_path: Path) -> None:
    """CLI should write the same pure comparison report without running benchmarks."""

    old_path = tmp_path / "old.jsonl"
    new_path = tmp_path / "new.jsonl"
    manifest_path = tmp_path / "artifacts.json"
    out_path = tmp_path / "report.json"
    old_path.write_text(
        json.dumps(_ledger_row("episode-1", collision=False)) + "\n", encoding="utf-8"
    )
    new_path.write_text(
        json.dumps(_ledger_row("episode-1", collision=True)) + "\n", encoding="utf-8"
    )
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "artifact_id": "table:safety-summary",
                    "artifact_type": "table",
                    "consumes_event_fields": ["exact_events.collision"],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--old-ledgers",
            str(old_path),
            "--new-ledgers",
            str(new_path),
            "--artifact-manifest",
            str(manifest_path),
            "--out",
            str(out_path),
        ],
        check=False,
        cwd=SCRIPT_PATH.parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["summary"]["changed_row_count"] == 1
    assert report["affected_artifacts"][0]["status"] == "affected_reconciliation_required"
