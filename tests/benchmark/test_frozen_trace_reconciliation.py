"""Tests for frozen-trace before/after event-ledger reconciliation reports."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from robot_sf.benchmark.frozen_trace_reconciliation import (
    FROZEN_TRACE_MISSING_EXPORT_SCHEMA,
    FROZEN_TRACE_RECONCILIATION_SCHEMA,
    build_frozen_trace_reconciliation_report,
    build_missing_frozen_trace_export_report,
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


def _raw_ledger_row(episode_id: str, *, collision: bool = False) -> dict:
    """Build a raw durable EpisodeEventLedger.v1 row."""

    return _ledger_row(episode_id, collision=collision)["event_ledger"]


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
    assert statuses["claim:collision-free"]["affected_changed_row_count"] == 1
    assert statuses["claim:collision-free"]["affected_added_row_count"] == 0
    assert statuses["claim:collision-free"]["affected_removed_row_count"] == 0
    assert statuses["table:safety-summary"]["status"] == "affected_reconciliation_required"
    assert statuses["figure:near-miss"]["status"] == "unchanged_by_event_delta"
    assert report["affected_artifact_summary"] == {
        "total_artifacts": 3,
        "by_type": {
            "claim": {"affected_reconciliation_required": 1},
            "figure": {"unchanged_by_event_delta": 1},
            "table": {"affected_reconciliation_required": 1},
        },
        "by_status": {
            "affected_reconciliation_required": 2,
            "unchanged_by_event_delta": 1,
        },
    }


def test_raw_durable_event_ledger_rows_are_comparable() -> None:
    """Frozen durable exports can carry raw EpisodeEventLedger.v1 rows."""

    report = build_frozen_trace_reconciliation_report(
        [_raw_ledger_row("episode-1")],
        [_raw_ledger_row("episode-1", collision=True)],
    )

    assert report["summary"]["old_row_count"] == 1
    assert report["summary"]["new_row_count"] == 1
    assert report["summary"]["changed_row_count"] == 1
    assert report["old_event_counts"]["exact_events.collision"] == 0
    assert report["new_event_counts"]["exact_events.collision"] == 1


def test_metric_semantics_export_without_ledger_payload_is_not_comparable() -> None:
    """Metric-semantics exports name the ledger but do not carry event booleans."""

    semantics_export = {
        "schema_version": "event_ledger_reconciliation_export.v1",
        "episode_id": "episode-1",
        "scenario_id": "scenario-a",
        "seed": 7,
        "planner": "demo",
        "software_commit": "abc123",
        "event_ledger_schema_version": "EpisodeEventLedger.v1",
        "reconciliation_table": {"rows": []},
    }

    with pytest.raises(ValueError, match="metric-semantics exports"):
        build_frozen_trace_reconciliation_report([semantics_export], [semantics_export])


def test_missing_export_report_marks_artifacts_not_evaluable() -> None:
    """Missing durable exports produce a blocker report without inventing row counts."""

    report = build_missing_frozen_trace_export_report(
        missing_exports=[
            {
                "label": "0.0.2-before",
                "path": "docs/context/evidence/missing-before.jsonl",
                "required_contract": "EpisodeEventLedger.v1 exact/surrogate event JSONL",
            },
            {
                "label": "0.0.2-after",
                "path": "docs/context/evidence/missing-after.jsonl",
                "required_contract": "EpisodeEventLedger.v1 exact/surrogate event JSONL",
            },
        ],
        artifact_manifest=[
            {
                "artifact_id": "table:safety-summary",
                "artifact_type": "table",
                "consumes_event_fields": ["exact_events.collision"],
            }
        ],
        old_label="0.0.2-before",
        new_label="0.0.2-after",
    )

    assert report["schema_version"] == FROZEN_TRACE_MISSING_EXPORT_SCHEMA
    assert report["status"] == "blocked_missing_durable_event_ledger_exports"
    assert report["summary"] == {
        "old_label": "0.0.2-before",
        "new_label": "0.0.2-after",
        "old_row_count": None,
        "new_row_count": None,
        "changed_row_count": None,
        "unchanged_row_count": None,
        "added_row_count": None,
        "removed_row_count": None,
        "missing_export_count": 2,
    }
    assert report["affected_artifacts"] == [
        {
            "artifact_id": "table:safety-summary",
            "artifact_type": "table",
            "status": "not_evaluable_missing_event_ledger_export",
            "consumes_event_fields": ["exact_events.collision"],
            "affected_event_fields": [],
            "affected_changed_row_count": None,
            "affected_added_row_count": None,
            "affected_removed_row_count": None,
        }
    ]
    assert report["affected_artifact_summary"] == {
        "total_artifacts": 1,
        "by_type": {"table": {"not_evaluable_missing_event_ledger_export": 1}},
        "by_status": {"not_evaluable_missing_event_ledger_export": 1},
    }
    assert report["semantics"]["claim_promotion"] == "none"


def test_added_and_removed_rows_mark_consuming_artifacts_affected() -> None:
    """Added or removed frozen rows still require artifact reconciliation."""

    report = build_frozen_trace_reconciliation_report(
        [_ledger_row("removed-episode", collision=True)],
        [_ledger_row("added-episode", collision=True)],
        artifact_manifest=[
            {
                "artifact_id": "claim:collision-count",
                "artifact_type": "claim",
                "consumes_event_fields": ["exact_events.collision"],
            }
        ],
    )

    assert report["summary"]["changed_row_count"] == 0
    assert report["summary"]["added_row_count"] == 1
    assert report["summary"]["removed_row_count"] == 1
    assert report["added_rows"] == [
        {
            "episode_key": "scenario-a|added-episode|7|demo",
            "event_fields": [
                "exact_events.collision",
                "exact_events.goal_reached",
                "exact_events.invalid_run",
                "exact_events.timeout",
                "surrogate_events.clearance_breach",
                "surrogate_events.late_evasive",
                "surrogate_events.near_miss",
                "surrogate_events.occlusion_near_miss",
                "surrogate_events.oscillation",
                "surrogate_events.ttc_breach",
            ],
        }
    ]
    assert report["removed_rows"] == [
        {
            "episode_key": "scenario-a|removed-episode|7|demo",
            "event_fields": [
                "exact_events.collision",
                "exact_events.goal_reached",
                "exact_events.invalid_run",
                "exact_events.timeout",
                "surrogate_events.clearance_breach",
                "surrogate_events.late_evasive",
                "surrogate_events.near_miss",
                "surrogate_events.occlusion_near_miss",
                "surrogate_events.oscillation",
                "surrogate_events.ttc_breach",
            ],
        }
    ]
    assert report["affected_artifacts"] == [
        {
            "artifact_id": "claim:collision-count",
            "artifact_type": "claim",
            "status": "affected_reconciliation_required",
            "consumes_event_fields": ["exact_events.collision"],
            "affected_event_fields": ["exact_events.collision"],
            "affected_changed_row_count": 0,
            "affected_added_row_count": 1,
            "affected_removed_row_count": 1,
        }
    ]
    assert report["affected_artifact_summary"] == {
        "total_artifacts": 1,
        "by_type": {"claim": {"affected_reconciliation_required": 1}},
        "by_status": {"affected_reconciliation_required": 1},
    }


def test_artifact_summary_treats_none_as_missing_but_preserves_falsy_values() -> None:
    """Artifact summary coercion preserves valid falsy values while handling missing fields."""

    report = build_frozen_trace_reconciliation_report(
        [_ledger_row("episode-1")],
        [_ledger_row("episode-1", collision=True)],
        artifact_manifest=[
            {
                "artifact_id": None,
                "artifact_type": None,
                "consumes_event_fields": ["exact_events.collision"],
            },
            {
                "artifact_id": 0,
                "artifact_type": 0,
                "consumes_event_fields": ["exact_events.collision"],
            },
        ],
    )

    assert report["affected_artifacts"][0]["artifact_id"] == "unknown"
    assert report["affected_artifacts"][0]["artifact_type"] == "unknown"
    assert report["affected_artifacts"][1]["artifact_id"] == "0"
    assert report["affected_artifacts"][1]["artifact_type"] == "0"
    assert report["affected_artifact_summary"] == {
        "total_artifacts": 2,
        "by_type": {
            "0": {"affected_reconciliation_required": 1},
            "unknown": {"affected_reconciliation_required": 1},
        },
        "by_status": {"affected_reconciliation_required": 2},
    }


def test_episode_keys_preserve_valid_falsy_identifiers() -> None:
    """Episode pairing keys should preserve valid falsy identifiers like zero."""

    row = _ledger_row("unused")
    ledger = row["event_ledger"]
    ledger["scenario_id"] = 0
    ledger["concrete_case_id"] = 0
    ledger["seed"] = 0
    ledger["planner"] = 0

    report = build_frozen_trace_reconciliation_report([row], [row])

    assert report["unchanged_rows"] == [{"episode_key": "0|0|0|0"}]


def test_v1_durable_rows_pass_and_unsupported_schema_fails_closed() -> None:
    """Frozen durable v1 ledgers remain supported, unknown schema versions do not."""

    v1_row = _ledger_row("episode-1")
    report = build_frozen_trace_reconciliation_report([v1_row], [v1_row])

    assert report["unchanged_rows"] == [{"episode_key": "scenario-a|episode-1|7|demo"}]

    unsupported_row = _ledger_row("episode-1")
    unsupported_row["event_ledger"]["schema_version"] = "EpisodeEventLedger.v3"
    with pytest.raises(ValueError, match="supported EpisodeEventLedger payload"):
        build_frozen_trace_reconciliation_report([unsupported_row], [v1_row])


def test_cli_jsonl_parse_error_names_path_and_line(tmp_path: Path) -> None:
    """Malformed JSONL input errors should identify the path and line."""

    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\n{"broken":\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.jsonl:2: invalid JSON"):
        _MODULE._read_jsonl(path)


def test_cli_manifest_parse_error_names_path(tmp_path: Path) -> None:
    """Malformed artifact manifests should identify the path."""

    path = tmp_path / "bad_manifest.json"
    path.write_text('{"artifacts": [', encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid JSON in artifact manifest .*bad_manifest\.json"):
        _MODULE._read_artifact_manifest(path)


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


def test_cli_can_write_missing_export_diagnostic(tmp_path: Path) -> None:
    """CLI can document missing durable frozen exports without fabricating comparison data."""

    manifest_path = tmp_path / "artifact_manifest.json"
    out_path = tmp_path / "missing_export_report.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "artifact_id": "claim:collision-count",
                        "artifact_type": "claim",
                        "consumes_event_fields": ["exact_events.collision"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--old-ledgers",
            str(tmp_path / "missing-before.jsonl"),
            "--new-ledgers",
            str(tmp_path / "missing-after.jsonl"),
            "--artifact-manifest",
            str(manifest_path),
            "--out",
            str(out_path),
            "--old-label",
            "0.0.2-before",
            "--new-label",
            "0.0.2-after",
            "--diagnose-missing-exports",
        ],
        check=False,
        cwd=SCRIPT_PATH.parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == FROZEN_TRACE_MISSING_EXPORT_SCHEMA
    assert report["status"] == "blocked_missing_durable_event_ledger_exports"
    assert report["summary"]["missing_export_count"] == 2
    assert report["affected_artifact_summary"] == {
        "total_artifacts": 1,
        "by_type": {"claim": {"not_evaluable_missing_event_ledger_export": 1}},
        "by_status": {"not_evaluable_missing_event_ledger_export": 1},
    }
