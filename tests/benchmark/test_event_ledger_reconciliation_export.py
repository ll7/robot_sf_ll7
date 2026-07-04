"""Tests for exporting event-ledger reconciliation audit tables."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "benchmark"
    / "export_event_ledger_reconciliation.py"
)
_SPEC = importlib.util.spec_from_file_location("export_event_ledger_reconciliation", SCRIPT_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

EXPORT_SCHEMA_VERSION = _MODULE.EXPORT_SCHEMA_VERSION
build_export_row = _MODULE.build_export_row


def _record(*, collision_event: bool = False, collisions: float = 0.0) -> dict:
    """Build a minimal benchmark episode record."""

    return {
        "episode_id": "episode-1",
        "scenario_id": "scenario-1",
        "seed": 7,
        "algo": "goal",
        "git_hash": "abc123",
        "metrics": {
            "success": 0.0 if collision_event else 1.0,
            "collisions": collisions,
        },
        "termination_reason": "collision" if collision_event else "success",
        "outcome": {
            "route_complete": not collision_event,
            "collision_event": collision_event,
        },
    }


def test_build_export_row_wraps_episode_identity_and_audit_table() -> None:
    """Export rows should preserve episode identity around the canonical audit table."""

    row = build_export_row(_record(collision_event=True, collisions=1.0))

    assert row["schema_version"] == EXPORT_SCHEMA_VERSION
    assert row["episode_id"] == "episode-1"
    assert row["scenario_id"] == "scenario-1"
    assert row["seed"] == 7
    assert row["planner"] == "goal"
    assert row["software_commit"] == "abc123"
    assert row["event_ledger_schema_version"] == "EpisodeEventLedger.v2"
    assert row["reconciliation_table"]["reconciles"] is True
    assert {entry["field"] for entry in row["reconciliation_table"]["rows"]} >= {
        "collision_count",
        "near_miss",
        "clearance_breach",
        "ttc_breach",
    }


def test_cli_exports_disagreement_without_silently_resolving_it(tmp_path: Path) -> None:
    """Exact-vs-derived disagreements should stay visible in exported rows."""

    episodes = tmp_path / "episodes.jsonl"
    out = tmp_path / "event_ledger_reconciliation.jsonl"
    episodes.write_text(
        json.dumps(_record(collision_event=True, collisions=0.0)) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--episodes",
            str(episodes),
            "--out",
            str(out),
        ],
        check=False,
        cwd=SCRIPT_PATH.parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    exported = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 1
    table = exported[0]["reconciliation_table"]
    assert table["reconciles"] is False
    assert any("collision" in violation for violation in table["violations"])


def test_cli_can_fail_closed_on_exported_reconciliation_violations(tmp_path: Path) -> None:
    """Pipelines can opt into a non-zero exit when exported rows contain violations."""

    episodes = tmp_path / "episodes.jsonl"
    out = tmp_path / "event_ledger_reconciliation.jsonl"
    episodes.write_text(
        json.dumps(_record(collision_event=True, collisions=0.0)) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--episodes",
            str(episodes),
            "--out",
            str(out),
            "--fail-on-violations",
        ],
        check=False,
        cwd=SCRIPT_PATH.parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert out.exists()
