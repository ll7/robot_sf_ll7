"""Tests the CI-facing event-ledger reconciliation guard."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = ROOT / "scripts" / "benchmark" / "export_event_ledger_reconciliation.py"
FIXTURE_DIR = ROOT / "tests" / "benchmark" / "fixtures" / "event_ledger_reconciliation"
RECONCILED_FIXTURE = FIXTURE_DIR / "reconciled_episodes.jsonl"
VIOLATING_FIXTURE = FIXTURE_DIR / "violating_exact_collision_zero_count.jsonl"


def _run_export(fixture: Path, out: Path) -> subprocess.CompletedProcess[str]:
    """Run the reconciliation exporter in fail-closed mode."""
    return subprocess.run(
        [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--episodes",
            str(fixture),
            "--out",
            str(out),
            "--fail-on-violations",
        ],
        check=False,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_reconciliation_guard_fixture_passes(tmp_path: Path) -> None:
    """The fixture wired into CI should represent a reconciled collision ledger."""
    result = _run_export(RECONCILED_FIXTURE, tmp_path / "reconciled.jsonl")

    assert result.returncode == 0, result.stderr


def test_reconciliation_guard_fixture_fails_closed_on_exact_collision_zero_count(
    tmp_path: Path,
) -> None:
    """Exact collision events with zero collision count remain hard failures."""
    out = tmp_path / "violating.jsonl"
    result = _run_export(VIOLATING_FIXTURE, out)

    assert result.returncode == 1
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["reconciliation_table"]["violations"] == [
        "exact collision event requires collision metric > 0"
    ]


def test_reconciliation_guard_is_wired_into_ci_test_phase() -> None:
    """The CI test phase must invoke the guard so it cannot become agent-only guidance."""
    ci_driver = (ROOT / "scripts" / "dev" / "ci_driver.sh").read_text(encoding="utf-8")

    assert "check_event_ledger_reconciliation_guard.sh" in ci_driver, (
        "event-ledger reconciliation guard is not invoked by ci_driver.sh; "
        "exact/derived benchmark mismatches could bypass CI"
    )
