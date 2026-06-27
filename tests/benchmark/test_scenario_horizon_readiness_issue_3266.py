"""Fail-closed readiness tests for scenario-horizon Results evidence (issue #3266).

These tests use small Markdown/JSON fixtures plus the real re-exported
scenario-horizon bundle (issue #3203, PR #3263) to prove the readiness check
reports valid / diagnostic-only / blocked without rerunning any campaign.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.scenario_horizon_readiness import (
    BLOCKED,
    DIAGNOSTIC_ONLY,
    VALID,
    classify_scenario_horizon_readiness,
)
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from pathlib import Path

# A minimal two-column table is enough to exercise status + SNQI classification.
_VALID_TABLE = (
    "| planner_key | status | benchmark_success | snqi_contract_status |\n"
    "|---|---|---|---|\n"
    "| goal | ok | true | pass |\n"
    "| ppo | ok | true | pass |\n"
)

_PPO_PARTIAL_FAILURE_TABLE = (
    "| planner_key | status | most_likely_failure_reason | snqi_contract_status |\n"
    "|---|---|---|---|\n"
    "| goal | ok |  | pass |\n"
    "| ppo | partial-failure | Missing required dict observation keys | pass |\n"
)

_NO_SNQI_STATUS_TABLE = "| planner_key | status |\n|---|---|\n| goal | ok |\n| ppo | ok |\n"


def _write(tmp_path: Path, name: str, text: str) -> Path:
    """Write fixture text to a temp file and return its path.

    Returns:
        Path to the written fixture.
    """
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_table_is_benchmark_valid(tmp_path: Path) -> None:
    """All-ok rows with an asserted SNQI pass status classify as valid."""
    artifact = _write(tmp_path, "campaign_table.md", _VALID_TABLE)
    readiness = classify_scenario_horizon_readiness(artifact)
    assert readiness.status == VALID
    assert readiness.is_valid
    assert readiness.ppo_status == "ok"
    assert readiness.snqi_contract_status == "pass"
    assert readiness.blockers == []


def test_ppo_partial_failure_is_diagnostic_only(tmp_path: Path) -> None:
    """A PPO partial-failure row caps the verdict at diagnostic-only."""
    artifact = _write(tmp_path, "campaign_table.md", _PPO_PARTIAL_FAILURE_TABLE)
    readiness = classify_scenario_horizon_readiness(artifact)
    assert readiness.status == DIAGNOSTIC_ONLY
    assert not readiness.is_valid
    assert readiness.ppo_status == "unexpected_failure"
    assert any("ppo" in blocker.lower() for blocker in readiness.blockers)
    # The failure reason is surfaced for the diagnostic.
    assert any("Missing required dict observation keys" in b for b in readiness.blockers)


def test_missing_snqi_contract_status_blocks_valid(tmp_path: Path) -> None:
    """An artifact that asserts no SNQI contract status cannot be valid (fail-closed)."""
    artifact = _write(tmp_path, "campaign_table.md", _NO_SNQI_STATUS_TABLE)
    readiness = classify_scenario_horizon_readiness(artifact)
    assert readiness.status == DIAGNOSTIC_ONLY
    assert readiness.snqi_contract_status is None
    assert any("snqi" in blocker.lower() for blocker in readiness.blockers)


def test_failing_snqi_contract_status_is_diagnostic_only(tmp_path: Path) -> None:
    """An explicit SNQI fail status caps the verdict at diagnostic-only."""
    table = _VALID_TABLE.replace(
        "| pass |\n| ppo | ok | true | pass |", "| fail |\n| ppo | ok | true | fail |"
    )
    artifact = _write(tmp_path, "campaign_table.md", table)
    readiness = classify_scenario_horizon_readiness(artifact)
    assert readiness.status == DIAGNOSTIC_ONLY
    assert readiness.snqi_contract_status == "fail"
    assert any("not pass" in blocker.lower() for blocker in readiness.blockers)


def test_missing_artifact_is_blocked(tmp_path: Path) -> None:
    """A non-existent artifact path is blocked, never silently valid."""
    readiness = classify_scenario_horizon_readiness(tmp_path / "does_not_exist.md")
    assert readiness.status == BLOCKED
    assert readiness.is_blocked
    assert any("not found" in blocker.lower() for blocker in readiness.blockers)


def test_unparseable_artifact_is_blocked(tmp_path: Path) -> None:
    """An artifact with no campaign table is blocked."""
    artifact = _write(tmp_path, "campaign_table.md", "# Heading\n\nNo table here.\n")
    readiness = classify_scenario_horizon_readiness(artifact)
    assert readiness.status == BLOCKED
    assert any(
        "planner rows" in b.lower() or "campaign table" in b.lower() for b in readiness.blockers
    )


def test_json_summary_artifact_is_supported(tmp_path: Path) -> None:
    """A campaign-summary JSON with planner_rows is parsed like the Markdown table."""
    payload = {
        "planner_rows": [
            {"planner_key": "goal", "status": "ok", "snqi_contract_status": "pass"},
            {"planner_key": "ppo", "status": "partial-failure", "snqi_contract_status": "pass"},
        ]
    }
    artifact = _write(tmp_path, "summary.json", json.dumps(payload))
    readiness = classify_scenario_horizon_readiness(artifact)
    assert readiness.status == DIAGNOSTIC_ONLY
    assert readiness.ppo_status == "unexpected_failure"


def test_to_payload_is_json_serializable(tmp_path: Path) -> None:
    """The verdict payload round-trips through JSON."""
    artifact = _write(tmp_path, "campaign_table.md", _VALID_TABLE)
    readiness = classify_scenario_horizon_readiness(artifact)
    payload = readiness.to_payload()
    assert payload["status"] == VALID
    assert json.loads(json.dumps(payload))["schema_version"] == "scenario_horizon_readiness.v1"


def test_real_scenario_horizon_bundle_is_diagnostic_only() -> None:
    """The real re-exported issue #3203 bundle is diagnostic-only, not benchmark-valid.

    PPO partial-failed (missing dict observation keys) and the re-exported Markdown
    carries no SNQI contract status, so the evidence must not be promotable to
    Results wording. This anchors the check on durable on-disk evidence.
    """
    artifact = (
        get_repository_root()
        / "docs/context/evidence/issue_3203_scenario_horizon_reexport_2026-06-20"
        / "reports/campaign_table.md"
    )
    if not artifact.exists():
        pytest.skip("issue #3203 scenario-horizon re-export bundle not present in this checkout")
    readiness = classify_scenario_horizon_readiness(artifact)
    assert readiness.status == DIAGNOSTIC_ONLY
    assert readiness.ppo_status == "unexpected_failure"
    assert readiness.planner_rows >= 1
    # Both the PPO failure and the absent SNQI contract status are recorded.
    assert any("ppo" in b.lower() for b in readiness.blockers)
    assert any("snqi" in b.lower() for b in readiness.blockers)
