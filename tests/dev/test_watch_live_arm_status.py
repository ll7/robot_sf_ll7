"""Tests live arm-status private-ops cancellation decisions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.dev.watch_live_arm_status import (
    CANCEL_EXIT_CODE,
    decide_arm_status_action,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_failed_event_recommends_scancel_without_executing(tmp_path: Path) -> None:
    """Fail-closed arm events should produce a deterministic cancellation decision."""

    output_root = tmp_path / "campaign"
    output_root.mkdir()
    _write_json(
        output_root / "live_arm_status.json",
        {
            "schema_version": "robot_sf.issue_4205.static_constriction_codesign_campaign.v1.live_arm_status.v1",
            "phase": "full",
            "arms": {
                "ppo_frozen_cbf_on": {"status": "running"},
                "ppo_frozen_wrapper_on": {"status": "pending"},
            },
        },
    )
    _write_jsonl(
        output_root / "arm_status.jsonl",
        [
            {"phase": "full", "arm": "ppo_frozen_cbf_on", "state": "running"},
            {
                "phase": "full",
                "arm": "ppo_frozen_cbf_on",
                "state": "failed",
                "details": {"error": "ContractError: map runner failed closed"},
            },
        ],
    )

    decision = decide_arm_status_action(
        live_status_path=output_root / "live_arm_status.json",
        event_log_path=output_root / "arm_status.jsonl",
        job_id="13320",
    )

    assert decision.action == "cancel"
    assert decision.reason == "fail_closed_arm_event"
    assert decision.failing_arms == ["ppo_frozen_cbf_on"]
    assert decision.scancel_command == ["scancel", "13320"]
    assert decision.executed is False


def test_completed_live_status_is_noop(tmp_path: Path) -> None:
    """Healthy completed arms should not request cancellation."""

    output_root = tmp_path / "campaign"
    output_root.mkdir()
    _write_json(
        output_root / "live_arm_status.json",
        {
            "schema_version": "robot_sf.issue_4205.static_constriction_codesign_campaign.v1.live_arm_status.v1",
            "phase": "smoke",
            "arms": {
                "ppo_frozen_cbf_on": {"status": "completed"},
                "ppo_frozen_wrapper_on": {"status": "completed"},
                "ppo_frozen": {"status": "completed"},
            },
        },
    )
    _write_jsonl(output_root / "arm_status.jsonl", [])

    decision = decide_arm_status_action(
        live_status_path=output_root / "live_arm_status.json",
        event_log_path=output_root / "arm_status.jsonl",
        job_id="13320",
    )

    assert decision.action == "complete"
    assert decision.scancel_command is None


def test_cli_json_returns_cancel_exit_code(tmp_path: Path, capsys) -> None:
    """Ops wrappers can poll once and branch on a stable nonzero cancel exit."""

    output_root = tmp_path / "campaign"
    output_root.mkdir()
    _write_json(
        output_root / "live_arm_status.json",
        {
            "schema_version": "robot_sf.issue_4205.static_constriction_codesign_campaign.v1.live_arm_status.v1",
            "phase": "phase0",
            "arms": {"ppo_frozen": {"status": "failed"}},
        },
    )

    exit_code = main(["--output-root", str(output_root), "--job-id", "13300", "--json"])

    assert exit_code == CANCEL_EXIT_CODE
    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "cancel"
    assert payload["reason"] == "fail_closed_live_status"
    assert payload["scancel_command"] == ["scancel", "13300"]
