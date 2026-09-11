"""Focused contract tests for the sanitized scheduler/allocation receipt tool (#8916)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.tools import scheduler_allocation_receipt as tool

if TYPE_CHECKING:
    from pathlib import Path

DIGEST = "a" * 64
STARTED, ENDED = "2026-09-01T08:00:00+02:00", "2026-09-01T09:00:00+02:00"
LATER = "2026-09-01T10:00:00+02:00"
PRIVATE_HOST, PRIVATE_USER = "gpu-node-17.cluster.invalid", "researcher42"
PRIVATE_CONTEXT = json.loads(
    '{"hostname": "gpu-node-17.cluster.invalid", "username": "researcher42", '
    '"account": "proj-rm", "partition": "gpu-a100", "node_list": ["node[01-04]"]}'
)
TAIL = ("ended_at", "exit_code", "derived_exit_code", "signal", "termination_class")
HISTORIES = {
    "completed": ["pending", "running", "completed"],
    "failed": ["running", "failed"],
    "timeout": ["running", "timeout"],
    "cancelled": ["pending", "cancelled"],
}


NO_START = dict.fromkeys(("started_at", "elapsed_seconds", *TAIL), "not_observed")
NO_TAIL = dict.fromkeys(TAIL, "not_observed")
NO_ALLOCATION = dict.fromkeys(tool.RESOURCE_CLASSES, "not_observed")
RESOURCES = {"cpu_class": "cpu_8", "memory_class": "mem_64gb", "accelerator_class": "gpu_1"}


def _private(state: str = "completed", **execution_overrides: Any) -> dict[str, Any]:
    execution = {
        "requested_resources": dict(RESOURCES),
        "allocated_resources": dict(RESOURCES),
        "started_at": STARTED,
        "ended_at": ENDED,
        "elapsed_seconds": 3600,
        "time_limit_seconds": 7200,
        "exit_code": 0,
        "derived_exit_code": 0,
        "signal": "not_applicable",
        "termination_class": "normal",
        "environment_receipt_digest": DIGEST,
    }
    execution.update(execution_overrides)
    return {
        "schema": tool.PRIVATE_INPUT_SCHEMA,
        "job_alias": "custody-0007",
        "campaign_id": "camera-ready-2026-09",
        "scheduler_state": state,
        "state_history": HISTORIES.get(state, [state]),
        "submission": {
            "intent_digest": DIGEST,
            "source_config_command_digest": DIGEST,
            "submitted_at": "2026-09-01T07:59:00+02:00",
            "acknowledgement": "lost" if state == "submission_unacknowledged" else "acknowledged",
        },
        "execution": execution,
        "artifacts": {
            "result_root_identity": "campaign_results/custody-0007",
            "manifest_digest": DIGEST,
        },
        "private_context": PRIVATE_CONTEXT,
    }


def _codes(error: tool.ReceiptError) -> set[str]:
    return {issue.code for issue in error.issues}


def _terminated(termination: str, **extra: Any) -> dict[str, Any]:
    return dict(
        exit_code=143, derived_exit_code=143, signal=15, termination_class=termination, **extra
    )


VALID_SCENARIOS = {
    "pending": ("pending", {**NO_START, "allocated_resources": NO_ALLOCATION}),
    "running": ("running", dict(NO_TAIL)),
    "completed": ("completed", {}),
    "failed": ("failed", {"exit_code": 1, "derived_exit_code": 1, "termination_class": "failed"}),
    "timeout": ("timeout", _terminated("time_limit", ended_at=LATER, elapsed_seconds=7200)),
    "cancelled": (
        "cancelled",
        _terminated("cancelled", started_at="not_applicable", elapsed_seconds="not_applicable"),
    ),
    "lost_acknowledgement": (
        "submission_unacknowledged",
        {**NO_START, "allocated_resources": NO_ALLOCATION},
    ),
    "missing_derived_exit": (
        "failed",
        {"exit_code": 1, "derived_exit_code": "unavailable", "termination_class": "failed"},
    ),
}


@pytest.mark.parametrize(("state", "overrides"), VALID_SCENARIOS.values(), ids=VALID_SCENARIOS)
def test_state_fixtures_convert_validate_and_stay_private(
    state: str, overrides: dict[str, Any], tmp_path: Path
) -> None:
    """Each lifecycle fixture round-trips through project, validate, and CLI without PII."""
    receipt = tool.project_receipt(_private(state, **overrides))
    assert receipt["scheduler_state"] == state and receipt["state_history"][-1] == state
    text = tool.render_receipt_json(receipt)
    (tmp_path / "receipt.json").write_text(text, encoding="utf-8")
    assert tool.main(["validate", "--receipt", str(tmp_path / "receipt.json"), "--json"]) == 0
    assert PRIVATE_HOST not in text and PRIVATE_USER not in text


def test_redacted_allocation_and_deterministic_normalized_output() -> None:
    """Redaction stays explicit, timestamps normalize, and repeated conversion is byte-stable."""
    private = _private("completed")
    private["redactions"] = ["execution.allocated_resources.accelerator_class"]
    first = tool.project_receipt(private)
    second = tool.project_receipt(json.loads(json.dumps(private)))
    assert tool.render_receipt_json(first) == tool.render_receipt_json(second)
    assert first["execution"]["allocated_resources"]["accelerator_class"] == "redacted"
    assert first["execution"]["allocated_resources"]["cpu_class"] == "cpu_8"
    assert first["submission"]["submitted_at"] == "2026-09-01T05:59:00Z"
    assert tool.CLAIM_NOTE in tool.render_markdown(first)


FAIL_CLOSED = [
    ({"execution": {"ended_at": "2026-09-01T07:00:00+02:00"}}, "timestamp_order"),
    ({"execution": {"elapsed_seconds": 99999}}, "elapsed_mismatch"),
    ({"execution": {"signal": 9}}, "exit_signal_mismatch"),
    ({"execution": {"exit_code": 9, "derived_exit_code": 9, "signal": 9}}, "derived_exit_mismatch"),
    ({"submission": {"campaign_id": "other-campaign"}}, "campaign_identity_mismatch"),
    (
        {
            "submission": {"intent_digest": "b" * 64},
            "execution": {"submission_intent_digest": DIGEST},
        },
        "source_identity_mismatch",
    ),
    ({"state_history": ["completed", "running"]}, "illegal_transition"),
    (
        {"scheduler_state": "failed", "state_history": ["running", "completed"]},
        "history_state_mismatch",
    ),
]


@pytest.mark.parametrize(("patch", "expected"), FAIL_CLOSED)
def test_contradictions_fail_closed(patch: dict[str, Any], expected: str) -> None:
    """Contradictory transitions, timestamps, and identities abort with a code."""
    patch = dict(patch)
    private = _private(patch.pop("scheduler_state", "completed"))
    for section, fields in patch.items():
        if section == "state_history":
            private[section] = fields
        else:
            private[section].update(fields)
    with pytest.raises(tool.ReceiptError) as error:
        tool.project_receipt(private)
    assert expected in _codes(error.value)


def test_forbidden_private_fields_and_values_fail_closed() -> None:
    """Private paths and topology never reach public output and are rejected."""
    leaked = _private("completed")
    leaked["artifacts"]["result_root_identity"] = f"/scratch/{PRIVATE_USER}/results"
    with pytest.raises(tool.ReceiptError) as error:
        tool.project_receipt(leaked)
    assert _codes(error.value) & {"private_value_leak", "invalid_field"}
    assert PRIVATE_USER not in str(error.value)

    public = tool.project_receipt(_private("completed"))
    public["hostname"] = PRIVATE_HOST
    public["artifacts"]["result_root_identity"] = "https://x.invalid/r?token=abc"
    codes = {issue.code for issue in tool.validate_receipt(public)}
    assert {"forbidden_field", "forbidden_value"} <= codes


def test_cli_project_writes_public_pair_and_rejects_private_input(
    tmp_path: Path, capsys: Any
) -> None:
    """The CLI writes deterministic JSON plus Markdown and exits 2 on private topology."""
    source = tmp_path / "private.json"
    source.write_text(json.dumps(_private("completed")), encoding="utf-8")
    output = tmp_path / "public.json"
    assert tool.main(["project", "--private-input", str(source), "--output", str(output)]) == 0
    assert output.is_file() and output.with_suffix(".md").is_file()
    assert tool.main(["validate", "--receipt", str(output), "--json"]) == 0

    bad = _private("completed")
    bad["hostname"] = PRIVATE_HOST
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    never = tmp_path / "never.json"
    args = ["project", "--private-input", str(tmp_path / "bad.json"), "--output", str(never)]
    assert tool.main(args) == 2
    assert PRIVATE_HOST not in "".join(capsys.readouterr()) and not never.exists()
