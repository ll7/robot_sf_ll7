"""Focused synthetic-projection tests for the scheduler-job reconciler (#8838)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.slurm_launch_manifest import SCHEMA_VERSION as LAUNCH_MANIFEST_SCHEMA
from scripts.tools import reconcile_scheduler_jobs as tool

FIXTURES = Path(__file__).parent / "fixtures" / "scheduler_reconcile"
RESOURCES = {"cpu_class": "cpu_8", "memory_class": "mem_64gb", "accelerator_class": "gpu_1"}
_EXPLICIT_TAILS = dict.fromkeys(
    ("exit_code", "derived_exit_code", "signal", "termination_class"), "not_observed"
)


def _alias(prefix: str, seed: int) -> str:
    return f"{prefix}-{seed:016x}"


def _packet(seed: int = 1) -> str:
    return f"{seed:064x}"


def _receipt(
    job: str,
    task: str,
    state: str = "running",
    *,
    packet: str | None = None,
    root: str | None = None,
) -> dict[str, Any]:
    packet = packet or _packet(1)
    root = root or _alias("artifact-root", 2)
    allocated, started, elapsed = dict(RESOURCES), "2026-09-01T08:00:00Z", 1200
    tails: dict[str, Any] = dict(_EXPLICIT_TAILS)
    history = ["pending", state]
    if state == "pending":
        allocated = dict.fromkeys(RESOURCES, "not_observed")
        started, elapsed = "not_observed", "not_observed"
    if state == "completed":
        elapsed = 3600
        tails = {
            "exit_code": 0,
            "derived_exit_code": 0,
            "signal": "not_applicable",
            "termination_class": "normal",
        }
        history = ["pending", "running", "completed"]
    return {
        "schema": "robot_sf.scheduler_allocation_receipt.v1",
        "job_alias": job,
        "campaign_id": task,
        "scheduler_state": state,
        "state_history": history,
        "submission": {
            "intent_digest": _packet(7),
            "source_config_command_digest": packet,
            "acknowledgement": "acknowledged",
            "submitted_at": "2026-09-01T07:59:00Z",
        },
        "execution": {
            "requested_resources": dict(RESOURCES),
            "allocated_resources": allocated,
            "started_at": started,
            "ended_at": "not_observed",
            "elapsed_seconds": elapsed,
            "time_limit_seconds": 7200,
            "environment_receipt_digest": _packet(8),
            **tails,
        },
        "artifacts": {"result_root_identity": root, "manifest_digest": _packet(9)},
        "claim_boundary": "Scheduler and allocation custody only.",
    }


_UNSET = object()


def _task(
    task_id: str,
    *,
    packet: str | None = None,
    root: str | None = None,
    owner: Any = _UNSET,
    ref: Any = _UNSET,
    harvest: str = "pending",
    transfer: str = "pending",
    durable_pointer: bool = False,
) -> dict[str, Any]:
    if ref is _UNSET:
        ref = {"kind": "issue", "number": 8838}
    if owner is _UNSET:
        owner = _alias("owner", 4)
    return {
        "task_id": task_id,
        "task_class": "campaign",
        "public_ref": ref,
        "launch_manifest": {
            "schema_version": LAUNCH_MANIFEST_SCHEMA,
            "manifest_sha256": _packet(5),
            "packet_digest": packet or _packet(1),
            "expected_episode_cells": 20160,
        },
        "artifact_root_identity": root or _alias("artifact-root", 2),
        "owner": owner,
        "harvest_state": harvest,
        "transfer_state": transfer,
        "durable_pointer": durable_pointer,
    }


def _projection(
    jobs: list[Any],
    tasks: list[Any] | None = None,
    *,
    issues: list[Any] | None = None,
    status: str = "complete",
    generated_at: str = "2026-09-11T00:00:00Z",
) -> dict[str, Any]:
    return {
        "schema": tool.PROJECTION_SCHEMA,
        "projection_status": status,
        "generated_at": generated_at,
        "jobs": jobs,
        "public_metadata": {
            "generated_at": generated_at,
            "issues": issues if issues is not None else [{"number": 8838, "state": "open"}],
            "pull_requests": [],
            "tasks": tasks or [],
        },
    }


def _rows(projection: dict[str, Any], **kwargs: Any) -> dict[str, dict[str, Any]]:
    report = tool.reconcile_projection(projection, **kwargs)
    return {row["job_alias"]: row for row in report["rows"]}


def test_owned_active_binds_exact_identity() -> None:
    task_id, job = _alias("campaign", 1), _alias("job", 1)
    projection = _projection(
        [{"receipt": _receipt(job, task_id), "issue_ref": {"kind": "issue", "number": 8838}}],
        [_task(task_id, ref={"kind": "issue", "number": 8838})],
        issues=[{"number": 8838, "state": "open"}],
    )
    report = tool.reconcile_projection(projection)
    row = report["rows"][0]
    assert row["classification"] == "owned_active"
    assert row["public_owner"] == {
        "kind": "issue",
        "number": 8838,
        "state": "open",
        "url": "https://github.com/ll7/robot_sf_ll7/issues/8838",
    }
    assert row["packet_digest"] == _packet(1)
    assert row["expected_row_count"] == 20160
    assert row["manifest_sha256"] == _packet(5)
    assert row["output_root_identity"] == _alias("artifact-root", 2)
    assert row["artifact_owner"] == _alias("owner", 4)
    assert row["scientific_status"] == "not_evaluated"
    assert row["next_action"] == "monitor_active_job"
    assert report["action_required_count"] == 0
    assert report["volatile_fields_normalized"] == ["generated_at"]


def test_terminal_and_harvested_rows_stay_custody_only() -> None:
    task_h, task_u = _alias("campaign", 2), _alias("campaign", 3)
    jobs = [
        {"receipt": _receipt(_alias("job", 2), task_h, "completed")},
        {"receipt": _receipt(_alias("job", 3), task_u, "completed")},
    ]
    tasks = [
        _task(task_h, harvest="harvested", transfer="transferred", durable_pointer=True),
        _task(task_u),
    ]
    rows = _rows(_projection(jobs, tasks))
    assert rows[_alias("job", 2)]["classification"] == "owned_harvested"
    assert rows[_alias("job", 2)]["durability"] == "durable"
    unharvested = rows[_alias("job", 3)]
    assert unharvested["classification"] == "owned_terminal_unharvested"
    assert unharvested["scheduler_state"] == "completed"
    assert unharvested["durability"] == "not_applicable"
    assert unharvested["scientific_status"] == "not_evaluated"
    assert unharvested["next_action"] == "harvest_terminal_output"


def test_orphan_jobs_without_exact_public_owner_stay_unknown() -> None:
    unmatched = _alias("campaign", 5)
    bound_task = _alias("campaign", 6)
    jobs = [
        {"receipt": _receipt(_alias("job", 5), unmatched)},
        {"receipt": _receipt(_alias("job", 6), bound_task)},
    ]
    projection = _projection(jobs, [_task(bound_task, ref=None)])
    rows = _rows(projection)
    assert rows[_alias("job", 5)]["classification"] == "orphan_unknown"
    assert rows[_alias("job", 6)]["classification"] == "orphan_unknown"
    assert rows[_alias("job", 6)]["next_action"] == "attach_exact_public_owner"


def test_stale_packet_digest_is_stale_input() -> None:
    task_id = _alias("campaign", 7)
    projection = _projection(
        [{"receipt": _receipt(_alias("job", 7), task_id, packet=_packet(2))}],
        [_task(task_id, packet=_packet(3))],
    )
    row = tool.reconcile_projection(projection)["rows"][0]
    assert row["classification"] == "stale_input"
    assert any(item["code"] == "stale_packet_digest" for item in row["findings"])
    assert row["next_action"] == "refresh_immutable_input_packet"


def test_missing_artifact_owner_is_explicit() -> None:
    task_id = _alias("campaign", 8)
    projection = _projection(
        [{"receipt": _receipt(_alias("job", 8), task_id)}], [_task(task_id, owner=None)]
    )
    row = tool.reconcile_projection(projection)["rows"][0]
    assert row["classification"] == "missing_output_owner"
    assert row["artifact_owner"] is None
    assert row["next_action"] == "assign_artifact_owner"


def test_duplicate_active_jobs_but_not_cancelled_replacement() -> None:
    task_id = _alias("campaign", 9)
    jobs = [
        {"receipt": _receipt(_alias("job", 91), task_id)},
        {"receipt": _receipt(_alias("job", 92), task_id)},
        {"receipt": _receipt(_alias("job", 93), task_id, "cancelled")},
    ]
    rows = _rows(_projection(jobs, [_task(task_id)]))
    assert rows[_alias("job", 91)]["classification"] == "duplicate_candidate"
    assert rows[_alias("job", 92)]["classification"] == "duplicate_candidate"
    assert rows[_alias("job", 93)]["classification"] == "owned_terminal_unharvested"
    assert all(
        item["code"] == "duplicate_active_jobs"
        for row in (rows[_alias("job", 91)], rows[_alias("job", 92)])
        for item in row["findings"]
    )


def test_resumed_job_with_parent_lineage_is_not_duplicate() -> None:
    task_id = _alias("campaign", 10)
    parent, resumed = _alias("job", 101), _alias("job", 102)
    jobs = [
        {"receipt": _receipt(parent, task_id, "cancelled")},
        {"receipt": _receipt(resumed, task_id), "parent_job_alias": parent},
    ]
    rows = _rows(_projection(jobs, [_task(task_id)]))
    assert rows[resumed]["classification"] == "owned_active"
    assert rows[resumed]["parent_job_alias"] == parent
    assert rows[parent]["classification"] == "owned_terminal_unharvested"


def test_conflicting_public_owner_is_duplicate_candidate() -> None:
    task_id, job = _alias("campaign", 11), _alias("job", 11)
    jobs = [
        {"receipt": _receipt(job, task_id), "issue_ref": {"kind": "issue", "number": 8838}},
        {"receipt": _receipt(job, task_id), "issue_ref": {"kind": "issue", "number": 8839}},
    ]
    rows = _rows(_projection(jobs, [_task(task_id)]))
    assert rows[job]["classification"] == "duplicate_candidate"
    assert any(item["code"] == "conflicting_public_owner" for item in rows[job]["findings"])


def test_distinct_array_indices_are_not_equivalent_duplicates() -> None:
    task_id = _alias("campaign", 12)
    jobs = [
        {"receipt": _receipt(_alias("job", 121), task_id), "array_index": 0},
        {"receipt": _receipt(_alias("job", 122), task_id), "array_index": 1},
    ]
    rows = _rows(_projection(jobs, [_task(task_id)]))
    assert {row["classification"] for row in rows.values()} == {"owned_active"}


def test_projection_unavailable_and_invalid_receipt_fail_closed() -> None:
    task_id = _alias("campaign", 13)
    unavailable = _projection(
        [{"receipt": _receipt(_alias("job", 13), task_id)}], [_task(task_id)], status="partial"
    )
    assert tool.reconcile_projection(unavailable)["rows"][0]["classification"] == (
        "projection_unavailable"
    )
    malformed = _projection([{"receipt": {"schema": "robot_sf.scheduler_allocation_receipt.v1"}}])
    row = tool.reconcile_projection(malformed)["rows"][0]
    assert row["classification"] == "projection_unavailable"
    assert row["findings"][0]["code"] == "invalid_receipt"


def test_under_redacted_value_fails_closed_without_echoing_it() -> None:
    task_id, secret = _alias("campaign", 14), "/home/researcher42/output"
    receipt = _receipt(_alias("job", 14), task_id)
    receipt["claim_boundary"] = f"run at {secret}"
    report = tool.reconcile_projection(_projection([{"receipt": receipt}], [_task(task_id)]))
    assert report["rows"][0]["classification"] == "projection_unavailable"
    assert secret not in tool.render_report_json(report)
    assert secret not in tool.render_report_markdown(report)
    assert "# Scheduler Job Reconciliation" in tool.render_report_markdown(report)
    forbidden = {"receipt": _receipt(_alias("job", 15), task_id), "hostname": "gpu-node-17"}
    report = tool.reconcile_projection(_projection([forbidden], [_task(task_id)]))
    assert report["rows"][0]["classification"] == "projection_unavailable"
    assert any(item["code"] == "forbidden_field_name" for item in report["rows"][0]["findings"])


def test_report_is_byte_stable_and_order_independent() -> None:
    task_a, task_b = _alias("campaign", 16), _alias("campaign", 17)
    job_a, job_b = _alias("job", 161), _alias("job", 171)
    jobs = [
        {"receipt": _receipt(job_a, task_a)},
        {"receipt": _receipt(job_b, task_b, "completed")},
    ]
    tasks = [_task(task_a), _task(task_b, harvest="harvested", transfer="transferred")]
    first = tool.render_report_json(tool.reconcile_projection(_projection(jobs, tasks)))
    shuffled = tool.render_report_json(
        tool.reconcile_projection(
            _projection(
                [jobs[1], jobs[0]], [tasks[1], tasks[0]], generated_at="2027-01-01T00:00:00Z"
            )
        )
    )
    assert first == shuffled
    assert first == tool.render_report_json(tool.reconcile_projection(_projection(jobs, tasks)))


def test_cli_check_mode_exit_codes_and_json(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    task_id = _alias("campaign", 18)
    good = tmp_path / "good.projection.json"
    good.write_text(
        json.dumps(
            _projection([{"receipt": _receipt(_alias("job", 18), task_id)}], [_task(task_id)])
        ),
        encoding="utf-8",
    )
    assert tool.main(["--check", "--projection", str(good), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["row_count"] == 1 and payload["check_only"] is True
    orphan = tmp_path / "orphan.projection.json"
    orphan.write_text(
        json.dumps(_projection([{"receipt": _receipt(_alias("job", 19), task_id)}])),
        encoding="utf-8",
    )
    assert tool.main(["--check", "--projection", str(orphan)]) == 1
    capsys.readouterr()
    bad = tmp_path / "bad.projection.json"
    bad.write_text("{", encoding="utf-8")
    assert tool.main(["--check", "--projection", str(bad)]) == 2
    with pytest.raises(SystemExit) as excinfo:
        tool.main(["--projection", str(good)])
    assert excinfo.value.code == 2


def test_cli_public_override_joins_separate_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    task_id = _alias("campaign", 20)
    projection = tmp_path / "projection.json"
    projection.write_text(
        json.dumps(_projection([{"receipt": _receipt(_alias("job", 20), task_id)}])),
        encoding="utf-8",
    )
    snapshot = tmp_path / "public.json"
    snapshot.write_text(json.dumps({"tasks": [_task(task_id)]}), encoding="utf-8")
    assert (
        tool.main(
            [
                "--check",
                "--projection",
                str(projection),
                "--public",
                str(snapshot),
                "--format",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["public_metadata_source"] == "override"
    assert payload["rows"][0]["classification"] == "owned_active"


@pytest.mark.parametrize(
    ("fixture", "exit_code", "classifications"),
    [
        ("bound", 0, {"owned_active", "owned_harvested"}),
        (
            "mixed",
            1,
            {
                "owned_active",
                "owned_harvested",
                "owned_terminal_unharvested",
                "duplicate_candidate",
                "orphan_unknown",
                "stale_input",
                "missing_output_owner",
            },
        ),
        ("unavailable", 1, {"projection_unavailable"}),
    ],
)
def test_checked_in_fixtures_cover_required_cases(
    fixture: str, exit_code: int, classifications: set[str], capsys: pytest.CaptureFixture
) -> None:
    path = FIXTURES / f"{fixture}.projection.json"
    assert tool.main(["--check", "--projection", str(path), "--format", "json"]) == exit_code
    report = json.loads(capsys.readouterr().out)
    assert {row["classification"] for row in report["rows"]} == classifications
    assert (report["action_required_count"] > 0) == bool(exit_code)
