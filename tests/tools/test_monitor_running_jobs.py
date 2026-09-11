"""Focused contract tests for the read-only running-job monitor (#8855)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from scripts.tools import monitor_running_jobs as tool

if TYPE_CHECKING:
    from pathlib import Path

COMMIT, CONFIG, RECEIPT = "a" * 40, "b" * 64, "c" * 64


def _identity(job_id: str = "8811") -> dict[str, Any]:
    return {
        "job_id": job_id,
        "issue": 8811,
        "owner": "ll7",
        "campaign_id": "camp-8811",
        "commit": COMMIT,
        "config_sha256": CONFIG,
        "receipt_sha256": RECEIPT,
    }


def _obs(
    state: str,
    minute: int = 0,
    *,
    identity: dict[str, Any] | None = None,
    array: dict[str, str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    identity = identity or _identity()
    payload: dict[str, Any] = {
        "schema_version": tool.OBSERVATION_SCHEMA,
        "job_id": identity["job_id"],
        "observed_at": f"2026-09-11T00:{minute:02d}:00+00:00",
        "state": state,
        "identity_sha256": tool.identity_digest(identity),
    }
    if array is not None:
        payload["array"] = array
    payload.update(extra)
    return payload


def _job(
    identity: dict[str, Any] | None = None,
    observations: list[Any] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    identity = identity or _identity()
    job = {
        **identity,
        "artifacts": ["rows", "manifest"],
        "rows": ["row-0001"],
        "harvest_request": f"output/harvest/{identity['job_id']}/request.json",
        "harvest_artifact_root": "output/benchmarks/camp-8811",
        "observations": observations or [],
    }
    job.update(overrides)
    return job


def _projection(*jobs: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": tool.PROJECTION_SCHEMA, "jobs": list(jobs)}


def _record(job: dict[str, Any]) -> dict[str, Any]:
    return tool.monitor_once(_projection(job))["jobs"][0]


def _codes(record: dict[str, Any]) -> set[str]:
    return {problem["code"] for problem in record["problems"]}


def test_queued_running_complete_records_transitions_and_handoff() -> None:
    identity = _identity()
    record = _record(
        _job(
            identity,
            [
                _obs("PENDING", 0, identity=identity),
                _obs("RUNNING", 1, identity=identity),
                _obs("COMPLETED", 2, identity=identity),
            ],
        )
    )
    assert [t["to"] for t in record["transitions"]] == ["pending", "running", "completed"]
    assert all(len(t["evidence_sha256"]) == 64 for t in record["transitions"])
    assert record["scheduler"]["first_terminal_state"] == "completed"
    assert record["result_validity"] == "not_evaluated"
    handoff = record["handoff"]
    assert handoff["schema_version"] == tool.HANDOFF_SCHEMA
    assert handoff["job_id"] == identity["job_id"]
    assert handoff["expected_artifacts"] == ["rows", "manifest"]
    assert handoff["expected_rows"] == ["row-0001"]
    assert handoff["submission_receipt_sha256"] == RECEIPT
    assert handoff["harvest_request_schema"] == "terminal_job_harvest_request.v1"
    assert "harvest_terminal_job.py --check" in handoff["next_command"]
    assert "output/harvest/8811/request.json" in handoff["next_command"]
    assert handoff["result_validity"] == "not_evaluated"


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("FAILED", "failed"),
        ("CANCELLED", "cancelled"),
        ("PREEMPTED", "cancelled"),
        ("TIMEOUT", "timeout"),
    ),
)
def test_terminal_states_emit_handoff_but_never_success(raw: str, expected: str) -> None:
    report = tool.monitor_once(_projection(_job(observations=[_obs("RUNNING", 0), _obs(raw, 1)])))
    record = report["jobs"][0]
    assert report["status"] == "terminal"
    assert record["scheduler"]["first_terminal_state"] == expected
    assert record["handoff"]["observed_terminal_state"] == expected
    assert record["result_validity"] == "not_evaluated"


def test_array_mixed_state_stays_running_until_all_elements_terminal() -> None:
    record = _record(
        _job(
            observations=[
                _obs("RUNNING", 0, array={"0": "COMPLETED", "1": "RUNNING"}),
                _obs("FAILED", 1, array={"0": "COMPLETED", "1": "FAILED"}),
            ]
        )
    )
    assert [t["to"] for t in record["transitions"]] == ["running", "failed"]
    assert record["transitions"][0]["array_digest"] is not None
    summary = record["array_summary"]
    assert summary["states"] == {"completed": 1, "failed": 1}
    assert summary["elements"] == 2 and summary["mixed"] is True


def test_requeue_is_legal_and_changed_identity_is_rejected() -> None:
    identity = _identity()
    record = _record(
        _job(
            identity,
            [
                _obs("RUNNING", 0, identity=identity),
                _obs("REQUEUED", 1, identity=identity),
                _obs("RUNNING", 2, identity=identity),
                _obs("COMPLETED", 3, identity=identity),
            ],
        )
    )
    assert [t["to"] for t in record["transitions"]] == [
        "running",
        "requeued",
        "running",
        "completed",
    ]
    replaced = _job(
        identity,
        [
            _obs("RUNNING", 0, identity=identity),
            {**_obs("RUNNING", 1, identity=identity), "job_id": "9911"},
        ],
    )
    report = tool.monitor_once(_projection(replaced))
    assert report["status"] == "monitor_unavailable"
    assert "job_identity_changed" in report["reason_codes"]
    assert report["jobs"][0]["handoff"] is None


def test_terminal_state_remains_latched_after_unavailable_observation() -> None:
    record = _record(_job(observations=[_obs("COMPLETED", 0), _obs("MISSING", 1)]))
    assert record["scheduler"]["state"] == "completed"
    assert record["scheduler"]["terminal"] is True
    assert record["scheduler"]["first_terminal_state"] == "completed"
    assert record["handoff"]["observed_terminal_state"] == "completed"
    assert "contradictory_states" in _codes(record)


REJECTIONS = (
    ("clock_regression", (_obs("RUNNING", 2), _obs("COMPLETED", 1))),
    ("contradictory_states", (_obs("COMPLETED", 0), _obs("RUNNING", 1))),
    ("contradictory_states", (_obs("RUNNING", 0), _obs("PENDING", 1))),
    ("unsupported_scheduler_state", (_obs("RUNNING", 0), _obs("NOT_A_STATE", 1))),
    (
        "contradictory_states",
        (
            _obs("RUNNING", 0, array={"0": "COMPLETED", "1": "RUNNING"}),
            _obs("COMPLETED", 1, array={"0": "COMPLETED", "1": "RUNNING"}),
        ),
    ),
    ("malformed_response", (_obs("RUNNING", 0), _obs("RUNNING", 1, host="gpu-01"))),
)


@pytest.mark.parametrize(("code", "observations"), REJECTIONS, ids=[row[0] for row in REJECTIONS])
def test_invalid_observations_are_rejected(code: str, observations: tuple[Any, ...]) -> None:
    record = _record(_job(observations=list(observations)))
    assert code in _codes(record)
    assert record["result_validity"] == "not_evaluated"


def test_private_values_are_rejected_and_never_rendered() -> None:
    secret = "https://cluster.invalid/log?sig=SECRETVALUE"
    report = tool.monitor_once(
        _projection(_job(observations=[_obs("RUNNING", 0), _obs("RUNNING", 1, log_excerpt=secret)]))
    )
    assert report["status"] == "monitor_unavailable"
    assert "private_value_rejected" in _codes(report["jobs"][0])
    blob = tool.render_report_json(report) + tool.render_report_text(report)
    assert "SECRETVALUE" not in blob and "cluster.invalid" not in blob


def test_reduction_and_rendering_are_deterministic() -> None:
    projection = _projection(
        _job(observations=[_obs("PENDING", 0), _obs("RUNNING", 1), _obs("COMPLETED", 2)])
    )
    first = tool.monitor_once(projection)
    assert first == tool.monitor_once(json.loads(json.dumps(projection)))
    rendered = tool.render_report_json(first)
    assert rendered == tool.render_report_json(first)
    assert json.loads(rendered) == first and rendered.endswith("\n")


def test_monitor_window_expiry_is_not_terminal() -> None:
    identity = _identity()
    counter = iter(range(10))
    ticks = [0.0, 0.5, 0.5, 10.0]
    slept: list[float] = []
    report = tool.monitor_live(
        _projection(_job(identity)),
        lambda _job_id: _obs("RUNNING", next(counter), identity=identity),
        interval=0.5,
        max_wall_seconds=10.0,
        clock=lambda: ticks.pop(0) if ticks else 10.0,
        sleeper=slept.append,
    )
    record = report["jobs"][0]
    assert report["status"] == "monitor_window_expired"
    assert report["window"] == {
        "interval_seconds": 0.5,
        "max_wall_seconds": 10.0,
        "expired": True,
    }
    assert record["scheduler"]["terminal"] is False
    assert record["handoff"] is None and record["observation_count"] == 3
    assert slept == [0.5, 0.5]


def test_query_outage_stays_non_terminal_and_explicit() -> None:
    elapsed = {"value": 0.0}

    def clock() -> float:
        elapsed["value"] += 1.0
        return elapsed["value"]

    report = tool.monitor_live(
        _projection(_job()),
        lambda _job_id: None,
        interval=0.5,
        max_wall_seconds=2.0,
        clock=clock,
        sleeper=lambda _seconds: None,
    )
    record = report["jobs"][0]
    assert report["status"] == "monitor_unavailable"
    assert "query_unavailable" in report["reason_codes"]
    assert record["query_failure_count"] == 2
    assert record["scheduler"]["terminal"] is False and record["handoff"] is None


def test_projection_failures_are_explicit() -> None:
    invalid = _job()
    invalid.pop("commit")
    report = tool.monitor_once(_projection(invalid))
    assert report["status"] == "monitor_unavailable"
    assert "invalid_job_identity" in report["reason_codes"] and report["jobs"] == []
    duplicate = tool.monitor_once(_projection(_job(), _job()))
    assert "duplicate_job_identity" in duplicate["reason_codes"]
    with pytest.raises(tool.MonitorContractError):
        tool.parse_projection({"schema_version": "wrong", "jobs": []})
    with pytest.raises(tool.MonitorContractError):
        tool.monitor_once({"schema_version": tool.PROJECTION_SCHEMA, "jobs": []})


def test_cli_once_json_text_and_malformed(tmp_path: Path, capsys: Any) -> None:
    projection = _projection(_job(observations=[_obs("PENDING", 0), _obs("COMPLETED", 1)]))
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(projection), encoding="utf-8")
    assert tool.main(["--check", "--projection", str(path), "--once", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "terminal"
    assert tool.main(["--check", "--projection", str(path), "--once", "--format", "text"]) == 0
    assert capsys.readouterr().out.startswith("status=terminal")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert tool.main(["--check", "--projection", str(bad), "--once"]) == 2
    assert "malformed input" in capsys.readouterr().err
    assert (
        tool.main(["--check", "--projection", str(path), "--once", "--max-wall-seconds", "0"]) == 2
    )


def test_cli_live_state_query_is_bounded(tmp_path: Path, capsys: Any) -> None:
    identity = _identity()
    response = tmp_path / identity["job_id"]
    response.write_text(json.dumps(_obs("RUNNING", 0, identity=identity)), encoding="utf-8")
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(_projection(_job(identity))), encoding="utf-8")
    code = tool.main(
        [
            "--check",
            "--projection",
            str(path),
            "--state-query",
            f"cat {tmp_path}/{{job_id}}",
            "--interval",
            "0.01",
            "--max-wall-seconds",
            "0.15",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["status"] == "monitor_window_expired"
    assert report["jobs"][0]["scheduler"]["terminal"] is False


def test_cli_state_query_requires_job_id_binding(tmp_path: Path, capsys: Any) -> None:
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(_projection(_job())), encoding="utf-8")
    code = tool.main(
        ["--check", "--projection", str(path), "--state-query", "true", "--max-wall-seconds", "1"]
    )
    assert code == 2
    assert "{job_id} placeholder" in capsys.readouterr().err
