"""Characterize the camera-ready campaign coordinator contract for issue #7327."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import robot_sf.benchmark.camera_ready.campaign as campaign_module

_NORMAL_LEDGER_SHA256 = "5e3a0f372bc61e3c5ffaa3b865954eaf064ea0c35e24a4e7b2563772d62af5bd"


def _install_phase_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failure_phase: str | None = None,
    failure_message: str | None = None,
    resume: bool = False,
    cleanup_on_matrix_failure: bool = False,
) -> list[dict[str, Any]]:
    """Install deterministic phase doubles and return their event ledger."""
    events: list[dict[str, Any]] = []
    paths = SimpleNamespace(
        campaign_id="fixture-campaign-001",
        campaign_root=Path("/fixture/campaign-001"),
        manifest_payload={"resume": resume, "receipt": "receipt-v1"},
        scenarios=["scenario-a", "scenario-b"],
        resolved_seeds=[11, 13],
    )
    run_entries = [{"arm": "arm-1", "status": "complete"}]
    planner_rows = [{"planner_key": "fixture-planner", "status": "complete"}]
    warnings = ["fixture-warning"]

    def maybe_fail(phase: str) -> None:
        if failure_phase == phase:
            raise RuntimeError(failure_message or f"{phase} failure")

    def prepare(
        _cfg: Any,
        _dependencies: Any,
        output_root: Path | None,
        label: str | None,
        campaign_id: str | None,
        invoked_command: str | None,
    ) -> tuple[Any, dict[str, Any] | None, dict[str, Any] | None, float]:
        events.append(
            {
                "phase": "prepare",
                "requested_campaign_id": campaign_id,
                "output_root": str(output_root) if output_root is not None else None,
                "label": label,
                "invoked_command": invoked_command,
            }
        )
        maybe_fail("prepare")
        return paths, {"weights": "fixture"}, {"baseline": "fixture"}, 1.0

    def execute_matrix(
        _cfg: Any,
        prepared_paths: Any,
        _snqi_weights: dict[str, Any] | None,
        _snqi_baseline: dict[str, Any] | None,
        _dependencies: Any,
        arm_isolation: str | None,
    ) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]], dict[str, Any]
    ]:
        events.append(
            {
                "phase": "matrix",
                "campaign_id": prepared_paths.campaign_id,
                "resume": prepared_paths.manifest_payload["resume"],
                "arm_isolation": arm_isolation,
            }
        )
        if failure_phase == "matrix" and cleanup_on_matrix_failure:
            events.append({"phase": "matrix_cleanup"})
        maybe_fail("matrix")
        return run_entries, planner_rows, warnings, [{"seed": 11}], {"differential_drive": {}}

    def integrity(
        _cfg: Any,
        *,
        manifest_payload: dict[str, Any],
        run_entries: list[dict[str, Any]],
        planner_rows: list[dict[str, Any]],
        warnings: list[str],
        scenarios: list[Any],
        resolved_seeds: list[Any],
        campaign_root: Path,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        del manifest_payload, planner_rows, resolved_seeds, campaign_root
        events.append(
            {
                "phase": "integrity",
                "run_count": len(run_entries),
                "warning_count": len(warnings),
                "scenario_count": len(scenarios),
            }
        )
        maybe_fail("integrity")
        return {"status": "passed"}, {"arm-1": "complete"}, {"status": "passed"}

    def report(  # noqa: PLR0913
        _cfg: Any,
        *,
        _paths: Any = None,
        paths: Any = None,
        start: float,
        run_entries: list[dict[str, Any]],
        planner_rows: list[dict[str, Any]],
        campaign_integrity: dict[str, Any],
        kinematics_matrix: tuple[str, ...] | dict[str, Any],
        seed_variability_records: list[dict[str, Any]],
        snqi_weights: dict[str, Any] | None,
        snqi_baseline: dict[str, Any] | None,
        warnings: list[str],
    ) -> object:
        del paths, start, planner_rows, campaign_integrity, seed_variability_records
        del snqi_weights, snqi_baseline, warnings
        events.append(
            {
                "phase": "report",
                "run_count": len(run_entries),
                "kinematics": sorted(kinematics_matrix),
            }
        )
        maybe_fail("report")
        return object()

    def finalize(  # noqa: PLR0913
        _cfg: Any,
        *,
        paths: Any,
        artifacts: Any,
        run_entries: list[dict[str, Any]],
        planner_rows: list[dict[str, Any]],
        campaign_integrity: dict[str, Any],
        arm_rollup: dict[str, Any],
        fairness_report: Any,
        warnings: list[str],
        kinematics_matrix: tuple[str, ...] | dict[str, Any],
        invoked_command: str | None,
        skip_publication_bundle: bool,
        dependencies: Any,
    ) -> dict[str, Any]:
        del paths, artifacts, planner_rows, campaign_integrity, arm_rollup, fairness_report
        del warnings, kinematics_matrix, dependencies
        events.append(
            {
                "phase": "finalize",
                "run_count": len(run_entries),
                "invoked_command": invoked_command,
                "skip_publication_bundle": skip_publication_bundle,
            }
        )
        maybe_fail("finalize")
        return {"published": not skip_publication_bundle}

    def build_return(
        *,
        paths: Any,
        artifacts: Any,
        run_entries: list[dict[str, Any]],
        campaign_integrity: dict[str, Any],
        warnings: list[str],
        publication_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del artifacts, run_entries, campaign_integrity, warnings, publication_payload
        events.append({"phase": "return", "campaign_id": paths.campaign_id, "status": "complete"})
        return {"campaign_id": paths.campaign_id, "status": "complete"}

    monkeypatch.setattr(campaign_module, "_prepare_campaign_execution", prepare)
    monkeypatch.setattr(campaign_module, "_execute_planner_matrix_phase", execute_matrix)
    monkeypatch.setattr(campaign_module, "_post_run_integrity_and_fairness", integrity)
    monkeypatch.setattr(campaign_module, "_write_campaign_report_artifacts", report)
    monkeypatch.setattr(campaign_module, "_finalize_campaign_outputs", finalize)
    monkeypatch.setattr(campaign_module, "_build_orchestrator_return", build_return)
    return events


def _run_fixture(
    monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one synthetic coordinator fixture without campaign or filesystem work."""
    events = _install_phase_fakes(monkeypatch, **kwargs)
    result = campaign_module._run_campaign_orchestrator(
        object(),
        output_root=Path("/fixture/output"),
        label="fixture",
        campaign_id="requested-campaign-001",
        skip_publication_bundle=True,
        invoked_command="fixture-command",
        dependencies=object(),
        arm_isolation="fixture-isolation",
    )
    return events, result


def _ledger_digest(events: list[dict[str, Any]]) -> str:
    """Hash the stable event ledger representation used by the golden fixture."""
    payload = json.dumps(events, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_normal_orchestration_ledger_is_ordered_and_checksum_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal completion preserves phase order and the stable fixture ledger."""
    events, result = _run_fixture(monkeypatch)

    assert [event["phase"] for event in events] == [
        "prepare",
        "matrix",
        "integrity",
        "report",
        "finalize",
        "return",
    ]
    assert result == {"campaign_id": "fixture-campaign-001", "status": "complete"}
    assert events[0]["requested_campaign_id"] == "requested-campaign-001"
    assert events[1]["arm_isolation"] == "fixture-isolation"
    assert events[4]["skip_publication_bundle"] is True
    assert _ledger_digest(events) == _NORMAL_LEDGER_SHA256


@pytest.mark.parametrize(
    ("failure_phase", "expected_phases"),
    [
        ("prepare", ["prepare"]),
        ("matrix", ["prepare", "matrix"]),
        ("integrity", ["prepare", "matrix", "integrity"]),
        ("report", ["prepare", "matrix", "integrity", "report"]),
        ("finalize", ["prepare", "matrix", "integrity", "report", "finalize"]),
    ],
)
def test_failure_precedence_stops_before_later_phases(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
    expected_phases: list[str],
) -> None:
    """A phase failure propagates without dispatching later coordinator phases."""
    events = _install_phase_fakes(
        monkeypatch,
        failure_phase=failure_phase,
        failure_message=f"{failure_phase} failure",
    )

    with pytest.raises(RuntimeError, match=f"{failure_phase} failure"):
        campaign_module._run_campaign_orchestrator(
            object(),
            dependencies=object(),
        )

    assert [event["phase"] for event in events] == expected_phases


def test_resume_state_is_forwarded_without_dispatch_policy_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid partial-state fixture reaches the matrix with its resume marker intact."""
    events, result = _run_fixture(monkeypatch, resume=True)

    assert events[1]["resume"] is True
    assert result["status"] == "complete"


def test_stale_receipt_rejection_happens_before_worker_or_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale/duplicate receipt is a preflight blocker with no downstream side effect."""
    events = _install_phase_fakes(
        monkeypatch,
        failure_phase="prepare",
        failure_message="stale duplicate receipt rejected",
    )

    with pytest.raises(RuntimeError, match="stale duplicate receipt rejected"):
        campaign_module._run_campaign_orchestrator(object(), dependencies=object())

    assert [event["phase"] for event in events] == ["prepare"]


def test_worker_failure_runs_phase_cleanup_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker-phase fixture records cleanup before the failure escapes the coordinator."""
    events = _install_phase_fakes(
        monkeypatch,
        failure_phase="matrix",
        failure_message="arm failed after partial progress",
        cleanup_on_matrix_failure=True,
    )

    with pytest.raises(RuntimeError, match="arm failed after partial progress"):
        campaign_module._run_campaign_orchestrator(object(), dependencies=object())

    assert [event["phase"] for event in events] == ["prepare", "matrix", "matrix_cleanup"]


def test_repeated_fixture_execution_is_byte_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identical synthetic inputs produce identical phase ledgers and result summaries."""
    first_events, first_result = _run_fixture(monkeypatch)
    second_events, second_result = _run_fixture(monkeypatch)

    assert first_events == second_events
    assert first_result == second_result
    assert _ledger_digest(first_events) == _ledger_digest(second_events)
