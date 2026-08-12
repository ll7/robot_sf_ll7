"""Tests for the camera-ready ``report_crosswalk.v1`` producer seam."""

from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import robot_sf.benchmark.camera_ready.campaign as campaign_module
from robot_sf.analysis_workbench.trace_failure_predicates import TraceFailurePredicate
from robot_sf.benchmark.camera_ready._crosswalk_producer import (
    CROSSWALK_SIDECAR_FILENAME,
    write_crosswalk_sidecar,
)
from robot_sf.benchmark.failure_diagnosis import (
    build_failure_diagnosis_payload,
    diagnose_from_trace_failure_predicate,
)
from robot_sf.benchmark.report_crosswalk import (
    build_crosswalk_example_fixture,
    validate_campaign_diagnostic_summary,
    validate_episode_diagnostic_summary,
)

if TYPE_CHECKING:
    from pathlib import Path


def _episode_record(
    *,
    episode_id: str,
    scenario_id: str,
    seed: int,
    success: bool,
    collision: bool,
    comfort: float,
    **extra: Any,
) -> dict[str, Any]:
    """Build a canonical episode row with complete source provenance."""
    return {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "metrics": {"comfort_exposure": comfort},
        "outcome": {
            "route_complete": success,
            "collision_event": collision,
            "timeout_event": not success and not collision,
        },
        "result_provenance": {
            "scenario_id": scenario_id,
            "seed": seed,
            "config_hash": f"config-{episode_id}",
            "repo_commit": "commit-test",
        },
        **extra,
    }


def _run_entry(path: Path, *, planner_key: str = "orca") -> dict[str, Any]:
    """Build the campaign run-entry contract used by the producer."""
    return {
        "status": "ok",
        "planner": {
            "key": planner_key,
            "algo": "orca",
            "kinematics": "differential_drive",
        },
        "episodes_path": path.relative_to(path.parents[2]).as_posix(),
    }


def _status_payload(status: str) -> dict[str, Any]:
    """Build a canonical diagnosis payload with a non-validity status."""
    predicate = TraceFailurePredicate(
        predicate_id="collision",
        time_interval_s=[1.0, 1.5],
        steps=[10, 15],
        involved_actors=["robot", "ped_0"],
        scenario_family="crosswalk",
        planner_id="orca",
        evidence_fields={"min_clearance_m": 0.1},
        severity="critical",
        validity_status=status,
    )
    record = diagnose_from_trace_failure_predicate(predicate)
    return build_failure_diagnosis_payload(
        [record],
        generated_at_utc="2026-01-01T00:00:00+00:00",
    )


def test_producer_writes_validated_sidecar_with_exact_provenance(tmp_path: Path) -> None:
    """The canonical helper preserves core fields and hashes its JSONL source."""
    episodes_path = tmp_path / "runs" / "orca" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True)
    rows = [
        _episode_record(
            episode_id="episode-0",
            scenario_id="corridor",
            seed=3,
            success=True,
            collision=False,
            comfort=0.25,
            diagnosis_payload=copy.deepcopy(
                build_crosswalk_example_fixture()["episodes"][0]["diagnosis_payload"]
            ),
        ),
        _episode_record(
            episode_id="episode-1",
            scenario_id="crossing",
            seed=4,
            success=False,
            collision=True,
            comfort=0.5,
        ),
    ]
    episodes_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    run_entries = [_run_entry(episodes_path)]

    sidecar_path = write_crosswalk_sidecar(
        tmp_path / "reports",
        campaign_id="campaign-test",
        run_entries=run_entries,
        repo_root=tmp_path,
    )
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))

    assert sidecar_path.name == CROSSWALK_SIDECAR_FILENAME
    assert payload["episode_count"] == 2
    assert payload["provenance"]["status"] == "complete"
    assert payload["input_quality"]["status"] == "valid"
    assert (
        payload["provenance"]["source_artifacts"][0]["episodes_sha256"]
        == hashlib.sha256(episodes_path.read_bytes()).hexdigest()
    )
    assert all(episode["provenance"]["status"] == "complete" for episode in payload["episodes"])

    by_id = {episode["episode_id"]: episode for episode in payload["episodes"]}
    assert by_id["episode-0"]["core_metrics"] == {
        "success": True,
        "collision": False,
        "comfort": 0.25,
    }
    assert by_id["episode-0"]["diagnosis"]["validity_state"] == "available"
    assert by_id["episode-1"]["diagnosis"]["validity_state"] == "unavailable"
    assert payload["campaign"]["core_metrics"] == {
        "success_rate": pytest.approx(0.5),
        "collision_rate": pytest.approx(0.5),
        "comfort_mean": pytest.approx(0.375),
    }
    validate_episode_diagnostic_summary(by_id["episode-0"])
    validate_campaign_diagnostic_summary(payload["campaign"])


def test_producer_marks_invalid_fallback_degraded_and_unsupported_inputs(tmp_path: Path) -> None:
    """Malformed or non-native inputs remain explicit and do not become available results."""
    episodes_path = tmp_path / "runs" / "diagnostics" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True)
    rows = [
        _episode_record(
            episode_id="invalid",
            scenario_id="invalid-scenario",
            seed=1,
            success=False,
            collision=False,
            comfort=0.0,
            diagnosis_payload={
                "schema_version": "failure_diagnosis.v0",
                "diagnosis_source": "wrong-source",
                "records": [],
            },
        ),
        _episode_record(
            episode_id="fallback",
            scenario_id="fallback-scenario",
            seed=2,
            success=False,
            collision=False,
            comfort=0.1,
            diagnosis_payload=_status_payload("fallback"),
        ),
        _episode_record(
            episode_id="degraded",
            scenario_id="degraded-scenario",
            seed=3,
            success=False,
            collision=False,
            comfort=0.2,
            diagnosis_payload=_status_payload("degraded"),
            execution_deviation={"schema_version": "execution_deviation.v1"},
        ),
    ]
    episodes_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\nnot-json\n",
        encoding="utf-8",
    )

    payload = json.loads(
        write_crosswalk_sidecar(
            tmp_path / "reports",
            campaign_id="campaign-invalid-inputs",
            run_entries=[_run_entry(episodes_path, planner_key="diagnostic-orca")],
            repo_root=tmp_path,
        ).read_text(encoding="utf-8")
    )
    by_id = {episode["episode_id"]: episode for episode in payload["episodes"]}

    assert payload["episode_count"] == 3
    assert payload["provenance"]["status"] == "incomplete"
    assert payload["input_quality"]["status"] == "invalid"
    assert payload["input_quality"]["invalid_source_record_count"] == 1
    assert by_id["invalid"]["diagnosis"]["validity_state"] == "invalid"
    assert by_id["invalid"]["diagnosis"]["provenance"] == "incomplete"
    assert by_id["fallback"]["diagnosis"]["validity_state"] == "fallback"
    assert by_id["degraded"]["diagnosis"]["validity_state"] == "degraded"
    assert by_id["degraded"]["execution_deviation"]["available"] is False
    assert by_id["degraded"]["execution_deviation"]["validity_state"] == "invalid"
    assert by_id["degraded"]["execution_deviation"]["provenance"] == "incomplete"
    assert payload["campaign"]["execution_deviation"]["available_count"] == 0


def test_campaign_finalizer_registers_crosswalk_artifact(monkeypatch, tmp_path: Path) -> None:
    """The real camera-ready finalization seam propagates the sidecar pointer."""
    summary = {"artifacts": {}}
    calls: dict[str, Any] = {}
    sidecar_path = tmp_path / "reports" / CROSSWALK_SIDECAR_FILENAME

    def fake_build(*_args: Any, **_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return summary, {}

    def fake_write(*_args: Any, **_kwargs: Any) -> Path:
        calls["writer"] = {
            "campaign_id": _kwargs["campaign_id"],
            "run_entries": _kwargs["run_entries"],
        }
        return sidecar_path

    def fake_export(*args: Any, **_kwargs: Any) -> str:
        calls["summary"] = args[3]
        return "publication-payload"

    monkeypatch.setattr(campaign_module, "_build_summary_and_write_run_files", fake_build)
    monkeypatch.setattr(campaign_module, "write_crosswalk_sidecar", fake_write)
    monkeypatch.setattr(campaign_module, "_export_and_write_final_artifacts", fake_export)

    result = campaign_module._finalize_campaign_outputs(
        object(),
        paths=SimpleNamespace(campaign_id="campaign-seam", reports_dir=tmp_path / "reports"),
        artifacts=SimpleNamespace(),
        run_entries=[{"status": "ok"}],
        planner_rows=[],
        campaign_integrity={},
        arm_rollup={},
        fairness_report=None,
        warnings=[],
        kinematics_matrix=("differential_drive",),
        invoked_command=None,
        skip_publication_bundle=True,
        dependencies=SimpleNamespace(),
    )

    assert result == "publication-payload"
    assert calls["writer"] == {"campaign_id": "campaign-seam", "run_entries": [{"status": "ok"}]}
    assert calls["summary"] is summary
    assert summary["artifacts"]["report_crosswalk_json"].endswith(CROSSWALK_SIDECAR_FILENAME)
