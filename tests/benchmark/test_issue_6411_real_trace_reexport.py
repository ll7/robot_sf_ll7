"""Contract tests for issue #6411 real-arm binding and trace normalization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.trace_reexport_packaging import (
    EXECUTION_COMMIT,
    REAL_REEXPORT_ARMS,
    REAL_REEXPORT_EXCEPTION_SEEDS,
    RealReexportBindingError,
    bind_real_reexport_arms,
)
from scripts.tools.build_simulation_trace_export import (
    ALLOWLISTED_METADATA_FIELDS,
    SimulationTraceNormalizationError,
    build_simulation_trace_export_with_receipt,
)


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outcome(
    *, planner: str, scenario_id: str, seed: int, mismatch: bool = False
) -> dict[str, bool]:
    success = planner == "goal" or (scenario_id == "classic_doorway_medium" and seed % 2 == 1)
    if mismatch:
        success = not success
    return {
        "success": success,
        "route_complete": success,
        "collision_event": not success,
        "timeout_event": False,
    }


def _trace_frame(*, unknown_field: str | None = None) -> dict[str, Any]:
    pedestrian: dict[str, Any] = {
        "id": "ped-0",
        "position": [1.0, 0.0],
        "velocity": [0.0, 0.0],
        "track_confidence": 0.9,
        "visibility_evidence_reason": "synthetic-test",
        "visibility_evidence_status": "available",
        "visibility_state": "visible",
    }
    if unknown_field is not None:
        pedestrian[unknown_field] = "reject-me"
    return {
        "step": 0,
        "time_s": 0.1,
        "robot": {"position": [0.0, 0.0], "heading": 0.0, "velocity": [0.0, 0.0]},
        "pedestrians": [pedestrian],
        "planner": {
            "event": "step",
            "event_id": "frame-0000",
            "selected_action": {"linear_velocity": 0.0, "angular_velocity": 0.0},
        },
    }


def _row(
    *,
    planner: str,
    scenario_id: str,
    seed: int,
    mismatch: bool = False,
    unknown_field: str | None = None,
) -> dict[str, Any]:
    outcome = _outcome(
        planner=planner,
        scenario_id=scenario_id,
        seed=seed,
        mismatch=mismatch,
    )
    config_hash = f"run-config-{planner}-{scenario_id}"
    return {
        "episode_id": f"rerun-{planner}-{scenario_id}-{seed}",
        "scenario_id": scenario_id,
        "seed": seed,
        "algo": planner,
        "git_hash": EXECUTION_COMMIT,
        "config_hash": config_hash,
        "scenario_params": {"algo": planner, "id": scenario_id},
        "metrics": {"success": outcome["success"]},
        "outcome": {key: value for key, value in outcome.items() if key != "success"},
        "algorithm_metadata": {
            "planner_kinematics": {"robot_kinematics": "differential_drive"},
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v1",
                "steps": [_trace_frame(unknown_field=unknown_field)],
            },
        },
    }


@pytest.fixture
def real_arm_inputs(tmp_path: Path) -> dict[str, Any]:
    """Build manifest/config/row fixtures with the real three-arm identities."""

    roots: dict[str, Path] = {}
    config_evidence: dict[str, dict[str, str]] = {}
    release_outcomes: dict[tuple[str, str, int], dict[str, bool]] = {}
    request_rows: list[dict[str, Any]] = []

    for arm in REAL_REEXPORT_ARMS:
        root = tmp_path / "sources" / arm.key
        roots[arm.key] = root
        config_source = tmp_path / "configs" / f"{arm.key}.yaml"
        config_source.parent.mkdir(parents=True, exist_ok=True)
        config_source.write_text(f"name: {arm.config_name}\n", encoding="utf-8")
        config_hash = f"run-config-{arm.planner}-{arm.scenario_id}"
        config_evidence[arm.key] = {
            "config_name": arm.config_name,
            "config_path": arm.config_path,
            "source_path": str(config_source),
            "config_hash": config_hash,
            "sha256": _sha256(config_source),
        }
        _write_json(
            root / "campaign_manifest.json",
            {
                "campaign_id": f"campaign-{arm.key}",
                "job_id": arm.job_id,
                "name": arm.config_name,
                "config_path": arm.config_path,
                "config_hash": config_hash,
                "scenario_matrix": "configs/scenarios/classic_interactions_francis2023.yaml",
                "scenario_candidates": [arm.scenario_id],
                "seed_policy": {"resolved_seeds": list(arm.seeds)},
                "git": {"commit": EXECUTION_COMMIT},
                "planners": [{"key": arm.planner}],
            },
        )
        rows = []
        for seed in arm.seeds:
            mismatch = seed in arm.not_admitted_seeds
            row = _row(
                planner=arm.planner,
                scenario_id=arm.scenario_id,
                seed=seed,
                mismatch=mismatch,
            )
            rows.append(row)
            release_outcomes[(arm.planner, arm.scenario_id, seed)] = _outcome(
                planner=arm.planner,
                scenario_id=arm.scenario_id,
                seed=seed,
            )
            request_rows.append(
                {
                    "planner": arm.planner,
                    "scenario_id": arm.scenario_id,
                    "seed": seed,
                }
            )
        episodes = root / "runs" / f"{arm.planner}__differential_drive" / "episodes.jsonl"
        episodes.parent.mkdir(parents=True, exist_ok=True)
        episodes.write_bytes(b"".join(_json_bytes(row) for row in rows))

    request_manifest = tmp_path / "request_manifest.json"
    _write_json(
        request_manifest,
        {
            "schema_version": "issue_5446_trace_reexport_list.v1",
            "n_tuples": 90,
            "tuples": request_rows,
        },
    )
    return {
        "roots": roots,
        "config_evidence": config_evidence,
        "release_outcomes": release_outcomes,
        "request_manifest": request_manifest,
        "tmp_path": tmp_path,
    }


def test_normalization_receipt_is_allowlisted_and_digest_bound(tmp_path: Path) -> None:
    """Only the four approved pedestrian metadata fields disappear."""

    source = tmp_path / "row.jsonl"
    row = _row(planner="ppo", scenario_id="classic_doorway_medium", seed=113)
    source.write_bytes(_json_bytes(row))

    normalized, receipt = build_simulation_trace_export_with_receipt(source)

    assert receipt["schema_version"] == "simulation_trace_export.normalization_receipt.v1"
    assert receipt["trace_schema_version"] == "simulation_trace_export.v1"
    assert receipt["raw_trace_sha256"] == _sha256(source)
    assert receipt["normalized_trace_sha256"] == hashlib.sha256(_json_bytes(normalized)).hexdigest()
    removed = receipt["removed_fields"]
    assert {item["field"] for item in removed} == set(ALLOWLISTED_METADATA_FIELDS)
    assert all(item["reason"] for item in removed)
    assert normalized["frames"][0]["pedestrians"][0] == {
        "id": "ped-0",
        "position": [1.0, 0.0],
        "velocity": [0.0, 0.0],
    }
    assert normalized["source"] == {
        "scenario_id": "classic_doorway_medium",
        "seed": 113,
        "planner_id": "ppo",
        "episode_id": "rerun-ppo-classic_doorway_medium-113",
        "generated_by": normalized["source"]["generated_by"],
    }
    assert normalized["frames"][0]["step"] == 0
    assert normalized["frames"][0]["time_s"] == 0.1
    assert normalized["frames"][0]["robot"] == {
        "position": [0.0, 0.0],
        "heading": 0.0,
        "velocity": [0.0, 0.0],
    }
    assert normalized["frames"][0]["planner"] == {
        "event": "step",
        "event_id": "frame-0000",
        "selected_action": {"linear_velocity": 0.0, "angular_velocity": 0.0},
    }
    assert receipt["semantic_payload_unchanged"] is True


def test_normalization_rejects_unknown_extra_field(tmp_path: Path) -> None:
    """A fifth pedestrian metadata field cannot be silently discarded."""

    source = tmp_path / "row.jsonl"
    source.write_bytes(
        _json_bytes(
            _row(
                planner="ppo",
                scenario_id="classic_doorway_medium",
                seed=113,
                unknown_field="unallowlisted_field",
            )
        )
    )

    with pytest.raises(SimulationTraceNormalizationError, match="unallowlisted"):
        build_simulation_trace_export_with_receipt(source)


def test_normalization_rejects_raw_digest_drift(tmp_path: Path) -> None:
    """A supplied raw digest must match the bytes actually normalized."""

    source = tmp_path / "row.jsonl"
    source.write_bytes(
        _json_bytes(_row(planner="ppo", scenario_id="classic_doorway_medium", seed=113))
    )

    with pytest.raises(SimulationTraceNormalizationError, match="disagrees"):
        build_simulation_trace_export_with_receipt(source, source_signature="0" * 64)


def test_real_arm_binding_emits_90_receipts_and_88_plus_2_boundary(
    real_arm_inputs: dict[str, Any],
) -> None:
    """All three real identities bind, while doorway exceptions remain excluded."""

    output_dir = real_arm_inputs["tmp_path"] / "normalized"
    receipt_path = real_arm_inputs["tmp_path"] / "binding_receipt.json"
    receipt = bind_real_reexport_arms(
        real_arm_inputs["roots"],
        expected_outcomes=real_arm_inputs["release_outcomes"],
        config_evidence=real_arm_inputs["config_evidence"],
        request_manifest=real_arm_inputs["request_manifest"],
        normalized_output_dir=output_dir,
        receipt_path=receipt_path,
    )

    assert receipt["schema_version"] == "issue_6411_real_reexport_binding.v1"
    assert receipt["trace_schema_version"] == "simulation_trace_export.v1"
    assert receipt["summary"] == {"n_rows": 90, "n_admitted": 88, "n_not_admitted": 2}
    assert {
        (row["planner"], row["scenario_id"], row["seed"])
        for row in receipt["rows"]
        if row["admission_status"] == "not_admitted"
    } == {("ppo", "classic_doorway_medium", seed) for seed in REAL_REEXPORT_EXCEPTION_SEEDS}
    assert len(list(output_dir.rglob("*.json"))) == 90
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    for row in receipt["rows"]:
        trace_path = Path(row["normalized_trace_path"])
        assert trace_path.is_file()
        assert _sha256(trace_path) == row["normalized_trace_sha256"]
        assert {item["field"] for item in row["removed_fields"]} == set(ALLOWLISTED_METADATA_FIELDS)


def test_real_arm_binding_rejects_wrong_job_before_normalized_output(
    real_arm_inputs: dict[str, Any],
) -> None:
    """A wrong-arm manifest cannot leave a partial normalized tree."""

    doorway_manifest = real_arm_inputs["roots"]["doorway_ppo"] / "campaign_manifest.json"
    manifest = json.loads(doorway_manifest.read_text(encoding="utf-8"))
    manifest["job_id"] = "wrong-job"
    _write_json(doorway_manifest, manifest)
    output_dir = real_arm_inputs["tmp_path"] / "normalized"

    with pytest.raises(RealReexportBindingError, match="job mismatch"):
        bind_real_reexport_arms(
            real_arm_inputs["roots"],
            expected_outcomes=real_arm_inputs["release_outcomes"],
            config_evidence=real_arm_inputs["config_evidence"],
            normalized_output_dir=output_dir,
        )
    assert not output_dir.exists()


def test_real_arm_binding_rejects_config_hash_drift(
    real_arm_inputs: dict[str, Any],
) -> None:
    """A config descriptor cannot silently drift from its campaign manifest."""

    config_evidence = {
        key: dict(value) for key, value in real_arm_inputs["config_evidence"].items()
    }
    config_evidence["doorway_ppo"]["config_hash"] = "wrong-config-hash"

    with pytest.raises(RealReexportBindingError, match="config hash evidence mismatch"):
        bind_real_reexport_arms(
            real_arm_inputs["roots"],
            expected_outcomes=real_arm_inputs["release_outcomes"],
            config_evidence=config_evidence,
        )
