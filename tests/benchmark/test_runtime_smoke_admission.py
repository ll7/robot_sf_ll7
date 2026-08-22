"""Tests for exact-source runtime-smoke release admission."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

import robot_sf.benchmark.runtime_smoke_admission as admission_module
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.runtime_smoke_admission import (
    RUNTIME_SMOKE_CONFIG,
    RUNTIME_SMOKE_MANIFEST,
    RUNTIME_SMOKE_RELEASE_ID,
    RuntimeSmokeAdmissionError,
    validate_runtime_smoke_result,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, tuple[str, ...]]:
    planners = tuple(f"planner-{index}" for index in range(14))
    manifest = tmp_path / RUNTIME_SMOKE_MANIFEST
    config = tmp_path / RUNTIME_SMOKE_CONFIG
    scenario = tmp_path / "configs/scenarios/single/runtime_smoke.yaml"
    _write_yaml(scenario, {"scenarios": [{"name": "runtime-smoke-scenario"}]})
    config_payload = {
        "horizon": 600,
        "kinematics_matrix": ["differential_drive"],
        "seed_policy": {"mode": "fixed-list", "seeds": [111]},
        "planners": [{"key": key, "algo": f"algo-{index}"} for index, key in enumerate(planners)],
    }
    _write_yaml(config, config_payload)
    manifest_payload = {
        "release_id": RUNTIME_SMOKE_RELEASE_ID,
        "campaign_config_sha256": sha256_file(config),
        "scenario": {
            "matrix_path": "../../scenarios/single/runtime_smoke.yaml",
            "matrix_sha256": sha256_file(scenario),
        },
        "planners": {"keys": list(planners)},
        "kinematics": {"matrix": ["differential_drive"]},
    }
    _write_yaml(manifest, manifest_payload)

    root = tmp_path / "output/benchmarks/camera_ready/smoke"
    result_path = root / "release/release_result.json"
    checkpoint_receipt = tmp_path / "output/release/checkpoints/smoke.json"
    _write_json(checkpoint_receipt, {"submit_safe": True})
    finished = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    runs: list[dict] = []
    planner_rows: list[dict] = []
    for index, planner in enumerate(planners):
        episodes = root / "runs" / f"{planner}__differential_drive" / "episodes.jsonl"
        _write_json(
            episodes,
            {
                "scenario_id": "runtime-smoke-scenario",
                "seed": 111,
                "horizon": 600,
                "result_provenance": {"repo_commit": "a" * 40},
                "algorithm_metadata": {"algorithm": f"algo-{index}", "status": "ok"},
            },
        )
        runs.append(
            {
                "planner": {
                    "key": planner,
                    "algo": f"algo-{index}",
                    "kinematics": "differential_drive",
                    "horizon": 600,
                },
                "status": "ok",
                "episodes_path": str(episodes),
                "summary": {"written": 1, "failed_jobs": 0, "failures": []},
            }
        )
        planner_rows.append(
            {
                "planner_key": planner,
                "kinematics": "differential_drive",
                "status": "ok",
                "episodes": 1,
            }
        )
    result = {
        "campaign_id": "smoke",
        "benchmark_release": {
            "release_id": RUNTIME_SMOKE_RELEASE_ID,
            "manifest_path": RUNTIME_SMOKE_MANIFEST.as_posix(),
            "manifest_sha256": sha256_file(manifest),
            "canonical_campaign_config": RUNTIME_SMOKE_CONFIG.as_posix(),
            "canonical_campaign_config_sha256": sha256_file(config),
        },
        "resolved_manifest": {"planners": {"keys": list(planners)}},
        "total_runs": 14,
        "successful_runs": 14,
        "total_episodes": 14,
        "non_success_runs": 0,
        "accepted_unavailable_runs": 0,
        "unexpected_failed_runs": 0,
        "row_status_summary": {
            "successful_evidence_rows": 14,
            "accepted_unavailable_rows": 0,
            "unexpected_failed_rows": 0,
            "fallback_or_degraded_rows": 0,
        },
        "campaign_integrity": {"status": "valid", "checked_arm_count": 14},
        "checkpoint_staging_receipt": {
            "path": checkpoint_receipt.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(checkpoint_receipt),
            "submit_safe": True,
        },
        "release_benchmark_success": True,
        "release_status": "ok",
        "release_exit_code": 0,
    }
    _write_json(result_path, result)
    _write_json(
        root / "run_meta.json",
        {"repo": {"commit": "a" * 40}, "campaign_id": "smoke", "finished_at_utc": finished},
    )
    _write_json(
        root / "reports/campaign_summary.json",
        {
            "campaign": {
                "campaign_id": "smoke",
                "git_hash": "a" * 40,
                "total_runs": 14,
                "successful_runs": 14,
                "total_episodes": 14,
                "non_success_runs": 0,
                "accepted_unavailable_runs": 0,
                "unexpected_failed_runs": 0,
                "benchmark_success": True,
                "campaign_execution_status": "completed",
                "evidence_status": "valid",
                "row_status_summary": {
                    "successful_evidence_rows": 14,
                    "accepted_unavailable_rows": 0,
                    "unexpected_failed_rows": 0,
                    "fallback_or_degraded_rows": 0,
                },
            },
            "runs": runs,
            "planner_rows": planner_rows,
        },
    )
    monkeypatch.setattr(admission_module, "load_campaign_config", lambda _path: object())
    monkeypatch.setattr(
        admission_module,
        "validate_checkpoint_staging_receipt",
        lambda *_args, **_kwargs: {"submit_safe": True},
    )
    return result_path, planners


def _admit(result: Path, planners: tuple[str, ...], tmp_path: Path) -> dict:
    return validate_runtime_smoke_result(
        result,
        repo_root=tmp_path,
        expected_source_commit="a" * 40,
        expected_planner_keys=planners,
    )


def test_runtime_smoke_admits_exact_source_complete_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)

    admitted = _admit(result, planners, tmp_path)

    assert admitted["status"] == "admitted"
    assert admitted["planner_arms"] == 14
    assert admitted["episode_cells"] == 14
    assert admitted["fallback_or_degraded_rows"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(total_runs=13), "total_runs mismatch"),
        (
            lambda payload: payload["row_status_summary"].update(fallback_or_degraded_rows=1),
            "fallback_or_degraded_rows mismatch",
        ),
        (
            lambda payload: payload.update(release_benchmark_success=False),
            "release success mismatch",
        ),
    ],
)
def test_runtime_smoke_rejects_incomplete_or_fallback_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation, message: str
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    payload = json.loads(result.read_text(encoding="utf-8"))
    mutation(payload)
    _write_json(result, payload)

    with pytest.raises(RuntimeSmokeAdmissionError, match=message):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_different_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)

    with pytest.raises(RuntimeSmokeAdmissionError, match="source commit mismatch"):
        validate_runtime_smoke_result(
            result,
            repo_root=tmp_path,
            expected_source_commit="b" * 40,
            expected_planner_keys=planners,
        )


def test_runtime_smoke_rejects_forged_green_summary_without_raw_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary_path = result.parent.parent / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"] = []
    summary["planner_rows"] = []
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeSmokeAdmissionError, match="raw planner arms"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_raw_episode_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary = json.loads(
        (result.parent.parent / "reports/campaign_summary.json").read_text(encoding="utf-8")
    )
    episode_path = Path(summary["runs"][0]["episodes_path"])
    row = json.loads(episode_path.read_text(encoding="utf-8"))
    row["algorithm_metadata"]["fallback_or_degraded"] = True
    _write_json(episode_path, row)

    with pytest.raises(RuntimeSmokeAdmissionError, match="fallback or degraded"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_nested_foresight_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary = json.loads(
        (result.parent.parent / "reports/campaign_summary.json").read_text(encoding="utf-8")
    )
    episode_path = Path(summary["runs"][0]["episodes_path"])
    row = json.loads(episode_path.read_text(encoding="utf-8"))
    row["algorithm_metadata"]["foresight_prediction"] = {"fallback_used": True}
    _write_json(episode_path, row)

    with pytest.raises(RuntimeSmokeAdmissionError, match="fallback or degraded"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_checkpoint_receipt_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["checkpoint_staging_receipt"]["sha256"] = "0" * 64
    _write_json(result, payload)

    with pytest.raises(RuntimeSmokeAdmissionError, match="checkpoint staging receipt hash"):
        _admit(result, planners, tmp_path)
