"""Tests for exact-source runtime-smoke release admission."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

import robot_sf.benchmark.runtime_smoke_admission as admission_module
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.result_provenance import build_result_provenance_manifest
from robot_sf.benchmark.runtime_smoke_admission import (
    RUNTIME_SMOKE_CONFIG,
    RUNTIME_SMOKE_MANIFEST,
    RUNTIME_SMOKE_PLANNER_KEYS,
    RUNTIME_SMOKE_RELEASE_ID,
    RuntimeSmokeAdmissionError,
    _episode_horizon,
    _read_episode_rows,
    _read_object,
    _read_yaml_object,
    _source_commit,
    _strict_int,
    _validate_age,
    validate_runtime_smoke_result,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, tuple[str, ...]]:
    planners = RUNTIME_SMOKE_PLANNER_KEYS
    manifest = tmp_path / RUNTIME_SMOKE_MANIFEST
    config = tmp_path / RUNTIME_SMOKE_CONFIG
    scenario = tmp_path / "configs/scenarios/single/francis2023_blind_corner.yaml"
    _write_yaml(scenario, {"scenarios": [{"name": "runtime-smoke-scenario"}]})
    schema = tmp_path / "robot_sf/benchmark/schemas/episode.schema.v1.json"
    _write_json(schema, {"$schema": "https://json-schema.org/draft/2020-12/schema"})
    planner_configs: dict[str, str] = {}
    for index, planner in enumerate(planners):
        config_rel = f"configs/algos/runtime-smoke-{index}.yaml"
        _write_yaml(tmp_path / config_rel, {"planner_key": planner})
        planner_configs[planner] = config_rel
    config_payload = {
        "horizon": 600,
        "kinematics_matrix": ["differential_drive"],
        "seed_policy": {"mode": "fixed-list", "seeds": [111]},
        "planners": [
            {
                "key": key,
                "algo": f"algo-{index}",
                "algo_config": planner_configs[key],
            }
            for index, key in enumerate(planners)
        ],
    }
    _write_yaml(config, config_payload)
    manifest_payload = {
        "release_id": RUNTIME_SMOKE_RELEASE_ID,
        "campaign_config_sha256": sha256_file(config),
        "scenario": {
            "matrix_path": "../../scenarios/single/francis2023_blind_corner.yaml",
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
        episode_row = {
            "algo": f"algo-{index}",
            "episode_id": f"runtime-smoke-scenario--111--{index}",
            "scenario_id": "runtime-smoke-scenario",
            "seed": 111,
            "horizon": 600,
            "config_hash": f"config-{index}",
            "git_hash": "a" * 40,
            "result_provenance": {
                "repo_commit": "a" * 40,
                "scenario_id": "runtime-smoke-scenario",
                "seed": 111,
                "config_hash": f"config-{index}",
            },
            "algorithm_metadata": {
                "algorithm": "ppo" if planner == "guarded_ppo" else f"algo-{index}",
                "canonical_algorithm": f"algo-{index}",
                "planner_contract": {"planner_id": f"algo-{index}"},
                "status": "ok",
            },
        }
        _write_json(episodes, episode_row)
        sidecar = build_result_provenance_manifest(
            out_path=episodes,
            episode_records=[episode_row],
            schema_path=schema,
            scenario_path=scenario,
            scenarios=[
                {
                    "name": "runtime-smoke-scenario",
                    "seeds": [111],
                    "robot_config": {"type": "differential_drive"},
                }
            ],
            algo=f"algo-{index}",
            algo_config_path=tmp_path / planner_configs[planner],
            benchmark_profile="baseline-safe",
            suite_key="francis2023",
            total_jobs=1,
            written=1,
            horizon=600,
            dt=0.1,
            record_forces=True,
            active_observation_mode="socnav_state",
            active_observation_level="tracked_agents_no_noise",
        )
        sidecar["run"]["repo_commit"] = "a" * 40
        sidecar["rows"][0]["repo_commit"] = "a" * 40
        sidecar["rows"][0]["config_hash"] = f"config-{index}"
        _write_json(episodes.with_name(f"{episodes.name}.provenance.json"), sidecar)
        arm_summary = {
            "status": "ok",
            "written": 1,
            "episodes_total": 1,
            "failed_jobs": 0,
            "failures": [],
            "out_path": str(episodes),
        }
        summary_path = episodes.parent / "summary.json"
        _write_json(summary_path, arm_summary)
        runs.append(
            {
                "planner": {
                    "key": planner,
                    "algo": f"algo-{index}",
                    "algo_config_path": planner_configs[planner],
                    "kinematics": "differential_drive",
                    "horizon": 600,
                },
                "status": "ok",
                "episodes_path": str(episodes),
                "summary_path": str(summary_path),
                "summary": arm_summary,
            }
        )
        planner_rows.append(
            {
                "planner_key": planner,
                "kinematics": "differential_drive",
                "status": "ok",
                "episodes": 1,
                "benchmark_success": "true",
            }
        )
    result = {
        "campaign_id": "smoke",
        "campaign_root": str(root),
        "summary_json": str(root / "reports/campaign_summary.json"),
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
        "benchmark_success": True,
        "status": "benchmark_success",
        "evidence_status": "valid",
        "campaign_execution_status": "completed",
        "exit_code": 0,
    }
    _write_json(
        root / "campaign_manifest.json",
        {
            "campaign_id": "smoke",
            "git": {"commit": "a" * 40},
            "scenario_matrix": "configs/scenarios/single/francis2023_blind_corner.yaml",
            "seed_policy": {"resolved_seeds": [111]},
            "planners": [
                {
                    "key": planner,
                    "algo": f"algo-{index}",
                    "algo_config_path": planner_configs[planner],
                    "checkpoint_provenance": {
                        "status": "not_run",
                        "load_succeeded": None,
                        "fallback_triggered": None,
                    },
                }
                for index, planner in enumerate(planners)
            ],
            "benchmark_release": result["benchmark_release"],
        },
    )
    _write_json(
        root / "manifest.json",
        {
            "git_hash": "a" * 40,
            "benchmark_release": result["benchmark_release"],
        },
    )
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
                "status": "benchmark_success",
                "campaign_execution_status": "completed",
                "evidence_status": "valid",
                "exit_code": 0,
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


def _set_episode_runtime(result: Path, planner: str, runtime: dict[str, object]) -> None:
    """Update one fixture row and its provenance-sidecar byte hash."""
    episodes_path = result.parent.parent / "runs" / f"{planner}__differential_drive/episodes.jsonl"
    row = json.loads(episodes_path.read_text(encoding="utf-8"))
    row["algorithm_metadata"]["planner_runtime"] = runtime
    _write_json(episodes_path, row)
    sidecar_path = episodes_path.with_name(f"{episodes_path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    artifact = next(
        item for item in sidecar["raw_artifacts"] if item.get("kind") == "episodes_jsonl"
    )
    artifact["sha256"] = sha256_file(episodes_path)
    _write_json(sidecar_path, sidecar)


def test_runtime_smoke_parsers_reject_malformed_or_wrong_shaped_inputs(tmp_path: Path) -> None:
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeSmokeAdmissionError, match="missing or invalid"):
        _read_object(invalid_json, "test JSON")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeSmokeAdmissionError, match="not a JSON object"):
        _read_object(list_json, "test JSON")

    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("[", encoding="utf-8")
    with pytest.raises(RuntimeSmokeAdmissionError, match="missing or invalid"):
        _read_yaml_object(invalid_yaml, "test YAML")
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeSmokeAdmissionError, match="not a YAML object"):
        _read_yaml_object(list_yaml, "test YAML")


def test_runtime_smoke_scalar_provenance_helpers_fail_closed() -> None:
    assert _strict_int(True) is None
    assert _strict_int(" 600 ") == 600
    assert _strict_int(1.5) is None
    assert _source_commit({"git_hash": "ABC"}) == "abc"
    assert (
        _episode_horizon({"result_provenance": {"simulator_settings": {"horizon": "600"}}}) == 600
    )
    assert _episode_horizon({}) is None


def test_runtime_smoke_age_rejects_invalid_and_stale_timestamps() -> None:
    with pytest.raises(RuntimeSmokeAdmissionError, match="timestamp is invalid"):
        _validate_age({"finished_at_utc": "not-a-time"}, max_age_hours=24)
    stale = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    with pytest.raises(RuntimeSmokeAdmissionError, match="stale or future-dated"):
        _validate_age({"finished_at_utc": stale}, max_age_hours=24)
    naive = datetime.now(UTC).replace(tzinfo=None, microsecond=0).isoformat()
    assert _validate_age({"finished_at_utc": naive}, max_age_hours=24).endswith("Z")
    for invalid_limit in (float("nan"), float("inf"), 0.0, -1.0, 24.1):
        with pytest.raises(RuntimeSmokeAdmissionError, match="maximum age"):
            _validate_age({"finished_at_utc": naive}, max_age_hours=invalid_limit)


def test_runtime_smoke_episode_reader_handles_blank_and_rejects_malformed_rows(
    tmp_path: Path,
) -> None:
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text('\n{"ok": true}\n', encoding="utf-8")
    assert _read_episode_rows(rows_path) == [{"ok": True}]
    rows_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(RuntimeSmokeAdmissionError, match="is not an object"):
        _read_episode_rows(rows_path)
    rows_path.write_text("{\n", encoding="utf-8")
    with pytest.raises(RuntimeSmokeAdmissionError, match="missing or invalid"):
        _read_episode_rows(rows_path)


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


def test_runtime_smoke_rejects_reused_episode_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary_path = result.parent.parent / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"][1]["episodes_path"] = summary["runs"][0]["episodes_path"]
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeSmokeAdmissionError, match="episode artifact rejected"):
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


def test_runtime_smoke_rejects_fallback_hidden_in_run_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary_path = result.parent.parent / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"][0]["summary"]["benchmark_availability"] = {"status": "fallback"}
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeSmokeAdmissionError, match="fallback or degraded"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_nested_preflight_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary_path = result.parent.parent / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"][0]["summary"]["preflight"] = {"learned_policy_contract": {"status": "fallback"}}
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeSmokeAdmissionError, match="fallback or degraded"):
        _admit(result, planners, tmp_path)


@pytest.mark.parametrize("reason", [None, ""])
@pytest.mark.parametrize("nested", [False, True])
def test_runtime_smoke_allows_empty_foresight_fallback_reason_when_unused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: object,
    nested: bool,
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    episode_path = result.parent.parent / "runs/goal__differential_drive/episodes.jsonl"
    row = json.loads(episode_path.read_text(encoding="utf-8"))
    metadata = row["algorithm_metadata"]
    if nested:
        metadata = metadata.setdefault("planner_runtime", {})
    metadata["foresight_prediction"] = {
        "fallback_used": False,
        "fallback_reason": reason,
    }
    _write_json(episode_path, row)
    sidecar_path = episode_path.with_name(f"{episode_path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["raw_artifacts"][0]["sha256"] = sha256_file(episode_path)
    _write_json(sidecar_path, sidecar)

    assert _admit(result, planners, tmp_path)["status"] == "admitted"


@pytest.mark.parametrize("reason", [None, ""])
def test_runtime_smoke_allows_empty_summary_foresight_reason_when_unused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: object,
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary_path = result.parent.parent / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"][0]["summary"]["algorithm_metadata_contract"] = {
        "planner_runtime": {
            "foresight_prediction": {
                "fallback_used": False,
                "fallback_reason": reason,
            }
        }
    }
    _write_json(summary_path, summary)

    assert _admit(result, planners, tmp_path)["status"] == "admitted"


def test_runtime_smoke_allows_planner_aggregate_not_applicable_learned_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary_path = result.parent.parent / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["planner_rows"][0]["learned_policy_contract_status"] = "not_applicable"
    _write_json(summary_path, summary)

    assert _admit(result, planners, tmp_path)["status"] == "admitted"


@pytest.mark.parametrize("reason", [None, ""])
@pytest.mark.parametrize("surface", ["summary", "row"])
def test_runtime_smoke_allows_empty_generic_runtime_reason_when_unused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: object,
    surface: str,
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary_path = result.parent.parent / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime = {
        "fallback_triggered": False,
        "fallback_reason": reason,
    }
    if surface == "summary":
        summary["runs"][0]["summary"]["algorithm_metadata_contract"] = {"planner_runtime": runtime}
        _write_json(summary_path, summary)
    else:
        _set_episode_runtime(result, planners[0], runtime)

    assert _admit(result, planners, tmp_path)["status"] == "admitted"


@pytest.mark.parametrize("surface", ["summary", "row"])
def test_runtime_smoke_rejects_nonempty_generic_runtime_reason_when_unused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary_path = result.parent.parent / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runtime = {
        "fallback_triggered": False,
        "fallback_reason": "unexpected runtime fallback",
    }
    if surface == "summary":
        summary["runs"][0]["summary"]["algorithm_metadata_contract"] = {"planner_runtime": runtime}
        _write_json(summary_path, summary)
    else:
        _set_episode_runtime(result, planners[0], runtime)

    with pytest.raises(RuntimeSmokeAdmissionError, match="fallback or degraded"):
        _admit(result, planners, tmp_path)


@pytest.mark.parametrize("nested", [False, True])
def test_runtime_smoke_rejects_nonempty_foresight_fallback_reason_when_unused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested: bool
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    episode_path = result.parent.parent / "runs/goal__differential_drive/episodes.jsonl"
    row = json.loads(episode_path.read_text(encoding="utf-8"))
    metadata = row["algorithm_metadata"]
    if nested:
        metadata = metadata.setdefault("planner_runtime", {})
    metadata["foresight_prediction"] = {
        "fallback_used": False,
        "fallback_reason": "unexpected runtime fallback",
    }
    _write_json(episode_path, row)
    sidecar_path = episode_path.with_name(f"{episode_path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["raw_artifacts"][0]["sha256"] = sha256_file(episode_path)
    _write_json(sidecar_path, sidecar)

    with pytest.raises(RuntimeSmokeAdmissionError, match="fallback or degraded"):
        _admit(result, planners, tmp_path)


@pytest.mark.parametrize(
    ("runtime_marker", "value"),
    [
        ("fallback_count", "1"),
        ("fallback_used", "true"),
        ("benchmark_success", False),
        ("load_succeeded", False),
        ("status", "skipped"),
        ("availability_status", "unknown"),
        ("execution_mode", "partial"),
    ],
)
def test_runtime_smoke_rejects_malformed_or_false_runtime_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_marker: str,
    value: object,
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    episode_path = result.parent.parent / "runs/goal__differential_drive/episodes.jsonl"
    row = json.loads(episode_path.read_text(encoding="utf-8"))
    row["algorithm_metadata"]["runtime"] = {runtime_marker: value}
    _write_json(episode_path, row)
    sidecar_path = episode_path.with_name(f"{episode_path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["raw_artifacts"][0]["sha256"] = sha256_file(episode_path)
    _write_json(sidecar_path, sidecar)

    with pytest.raises(RuntimeSmokeAdmissionError, match="fallback or degraded"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_allows_nonlearned_checkpoint_observation_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    episode_path = result.parent.parent / "runs/goal__differential_drive/episodes.jsonl"
    row = json.loads(episode_path.read_text(encoding="utf-8"))
    row["algorithm_metadata"]["learned_checkpoint_observation_contract"] = {
        "status": "not_applicable"
    }
    _write_json(episode_path, row)
    sidecar_path = episode_path.with_name(f"{episode_path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["raw_artifacts"][0]["sha256"] = sha256_file(episode_path)
    _write_json(sidecar_path, sidecar)

    assert _admit(result, planners, tmp_path)["status"] == "admitted"


@pytest.mark.parametrize("field", ["benchmark_success", "availability_status", "nested"])
def test_runtime_smoke_rejects_not_applicable_outside_exact_learned_policy_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary_path = result.parent.parent / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    contract = (
        summary["runs"][0]["summary"]
        .setdefault("preflight", {})
        .setdefault("learned_policy_contract", {})
    )
    if field == "nested":
        contract["nested"] = {"status": "not_applicable"}
    else:
        contract[field] = "not_applicable"
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeSmokeAdmissionError, match="fallback or degraded"):
        _admit(result, planners, tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("benchmark_success", None),
        ("availability_status", "not_run"),
        ("execution_mode", "not_run"),
    ],
)
def test_runtime_smoke_rejects_noncanonical_checkpoint_preflight_placeholders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    manifest_path = result.parent.parent / "campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["planners"][0]["checkpoint_provenance"][field] = value
    _write_json(manifest_path, manifest)

    with pytest.raises(RuntimeSmokeAdmissionError, match="forbidden status marker"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_allows_declarative_guarded_ppo_fallback_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    episode_path = result.parent.parent / "runs/guarded_ppo__differential_drive/episodes.jsonl"
    row = json.loads(episode_path.read_text(encoding="utf-8"))
    row["algorithm_metadata"]["config"] = {"fallback_to_goal": True}
    _write_json(episode_path, row)
    sidecar_path = episode_path.with_name(f"{episode_path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["raw_artifacts"][0]["sha256"] = sha256_file(episode_path)
    _write_json(sidecar_path, sidecar)

    admitted = _admit(result, planners, tmp_path)

    assert admitted["status"] == "admitted"


def test_runtime_smoke_allows_guarded_ppo_intrinsic_safe_shield_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    episode_path = result.parent.parent / "runs/guarded_ppo__differential_drive/episodes.jsonl"
    row = json.loads(episode_path.read_text(encoding="utf-8"))
    row["algorithm_metadata"]["guard_stats"] = {"fallback_safe": 1}
    row["algorithm_metadata"]["shield_stats"] = {
        "decision_counts": {"fallback_safe": 1},
        "last_decision": {
            "fallback_controller_state": {
                "policy": "RiskDWAPlannerAdapter",
                "prior_available": False,
            }
        },
    }
    _write_json(episode_path, row)
    sidecar_path = episode_path.with_name(f"{episode_path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["raw_artifacts"][0]["sha256"] = sha256_file(episode_path)
    _write_json(sidecar_path, sidecar)

    assert _admit(result, planners, tmp_path)["status"] == "admitted"


@pytest.mark.parametrize(
    ("planner", "guard_stats"),
    [
        ("guarded_ppo", {"fallback_best_effort": 1}),
        ("prediction_planner", {"fallback_safe": 1}),
    ],
)
def test_runtime_smoke_rejects_noncanonical_guard_fallback_counter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    planner: str,
    guard_stats: dict[str, int],
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    episode_path = result.parent.parent / "runs" / f"{planner}__differential_drive/episodes.jsonl"
    row = json.loads(episode_path.read_text(encoding="utf-8"))
    row["algorithm_metadata"]["guard_stats"] = guard_stats
    _write_json(episode_path, row)
    sidecar_path = episode_path.with_name(f"{episode_path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["raw_artifacts"][0]["sha256"] = sha256_file(episode_path)
    _write_json(sidecar_path, sidecar)

    with pytest.raises(RuntimeSmokeAdmissionError, match="fallback or degraded"):
        _admit(result, planners, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("benchmark_success", False, "result benchmark_success mismatch"),
        ("status", "failed", "result status mismatch"),
        ("evidence_status", "partial", "result evidence_status mismatch"),
    ],
)
def test_runtime_smoke_rejects_forged_result_status_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload[field] = value
    _write_json(result, payload)

    with pytest.raises(RuntimeSmokeAdmissionError, match=message):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_forged_campaign_status_surface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary_path = result.parent.parent / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["campaign"]["status"] = "failed"
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeSmokeAdmissionError, match="campaign status mismatch"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_caller_supplied_noncanonical_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    forged_roster = planners[:-1] + ("forged-arm",)

    with pytest.raises(RuntimeSmokeAdmissionError, match="canonical 14-arm roster"):
        _admit(result, forged_roster, tmp_path)


def test_runtime_smoke_rejects_crosswired_hybrid_arm_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary_path = result.parent.parent / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"][7]["planner"]["key"] = "scenario_adaptive_hybrid_orca_v2_collision_guard"
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeSmokeAdmissionError, match="episode artifact rejected"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_crosswired_hybrid_config_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    root = result.parent.parent / "runs"
    first = root / f"{planners[7]}__differential_drive/episodes.jsonl.provenance.json"
    second = root / f"{planners[8]}__differential_drive/episodes.jsonl.provenance.json"
    first.write_text(second.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(RuntimeSmokeAdmissionError, match="sidecar algorithm config"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_old_repo_root_episode_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    summary_path = result.parent.parent / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    current_episode = Path(summary["runs"][0]["episodes_path"])
    old_episode = (
        tmp_path
        / "output/benchmarks/camera_ready/old-smoke/runs/prediction_planner__differential_drive/episodes.jsonl"
    )
    old_episode.parent.mkdir(parents=True, exist_ok=True)
    old_episode.write_text(current_episode.read_text(encoding="utf-8"), encoding="utf-8")
    summary["runs"][0]["episodes_path"] = str(old_episode)
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeSmokeAdmissionError, match="episode artifact rejected"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_parent_directory_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    campaign_root = result.parent.parent
    runs = campaign_root / "runs"
    relocated = tmp_path / "old-campaign-runs"
    runs.rename(relocated)
    runs.symlink_to(relocated, target_is_directory=True)

    with pytest.raises(RuntimeSmokeAdmissionError, match="path contains a symlink"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_raw_artifact_path_with_intermediate_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    campaign_root = result.parent.parent
    alias = campaign_root / "alias-runs"
    alias.symlink_to(campaign_root / "runs", target_is_directory=True)
    summary_path = campaign_root / "reports/campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw = Path(summary["runs"][0]["episodes_path"])
    summary["runs"][0]["episodes_path"] = str(alias / raw.relative_to(campaign_root / "runs"))
    _write_json(summary_path, summary)

    with pytest.raises(RuntimeSmokeAdmissionError, match="episode artifact rejected"):
        _admit(result, planners, tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "campaign_manifest.json",
        "manifest.json",
        "run_meta.json",
        "reports/campaign_summary.json",
    ],
)
def test_runtime_smoke_rejects_canonical_metadata_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    metadata = result.parent.parent / relative_path
    saved = tmp_path / f"external-{metadata.name}"
    metadata.replace(saved)
    metadata.symlink_to(saved)

    with pytest.raises(RuntimeSmokeAdmissionError, match="path contains a symlink"):
        _admit(result, planners, tmp_path)


@pytest.mark.parametrize("relative_path", [RUNTIME_SMOKE_MANIFEST, RUNTIME_SMOKE_CONFIG])
def test_runtime_smoke_rejects_canonical_source_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: Path,
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    source = tmp_path / relative_path
    saved = tmp_path / f"external-{source.name}"
    source.replace(saved)
    source.symlink_to(saved)

    with pytest.raises(RuntimeSmokeAdmissionError, match="path contains a symlink"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_raw_artifact_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    episode_path = (
        result.parent.parent / "runs/prediction_planner__differential_drive/episodes.jsonl"
    )
    episode_path.write_text(
        episode_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeSmokeAdmissionError, match="raw artifact hash mismatch"):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_rejects_sidecar_scenario_matrix_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    sidecar_path = (
        result.parent.parent
        / "runs/prediction_planner__differential_drive/episodes.jsonl.provenance.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["inputs"]["scenario_matrix"]["sha256"] = "0" * 64
    _write_json(sidecar_path, sidecar)

    with pytest.raises(RuntimeSmokeAdmissionError, match="scenario matrix hash mismatch"):
        _admit(result, planners, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version="forged.v1"), "provenance schema rejected"),
        (
            lambda payload: payload["campaign_identity"].update(suite_key="forged"),
            "sidecar suite mismatch",
        ),
        (
            lambda payload: payload["raw_artifacts"][0].update(artifact_status="missing"),
            "raw artifact status mismatch",
        ),
        (lambda payload: payload["rows"][0].update(episode_id="forged"), "episode id mismatch"),
        (lambda payload: payload["rows"][0].update(jsonl_line=1), "JSONL line mismatch"),
        (
            lambda payload: payload.update(completeness={}),
            "sidecar completeness mismatch",
        ),
        (
            lambda payload: payload["campaign_identity"].update(config_hash="forged"),
            "campaign config hash mismatch",
        ),
        (
            lambda payload: payload["campaign_identity"].update(scenario_matrix_hash="forged"),
            "scenario identity hash mismatch",
        ),
        (lambda payload: payload["run"].update(runner="forged"), "sidecar runner mismatch"),
    ],
)
def test_runtime_smoke_rejects_incomplete_sidecar_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    message: str,
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    sidecar_path = (
        result.parent.parent
        / "runs/prediction_planner__differential_drive/episodes.jsonl.provenance.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    mutation(sidecar)
    _write_json(sidecar_path, sidecar)

    with pytest.raises(RuntimeSmokeAdmissionError, match=message):
        _admit(result, planners, tmp_path)


def test_runtime_smoke_admits_consistently_relocated_campaign_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_repo = tmp_path / "original"
    result, planners = _fixture(original_repo, monkeypatch)
    relocated_repo = tmp_path / "relocated"
    shutil.copytree(original_repo, relocated_repo)
    relocated_result = relocated_repo / result.relative_to(original_repo)

    admitted = validate_runtime_smoke_result(
        relocated_result,
        repo_root=relocated_repo,
        expected_source_commit="a" * 40,
        expected_planner_keys=planners,
    )

    assert admitted["status"] == "admitted"


def test_runtime_smoke_rejects_relocated_raw_path_with_symlinked_old_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_repo = tmp_path / "original"
    result, planners = _fixture(original_repo, monkeypatch)
    relocated_repo = tmp_path / "relocated"
    shutil.copytree(original_repo, relocated_repo)
    relocated_result = relocated_repo / result.relative_to(original_repo)
    old_runs = result.parent.parent / "runs"
    saved_runs = tmp_path / "saved-old-runs"
    old_runs.rename(saved_runs)
    old_runs.symlink_to(saved_runs, target_is_directory=True)

    with pytest.raises(RuntimeSmokeAdmissionError, match="path contains a symlink"):
        validate_runtime_smoke_result(
            relocated_result,
            repo_root=relocated_repo,
            expected_source_commit="a" * 40,
            expected_planner_keys=planners,
        )


def test_runtime_smoke_rejects_arbitrary_relocated_canonical_input_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_repo = tmp_path / "original"
    result, planners = _fixture(original_repo, monkeypatch)
    relocated_repo = tmp_path / "relocated"
    shutil.copytree(original_repo, relocated_repo)
    relocated_result = relocated_repo / result.relative_to(original_repo)
    sidecar_path = (
        relocated_result.parent.parent
        / "runs/prediction_planner__differential_drive/episodes.jsonl.provenance.json"
    )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected_suffix = Path(sidecar["inputs"]["scenario_matrix"]["path"]).relative_to(original_repo)
    sidecar["inputs"]["scenario_matrix"]["path"] = str(
        Path("/arbitrary/untrusted/prefix") / expected_suffix
    )
    _write_json(sidecar_path, sidecar)

    with pytest.raises(RuntimeSmokeAdmissionError, match="sidecar scenario matrix"):
        validate_runtime_smoke_result(
            relocated_result,
            repo_root=relocated_repo,
            expected_source_commit="a" * 40,
            expected_planner_keys=planners,
        )


def test_runtime_smoke_rejects_symlinked_checkpoint_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, planners = _fixture(tmp_path, monkeypatch)
    payload = json.loads(result.read_text(encoding="utf-8"))
    receipt = tmp_path / payload["checkpoint_staging_receipt"]["path"]
    saved = tmp_path / "saved-checkpoint-receipt.json"
    receipt.rename(saved)
    receipt.symlink_to(saved)

    with pytest.raises(RuntimeSmokeAdmissionError, match="checkpoint staging receipt path"):
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
