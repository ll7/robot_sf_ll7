"""Acceptance-level regression tests for the fixed hybrid stress contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark import release_acceptance
from robot_sf.benchmark.camera_ready._config import (
    _load_campaign_scenarios,
    _scenario_with_kinematics,
)
from robot_sf.benchmark.camera_ready._preflight import _config_hash_payload, _scenario_matrix_hash
from robot_sf.benchmark.camera_ready._run_state import validate_campaign_integrity
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.benchmark.release_acceptance import validate_diagnostic_stress_smoke_acceptance
from robot_sf.benchmark.release_protocol import (
    STRESS_SMOKE_EXPECTED_SCENARIO_IDS,
    load_release_manifest,
    validate_stress_smoke_runtime_identity,
)
from robot_sf.benchmark.result_provenance import (
    build_result_provenance_manifest,
    write_result_provenance_manifest,
)
from robot_sf.benchmark.utils import _config_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / (
    "configs/benchmarks/releases/paper_experiment_matrix_v2_h600_s30_hybrid_stress_smoke_v0_1.yaml"
)
SOURCE_COMMIT = "a" * 40


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _repo_relative(path: Path) -> str:
    """Return a fixture path in the same form as camera-ready metadata."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _row(*, algo: str, scenario_id: str, seed: int) -> dict[str, Any]:
    scenario_params = {
        "algo": algo,
        "id": scenario_id,
        "robot_config": {"type": "differential_drive"},
        "run_dt": 0.1,
    }
    config_hash = _config_hash(scenario_params)
    return {
        "algo": algo,
        "config_hash": config_hash,
        "episode_id": f"{scenario_id}--{seed}",
        "event_ledger": {"software_commit": SOURCE_COMMIT},
        "git_hash": SOURCE_COMMIT,
        "horizon": 600,
        "scenario_id": scenario_id,
        "scenario_params": scenario_params,
        "seed": seed,
        "status": "success",
        "algorithm_metadata": {
            "algorithm": "ppo" if algo == "guarded_ppo" else algo,
            "canonical_algorithm": algo,
            "planner_kinematics": {
                "robot_kinematics": "differential_drive",
                "scenario_kinematics": ["differential_drive"],
            },
            "planner_contract": {
                "planner_id": algo,
                "action_contract": {"active_robot_kinematics": "differential_drive"},
            },
        },
        "provenance": {
            "git_hash": SOURCE_COMMIT,
            "config_hash": config_hash,
            "config_identity": {"algo": algo},
        },
        "result_provenance": {
            "config_hash": config_hash,
            "repo_commit": SOURCE_COMMIT,
            "scenario_id": scenario_id,
            "seed": seed,
            "simulator_settings": {"dt": 0.1, "horizon": 600, "kinematics": "differential_drive"},
        },
    }


@pytest.fixture
def stress_fixture(tmp_path: Path) -> tuple[Path, Any, Any]:
    """Build a complete accepted 14-arm stress campaign with tiny JSONL files."""
    manifest = load_release_manifest(MANIFEST_PATH)
    campaign_config = load_campaign_config(manifest.canonical_campaign_config_path)
    scenarios = _load_campaign_scenarios(campaign_config)
    effective_scenarios = [
        _scenario_with_kinematics(
            scenario,
            kinematics="differential_drive",
            holonomic_command_mode=campaign_config.holonomic_command_mode,
        )
        for scenario in scenarios
    ]
    scenario_ids = tuple(str(scenario["name"]) for scenario in scenarios)
    assert scenario_ids == STRESS_SMOKE_EXPECTED_SCENARIO_IDS

    root = tmp_path / "campaign"
    scenario_hash = _scenario_matrix_hash(scenarios)
    config_hash = _config_hash(_config_hash_payload(campaign_config))
    seed_policy = campaign_config.seed_policy
    campaign_manifest = {
        "git": {"commit": SOURCE_COMMIT},
        "config_hash": config_hash,
        "scenario_matrix": _repo_relative(campaign_config.scenario_matrix_path),
        "scenario_matrix_hash": scenario_hash,
        "seed_policy": {
            "mode": seed_policy.mode,
            "seed_set": seed_policy.seed_set,
            "seeds": list(seed_policy.seeds),
            "resolved_seeds": [116],
            "seed_sets_path": _repo_relative(seed_policy.seed_sets_path),
        },
        "route_clearance_certifications_path": _repo_relative(
            campaign_config.route_clearance_certifications_path
        ),
        "snqi_weights_path": _repo_relative(campaign_config.snqi_weights_path),
        "snqi_baseline_path": _repo_relative(campaign_config.snqi_baseline_path),
    }
    _write_json(root / "campaign_manifest.json", campaign_manifest)
    _write_json(
        root / "manifest.json",
        {
            "git_hash": SOURCE_COMMIT,
            "scenario_matrix_hash": scenario_hash,
        },
    )
    _write_json(
        root / "run_meta.json",
        {
            "repo": {"commit": SOURCE_COMMIT},
            "matrix_path": _repo_relative(campaign_config.scenario_matrix_path),
            "scenario_matrix_hash": scenario_hash,
            "seed_policy": campaign_manifest["seed_policy"],
        },
    )
    runs: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    for planner in campaign_config.planners:
        arm = f"{planner.key}__differential_drive"
        episodes_path = root / "runs" / arm / "episodes.jsonl"
        summary_path = root / "runs" / arm / "summary.json"
        rows = [
            _row(algo=planner.algo, scenario_id=scenario_id, seed=116)
            for scenario_id in scenario_ids
        ]
        episodes_path.parent.mkdir(parents=True, exist_ok=True)
        episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        provenance = build_result_provenance_manifest(
            out_path=episodes_path,
            episode_records=rows,
            schema_path=REPO_ROOT / "robot_sf/benchmark/schemas/episode.schema.v1.json",
            scenario_path=campaign_config.scenario_matrix_path,
            scenarios=effective_scenarios,
            algo=planner.algo,
            algo_config_path=planner.algo_config_path,
            benchmark_profile=planner.benchmark_profile,
            suite_key="default",
            total_jobs=5,
            written=5,
            horizon=600,
            dt=0.1,
            record_forces=True,
            active_observation_mode=None,
            active_observation_level=None,
        )
        provenance["run"]["repo_commit"] = SOURCE_COMMIT
        write_result_provenance_manifest(
            episodes_path.with_name(f"{episodes_path.name}.provenance.json"), provenance
        )
        _write_json(summary_path, {"status": "ok"})
        runs.append(
            {
                "planner": {
                    "key": planner.key,
                    "algo": planner.algo,
                    "kinematics": "differential_drive",
                    "horizon": 600,
                    "dt": 0.1,
                },
                "status": "ok",
                "episodes_path": f"runs/{arm}/episodes.jsonl",
                "summary_path": f"runs/{arm}/summary.json",
                "summary": {
                    "status": "ok",
                    "total_jobs": 5,
                    "written": 5,
                    "successful_jobs": 5,
                    "failed_jobs": 0,
                    "skipped_jobs": 0,
                    "failures": [],
                },
            }
        )
        planner_rows.append(
            {
                "planner_key": planner.key,
                "kinematics": "differential_drive",
                "status": "ok",
                "benchmark_success": "true",
                "episodes": 5,
                "failed_jobs": 0,
            }
        )
    integrity = validate_campaign_integrity(
        runs,
        scenarios=scenarios,
        resolved_seeds=[116],
        campaign_root=root,
        campaign_manifest=campaign_manifest,
    )
    _write_json(
        root / "reports" / "campaign_summary.json",
        {
            "campaign": {
                "git_hash": SOURCE_COMMIT,
                "scenario_matrix": _repo_relative(campaign_config.scenario_matrix_path),
                "scenario_matrix_hash": scenario_hash,
                "kinematics_matrix": ["differential_drive"],
                "snqi_weights_sha256": sha256_file(campaign_config.snqi_weights_path),
                "snqi_baseline_sha256": sha256_file(campaign_config.snqi_baseline_path),
                "benchmark_success": True,
                "status": "benchmark_success",
                "evidence_status": "valid",
                "campaign_execution_status": "completed",
            },
            "campaign_integrity": integrity,
            "runs": runs,
            "planner_rows": planner_rows,
        },
    )
    return root, manifest, campaign_config


def _acceptance(root: Path, manifest: Any, campaign_config: Any) -> dict[str, Any]:
    return validate_diagnostic_stress_smoke_acceptance(
        root,
        manifest=manifest,
        campaign_config=campaign_config,
        expected_source_commit=SOURCE_COMMIT,
    )


def _first_row_path(root: Path, planner_key: str) -> Path:
    return root / "runs" / f"{planner_key}__differential_drive" / "episodes.jsonl"


def _refresh_sidecar_raw_hash(path: Path) -> None:
    """Keep a deliberately edited fixture internally byte-consistent."""
    sidecar_path = path.with_name(f"{path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    for artifact in sidecar["raw_artifacts"]:
        if artifact["kind"] == "episodes_jsonl":
            artifact["sha256"] = sha256_file(path)
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")


def test_complete_stress_campaign_is_admitted(stress_fixture: tuple[Path, Any, Any]) -> None:
    root, manifest, campaign_config = stress_fixture

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "valid", report["blockers"]
    assert report["diagnostic_success"] is True
    assert report["observed_episode_rows"] == 70


@pytest.mark.parametrize("status", ("collision", "failure"))
def test_scientific_terminal_outcomes_do_not_fail_runtime_admission(
    stress_fixture: tuple[Path, Any, Any], status: str
) -> None:
    """A completed native episode need not reach the goal to prove runtime execution."""
    root, manifest, campaign_config = stress_fixture
    episodes_path = _first_row_path(root, "prediction_planner")
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    rows[0]["status"] = status
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_sidecar_raw_hash(episodes_path)

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "valid", report["blockers"]


def test_unknown_episode_status_fails_runtime_admission(
    stress_fixture: tuple[Path, Any, Any],
) -> None:
    root, manifest, campaign_config = stress_fixture
    episodes_path = _first_row_path(root, "prediction_planner")
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    rows[0]["status"] = "mystery"
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_sidecar_raw_hash(episodes_path)

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"
    assert any("terminal outcome" in blocker for blocker in report["blockers"])


@pytest.mark.parametrize(
    ("metadata_path", "value"),
    (
        (("fallback_reason",), "alternate planner invoked"),
        (("planner_diagnostics", "fallback_count"), 1),
        (("runtime", "fallback_count"), 1),
    ),
)
def test_fallback_markers_anywhere_in_runtime_algorithm_metadata_fail_closed(
    stress_fixture: tuple[Path, Any, Any],
    metadata_path: tuple[str, ...],
    value: Any,
) -> None:
    """Canonical and newly introduced runtime containers cannot bypass admission."""
    root, manifest, campaign_config = stress_fixture
    episodes_path = _first_row_path(root, "prediction_planner")
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    target = rows[0]["algorithm_metadata"]
    for key in metadata_path[:-1]:
        target = target.setdefault(key, {})
    target[metadata_path[-1]] = value
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_sidecar_raw_hash(episodes_path)

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"
    assert any("fallback" in blocker for blocker in report["blockers"])


def test_guarded_ppo_declared_safe_shield_intervention_is_native(
    stress_fixture: tuple[Path, Any, Any],
) -> None:
    """The exact safe Risk-DWA shield path is part of the declared composite arm."""
    root, manifest, campaign_config = stress_fixture
    episodes_path = _first_row_path(root, "guarded_ppo")
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    rows[0]["algorithm_metadata"].update(
        {
            "guard_stats": {"fallback_safe": 3, "fallback_best_effort": 0},
            "shield_stats": {
                "decision_counts": {"fallback_safe": 3, "fallback_best_effort": 0},
                "last_decision": {"fallback_controller_state": {"status": "ok"}},
            },
        }
    )
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_sidecar_raw_hash(episodes_path)

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "valid", report["blockers"]


def test_guarded_ppo_best_effort_fallback_still_fails_closed(
    stress_fixture: tuple[Path, Any, Any],
) -> None:
    root, manifest, campaign_config = stress_fixture
    episodes_path = _first_row_path(root, "guarded_ppo")
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    rows[0]["algorithm_metadata"]["guard_stats"] = {"fallback_best_effort": 1}
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_sidecar_raw_hash(episodes_path)

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"
    assert any("fallback_best_effort" in blocker for blocker in report["blockers"])


def test_guarded_ppo_safe_label_requires_exact_composite_identity(
    stress_fixture: tuple[Path, Any, Any],
) -> None:
    root, manifest, campaign_config = stress_fixture
    episodes_path = _first_row_path(root, "guarded_ppo")
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
    metadata = rows[0]["algorithm_metadata"]
    metadata["planner_contract"]["planner_id"] = "ppo"
    metadata["guard_stats"] = {"fallback_safe": 1}
    episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _refresh_sidecar_raw_hash(episodes_path)

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"
    assert any("fallback_safe" in blocker for blocker in report["blockers"])


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_sidecar",
        "sidecar_config",
        "integrity",
        "event_source",
        "robot_type",
        "run_dt",
        "row_config",
    ),
)
def test_stress_provenance_and_integrity_bypasses_fail_closed(
    stress_fixture: tuple[Path, Any, Any], mutation: str
) -> None:
    """Every release/stress provenance bypass has an executable negative fixture."""
    root, manifest, campaign_config = stress_fixture
    episodes_path = _first_row_path(root, "prediction_planner")
    sidecar_path = episodes_path.with_name(f"{episodes_path.name}.provenance.json")
    if mutation == "missing_sidecar":
        sidecar_path.unlink()
    elif mutation == "sidecar_config":
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["campaign_identity"]["config_hash"] = "0" * 16
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    elif mutation == "integrity":
        summary_path = root / "reports" / "campaign_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["campaign_integrity"]["status"] = "invalid"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
    else:
        rows = [json.loads(line) for line in episodes_path.read_text().splitlines()]
        if mutation == "event_source":
            rows[0]["event_ledger"]["software_commit"] = "b" * 40
        elif mutation == "robot_type":
            rows[0]["scenario_params"]["robot_config"]["type"] = "holonomic"
        elif mutation == "run_dt":
            rows[0]["scenario_params"]["run_dt"] = 0.2
        else:
            rows[0]["config_hash"] = "deadbeefdeadbeef"
            rows[0]["provenance"]["config_hash"] = "deadbeefdeadbeef"
            rows[0]["result_provenance"]["config_hash"] = "deadbeefdeadbeef"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["rows"][0]["config_hash"] = "deadbeefdeadbeef"
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        episodes_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        _refresh_sidecar_raw_hash(episodes_path)

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid", report["blockers"]
    assert report["diagnostic_success"] is False


def test_sidecar_runtime_identity_and_input_aliases_fail_closed_together(
    stress_fixture: tuple[Path, Any, Any],
) -> None:
    """A coherent-looking sidecar still fails when its independent aliases drift."""
    root, manifest, campaign_config = stress_fixture
    episodes_path = _first_row_path(root, "prediction_planner")
    sidecar_path = episodes_path.with_name(f"{episodes_path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["run"].update({"repo_commit": "b" * 40, "runner": "other.runner"})
    sidecar["completeness"]["status"] = "partial"
    sidecar["campaign_identity"].update(
        {
            "algorithm": "goal",
            "suite_key": "other",
            "scenario_matrix_hash": "wrong",
            "total_jobs": 4,
            "written": 4,
        }
    )
    sidecar["inputs"]["schema_path"].update({"path": "wrong/schema.json", "sha256": "wrong"})
    sidecar["inputs"]["scenario_matrix"].update({"path": "wrong/scenarios.yaml", "sha256": "wrong"})
    sidecar["raw_artifacts"] = []
    sidecar["rows"] = sidecar["rows"][:-1]
    _write_json(sidecar_path, sidecar)

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"
    assert any("sidecar source commit" in item for item in report["blockers"])
    assert any("sidecar must bind every episode row" in item for item in report["blockers"])


@pytest.mark.parametrize("payload", ("{bad\n", "[]\n"))
def test_malformed_sidecar_payload_fails_closed(
    stress_fixture: tuple[Path, Any, Any], payload: str
) -> None:
    """Unreadable and non-object provenance sidecars are rejected explicitly."""
    root, manifest, campaign_config = stress_fixture
    episodes_path = _first_row_path(root, "prediction_planner")
    sidecar_path = episodes_path.with_name(f"{episodes_path.name}.provenance.json")
    sidecar_path.write_text(payload, encoding="utf-8")

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"
    assert any("sidecar" in item for item in report["blockers"])


@pytest.mark.parametrize(
    "field",
    (
        "config_hash",
        "scenario_matrix_hash",
        "seed_policy",
        "route_clearance_certifications_path",
        "snqi_weights_sha256",
    ),
)
def test_campaign_bindings_cannot_be_repointed(
    stress_fixture: tuple[Path, Any, Any], field: str
) -> None:
    """Campaign-level config, matrix, seed, route, and SNQI pins are mandatory."""
    root, manifest, campaign_config = stress_fixture
    if field == "snqi_weights_sha256":
        summary_path = root / "reports" / "campaign_summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["campaign"][field] = "0" * 64
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        metadata_path = root / "campaign_manifest.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if field == "seed_policy":
            payload[field]["resolved_seeds"] = [117]
        elif field == "config_hash":
            payload[field] = "0" * 16
        elif field == "scenario_matrix_hash":
            payload[field] = "0" * 12
        else:
            payload[field] = "configs/benchmarks/not-the-pinned-route.yaml"
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid", report["blockers"]


def test_all_campaign_metadata_aliases_fail_closed_together(
    stress_fixture: tuple[Path, Any, Any],
) -> None:
    """Exercise every campaign-level alias instead of relying on one happy-path source."""
    root, manifest, campaign_config = stress_fixture
    campaign_manifest_path = root / "campaign_manifest.json"
    campaign_manifest = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
    campaign_manifest["git"]["commit"] = "b" * 40
    campaign_manifest["config_hash"] = "wrong"
    campaign_manifest["scenario_matrix"] = "wrong/scenarios.yaml"
    campaign_manifest["scenario_matrix_hash"] = "wrong"
    campaign_manifest["seed_policy"].update(
        {
            "mode": "random",
            "seed_set": "wrong",
            "seeds": [999],
            "resolved_seeds": [999],
            "seed_sets_path": "wrong/seed_sets.yaml",
        }
    )
    campaign_manifest["route_clearance_certifications_path"] = "wrong/routes.yaml"
    campaign_manifest["snqi_weights_path"] = "wrong/weights.yaml"
    campaign_manifest["snqi_baseline_path"] = "wrong/baseline.yaml"
    _write_json(campaign_manifest_path, campaign_manifest)

    run_meta_path = root / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    run_meta["repo"]["commit"] = "b" * 40
    run_meta["matrix_path"] = "wrong/scenarios.yaml"
    run_meta["scenario_matrix_hash"] = "wrong"
    run_meta["seed_policy"].update(
        {
            "mode": "random",
            "seed_set": "wrong",
            "seeds": [999],
            "resolved_seeds": [999],
            "seed_sets_path": "wrong/seed_sets.yaml",
        }
    )
    _write_json(run_meta_path, run_meta)

    run_manifest_path = root / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    run_manifest.update({"git_hash": "b" * 40, "scenario_matrix_hash": "wrong"})
    _write_json(run_manifest_path, run_manifest)

    summary_path = root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["campaign"].update(
        {
            "git_hash": "b" * 40,
            "scenario_matrix": "wrong/scenarios.yaml",
            "scenario_matrix_hash": "wrong",
            "kinematics_matrix": ["holonomic"],
            "snqi_weights_sha256": "wrong",
            "snqi_baseline_sha256": "wrong",
        }
    )
    _write_json(summary_path, summary)

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"
    assert len(report["blockers"]) >= 20


def test_missing_campaign_metadata_objects_fail_closed(
    stress_fixture: tuple[Path, Any, Any],
) -> None:
    """Every required campaign metadata object is independently mandatory."""
    root, manifest, campaign_config = stress_fixture
    for path in (
        root / "campaign_manifest.json",
        root / "run_meta.json",
        root / "manifest.json",
    ):
        path.unlink()

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"
    assert any("campaign_manifest.json" in item for item in report["blockers"])
    assert any("run_meta.json" in item for item in report["blockers"])

    (root / "reports" / "campaign_summary.json").unlink()
    missing_summary = _acceptance(root, manifest, campaign_config)
    assert missing_summary["status"] == "invalid"
    assert any("campaign_summary.json" in item for item in missing_summary["blockers"])


@pytest.mark.parametrize(
    "case",
    (
        "root_missing",
        "empty",
        "traversal",
        "incomplete_arm",
        "absolute_outside",
        "absolute_not_run",
        "absolute_wrong_arm",
        "missing_runs",
        "wrong_arm",
        "wrong_shape",
        "symlink_artifact",
        "missing_file",
    ),
)
def test_stress_artifact_path_rejects_ambiguous_or_escaping_inputs(  # noqa: C901
    tmp_path: Path, case: str
) -> None:
    """Artifact resolution accepts only the exact campaign arm file."""
    root = tmp_path / "campaign"
    root.mkdir()
    expected = root / "runs" / "planner__differential_drive" / "episodes.jsonl"
    expected.parent.mkdir(parents=True)
    expected.write_text("{}\n", encoding="utf-8")
    raw_path: str = "runs/planner__differential_drive/episodes.jsonl"
    arm: tuple[str, str] | None = ("planner", "differential_drive")

    if case == "root_missing":
        root = tmp_path / "missing"
    elif case == "empty":
        raw_path = ""
    elif case == "traversal":
        raw_path = "runs/../planner__differential_drive/episodes.jsonl"
    elif case == "incomplete_arm":
        arm = ("", "differential_drive")
    elif case == "absolute_outside":
        outside = tmp_path / "outside" / "episodes.jsonl"
        outside.parent.mkdir()
        outside.write_text("{}\n", encoding="utf-8")
        raw_path, arm = str(outside), None
    elif case == "absolute_not_run":
        wrong = root / "other" / "episodes.jsonl"
        wrong.parent.mkdir()
        wrong.write_text("{}\n", encoding="utf-8")
        raw_path, arm = str(wrong), None
    elif case == "absolute_wrong_arm":
        wrong = root / "runs" / "other__differential_drive" / "episodes.jsonl"
        wrong.parent.mkdir()
        wrong.write_text("{}\n", encoding="utf-8")
        raw_path = str(wrong)
    elif case == "missing_runs":
        raw_path = "planner__differential_drive/episodes.jsonl"
    elif case == "wrong_arm":
        raw_path = "runs/other__differential_drive/episodes.jsonl"
    elif case == "wrong_shape":
        raw_path, arm = "runs/planner__differential_drive/nested/episodes.jsonl", None
    elif case == "symlink_artifact":
        target = tmp_path / "target.jsonl"
        target.write_text("{}\n", encoding="utf-8")
        expected.unlink()
        expected.symlink_to(target)
    elif case == "missing_file":
        expected.unlink()

    with pytest.raises(ValueError):
        release_acceptance._resolve_stress_artifact_path(root, raw_path, arm=arm)


def test_stress_metadata_readers_reject_malformed_inputs(tmp_path: Path) -> None:
    """Malformed JSONL and metadata objects cannot enter stress acceptance."""
    invalid_jsonl = tmp_path / "invalid.jsonl"
    invalid_jsonl.write_text("{bad\n", encoding="utf-8")
    rows, error = release_acceptance._read_episode_rows(invalid_jsonl)
    assert rows == [] and "invalid JSON" in str(error)

    list_jsonl = tmp_path / "list.jsonl"
    list_jsonl.write_text("[]\n", encoding="utf-8")
    rows, error = release_acceptance._read_episode_rows(list_jsonl)
    assert rows == [] and "must be an object" in str(error)

    rows, error = release_acceptance._read_episode_rows(tmp_path / "missing.jsonl")
    assert rows == [] and "cannot read" in str(error)

    (tmp_path / "metadata.json").write_text("[]\n", encoding="utf-8")
    payload, error = release_acceptance._read_campaign_object(tmp_path, "metadata.json")
    assert payload is None and "must contain a JSON object" in str(error)


@pytest.mark.parametrize("kind", ("symlink", "directory"))
def test_campaign_summary_reader_requires_campaign_contained_regular_file(
    tmp_path: Path, kind: str
) -> None:
    """Summary reads fail closed before following links or opening directories."""
    root = tmp_path / "campaign"
    reports = root / "reports"
    reports.mkdir(parents=True)
    outside = tmp_path / "outside-summary.json"
    outside.write_text("{}\n", encoding="utf-8")
    summary_path = reports / "campaign_summary.json"
    if kind == "symlink":
        summary_path.symlink_to(outside)
    else:
        summary_path.mkdir()

    payload, error = release_acceptance._read_campaign_summary(root)

    assert payload is None
    assert error is not None
    assert "campaign summary cannot be read" in error
    if kind == "symlink":
        assert "symlink" in error


@pytest.mark.parametrize(
    "payload",
    (
        {"planner_runtime": {"emergency_stop_count": -1}},
        {"planner_runtime": {"emergency_stop": True}},
        {"algorithm_metadata": {"history": [{"planner_mode": "emergency_stop"}]}},
    ),
)
def test_emergency_marker_rejects_invalid_counter_boolean_and_nested_list(
    payload: dict[str, Any],
) -> None:
    """All legacy emergency marker shapes are detected."""
    assert release_acceptance._emergency_stop_marker(payload) is not None


def test_real_episode_schema_may_omit_repeated_kinematics_aliases(
    stress_fixture: tuple[Path, Any, Any],
) -> None:
    root, manifest, campaign_config = stress_fixture
    path = _first_row_path(root, "prediction_planner")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        planner_metadata = row["algorithm_metadata"]
        planner_metadata.pop("planner_kinematics")
        planner_metadata["planner_contract"]["action_contract"].pop("active_robot_kinematics")
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    _refresh_sidecar_raw_hash(path)

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "valid", report["blockers"]


@pytest.mark.parametrize(
    "mutation",
    ("planner", "kinematics", "config", "source", "emergency", "horizon", "dt", "status"),
)
def test_row_alias_and_runtime_bypasses_fail_closed(
    stress_fixture: tuple[Path, Any, Any], mutation: str
) -> None:
    root, manifest, campaign_config = stress_fixture
    path = _first_row_path(root, "prediction_planner")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    if mutation == "planner":
        row["algorithm_metadata"]["canonical_algorithm"] = "goal"
    elif mutation == "kinematics":
        row["algorithm_metadata"]["planner_kinematics"]["robot_kinematics"] = "holonomic"
    elif mutation == "config":
        row["result_provenance"]["config_hash"] = "d" * 16
    elif mutation == "source":
        row["provenance"]["git_hash"] = "b" * 40
    elif mutation == "emergency":
        row["algorithm_metadata"]["runtime"] = {"last_decision": {"planner_mode": "EMERGENCY_STOP"}}
    elif mutation == "horizon":
        row["result_provenance"]["simulator_settings"]["horizon"] = 599
    elif mutation == "dt":
        row["result_provenance"]["simulator_settings"]["dt"] = 0.2
    else:
        row["status"] = "yes"
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"
    assert report["diagnostic_success"] is False


@pytest.mark.parametrize("trace_hash_matches", (True, False))
def test_analysis_trace_config_hash_keeps_legacy_trace_binding(
    stress_fixture: tuple[Path, Any, Any], trace_hash_matches: bool
) -> None:
    """Trace-mode provenance retains its trace hash without weakening row identity."""
    root, manifest, campaign_config = stress_fixture
    path = _first_row_path(root, "prediction_planner")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    row = rows[0]
    trace_hash = "trace-config-hash"
    row["algorithm_metadata"]["analysis_trace"] = {"config_hash": trace_hash}
    row["provenance"]["config_hash"] = trace_hash if trace_hash_matches else "wrong-trace-hash"
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    _refresh_sidecar_raw_hash(path)

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == ("valid" if trace_hash_matches else "invalid"), report["blockers"]


def test_external_and_cross_campaign_paths_fail_closed(
    stress_fixture: tuple[Path, Any, Any], tmp_path: Path
) -> None:
    root, manifest, campaign_config = stress_fixture
    summary_path = root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"][0]["episodes_path"] = str(tmp_path / "other-campaign" / "episodes.jsonl")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"
    assert any("episodes_path rejected" in blocker for blocker in report["blockers"])


def test_unrelated_nested_business_metadata_does_not_trigger_emergency_marker(
    stress_fixture: tuple[Path, Any, Any],
) -> None:
    root, manifest, campaign_config = stress_fixture
    path = _first_row_path(root, "prediction_planner")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["scenario_params"]["metadata"] = {
        "planner_mode": "REORIENT",
        "selected_source": "static_reorient",
    }
    config_hash = _config_hash(rows[0]["scenario_params"])
    rows[0]["config_hash"] = config_hash
    rows[0]["provenance"]["config_hash"] = config_hash
    rows[0]["result_provenance"]["config_hash"] = config_hash
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
    sidecar_path = path.with_name(f"{path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["rows"][0]["config_hash"] = config_hash
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    _refresh_sidecar_raw_hash(path)

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "valid", report["blockers"]


@pytest.mark.parametrize(
    ("field", "value"),
    (("campaign_execution_status", "interrupted"), ("evidence_status", "blocked")),
)
def test_campaign_status_gate_requires_completed_valid_evidence(
    stress_fixture: tuple[Path, Any, Any], field: str, value: str
) -> None:
    root, manifest, campaign_config = stress_fixture
    summary_path = root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["campaign"][field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    report = _acceptance(root, manifest, campaign_config)

    assert report["status"] == "invalid"


def test_private_runtime_identity_requires_exact_launch_pin_and_clean_worktree() -> None:
    manifest = load_release_manifest(MANIFEST_PATH)
    runtime_commit = "b" * 40

    missing_pin = validate_stress_smoke_runtime_identity(
        manifest,
        current_source_commit=runtime_commit,
        require_launch_pin=True,
        worktree_clean=True,
        require_clean_worktree=True,
    )
    dirty = validate_stress_smoke_runtime_identity(
        manifest,
        current_source_commit=runtime_commit,
        launch_expected_source_commit=runtime_commit,
        require_launch_pin=True,
        worktree_clean=False,
        require_clean_worktree=True,
    )
    local_dirty = validate_stress_smoke_runtime_identity(
        manifest,
        current_source_commit=runtime_commit,
        require_launch_pin=False,
        worktree_clean=False,
        require_clean_worktree=True,
    )

    assert missing_pin["status"] == "invalid"
    assert dirty["status"] == "invalid"
    assert local_dirty["status"] == "invalid"
