"""Evidence checks for the frozen-release trace comparison."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.utils import _config_hash
from scripts.validation import check_issue_9671_force_bundle as bundle_checker
from scripts.validation import check_issue_9671_trace_reexport as checker
from scripts.validation import issue_9671_force_observer_sitecustomize as observer
from scripts.validation.check_issue_9671_trace_reexport import SOURCE_SHA, check


def _row(seed: int, status: str, *, trace: bool) -> dict:
    params = {
        "algo": "ppo",
        "run_horizon": 600,
        "run_dt": 0.1,
        "simulation_config": {
            "route_spawn_seed": seed,
            "goal_completion_policy": "goal_zone_entry_v1",
        },
    }
    if trace:
        params.update(
            record_forces=True,
            record_planner_decision_trace=True,
            record_simulation_step_trace=True,
        )
    config_hash = _config_hash(params)
    return {
        "episode_id": f"classic_doorway_medium--{seed}--{config_hash}",
        "config_hash": config_hash,
        "result_provenance": {"config_hash": config_hash},
        "algo": "ppo",
        "scenario_id": "classic_doorway_medium",
        "seed": seed,
        "status": status,
        "termination_reason": "collision" if status == "collision" else "success",
        "outcome": {
            "route_complete": status == "success",
            "collision_event": status == "collision",
            "timeout_event": False,
        },
        "steps": 1,
        "interaction_exposure": {"interaction_exposure_steps": 0},
        "git_hash": SOURCE_SHA,
        "scenario_params": params,
        "algorithm_metadata": {
            "planner_kinematics": {"execution_mode": "native"},
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v1",
                "dt": 0.1,
                "steps": [
                    {
                        "step": 0,
                        "time_s": 0.1,
                        "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0], "heading": 0.0},
                        "pedestrians": [],
                    }
                ],
            },
        },
    }


def test_canonical_hash_policy_is_shared_by_all_three_consumers() -> None:
    row = _row(113, "collision", trace=False)
    row["metrics"] = {"min_separation_corrupted_m": float("inf")}
    row["algorithm_metadata"]["tracking_precision"] = {"min_separation_corrupted_m": float("inf")}
    original_values = (
        row["metrics"]["min_separation_corrupted_m"],
        row["algorithm_metadata"]["tracking_precision"]["min_separation_corrupted_m"],
    )

    digests = {
        observer.canonical_episode_record_sha256(row),
        checker._canonical_row_sha(row),
        bundle_checker._canonical_row_sha(row),
    }

    assert len(digests) == 1
    assert original_values == (float("inf"), float("inf"))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("metrics", "min_separation_corrupted_m"), float("nan")),
        (("metrics", "min_separation_corrupted_m"), float("-inf")),
        (("metrics", "unrelated"), float("inf")),
    ],
)
def test_all_canonical_hash_consumers_reject_unsupported_non_finite_values(
    path: tuple[str, ...], value: float
) -> None:
    row = _row(113, "collision", trace=False)
    parent = row
    for key in path[:-1]:
        parent = parent.setdefault(key, {})
    parent[path[-1]] = value

    for hash_row in (
        observer.canonical_episode_record_sha256,
        checker._canonical_row_sha,
        bundle_checker._canonical_row_sha,
    ):
        with pytest.raises(ValueError, match="non-finite"):
            hash_row(row)


def _archive(path: Path, rows: list[dict]) -> None:
    data = b"\n".join(json.dumps(row).encode() for row in rows) + b"\n"
    member = tarfile.TarInfo("bundle/payload/runs/ppo__differential_drive/episodes.jsonl")
    member.size = len(data)
    with tarfile.open(path, "w:gz") as bundle:
        bundle.addfile(member, io.BytesIO(data))


def _bindings(tmp_path: Path, seeds: list[int]) -> tuple[dict, dict, dict, dict]:
    frozen_root = tmp_path / "frozen-root"
    configs = {}
    manifests = {}
    digests = {}
    effective = {"headon_group": "test-headon-hash", "doorway": "test-doorway-hash"}
    for name in effective:
        selected = seeds if name == "headon_group" else []
        config = {
            "horizon": 600,
            "dt": 0.1,
            "kinematics_matrix": ["differential_drive"],
            "record_forces": True,
            "record_planner_decision_trace": True,
            "record_simulation_step_trace": True,
            "scenario_matrix": "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml",
            "scenario_candidates": ["classic_doorway_medium"] if selected else [],
            "seed_policy": {"seeds": selected},
            "planners": [{"key": "ppo"}] if selected else [],
        }
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        configs[name] = config_path
        digests[name] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        manifest_path = tmp_path / f"{name}-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "git": {"commit": SOURCE_SHA},
                    "config_hash": effective[name],
                    "scenario_matrix": config["scenario_matrix"],
                    "campaign_id": checker.CAMPAIGN_ID[name],
                    "scenario_matrix_hash": "test-matrix-hash",
                    "invoked_command": (
                        "python scripts/tools/run_camera_ready_benchmark.py "
                        f"--config {frozen_root / checker.DIAGNOSTIC_CONFIG[name]} "
                        f"--mode run --campaign-id {checker.CAMPAIGN_ID[name]}"
                    ),
                    "seed_policy": {"resolved_seeds": selected},
                }
            )
        )
        manifests[name] = manifest_path
    return configs, manifests, digests, effective


def _producer_trace(tmp_path: Path, rows: list[dict]) -> Path:
    frozen_root = tmp_path / "frozen-root"
    path = tmp_path / "runs" / "ppo__differential_drive" / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        config_hash = _config_hash(row["scenario_params"])
        row["config_hash"] = config_hash
        row["result_provenance"]["config_hash"] = config_hash
        row["episode_id"] = f"classic_doorway_medium--{row['seed']}--{config_hash}"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    sidecar = {
        "run": {
            "repo_commit": SOURCE_SHA,
            "invocation": (
                "scripts/tools/run_camera_ready_benchmark.py "
                f"--config {frozen_root / checker.DIAGNOSTIC_CONFIG['headon_group']} "
                f"--mode run --campaign-id {checker.CAMPAIGN_ID['headon_group']}"
            ),
        },
        "inputs": {
            "schema_path": {
                "path": "robot_sf/benchmark/schemas/episode.schema.v1.json",
                "sha256": checker.FROZEN_INPUT_SHA256[
                    "robot_sf/benchmark/schemas/episode.schema.v1.json"
                ],
                "artifact_status": "available",
            },
            "scenario_matrix": {
                "path": "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml",
                "sha256": checker.FROZEN_INPUT_SHA256[
                    "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml"
                ],
                "artifact_status": "available",
            },
            "algo_config": {"path": None, "sha256": None, "artifact_status": "not_provided"},
        },
        "campaign_identity": {
            "scenario_matrix_hash": "test-arm-hash",
            "algorithm": "ppo",
        },
        "raw_artifacts": [
            {
                "kind": "episodes_jsonl",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
        "rows": [
            {
                "jsonl_line": index,
                "episode_id": row["episode_id"],
                "scenario_id": row["scenario_id"],
                "seed": row["seed"],
                "repo_commit": SOURCE_SHA,
                "config_hash": row["config_hash"],
            }
            for index, row in enumerate(rows)
        ],
    }
    path.with_name(path.name + ".provenance.json").write_text(json.dumps(sidecar))
    return path


def _check(
    archive: Path,
    traces: Path,
    tmp_path: Path,
    seeds: list[int],
    *,
    observer_sidecar_dirs: list[Path] | None = None,
) -> dict:
    configs, manifests, digests, effective = _bindings(tmp_path, seeds)
    traces = _producer_trace(
        tmp_path, [json.loads(line) for line in traces.read_text().splitlines()]
    )
    return check(
        archive,
        [traces],
        configs,
        manifests,
        expected={("ppo", "classic_doorway_medium", seed) for seed in seeds},
        expected_archive_sha256=None,
        expected_config_sha256=digests,
        expected_effective_hash=effective,
        observer_sidecar_dirs=observer_sidecar_dirs,
    )


def _observer_sidecar(row: dict, config_sha256: str) -> dict:
    observer_path = Path(checker.__file__).with_name("issue_9671_force_observer_sitecustomize.py")
    return {
        "episode_id": row["episode_id"],
        "episode_record_canonical_sha256": checker._canonical_row_sha(row),
        "observer_provenance": {
            "source_state_schema": checker.SOURCE_STATE_SCHEMA,
            "frozen_source_commit": SOURCE_SHA,
            "frozen_source_tree_oid": checker.FROZEN_SOURCE_TREE_OID,
            "frozen_source_worktree_clean": True,
            "frozen_source_worktree_status_sha256": checker.EMPTY_STATUS_SHA256,
            "diagnostic_config_sha256": config_sha256,
            "campaign_id": checker.CAMPAIGN_ID["headon_group"],
            "observer_sha256": hashlib.sha256(observer_path.read_bytes()).hexdigest(),
            "job_id": "123",
            "pid": "456",
        },
    }


def test_reports_mismatch_and_absent_release_row(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(
        "\n".join(
            json.dumps(row)
            for row in [_row(113, "success", trace=True), _row(22, "collision", trace=True)]
        )
        + "\n"
    )
    report = _check(archive, traces, tmp_path, [113, 22])
    assert report["source_state_validation"] == "not_supplied"
    assert [row["comparison"] for row in report["comparisons"]] == [
        "no_release_row",
        "mismatch",
    ]
    assert report["comparisons"][1]["release_interaction_exposure_steps"] == 0
    assert report["comparisons"][1]["trace_interaction_exposure_steps"] == 0


def test_validates_clean_frozen_source_state_against_every_episode(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    _, _, config_digests, _ = _bindings(tmp_path, [113])
    sidecar_dir = tmp_path / "robot-force-sidecars"
    sidecar_dir.mkdir()
    sidecar_path = sidecar_dir / f"{trace['episode_id']}.robot-force.json"
    sidecar_path.write_text(json.dumps(_observer_sidecar(trace, config_digests["headon_group"])))

    report = _check(archive, traces, tmp_path, [113], observer_sidecar_dirs=[sidecar_dir])

    assert report["source_state_validation"] == "verified"
    assert list(report["observer_sidecars_sha256"]) == [str(sidecar_path)]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frozen_source_tree_oid", "0" * 40),
        ("frozen_source_worktree_clean", False),
        ("frozen_source_worktree_status_sha256", "0" * 64),
    ],
)
def test_rejects_dirty_or_mismatched_observer_source_state(
    tmp_path: Path, field: str, value: object
) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    _, _, config_digests, _ = _bindings(tmp_path, [113])
    sidecar_dir = tmp_path / "robot-force-sidecars"
    sidecar_dir.mkdir()
    sidecar = _observer_sidecar(trace, config_digests["headon_group"])
    sidecar["observer_provenance"][field] = value
    (sidecar_dir / f"{trace['episode_id']}.robot-force.json").write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match="observer source-state receipt mismatch"):
        _check(archive, traces, tmp_path, [113], observer_sidecar_dirs=[sidecar_dir])


def test_rejects_observer_sidecar_bound_to_different_episode_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    _, _, config_digests, _ = _bindings(tmp_path, [113])
    sidecar_dir = tmp_path / "robot-force-sidecars"
    sidecar_dir.mkdir()
    sidecar = _observer_sidecar(trace, config_digests["headon_group"])
    sidecar["episode_record_canonical_sha256"] = "0" * 64
    (sidecar_dir / f"{trace['episode_id']}.robot-force.json").write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match="observer source-state receipt mismatch"):
        _check(archive, traces, tmp_path, [113], observer_sidecar_dirs=[sidecar_dir])


@pytest.mark.parametrize("field", ["outcome", "termination_reason"])
def test_paired_rows_compare_canonical_outcome_fields(tmp_path: Path, field: str) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    if field == "outcome":
        trace["outcome"]["collision_event"] = False
    else:
        trace["termination_reason"] = "terminated"
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")

    report = _check(archive, traces, tmp_path, [113])

    assert report["comparison_counts"] == {"match": 0, "mismatch": 1, "no_release_row": 0}
    comparison = report["comparisons"][0]
    assert comparison["comparison"] == "mismatch"
    assert (
        comparison["trace_outcome"] != comparison["release_outcome"]
        or comparison["trace_termination_reason"] != comparison["release_termination_reason"]
    )


@pytest.mark.parametrize("source", ["trace", "release"])
def test_rejects_contradictory_outcome_flags(tmp_path: Path, source: str) -> None:
    release_row = _row(113, "collision", trace=False)
    trace = _row(113, "collision", trace=True)
    target = release_row if source == "release" else trace
    target["outcome"]["route_complete"] = True
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [release_row])
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")

    with pytest.raises(ValueError, match="contradictory canonical outcome"):
        _check(archive, traces, tmp_path, [113])


@pytest.mark.parametrize("source", ["trace", "release"])
def test_rejects_mismatched_algorithm_identity(tmp_path: Path, source: str) -> None:
    release_row = _row(113, "collision", trace=False)
    trace = _row(113, "collision", trace=True)
    target = release_row if source == "release" else trace
    target["algo"] = "social_force"
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [release_row])
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")

    with pytest.raises(ValueError, match="top-level algo differs from scenario_params.algo"):
        _check(archive, traces, tmp_path, [113])


def test_accepts_native_no_fallback_planner_diagnostics(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    trace["algorithm_metadata"]["planner_diagnostics"] = {
        "planner_type": "SocialForcePlanner",
        "fallback": False,
        "fallback_count": 0,
        "fallback_reason": None,
        "fallback_reasons": {},
        "fast_pysf_wrapper": {
            "fallback": False,
            "fallback_count": 0,
            "fallback_reason": None,
            "fallback_reasons": {},
        },
    }
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")

    report = _check(archive, traces, tmp_path, [113])

    assert report["comparison_counts"] == {"match": 1, "mismatch": 0, "no_release_row": 0}


@pytest.mark.parametrize(
    "field",
    ["fallback_count", "fallback_reason", "fallback_reasons"],
)
def test_rejects_incomplete_native_no_fallback_diagnostics(tmp_path: Path, field: str) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    trace["algorithm_metadata"]["planner_diagnostics"] = {"fallback": False}
    trace["algorithm_metadata"]["planner_diagnostics"].pop(field, None)
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")

    with pytest.raises(ValueError, match="malformed Social Force no-fallback diagnostics"):
        _check(archive, traces, tmp_path, [113])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fallback_count", False),
        ("fallback_count", 1),
        ("fallback_reason", "unexpected"),
        ("fallback_reasons", {"kernel": 0}),
    ],
)
def test_rejects_malformed_native_no_fallback_diagnostics(
    tmp_path: Path, field: str, value: object
) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    trace["algorithm_metadata"]["planner_diagnostics"] = {
        "fallback": False,
        "fallback_count": 0,
        "fallback_reason": None,
        "fallback_reasons": {},
    }
    trace["algorithm_metadata"]["planner_diagnostics"][field] = value
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")

    with pytest.raises(ValueError, match="malformed Social Force no-fallback diagnostics"):
        _check(archive, traces, tmp_path, [113])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fallback", True),
        ("fallback_count", 1),
        ("fallback_reasons", {"kernel": 1}),
    ],
)
def test_rejects_positive_social_force_fallback_diagnostics(
    tmp_path: Path, field: str, value: object
) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    trace["algorithm_metadata"]["planner_diagnostics"] = {
        "fallback": False,
        "fallback_count": 0,
        "fallback_reason": None,
        "fallback_reasons": {},
        field: value,
    }
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")

    with pytest.raises(
        ValueError,
        match="inadmissible runtime mode|malformed Social Force no-fallback diagnostics",
    ):
        _check(archive, traces, tmp_path, [113])


@pytest.mark.parametrize(
    ("source", "field"),
    [
        ("trace", "outcome"),
        ("trace", "termination_reason"),
        ("release", "outcome"),
        ("release", "termination_reason"),
    ],
)
def test_rejects_missing_canonical_outcome_fields(tmp_path: Path, source: str, field: str) -> None:
    release_row = _row(113, "collision", trace=False)
    trace = _row(113, "collision", trace=True)
    if source == "release":
        del release_row[field]
    else:
        del trace[field]
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [release_row])
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")

    with pytest.raises(ValueError, match="missing or invalid"):
        _check(archive, traces, tmp_path, [113])


def test_rejects_missing_paired_release_row(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(
        "\n".join(
            json.dumps(row)
            for row in [_row(113, "collision", trace=True), _row(114, "collision", trace=True)]
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="required paired release rows are missing"):
        _check(archive, traces, tmp_path, [113, 114])


def test_accepts_separately_pinned_observer_campaign_ids(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    configs, manifests, digests, effective = _bindings(tmp_path, [113])
    trace = _producer_trace(tmp_path, [_row(113, "collision", trace=True)])
    new_ids = {
        "headon_group": "issue9671-observer-headon-test",
        "doorway": "issue9671-observer-doorway-test",
    }
    for name, new_id in new_ids.items():
        old_id = checker.CAMPAIGN_ID[name]
        campaign = json.loads(manifests[name].read_text())
        campaign["campaign_id"] = new_id
        campaign["invoked_command"] = campaign["invoked_command"].replace(old_id, new_id)
        manifests[name].write_text(json.dumps(campaign))
    producer_path = trace.with_name(trace.name + ".provenance.json")
    producer = json.loads(producer_path.read_text())
    producer["run"]["invocation"] = producer["run"]["invocation"].replace(
        checker.CAMPAIGN_ID["headon_group"], new_ids["headon_group"]
    )
    producer_path.write_text(json.dumps(producer))
    report = check(
        archive,
        [trace],
        configs,
        manifests,
        expected={("ppo", "classic_doorway_medium", 113)},
        expected_archive_sha256=None,
        expected_config_sha256=digests,
        expected_effective_hash=effective,
        expected_campaign_ids=new_ids,
    )
    assert report["comparison_counts"] == {"match": 1, "mismatch": 0, "no_release_row": 0}


@pytest.mark.parametrize("campaign_id", [None, "", "  "])
def test_rejects_missing_or_empty_expected_campaign_id(tmp_path: Path, campaign_id: object) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    configs, manifests, digests, effective = _bindings(tmp_path, [113])
    trace = _producer_trace(tmp_path, [_row(113, "collision", trace=True)])
    campaign_ids: dict[str, str | None] = dict(checker.CAMPAIGN_ID)
    campaign_ids["headon_group"] = campaign_id if isinstance(campaign_id, str) else None
    with pytest.raises(ValueError, match="campaign IDs.*non-empty"):
        check(
            archive,
            [trace],
            configs,
            manifests,
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_archive_sha256=None,
            expected_config_sha256=digests,
            expected_effective_hash=effective,
            expected_campaign_ids=campaign_ids,
        )


def test_rejects_expected_campaign_id_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    configs, manifests, digests, effective = _bindings(tmp_path, [113])
    trace = _producer_trace(tmp_path, [_row(113, "collision", trace=True)])
    campaign_ids = dict(checker.CAMPAIGN_ID)
    campaign_ids["headon_group"] = "issue9671-wrong-campaign"

    with pytest.raises(ValueError, match="campaign manifest identity mismatch"):
        check(
            archive,
            [trace],
            configs,
            manifests,
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_archive_sha256=None,
            expected_config_sha256=digests,
            expected_effective_hash=effective,
            expected_campaign_ids=campaign_ids,
        )


def test_rejects_parameter_drift(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    trace["scenario_params"]["run_dt"] = 0.2
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="changes release scenario/planner parameters"):
        _check(archive, traces, tmp_path, [113])


def test_rejects_missing_per_pedestrian_force(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(22, "collision", trace=True)
    trace["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["pedestrians"] = [
        {"position": [1.0, 1.0], "velocity": [0.0, 0.0]}
    ]
    trace["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["planner"] = {
        "ammv": {"pedestrian_force_vectors": [None]}
    }
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="lacks pedestrian forces"):
        _check(archive, traces, tmp_path, [22])


@pytest.mark.parametrize(
    ("metadata_key", "metadata_value"),
    [
        ("execution_mode", "fallback"),
        ("readiness_status", "fallback"),
        ("availability_status", "not_available"),
        ("planner_runtime", {"fallback_used": True}),
        ("policy_step_timeout", {"fallback_actions": 1}),
        ("fallback_reason", "policy_step_timeout"),
        ("status", "policy_step_timeout_fallback"),
        ("status", "policy_step_isolation_unavailable"),
        ("planner_runtime", {"status": "planner_unavailable"}),
    ],
)
def test_rejects_fallback_runtime(
    tmp_path: Path, metadata_key: str, metadata_value: object
) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    trace["algorithm_metadata"][metadata_key] = metadata_value
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="inadmissible runtime mode"):
        _check(archive, traces, tmp_path, [113])


@pytest.mark.parametrize("defect", ["short", "noncontiguous"])
def test_rejects_incomplete_step_trace(tmp_path: Path, defect: str) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    if defect == "short":
        trace["steps"] = 600
        trace["truncated"] = True
        match = "step count does not match episode"
    else:
        trace["steps"] = 2
        second = copy.deepcopy(trace["algorithm_metadata"]["simulation_step_trace"]["steps"][0])
        second["step"] = 2
        second["time_s"] = 0.3
        trace["algorithm_metadata"]["simulation_step_trace"]["steps"].append(second)
        match = "noncontiguous index/time"
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match=match):
        _check(archive, traces, tmp_path, [113])


def test_rejects_unpaired_scientific_parameter_drift(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(22, "collision", trace=True)
    trace["scenario_params"]["run_dt"] = 0.2
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="changes release scenario/planner parameters"):
        _check(archive, traces, tmp_path, [22])


def test_rejects_unbound_config_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(_row(113, "collision", trace=True)) + "\n")
    configs, manifests, digests, effective = _bindings(tmp_path, [113])
    traces = _producer_trace(tmp_path, [json.loads(traces.read_text())])
    configs["doorway"].write_text(configs["doorway"].read_text() + "# drift\n")
    with pytest.raises(ValueError, match="diagnostic config SHA-256 mismatch"):
        check(
            archive,
            [traces],
            configs,
            manifests,
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_archive_sha256=None,
            expected_config_sha256=digests,
            expected_effective_hash=effective,
        )


def test_rejects_unstaged_config_invocation(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    configs, manifests, digests, effective = _bindings(tmp_path, [113])
    traces = _producer_trace(tmp_path, [_row(113, "collision", trace=True)])
    manifest = json.loads(manifests["headon_group"].read_text())
    manifest["invoked_command"] = manifest["invoked_command"].replace(
        str(tmp_path / "frozen-root" / checker.DIAGNOSTIC_CONFIG["headon_group"]),
        "configs/benchmarks/issue_9671_trace_headon_group_v007.yaml",
    )
    manifests["headon_group"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="campaign manifest identity mismatch"):
        check(
            archive,
            [traces],
            configs,
            manifests,
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_archive_sha256=None,
            expected_config_sha256=digests,
            expected_effective_hash=effective,
        )


def test_rejects_mixed_trace_bytes_without_matching_producer_manifest(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    configs, manifests, digests, effective = _bindings(tmp_path, [113, 22])
    traces = _producer_trace(tmp_path, [_row(113, "collision", trace=True)])
    with traces.open("a") as stream:
        stream.write(json.dumps(_row(22, "collision", trace=True)) + "\n")
    with pytest.raises(ValueError, match="does not bind JSONL bytes"):
        check(
            archive,
            [traces],
            configs,
            manifests,
            expected={("ppo", "classic_doorway_medium", seed) for seed in (113, 22)},
            expected_archive_sha256=None,
            expected_config_sha256=digests,
            expected_effective_hash=effective,
        )


def test_rejects_coherent_file_and_sidecar_from_another_campaign(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    configs, manifests, digests, effective = _bindings(tmp_path, [113])
    traces = _producer_trace(tmp_path, [_row(113, "collision", trace=True)])
    sidecar_path = traces.with_name(traces.name + ".provenance.json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["run"]["invocation"] = (
        "scripts/tools/run_camera_ready_benchmark.py "
        "--config configs/benchmarks/other_campaign.yaml --mode run "
        "--campaign-id other_campaign"
    )
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="producer invocation does not match diagnostic campaign"):
        check(
            archive,
            [traces],
            configs,
            manifests,
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_archive_sha256=None,
            expected_config_sha256=digests,
            expected_effective_hash=effective,
        )


def test_cli_keeps_report_and_exits_nonzero_on_outcome_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {"comparison_counts": {"mismatch": 1}, "comparisons": [{"comparison": "mismatch"}]}
    observed: dict[str, object] = {}

    def fake_check(*_args: object, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return report

    monkeypatch.setattr(checker, "check", fake_check)
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_issue_9671_trace_reexport.py",
            "--release-archive",
            "unused.tar.gz",
            "--traces",
            "unused.jsonl",
            "--headon-config",
            "unused-headon.yaml",
            "--doorway-config",
            "unused-doorway.yaml",
            "--headon-manifest",
            "unused-headon.json",
            "--doorway-manifest",
            "unused-doorway.json",
            "--output",
            str(output),
        ],
    )
    assert checker.main() == 2
    assert json.loads(output.read_text()) == report
    assert observed["expected_campaign_ids"] == checker.CAMPAIGN_ID


def test_cli_passes_selected_campaign_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = {"comparison_counts": {"mismatch": 0}, "comparisons": []}
    observed: dict[str, object] = {}

    def fake_check(*_args: object, **kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return report

    monkeypatch.setattr(checker, "check", fake_check)
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_issue_9671_trace_reexport.py",
            "--release-archive",
            "unused.tar.gz",
            "--traces",
            "unused.jsonl",
            "--headon-config",
            "unused-headon.yaml",
            "--doorway-config",
            "unused-doorway.yaml",
            "--headon-manifest",
            "unused-headon.json",
            "--doorway-manifest",
            "unused-doorway.json",
            "--headon-campaign-id",
            "issue9671-headon-new",
            "--doorway-campaign-id",
            "issue9671-doorway-new",
            "--output",
            str(output),
        ],
    )

    assert checker.main() == 0
    assert observed["expected_campaign_ids"] == {
        "headon_group": "issue9671-headon-new",
        "doorway": "issue9671-doorway-new",
    }
