"""Focused contract tests for the strict issue #6814 provenance overlay."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark import issue_6814_trace_reexport as issue6814
from robot_sf.benchmark.issue_6814_trace_reexport import (
    EXECUTION_COMMIT,
    TraceIdentity,
    build_initial_state_record,
    build_issue_6814_trace_packet,
    build_issue_6814_trace_source_contract,
    build_static_run_config,
    enrich_simulation_trace_export,
    initial_state_digest,
    static_config_digest,
)
from robot_sf.benchmark.trace_reexport_packaging import (
    RealReexportBindingError,
    TraceReexportPackagingError,
    VerifiedRealReexportRowSource,
    load_verified_real_reexport_row_source,
)
from scripts.tools.build_simulation_trace_export import (
    SimulationTraceNormalizationError,
    apply_strict_metadata_projection,
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _frame(*, step: int, robot_x: float, actor_order: tuple[str, ...] = ("l1",)) -> dict[str, Any]:
    return {
        "step": step,
        "time_s": step * 0.1 + 0.1,
        "robot": {"position": [robot_x, 0.0], "heading": 0.0, "velocity": [1.0, 0.0]},
        "pedestrians": [
            {"id": actor_id, "position": [1.0 + index, 0.0], "velocity": [0.0, 0.0]}
            for index, actor_id in enumerate(actor_order)
        ],
        "planner": {
            "selected_action": {"linear_velocity": 1.0, "angular_velocity": 0.0},
            "event": "step",
        },
    }


def _trace(identity: TraceIdentity, *, actor_order: tuple[str, ...] = ("l1",)) -> dict[str, Any]:
    return {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": f"{identity.episode_id}-trace",
        "source": {
            "scenario_id": identity.scenario_id,
            "seed": identity.seed,
            "planner_id": identity.planner_id,
            "episode_id": identity.episode_id,
            "generated_by": "verified-source-test",
        },
        "evidence_boundary": "analysis_workbench_only",
        "coordinate_frame": "world",
        "units": {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
        "frames": [
            _frame(step=0, robot_x=0.0, actor_order=actor_order),
            _frame(step=1, robot_x=0.1, actor_order=actor_order),
        ],
    }


def _raw_row(identity: TraceIdentity, *, actor_order: tuple[str, ...] = ("l1",)) -> dict[str, Any]:
    trace = _trace(identity, actor_order=actor_order)
    return {
        "episode_id": identity.episode_id,
        "scenario_id": identity.scenario_id,
        "seed": identity.seed,
        "algo": identity.planner_id,
        "git_hash": EXECUTION_COMMIT,
        "config_hash": identity.row_config_hash,
        "simulator_settings": {"horizon": 600, "dt": 0.1},
        "scenario_params": {"algo": identity.planner_id, "run_horizon": 600, "run_dt": 0.1},
        "outcome": {"collision_event": False, "timeout_event": False, "route_complete": True},
        "algorithm_metadata": {
            "config_hash": identity.algorithm_config_hash,
            "config": {"planner": identity.planner_id, "hidden_size": 32},
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v1",
                "steps": trace["frames"],
            },
        },
    }


def _source(identity: TraceIdentity, *, outcome: bool = True) -> VerifiedRealReexportRowSource:
    raw = _raw_row(identity)
    if not outcome:
        raw.pop("outcome")
    scenario_matrix = (
        Path(__file__).parents[2] / "configs/scenarios/classic_interactions_francis2023.yaml"
    )
    return VerifiedRealReexportRowSource(
        arm=identity.arm,
        job_id=identity.job_id,
        row_index=identity.row_index,
        episode_id=identity.episode_id,
        scenario_id=identity.scenario_id,
        planner_id=identity.planner_id,
        seed=identity.seed,
        execution_commit=EXECUTION_COMMIT,
        raw_row=raw,
        raw_row_sha256=identity.raw_trace_sha256 or "a" * 64,
        prior_normalized_sha256=identity.prior_normalized_sha256 or "b" * 64,
        episodes_sha256="c" * 64,
        manifest_sha256="d" * 64,
        run_summary_sha256="e" * 64,
        preflight_sha256="f" * 64,
        result_provenance_sha256="1" * 64,
        result_provenance_row={
            "repo_commit": EXECUTION_COMMIT,
            "simulator_settings": {"horizon": 600, "dt": 0.1},
        },
        source_root_retrieval_key=f"synthetic/{identity.arm}",
        run_summary={"horizon": 600, "dt": 0.1},
        preflight={
            "horizon": 600,
            "time_step_s": 0.1,
            "algorithm_config_hash": identity.algorithm_config_hash,
        },
        result_provenance_manifest={
            "inputs": {
                "scenario_matrix": {
                    "path": "configs/scenarios/classic_interactions_francis2023.yaml",
                    "sha256": _sha256_bytes(scenario_matrix.read_bytes()),
                }
            }
        },
    )


def _base_identity(
    arm: str, planner: str, scenario: str, seed: int, row_index: int, job_id: str
) -> TraceIdentity:
    return TraceIdentity(
        arm=arm,
        job_id=job_id,
        row_index=row_index,
        episode_id=f"{scenario}--{seed}--{arm}",
        scenario_id=scenario,
        planner_id=planner,
        seed=seed,
        row_config_hash=f"row-{arm}-{seed}",
        algorithm_config_hash=f"algorithm-{planner}",
    )


def _make_source_contract() -> tuple[VerifiedRealReexportRowSource, dict[str, Any], dict[str, Any]]:
    identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    source = _source(identity)
    contract = build_issue_6814_trace_source_contract(
        source,
        execution_repository=Path(__file__).parents[2],
    )
    trace = _trace(identity)
    return source, contract, trace


def _make_synthetic_packet_inputs(
    tmp_path: Path,
) -> tuple[Path, dict[str, Path], tuple[TraceIdentity, ...], str]:
    """Create a four-row package with hashes computed from its temporary bytes."""

    bases = (
        _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483"),
        _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 114, 4, "13483"),
        _base_identity(
            "double_bottleneck_goal",
            "goal",
            "classic_realworld_double_bottleneck_high",
            118,
            8,
            "13487",
        ),
        _base_identity(
            "double_bottleneck_ppo",
            "ppo",
            "classic_realworld_double_bottleneck_high",
            118,
            8,
            "13488",
        ),
    )
    roots: dict[str, Path] = {}
    selected: list[TraceIdentity] = []
    package = tmp_path / "package"
    package.mkdir()
    for base in bases:
        root = tmp_path / "arms" / base.arm
        root.mkdir(parents=True, exist_ok=True)
        rows: list[bytes] = []
        selected_line = b""
        for row_index in range(1, base.row_index + 1):
            row_identity = (
                base
                if row_index == base.row_index
                else replace(base, episode_id=f"filler-{base.arm}-{row_index}")
            )
            line = (
                json.dumps(_raw_row(row_identity), sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )
            rows.append(line)
            if row_index == base.row_index:
                selected_line = line
        episodes_path = root / "episodes.jsonl"
        episodes = (
            episodes_path.read_bytes() + selected_line if episodes_path.exists() else b"".join(rows)
        )
        episodes_path.write_bytes(episodes)
        manifest_payload = {
            "job_id": base.job_id,
            "planner_id": base.planner_id,
            "scenario_id": base.scenario_id,
        }
        _write_json(root / "manifest.json", manifest_payload)
        (root / "run_summary.yaml").write_text("horizon: 600\ndt: 0.1\n", encoding="utf-8")
        _write_json(
            root / "validate_config.json",
            {
                "horizon": 600,
                "time_step_s": 0.1,
                "algorithm_config_hash": base.algorithm_config_hash,
            },
        )
        selected_hash = _sha256_bytes(selected_line)
        selected_identity = replace(base, raw_trace_sha256=selected_hash)
        prior_path = package / "traces" / f"{base.arm}_{base.seed}.json"
        _write_json(prior_path, _trace(selected_identity))
        selected_identity = replace(
            selected_identity, prior_normalized_sha256=_sha256_bytes(prior_path.read_bytes())
        )
        selected.append(selected_identity)
        roots[base.arm] = root

    source_pointer_arms: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for identity in selected:
        root = roots[identity.arm]
        pointer = {
            "arm": identity.arm,
            "job_id": identity.job_id,
            "planner": identity.planner_id,
            "scenario_id": identity.scenario_id,
            "retrieval_key": f"synthetic/{identity.arm}",
            "episodes_sha256": _sha256_bytes((root / "episodes.jsonl").read_bytes()),
            "manifest_sha256": _sha256_bytes((root / "manifest.json").read_bytes()),
            "run_summary_sha256": _sha256_bytes((root / "run_summary.yaml").read_bytes()),
            "preflight_sha256": _sha256_bytes((root / "validate_config.json").read_bytes()),
        }
        source_pointer_arms.append(pointer)
        mapping_rows.append(
            {
                "planner": identity.planner_id,
                "scenario_id": identity.scenario_id,
                "seed": identity.seed,
                "episode_id": identity.episode_id,
                "raw_trace_sha256": identity.raw_trace_sha256,
                "normalized_trace_sha256": identity.prior_normalized_sha256,
                "trace_artifact_uri": f"traces/{identity.arm}_{identity.seed}.json",
                "source_provenance": {
                    "arm": identity.arm,
                    "job_id": identity.job_id,
                    "source_row_index": identity.row_index,
                    "source_episode_id": identity.episode_id,
                    "source_episodes_sha256": pointer["episodes_sha256"],
                    "source_manifest_sha256": pointer["manifest_sha256"],
                    "run_summary_sha256": pointer["run_summary_sha256"],
                    "preflight_sha256": pointer["preflight_sha256"],
                    "execution_commit": EXECUTION_COMMIT,
                    "row_config_hash": identity.row_config_hash,
                    "algorithm_config_hash": identity.algorithm_config_hash,
                },
            }
        )
    while len(mapping_rows) < 90:
        mapping_rows.append({"planner": "dummy", "scenario_id": "dummy", "seed": len(mapping_rows)})
    _write_json(
        package / "mapping_receipt.json",
        {
            "schema_version": "issue_6412_trace_reexport_mapping_receipt.v1",
            "n_rows": 90,
            "rows": mapping_rows,
        },
    )
    _write_json(
        package / "source_pointer.json",
        {
            "schema_version": "issue_6412_source_pointer.v1",
            "retrieval_key": "synthetic",
            "arms": source_pointer_arms,
        },
    )
    _write_json(
        package / "package_manifest.json",
        {
            "schema_version": "issue_6412_real_reexport_package.v1",
            "execution_commit": EXECUTION_COMMIT,
            "n_requested": 90,
            "n_admitted": 88,
            "n_excluded": 2,
            "visualization_only": True,
            "excluded_tuples": [
                ["ppo", "classic_doorway_medium", 128],
                ["ppo", "classic_doorway_medium", 130],
            ],
        },
    )
    (package / "README.md").write_text("synthetic\n", encoding="utf-8")
    sums = "".join(
        f"{_sha256_bytes((package / name).read_bytes())}  {name}\n"
        for name in (
            "README.md",
            "package_manifest.json",
            "source_pointer.json",
            "mapping_receipt.json",
        )
    ).encode()
    (package / "SHA256SUMS").write_bytes(sums)
    package_sha = _sha256_bytes(sums)
    _write_json(
        package / "package_complete.json",
        {
            "schema_version": "issue_6412_package_complete.v1",
            "status": "complete",
            "visualization_only": True,
            "n_requested": 90,
            "n_admitted": 88,
            "n_excluded": 2,
            "sha256sums_sha256": package_sha,
        },
    )
    return package, roots, tuple(selected), package_sha


def _write_result_provenance_sidecar(root: Path, identity: TraceIdentity) -> Path:
    """Write one valid result-provenance sidecar for loader-link tests."""

    episodes = root / "episodes.jsonl"
    payload = {
        "schema_version": "benchmark_result_provenance.v1",
        "run": {
            "run_id": f"run-{identity.job_id}",
            "repo_commit": EXECUTION_COMMIT,
            "runner": "synthetic-test",
        },
        "inputs": {
            "schema_path": {"path": "schema.json"},
            "scenario_matrix": {
                "path": "configs/scenarios/classic_interactions_francis2023.yaml",
                "sha256": _sha256_bytes(
                    (
                        Path(__file__).parents[2]
                        / "configs/scenarios/classic_interactions_francis2023.yaml"
                    ).read_bytes()
                ),
            },
        },
        "campaign_identity": {
            "scenario_matrix_hash": "synthetic-matrix",
            "total_jobs": 30,
            "written": 30,
        },
        "raw_artifacts": [
            {
                "kind": "episodes_jsonl",
                "path": str(episodes),
                "sha256": _sha256_bytes(episodes.read_bytes()),
                "artifact_status": "available",
            }
        ],
        "rows": [
            {
                "episode_id": identity.episode_id,
                "scenario_id": identity.scenario_id,
                "seed": identity.seed,
                "config_hash": identity.row_config_hash,
                "repo_commit": EXECUTION_COMMIT,
                "raw_artifact": str(episodes),
                "jsonl_line": identity.row_index,
                "trace_artifact_sha256": _sha256_bytes(episodes.read_bytes()),
                "simulator_settings": {"horizon": 600, "dt": 0.1},
                "postprocessing": [],
            }
        ],
        "completeness": {"status": "complete"},
    }
    path = root / "episodes.jsonl.provenance.json"
    _write_json(path, payload)
    return path


def test_source_contract_selects_exact_6412_row_and_hashes() -> None:
    source, contract, _trace_payload = _make_source_contract()
    assert source.row_index == 3
    assert contract["trace_identity"]["episode_id"] == source.episode_id
    assert contract["fields"]["map_id"]["status"] == "available"
    assert len(contract["canonical_config"]["sha256"]) == 64


def test_source_contract_rejects_6412_package_digest_drift(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    _write_json(package / "package_complete.json", {"status": "complete"})
    with pytest.raises(RealReexportBindingError, match="schema"):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=tmp_path,
            expected_identity=_base_identity(
                "doorway_ppo", "ppo", "classic_doorway_medium", 113, 1, "13483"
            ),
        )


def test_source_contract_rejects_external_episodes_digest_drift(tmp_path: Path) -> None:
    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    episodes = roots[identities[0].arm] / "episodes.jsonl"
    episodes.write_bytes(episodes.read_bytes() + b"tampered\n")
    with pytest.raises(TraceReexportPackagingError, match="episodes_sha256"):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=roots[identities[0].arm],
            expected_identity=identities[0],
            expected_package_sha256=package_sha,
        )


@pytest.mark.parametrize(
    ("artifact_name", "message"),
    (
        ("manifest.json", "manifest_sha256"),
        ("run_summary.yaml", "run_summary_sha256"),
        ("validate_config.json", "preflight_sha256"),
    ),
)
def test_source_contract_rejects_external_artifact_digest_drift(
    tmp_path: Path, artifact_name: str, message: str
) -> None:
    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    artifact = roots[identities[0].arm] / artifact_name
    artifact.write_bytes(artifact.read_bytes() + b"tampered\n")
    with pytest.raises(TraceReexportPackagingError, match=message):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=roots[identities[0].arm],
            expected_identity=identities[0],
            expected_package_sha256=package_sha,
        )


def test_source_contract_validates_result_provenance_row_link(tmp_path: Path) -> None:
    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    _write_result_provenance_sidecar(roots[identities[0].arm], identities[0])
    source = load_verified_real_reexport_row_source(
        package_root=package,
        external_arm_root=roots[identities[0].arm],
        expected_identity=identities[0],
        expected_package_sha256=package_sha,
    )
    assert source.result_provenance_row is not None
    assert source.result_provenance_manifest is not None


def test_source_contract_rejects_result_provenance_row_mismatch(tmp_path: Path) -> None:
    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    sidecar = _write_result_provenance_sidecar(roots[identities[0].arm], identities[0])
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["rows"][0]["episode_id"] = "wrong-episode"
    _write_json(sidecar, payload)
    with pytest.raises(RealReexportBindingError, match="row link"):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=roots[identities[0].arm],
            expected_identity=identities[0],
            expected_package_sha256=package_sha,
        )


def test_static_config_digest_excludes_seed_and_realization() -> None:
    kwargs = {
        "scenario_id": "classic_doorway_medium",
        "scenario_matrix_sha256": "a" * 64,
        "scenario_definition_sha256": "b" * 64,
        "map_id": "maps/svg_maps/classic_doorway.svg",
        "map_sha256": "c" * 64,
        "horizon_steps": 600,
        "time_step_s": 0.1,
        "planner_id": "ppo",
        "planner_config_id": "config",
        "planner_config_sha256": "d" * 64,
        "source_algorithm_config_hash": "short",
    }
    left = build_static_run_config(**kwargs)
    right = build_static_run_config(**kwargs)
    assert static_config_digest(left) == static_config_digest(right)
    assert "seed" not in left and "episode_id" not in left


def test_static_config_digest_changes_with_planner_config() -> None:
    common = {
        "scenario_id": "classic_doorway_medium",
        "scenario_matrix_sha256": "a" * 64,
        "scenario_definition_sha256": "b" * 64,
        "map_id": "maps/svg_maps/classic_doorway.svg",
        "map_sha256": "c" * 64,
        "horizon_steps": 600,
        "time_step_s": 0.1,
        "planner_id": "ppo",
        "planner_config_id": "config",
        "source_algorithm_config_hash": "short",
    }
    assert static_config_digest(
        build_static_run_config(planner_config_sha256="d" * 64, **common)
    ) != static_config_digest(build_static_run_config(planner_config_sha256="e" * 64, **common))


def test_initial_state_digest_is_actor_order_independent() -> None:
    identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    first = build_initial_state_record(_trace(identity, actor_order=("l1", "l2")))
    second = copy.deepcopy(first)
    second["actors"].reverse()
    assert initial_state_digest(first) == initial_state_digest(second)


def test_initial_state_digest_requires_authoritative_actor_ids() -> None:
    identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    with pytest.raises(ValueError, match="actor identity"):
        build_initial_state_record(_trace(identity, actor_order=("ped-0",)))


def test_metadata_projection_changes_only_approved_json_paths() -> None:
    _source_record, contract, trace = _make_source_contract()
    enriched, receipt = enrich_simulation_trace_export(trace, source_contract=contract)
    assert receipt["semantic_payload_unchanged"] is True
    for before, after in zip(trace["frames"], enriched["frames"], strict=True):
        assert before["robot"] == after["robot"]
        assert before["pedestrians"] == after["pedestrians"]
        assert before["step"] == after["step"]
        assert before["time_s"] == after["time_s"]
        assert before["planner"]["selected_action"] == after["planner"]["selected_action"]
        assert before["planner"]["event"] == after["planner"]["event"]
        assert after["planner"]["run_config"] == contract["trace_projection"]["planner_run_config"]
    assert "outcome" in enriched["frames"][-1]["planner"]
    assert "outcome" not in enriched["frames"][0]["planner"]


def test_terminal_outcome_requires_exact_raw_row_authority() -> None:
    identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    source = _source(identity, outcome=False)
    contract = build_issue_6814_trace_source_contract(
        source, execution_repository=Path(__file__).parents[2]
    )
    assert contract["trace_projection"]["terminal_outcome"]["status"] == "unavailable"


def test_release_outcome_cannot_create_terminal_trace_evidence() -> None:
    identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    source = _source(identity, outcome=False)
    source = replace(source, run_summary={"outcome": "route_complete"})
    contract = build_issue_6814_trace_source_contract(
        source, execution_repository=Path(__file__).parents[2]
    )
    assert contract["trace_projection"]["terminal_outcome"]["status"] == "unavailable"


def test_source_contract_rejects_conflicting_horizon_values() -> None:
    source, _contract, _trace_payload = _make_source_contract()
    conflicted = replace(source, preflight={"horizon": 599, "time_step_s": 0.1})
    with pytest.raises(ValueError, match="horizon"):
        build_issue_6814_trace_source_contract(
            conflicted, execution_repository=Path(__file__).parents[2]
        )


def test_source_contract_rejects_conflicting_time_step_values() -> None:
    source, _contract, _trace_payload = _make_source_contract()
    conflicted = replace(source, preflight={"horizon": 600, "time_step_s": 0.2})
    with pytest.raises(ValueError, match="time-step"):
        build_issue_6814_trace_source_contract(
            conflicted, execution_repository=Path(__file__).parents[2]
        )


def test_source_contract_rejects_missing_planner_config_snapshot() -> None:
    source, _contract, _trace_payload = _make_source_contract()
    raw = copy.deepcopy(source.raw_row)
    raw["algorithm_metadata"].pop("config")
    raw["planner_config"] = None
    missing = replace(source, raw_row=raw, preflight={"horizon": 600, "time_step_s": 0.1})
    contract = build_issue_6814_trace_source_contract(
        missing, execution_repository=Path(__file__).parents[2]
    )
    assert contract["status"] == "unsupported"
    assert contract["fields"]["planner_config_sha256"]["status"] == "unavailable"


def test_source_contract_does_not_use_current_scenario_horizon_as_run_authority() -> None:
    source, _contract, _trace_payload = _make_source_contract()
    raw = copy.deepcopy(source.raw_row)
    raw["simulator_settings"] = {}
    raw["scenario_params"] = {}
    no_run_settings = replace(
        source,
        raw_row=raw,
        preflight={},
        run_summary={},
        result_provenance_row=None,
        result_provenance_sha256=None,
    )
    contract = build_issue_6814_trace_source_contract(
        no_run_settings, execution_repository=Path(__file__).parents[2]
    )
    assert contract["fields"]["horizon_steps"]["status"] == "unavailable"
    assert contract["fields"]["time_step_s"]["status"] == "unavailable"


def test_strict_export_rejects_unknown_trace_identity_fallbacks() -> None:
    trace = _trace(_base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483"))
    trace["source"]["planner_id"] = "unknown_planner"
    with pytest.raises(SimulationTraceNormalizationError, match="fallback"):
        apply_strict_metadata_projection(
            trace,
            run_config={
                "map_id": "map",
                "horizon": 600,
                "time_step_s": 0.1,
                "config_digest": "a" * 64,
            },
        )


def test_strict_export_rejects_generated_pedestrian_identity() -> None:
    trace = _trace(_base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483"))
    trace["frames"][0]["pedestrians"][0]["id"] = "ped-0"
    with pytest.raises(SimulationTraceNormalizationError, match="generated actor"):
        apply_strict_metadata_projection(
            trace,
            run_config={
                "map_id": "map",
                "horizon": 600,
                "time_step_s": 0.1,
                "config_digest": "a" * 64,
            },
        )


def test_metadata_delta_rejects_unapproved_state_change() -> None:
    trace = _trace(_base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483"))
    trace["frames"][0]["planner"]["run_config"] = {"bad": True}
    with pytest.raises(SimulationTraceNormalizationError, match="non-additive"):
        apply_strict_metadata_projection(
            trace,
            run_config={
                "map_id": "map",
                "horizon": 600,
                "time_step_s": 0.1,
                "config_digest": "a" * 64,
            },
        )


def test_live_package_block_is_not_reported_as_proven_unsupported(tmp_path: Path) -> None:
    output = tmp_path / "packet"
    with pytest.raises(ValueError, match="SHA256SUMS"):
        build_issue_6814_trace_packet(
            package_root=Path(__file__).parents[2]
            / "docs/context/evidence/issue_6412_real_reexport_package",
            arm_roots={
                "doorway_ppo": tmp_path,
                "double_bottleneck_goal": tmp_path,
                "double_bottleneck_ppo": tmp_path,
            },
            external_output_root=output,
        )
    assert not output.exists()


def test_packet_generation_is_atomic_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    monkeypatch.setattr(issue6814, "SELECTED_TRACE_IDENTITIES", identities)
    episodes = roots[identities[0].arm] / "episodes.jsonl"
    episodes.write_bytes(episodes.read_bytes() + b"tampered\n")
    output = tmp_path / "packet"
    with pytest.raises(ValueError, match="episodes_sha256"):
        build_issue_6814_trace_packet(
            package_root=package,
            arm_roots=roots,
            external_output_root=output,
            expected_package_sha256=package_sha,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".packet.staging-*"))


def test_packet_generation_is_content_hash_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    monkeypatch.setattr(issue6814, "SELECTED_TRACE_IDENTITIES", identities)
    output = tmp_path / "packet"
    manifest = build_issue_6814_trace_packet(
        package_root=package,
        arm_roots=roots,
        external_output_root=output,
        execution_repository=Path(__file__).parents[2],
        check_determinism=True,
        expected_package_sha256=package_sha,
    )
    assert manifest["disposition"] == "unsupported"
    assert output.is_dir()
    assert (output / "packet_manifest.json").is_file()
    doorway_receipt = json.loads(
        (output / "pair_receipts" / "doorway_ppo_113_114.json").read_text(encoding="utf-8")
    )
    double_receipt = json.loads(
        (output / "pair_receipts" / "double_bottleneck_goal_118_118.json").read_text(
            encoding="utf-8"
        )
    )
    assert doorway_receipt["comparison_grammar"] == "same_cell_seed_sensitivity"
    assert double_receipt["comparison_grammar"] == "matched_start"
