"""Focused contract tests for the strict issue #6814 provenance overlay."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench.simulation_trace_export import (
    SimulationTraceNormalizationError,
    apply_strict_metadata_projection,
)
from robot_sf.benchmark import issue_6814_trace_reexport as issue6814
from robot_sf.benchmark.issue_6814_trace_reexport import (
    EXECUTION_COMMIT,
    Issue6814Error,
    Issue6814SourceIntegrityError,
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


def _refresh_synthetic_package_sums(package: Path) -> str:
    """Refresh the synthetic package's listed-artifact digest and completion pin."""

    names = ("README.md", "package_manifest.json", "source_pointer.json", "mapping_receipt.json")
    sums = "".join(
        f"{_sha256_bytes((package / name).read_bytes())}  {name}\n" for name in names
    ).encode()
    (package / "SHA256SUMS").write_bytes(sums)
    package_sha = _sha256_bytes(sums)
    complete = json.loads((package / "package_complete.json").read_text(encoding="utf-8"))
    complete["sha256sums_sha256"] = package_sha
    _write_json(package / "package_complete.json", complete)
    return package_sha


def _mapping_identity(identity: TraceIdentity, *, seed: object | None = None) -> dict[str, object]:
    """Return the small mapping identity accepted by the strict package loader."""

    return {
        "arm": identity.arm,
        "planner_id": identity.planner_id,
        "scenario_id": identity.scenario_id,
        "seed": identity.seed if seed is None else seed,
    }


def test_source_contract_selects_exact_6412_row_and_hashes() -> None:
    """Verify the contract binds the approved row and emits a config digest."""

    source, contract, _trace_payload = _make_source_contract()
    assert source.row_index == 3
    assert contract["trace_identity"]["episode_id"] == source.episode_id
    assert contract["fields"]["map_id"]["status"] == "available"
    assert len(contract["canonical_config"]["sha256"]) == 64
    retrieval_keys = {
        artifact["retrieval_key"]
        for artifact in contract["source_artifacts"]
        if artifact["role"] in {"episodes_jsonl", "arm_manifest", "run_summary", "preflight"}
    }
    assert retrieval_keys == {
        "synthetic/doorway_ppo/episodes.jsonl",
        "synthetic/doorway_ppo/manifest.json",
        "synthetic/doorway_ppo/run_summary.yaml",
        "synthetic/doorway_ppo/validate_config.json",
    }


def test_source_contract_rejects_6412_package_digest_drift(tmp_path: Path) -> None:
    """Reject a package whose immutable completion record is not schema-valid."""

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


def test_source_loader_accepts_mapping_identity(tmp_path: Path) -> None:
    """Verify the strict loader accepts the packet's compact mapping identity."""

    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    source = load_verified_real_reexport_row_source(
        package_root=package,
        external_arm_root=roots[identities[0].arm],
        expected_identity=_mapping_identity(identities[0]),
        expected_package_sha256=package_sha,
    )
    assert source.episode_id == identities[0].episode_id
    assert source.episodes_retrieval_key == "synthetic/doorway_ppo/episodes.jsonl"
    assert source.manifest_retrieval_key == "synthetic/doorway_ppo/manifest.json"
    assert source.run_summary_retrieval_key == "synthetic/doorway_ppo/run_summary.yaml"
    assert source.preflight_retrieval_key == "synthetic/doorway_ppo/validate_config.json"


def test_source_loader_rejects_missing_package_root(tmp_path: Path) -> None:
    """Fail closed when the approved source package directory is absent."""

    identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    with pytest.raises(RealReexportBindingError, match="package is unavailable"):
        load_verified_real_reexport_row_source(
            package_root=tmp_path / "missing-package",
            external_arm_root=tmp_path,
            expected_identity=identity,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("n_admitted", 87, "count or evidence"),
        ("sha256sums_sha256", "0" * 64, "identity mismatch"),
    ),
)
def test_source_loader_rejects_package_complete_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """Reject count or SHA drift in the immutable package completion record."""

    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    complete = json.loads((package / "package_complete.json").read_text(encoding="utf-8"))
    complete[field] = value
    _write_json(package / "package_complete.json", complete)
    with pytest.raises(RealReexportBindingError, match=message):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=roots[identities[0].arm],
            expected_identity=identities[0],
            expected_package_sha256=package_sha,
        )


def test_source_loader_rejects_missing_package_sums(tmp_path: Path) -> None:
    """Reject a package with no integrity ledger before reading source rows."""

    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    (package / "SHA256SUMS").unlink()
    with pytest.raises(RealReexportBindingError, match="SHA256SUMS is unavailable"):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=roots[identities[0].arm],
            expected_identity=identities[0],
            expected_package_sha256=package_sha,
        )


def test_source_loader_accepts_blank_and_comment_checksum_lines(tmp_path: Path) -> None:
    """Accept standard SHA256SUMS comments and blank separators."""

    package, roots, identities, _package_sha = _make_synthetic_packet_inputs(tmp_path)
    sums_path = package / "SHA256SUMS"
    sums = b"# generated by the immutable package exporter\n\n" + sums_path.read_bytes()
    sums_path.write_bytes(sums)
    package_sha = _sha256_bytes(sums_path.read_bytes())
    complete = json.loads((package / "package_complete.json").read_text(encoding="utf-8"))
    complete["sha256sums_sha256"] = package_sha
    _write_json(package / "package_complete.json", complete)
    source = load_verified_real_reexport_row_source(
        package_root=package,
        external_arm_root=roots[identities[0].arm],
        expected_identity=identities[0],
        expected_package_sha256=package_sha,
    )
    assert source.n_rows > 0


def test_source_loader_rejects_unsafe_package_sums_path(tmp_path: Path) -> None:
    """Reject traversal paths in the package integrity ledger."""

    package, roots, identities, _package_sha = _make_synthetic_packet_inputs(tmp_path)
    sums = ("0" * 64 + "  ../outside\n").encode()
    (package / "SHA256SUMS").write_bytes(sums)
    complete = json.loads((package / "package_complete.json").read_text(encoding="utf-8"))
    complete["sha256sums_sha256"] = _sha256_bytes(sums)
    _write_json(package / "package_complete.json", complete)
    with pytest.raises(RealReexportBindingError, match="unsafe issue #6412 package path"):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=roots[identities[0].arm],
            expected_identity=identities[0],
            expected_package_sha256=_sha256_bytes(sums),
        )


def test_source_loader_rejects_missing_required_package_sum_entry(tmp_path: Path) -> None:
    """Reject an integrity ledger that omits a required package artifact."""

    package, roots, identities, _package_sha = _make_synthetic_packet_inputs(tmp_path)
    names = ("README.md", "package_manifest.json", "source_pointer.json")
    sums = "".join(
        f"{_sha256_bytes((package / name).read_bytes())}  {name}\n" for name in names
    ).encode()
    (package / "SHA256SUMS").write_bytes(sums)
    complete = json.loads((package / "package_complete.json").read_text(encoding="utf-8"))
    complete["sha256sums_sha256"] = _sha256_bytes(sums)
    _write_json(package / "package_complete.json", complete)
    with pytest.raises(RealReexportBindingError, match="omits required"):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=roots[identities[0].arm],
            expected_identity=identities[0],
            expected_package_sha256=_sha256_bytes(sums),
        )


@pytest.mark.parametrize(
    ("artifact", "field", "message"),
    (
        ("package_manifest.json", "schema_version", "package_manifest identity"),
        ("mapping_receipt.json", "schema_version", "mapping receipt schema"),
    ),
)
def test_source_loader_rejects_package_record_schema_drift(
    tmp_path: Path, artifact: str, field: str, message: str
) -> None:
    """Reject schema-version drift in a package record even after re-pinning sums."""

    package, roots, identities, _package_sha = _make_synthetic_packet_inputs(tmp_path)
    payload = json.loads((package / artifact).read_text(encoding="utf-8"))
    payload[field] = "wrong.schema.v1"
    _write_json(package / artifact, payload)
    package_sha = _refresh_synthetic_package_sums(package)
    with pytest.raises(RealReexportBindingError, match=message):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=roots[identities[0].arm],
            expected_identity=identities[0],
            expected_package_sha256=package_sha,
        )


def test_source_loader_rejects_non_integer_identity_seed(tmp_path: Path) -> None:
    """Reject mapping identities whose seed is not an exact integer."""

    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    with pytest.raises(RealReexportBindingError, match="seed identity"):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=roots[identities[0].arm],
            expected_identity=_mapping_identity(identities[0], seed="113"),
            expected_package_sha256=package_sha,
        )


def test_source_loader_rejects_ambiguous_optional_owner(tmp_path: Path) -> None:
    """Reject an optional geometry owner with an unapproved schema version."""

    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    root = roots[identities[0].arm]
    _write_json(
        root / "process_trace_geometry_registry.json",
        {"schema_version": "wrong.schema.v1"},
    )
    with pytest.raises(RealReexportBindingError, match="route geometry owner schema"):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=root,
            expected_identity=identities[0],
            expected_package_sha256=package_sha,
        )


def test_source_loader_rejects_ambiguous_result_provenance_sidecars(tmp_path: Path) -> None:
    """Reject multiple external provenance sidecars instead of guessing an owner."""

    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    root = roots[identities[0].arm]
    _write_json(root / "nested-a" / "episodes.jsonl.provenance.json", {})
    _write_json(root / "nested-b" / "episodes.jsonl.provenance.json", {})
    with pytest.raises(RealReexportBindingError, match="sidecar is ambiguous"):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=root,
            expected_identity=identities[0],
            expected_package_sha256=package_sha,
        )


def test_source_loader_rejects_invalid_result_provenance_sidecar(tmp_path: Path) -> None:
    """Reject a result-provenance sidecar that fails its manifest contract."""

    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    _write_json(roots[identities[0].arm] / "episodes.jsonl.provenance.json", {"invalid": True})
    with pytest.raises(RealReexportBindingError, match="sidecar rows are unavailable"):
        load_verified_real_reexport_row_source(
            package_root=package,
            external_arm_root=roots[identities[0].arm],
            expected_identity=identities[0],
            expected_package_sha256=package_sha,
        )


def test_source_contract_rejects_external_episodes_digest_drift(tmp_path: Path) -> None:
    """Reject external episode bytes that differ from the pinned package digest."""

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
    """Reject drift in any required external arm artifact."""

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
    """Verify a selected row links to exactly one validated provenance record."""

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
    """Reject a provenance sidecar row that does not identify the selected episode."""

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
    """Verify static configuration identity excludes realization-specific fields."""

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


def test_static_config_marks_missing_metric_settings_unavailable() -> None:
    """Do not claim metric settings are available when no mapping was supplied."""

    config = build_static_run_config(
        scenario_id="classic_doorway_medium",
        scenario_matrix_sha256="a" * 64,
        scenario_definition_sha256="b" * 64,
        map_id="maps/svg_maps/classic_doorway.svg",
        map_sha256="c" * 64,
        horizon_steps=600,
        time_step_s=0.1,
        planner_id="ppo",
        planner_config_id="config",
        planner_config_sha256="d" * 64,
        source_algorithm_config_hash="short",
    )
    assert config["metric_affecting_settings"] == {
        "status": "unavailable",
        "reason_code": "metric_affecting_settings_unavailable",
        "required_authority": "verified source settings",
    }


def test_static_config_digest_changes_with_planner_config() -> None:
    """Verify planner configuration changes alter the static configuration digest."""

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
    """Verify actor ordering does not change the canonical initial-state digest."""

    identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    first = build_initial_state_record(_trace(identity, actor_order=("l1", "l2")))
    second = copy.deepcopy(first)
    second["actors"].reverse()
    assert initial_state_digest(first) == initial_state_digest(second)


def test_initial_state_digest_requires_authoritative_actor_ids() -> None:
    """Reject generated actor identities before constructing initial-state evidence."""

    identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    with pytest.raises(ValueError, match="actor identity"):
        build_initial_state_record(_trace(identity, actor_order=("ped-0",)))


def test_initial_state_rejects_malformed_actor_state() -> None:
    """Reject actor rows missing position or velocity instead of leaking a KeyError."""

    identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    trace = _trace(identity)
    trace["frames"][0]["pedestrians"][0].pop("position")
    with pytest.raises(ValueError, match="actor position or velocity"):
        build_initial_state_record(trace)


def test_metadata_projection_changes_only_approved_json_paths() -> None:
    """Verify strict enrichment adds metadata without changing canonical state."""

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
    """Keep terminal outcome unavailable when the raw row lacks typed outcome data."""

    identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    source = _source(identity, outcome=False)
    contract = build_issue_6814_trace_source_contract(
        source, execution_repository=Path(__file__).parents[2]
    )
    assert contract["trace_projection"]["terminal_outcome"]["status"] == "unavailable"


def test_release_outcome_cannot_create_terminal_trace_evidence() -> None:
    """Prevent a release summary from manufacturing raw terminal trace authority."""

    identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    source = _source(identity, outcome=False)
    source = replace(source, run_summary={"outcome": "route_complete"})
    contract = build_issue_6814_trace_source_contract(
        source, execution_repository=Path(__file__).parents[2]
    )
    assert contract["trace_projection"]["terminal_outcome"]["status"] == "unavailable"


def test_source_contract_rejects_conflicting_horizon_values() -> None:
    """Reject conflicting authoritative horizon values across source artifacts."""

    source, _contract, _trace_payload = _make_source_contract()
    conflicted = replace(source, preflight={"horizon": 599, "time_step_s": 0.1})
    with pytest.raises(ValueError, match="horizon"):
        build_issue_6814_trace_source_contract(
            conflicted, execution_repository=Path(__file__).parents[2]
        )


def test_source_contract_rejects_conflicting_time_step_values() -> None:
    """Reject conflicting authoritative time-step values across source artifacts."""

    source, _contract, _trace_payload = _make_source_contract()
    conflicted = replace(source, preflight={"horizon": 600, "time_step_s": 0.2})
    with pytest.raises(ValueError, match="time-step"):
        build_issue_6814_trace_source_contract(
            conflicted, execution_repository=Path(__file__).parents[2]
        )


def test_source_contract_resolves_partial_settings_by_field_authority() -> None:
    """Retain independent horizon and time-step authorities from different artifacts."""

    source, _contract, _trace_payload = _make_source_contract()
    raw = copy.deepcopy(source.raw_row)
    raw["simulator_settings"] = {}
    raw["scenario_params"] = {}
    partial = replace(
        source,
        raw_row=raw,
        result_provenance_row=None,
        result_provenance_sha256=None,
        preflight={"horizon": 600},
        run_summary={"time_step_s": 0.1},
    )
    contract = build_issue_6814_trace_source_contract(
        partial, execution_repository=Path(__file__).parents[2]
    )
    assert contract["fields"]["horizon_steps"]["status"] == "available"
    assert contract["fields"]["time_step_s"]["status"] == "available"
    assert contract["fields"]["horizon_steps"]["authorities"][0]["json_pointer"] == "/"
    assert contract["fields"]["time_step_s"]["authorities"][0]["json_pointer"] == "/"


def test_source_contract_rejects_missing_planner_config_snapshot() -> None:
    """Keep planner configuration unavailable when the raw snapshot is missing."""

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
    assert contract["fields"]["horizon_steps"]["status"] == "available"
    assert contract["fields"]["time_step_s"]["status"] == "available"


def test_source_contract_does_not_use_current_scenario_horizon_as_run_authority() -> None:
    """Do not infer run settings from the current scenario definition."""

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
    """Reject fallback planner identities in provenance-bound strict exports."""

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


def _strict_run_config(**overrides: Any) -> dict[str, Any]:
    """Return a valid strict run configuration with selected fields overridden."""

    config: dict[str, Any] = {
        "map_id": "map",
        "horizon": 600,
        "time_step_s": 0.1,
        "config_digest": "a" * 64,
    }
    config.update(overrides)
    return config


def test_strict_projection_rejects_non_object_source_and_invalid_seed() -> None:
    """Reject missing source objects and non-integer source seeds before projection."""

    missing_source = _trace(
        _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    )
    missing_source["source"] = None
    invalid_seed = _trace(
        _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    )
    invalid_seed["source"]["seed"] = True

    with pytest.raises(SimulationTraceNormalizationError, match="source identity"):
        apply_strict_metadata_projection(missing_source, run_config=_strict_run_config())
    with pytest.raises(SimulationTraceNormalizationError, match="seed"):
        apply_strict_metadata_projection(invalid_seed, run_config=_strict_run_config())


def test_strict_identity_validation_skips_malformed_container_shapes() -> None:
    """Traverse malformed frame and pedestrian containers without inventing identities."""

    trace = _trace(_base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483"))
    trace["frames"] = [
        "not-a-frame",
        {"pedestrians": "not-a-list", "planner": {}},
        {"pedestrians": [None, {}], "planner": {}},
    ]

    with pytest.raises(SimulationTraceNormalizationError, match="missing"):
        apply_strict_metadata_projection(trace, run_config=_strict_run_config())


def test_strict_export_rejects_numeric_generated_pedestrian_identity() -> None:
    """Reject numeric actor identifiers generated from pedestrian ordering."""

    trace = _trace(_base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483"))
    trace["frames"][0]["pedestrians"][0]["id"] = "0"
    with pytest.raises(SimulationTraceNormalizationError, match="generated actor"):
        apply_strict_metadata_projection(trace, run_config=_strict_run_config())


def test_strict_projection_returns_a_semantic_delta_receipt() -> None:
    """Return typed metadata and prove the canonical trace state stayed unchanged."""

    trace = _trace(_base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483"))
    original = copy.deepcopy(trace)
    enriched, receipt = apply_strict_metadata_projection(
        trace,
        run_config=_strict_run_config(),
        terminal_outcome={
            "collision_event": False,
            "timeout_event": False,
            "route_complete": True,
        },
    )

    assert trace == original
    assert all(
        frame["planner"]["run_config"] == _strict_run_config() for frame in enriched["frames"]
    )
    assert enriched["frames"][-1]["planner"]["outcome"]["route_complete"] is True
    assert receipt["before_projection_sha256"] == receipt["after_projection_sha256"]
    assert receipt["terminal_outcome_path"] == "/frames/1/planner/outcome"
    assert receipt["semantic_payload_unchanged"] is True


@pytest.mark.parametrize(
    ("run_config", "error_text"),
    (
        ({"map_id": "map", "horizon": 600, "time_step_s": 0.1}, "fields"),
        ({**_strict_run_config(), "extra": True}, "fields"),
        (_strict_run_config(map_id=""), "map_id"),
        (_strict_run_config(horizon=0), "horizon"),
        (_strict_run_config(time_step_s=True), "numeric"),
        (_strict_run_config(time_step_s=0.0), "positive"),
        (_strict_run_config(config_digest="not-a-sha"), "SHA-256"),
    ),
)
def test_strict_projection_rejects_invalid_run_config(
    run_config: dict[str, Any], error_text: str
) -> None:
    """Reject missing, extra, malformed, and non-finite strict run settings."""

    trace = _trace(_base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483"))
    with pytest.raises(SimulationTraceNormalizationError, match=error_text):
        apply_strict_metadata_projection(trace, run_config=run_config)


def test_strict_projection_rejects_non_object_metadata() -> None:
    """Reject list-shaped metadata before attempting mapping operations."""

    trace = _trace(_base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483"))
    with pytest.raises(SimulationTraceNormalizationError, match="run_config must be an object"):
        apply_strict_metadata_projection(trace, run_config=["not", "an", "object"])  # type: ignore[arg-type]
    with pytest.raises(
        SimulationTraceNormalizationError, match="terminal_outcome must be an object"
    ):
        apply_strict_metadata_projection(
            trace,
            run_config=_strict_run_config(),
            terminal_outcome=["not", "an", "object"],  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("terminal_outcome", "error_text"),
    (
        ({}, "exactly"),
        (
            {"collision_event": 1, "timeout_event": False, "route_complete": True},
            "booleans",
        ),
    ),
)
def test_strict_projection_rejects_invalid_terminal_outcome(
    terminal_outcome: dict[str, Any], error_text: str
) -> None:
    """Require exactly typed terminal outcome fields before writing the final frame."""

    trace = _trace(_base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483"))
    with pytest.raises(SimulationTraceNormalizationError, match=error_text):
        apply_strict_metadata_projection(
            trace,
            run_config=_strict_run_config(),
            terminal_outcome=terminal_outcome,
        )


@pytest.mark.parametrize(
    "frames",
    (
        [],
        ["not-a-frame"],
        [{"planner": None}],
    ),
)
def test_strict_projection_rejects_empty_or_malformed_frames(frames: list[Any]) -> None:
    """Reject empty traces and frames without a planner object before mutation."""

    trace = _trace(_base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483"))
    trace["frames"] = frames
    with pytest.raises(SimulationTraceNormalizationError, match="frame|planner"):
        apply_strict_metadata_projection(trace, run_config=_strict_run_config())


def test_strict_export_rejects_generated_pedestrian_identity() -> None:
    """Reject generated pedestrian identifiers in provenance-bound strict exports."""

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
    """Reject strict projection when canonical planner state already contains metadata."""

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


def test_pair_receipt_does_not_hash_missing_process_trace() -> None:
    """Keep missing process provenance null and typed unavailable in pair receipts."""

    left_identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 113, 3, "13483")
    right_identity = _base_identity("doorway_ppo", "ppo", "classic_doorway_medium", 114, 4, "13483")
    left_source = _source(left_identity)
    right_source = _source(right_identity)
    left_contract = build_issue_6814_trace_source_contract(
        left_source, execution_repository=Path(__file__).parents[2]
    )
    right_contract = build_issue_6814_trace_source_contract(
        right_source, execution_repository=Path(__file__).parents[2]
    )
    receipt = issue6814._pair_receipt(
        left_identity,
        right_identity,
        left_contract,
        right_contract,
        _trace(left_identity),
        _trace(right_identity),
        None,
        None,
        "matched_realization_pair",
    )
    assert receipt["sources"]["left"]["process_trace_sha256"] is None
    assert receipt["semantic_inputs"]["terminal_event"] == {
        "status": "unavailable",
        "reason_code": "process_trace_unavailable",
    }


def test_live_package_block_is_not_reported_as_proven_unsupported(tmp_path: Path) -> None:
    """Classify the live package mismatch as source integrity failure, not unsupported data."""

    output = tmp_path / "packet"
    with pytest.raises(Issue6814SourceIntegrityError, match="SHA256SUMS"):
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
    """Leave no published or staging packet after an external integrity failure."""

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
    """Verify a synthetic packet rebuild has stable content hashes and dispositions."""

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
    assert manifest["generation_time"] == "deterministic-not-a-clock"
    assert manifest["check_results"] == {
        "package_digest_ok": True,
        "row_contract_digest_ok": True,
        "artifact_integrity_ok": True,
        "deterministic_rebuild_ok": True,
    }
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
    assert doorway_receipt["renderer_admission"]["reason_codes"] == [
        "pair_compatibility_incompatible",
        "required_renderer_input_unavailable",
        "process_trace_unavailable",
        "run_config_contract_unavailable",
    ]
    assert double_receipt["renderer_admission"]["reason_codes"] == [
        "pair_compatibility_incompatible",
        "required_renderer_input_unavailable",
        "process_trace_unavailable",
        "run_config_contract_unavailable",
    ]
    assert manifest["source_package"]["arms"]["doorway_ppo"]["n_rows"] == sum(
        1
        for line in (roots["doorway_ppo"] / "episodes.jsonl").read_bytes().splitlines()
        if line.strip()
    )
    doorway_arm = manifest["source_package"]["arms"]["doorway_ppo"]
    assert doorway_arm["manifest_uri"] == "synthetic/doorway_ppo/manifest.json"
    assert doorway_arm["episodes_uri"] == "synthetic/doorway_ppo/episodes.jsonl"
    assert doorway_arm["run_summary_uri"] == "synthetic/doorway_ppo/run_summary.yaml"
    assert doorway_arm["preflight_uri"] == "synthetic/doorway_ppo/validate_config.json"
    assert (
        len(
            {
                doorway_arm["manifest_uri"],
                doorway_arm["episodes_uri"],
                doorway_arm["run_summary_uri"],
                doorway_arm["preflight_uri"],
            }
        )
        == 4
    )


def test_packet_manifest_schema_is_fail_closed_on_coverage_and_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject incomplete source indexes, mismatched receipt lists, and false support claims."""

    package, roots, identities, package_sha = _make_synthetic_packet_inputs(tmp_path)
    monkeypatch.setattr(issue6814, "SELECTED_TRACE_IDENTITIES", identities)
    manifest = build_issue_6814_trace_packet(
        package_root=package,
        arm_roots=roots,
        external_output_root=tmp_path / "packet",
        execution_repository=Path(__file__).parents[2],
        check_determinism=True,
        expected_package_sha256=package_sha,
    )

    missing_source = copy.deepcopy(manifest)
    missing_source["source_contracts"].pop()
    with pytest.raises(Issue6814Error):
        issue6814._schema_validate(missing_source, "issue_6814_packet_manifest.v1.json")

    mismatched_receipts = copy.deepcopy(manifest)
    mismatched_receipts["output_hashes"]["pair_receipts"][0]["pair_id"] = "wrong"
    with pytest.raises(Issue6814Error, match="indexes disagree"):
        issue6814._schema_validate(mismatched_receipts, "issue_6814_packet_manifest.v1.json")

    false_supported = copy.deepcopy(manifest)
    false_supported["disposition"] = "supported"
    false_supported["check_results"]["artifact_integrity_ok"] = False
    with pytest.raises(Issue6814Error):
        issue6814._schema_validate(false_supported, "issue_6814_packet_manifest.v1.json")

    missing_reexport_digest = copy.deepcopy(manifest)
    missing_reexport_digest["source_contracts"][0]["status"] = "available"
    with pytest.raises(Issue6814Error):
        issue6814._schema_validate(missing_reexport_digest, "issue_6814_packet_manifest.v1.json")
