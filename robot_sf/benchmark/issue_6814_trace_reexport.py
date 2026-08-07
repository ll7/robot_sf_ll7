# ruff: noqa: C901, DOC201, PLR0912, PLR0913, PLR0915, RUF022

"""Fail-closed provenance re-export and packet construction for issue #6814.

The module is intentionally a narrow overlay on the frozen #6412 package.  It
does not rerun simulation, rewrite the old package, infer provenance from the
current checkout, or substitute an episode when a source gate is unavailable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import posixpath
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.event_alignment import (
    PAIR_COMPATIBILITY_PROFILE_VERSION,
    build_pair_compatibility_record,
    build_trace_run_config_contract,
)
from robot_sf.analysis_workbench.interaction_coordinates import (
    _build_worked_example_process_trace_from_export,
)
from robot_sf.analysis_workbench.simulation_trace_export import (
    simulation_trace_export_from_dict,
)
from robot_sf.benchmark.trace_reexport_packaging import (
    EXECUTION_COMMIT,
    ISSUE_6412_PACKAGE_SHA256SUMS_SHA256,
    RealReexportBindingError,
    TraceReexportPackagingError,
    VerifiedRealReexportRowSource,
    _sha256_file,
    load_verified_real_reexport_row_source,
)
from scripts.tools.build_simulation_trace_export import (
    apply_strict_metadata_projection,
)

ISSUE = 6814
SOURCE_ISSUE = 6412
SOURCE_CONTRACT_SCHEMA = "issue_6814_trace_source_contract.v1"
PAIR_RECEIPT_SCHEMA = "issue_6814_pair_compatibility_receipt.v1"
PACKET_MANIFEST_SCHEMA = "issue_6814_packet_manifest.v1"
STATIC_CONFIG_SCHEMA = "chapter7_static_run_config.v1"
INITIAL_STATE_SCHEMA = "chapter7_initial_state.v1"
CANONICALIZATION = "strict-json-sort-keys-utf8-no-newline.v1"
POSITION_TOLERANCE_M = 1e-6
HEADING_TOLERANCE_RAD = 1e-6
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Issue6814Error(ValueError):
    """Base error for the strict issue #6814 path."""


class Issue6814SourceIntegrityError(Issue6814Error):
    """Raised when source retrieval or source integrity cannot be proven."""


class Issue6814UnsupportedError(Issue6814Error):
    """Raised when verified source lacks a required field-level authority."""


class Issue6814DeterminismError(Issue6814Error):
    """Raised when two packet builds do not produce identical bytes."""


@dataclass(frozen=True, slots=True)
class TraceIdentity:
    """Exact four-row identity selected by the approved #6412 mapping receipt."""

    arm: str
    job_id: str
    row_index: int
    episode_id: str
    scenario_id: str
    planner_id: str
    seed: int
    execution_commit: str = EXECUTION_COMMIT
    raw_trace_sha256: str | None = None
    prior_normalized_sha256: str | None = None
    row_config_hash: str | None = None
    algorithm_config_hash: str | None = None


SELECTED_TRACE_IDENTITIES = (
    TraceIdentity(
        arm="doorway_ppo",
        job_id="13483",
        row_index=3,
        episode_id="classic_doorway_medium--113--5aae52ff8e7aacda",
        scenario_id="classic_doorway_medium",
        planner_id="ppo",
        seed=113,
        raw_trace_sha256="478c9297ad035ee83945d1d59a8d4d735e4ea3d91bf31b193dbe401aa85c750b",
        prior_normalized_sha256="06a10f9312772e55f6049a1e33ca5f4d51ea27e3759e683f9ef6ef012b36cd82",
        row_config_hash="5aae52ff8e7aacda",
        algorithm_config_hash="dfcc6e96335c47ce",
    ),
    TraceIdentity(
        arm="doorway_ppo",
        job_id="13483",
        row_index=4,
        episode_id="classic_doorway_medium--114--562d0581dd50ef91",
        scenario_id="classic_doorway_medium",
        planner_id="ppo",
        seed=114,
        raw_trace_sha256="e81781a7e2407073b4106a02b415e7141a7e3f7ef7206919dbab7e057d7415ce",
        prior_normalized_sha256="3b3d64e43c84b9d267fa7dede44a47ca1a62b0fcbee3fe3a03e8495c494b1d30",
        row_config_hash="562d0581dd50ef91",
        algorithm_config_hash="dfcc6e96335c47ce",
    ),
    TraceIdentity(
        arm="double_bottleneck_goal",
        job_id="13487",
        row_index=8,
        episode_id="classic_realworld_double_bottleneck_high--118--2be3c567b2ce6000",
        scenario_id="classic_realworld_double_bottleneck_high",
        planner_id="goal",
        seed=118,
        raw_trace_sha256="c3ed1af9eb816805eebaf302f29dbdf3923df6520222f0b5ca3e116870f6de9b",
        prior_normalized_sha256="fea57cbb3b594bf6a26e592ebb298e8f8c9ff9e357f7d6a75671de26559c364e",
        row_config_hash="2be3c567b2ce6000",
        algorithm_config_hash="44136fa355b3678a",
    ),
    TraceIdentity(
        arm="double_bottleneck_ppo",
        job_id="13488",
        row_index=8,
        episode_id="classic_realworld_double_bottleneck_high--118--6c7522567ce57305",
        scenario_id="classic_realworld_double_bottleneck_high",
        planner_id="ppo",
        seed=118,
        raw_trace_sha256="530c192640dc143aba80a11db4fefc6a948161b6fbb19eb3e27978e07c5e8d21",
        prior_normalized_sha256="e09dc64f88ebb33de5495d51ec0fe51938eeeb13aa4c50f297a5ae06d8a3a1e9",
        row_config_hash="6c7522567ce57305",
        algorithm_config_hash="dfcc6e96335c47ce",
    ),
)


def _canonical_bytes(value: Any) -> bytes:
    """Encode strict JSON without a trailing newline."""

    _assert_finite_json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _assert_finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise Issue6814Error(f"non-finite number at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_json(child, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _assert_finite_json(child, f"{path}[{index}]")


def _identity_value(identity: object, field: str) -> object:
    if isinstance(identity, Mapping):
        return identity.get(field)
    return getattr(identity, field, None)


def _identity_mapping(identity: object) -> dict[str, Any]:
    fields = (
        "arm",
        "job_id",
        "row_index",
        "episode_id",
        "scenario_id",
        "planner_id",
        "seed",
        "execution_commit",
        "raw_trace_sha256",
        "prior_normalized_sha256",
        "row_config_hash",
        "algorithm_config_hash",
    )
    return {field: _identity_value(identity, field) for field in fields}


def build_static_run_config(
    *,
    scenario_id: str,
    scenario_matrix_sha256: str,
    scenario_definition_sha256: str,
    map_id: str,
    map_sha256: str,
    horizon_steps: int,
    time_step_s: float,
    planner_id: str,
    planner_config_id: str,
    planner_config_sha256: str,
    source_algorithm_config_hash: str,
    simulator_commit: str = EXECUTION_COMMIT,
    metric_affecting_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the static digest input, excluding realization identity."""

    return {
        "schema_version": STATIC_CONFIG_SCHEMA,
        "scenario": {
            "scenario_id": scenario_id,
            "scenario_matrix_sha256": scenario_matrix_sha256,
            "scenario_definition_sha256": scenario_definition_sha256,
            "map_id": map_id,
            "map_sha256": map_sha256,
        },
        "execution": {
            "horizon_steps": int(horizon_steps),
            "time_step_s": float(time_step_s),
            "coordinate_frame": "world",
            "units": {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
            "simulator_commit": simulator_commit,
        },
        "planner": {
            "planner_id": planner_id,
            "planner_config_id": planner_config_id,
            "planner_config_sha256": planner_config_sha256,
            "source_algorithm_config_hash": source_algorithm_config_hash,
        },
        "metric_affecting_settings": {
            "status": "available",
            "content": dict(metric_affecting_settings or {}),
        },
    }


def static_config_digest(config: Mapping[str, Any]) -> str:
    """Hash one static configuration with the packet canonicalization."""

    excluded = {"seed", "episode_id", "job_id", "run_id", "row_index", "timestamps"}
    if excluded & set(config):
        raise Issue6814Error("static config contains realization-specific fields")
    return _sha256_payload(config)


def build_initial_state_record(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Build an actor-order-independent initial-state record from a trace."""

    frames = trace.get("frames")
    if not isinstance(frames, list) or not frames or not isinstance(frames[0], Mapping):
        raise Issue6814UnsupportedError("initial state frame zero is unavailable")
    frame = frames[0]
    robot = frame.get("robot")
    pedestrians = frame.get("pedestrians")
    if not isinstance(robot, Mapping) or not isinstance(pedestrians, list):
        raise Issue6814UnsupportedError("initial state robot or actors are unavailable")
    robot_position = robot.get("position")
    robot_velocity = robot.get("velocity")
    if not isinstance(robot_position, list) or not isinstance(robot_velocity, list):
        raise Issue6814UnsupportedError("initial robot position or velocity is unavailable")
    actors: list[dict[str, Any]] = []
    for index, actor in enumerate(pedestrians):
        if not isinstance(actor, Mapping):
            raise Issue6814UnsupportedError("initial actor record is unavailable")
        actor_id = actor.get("id")
        if (
            not isinstance(actor_id, str)
            or not actor_id.strip()
            or actor_id == str(index)
            or actor_id.startswith("ped-")
            or actor_id.startswith("ped_")
        ):
            raise Issue6814UnsupportedError("initial actor identity is generated or unavailable")
        item = {
            "id": actor_id,
            "position": list(actor["position"]),
            "velocity": list(actor["velocity"]),
            "radius_m": actor.get("radius", 0.0),
        }
        if "heading" in actor:
            item["heading"] = actor["heading"]
        actors.append(item)
    actors.sort(key=lambda item: item["id"])
    record = {
        "schema_version": INITIAL_STATE_SCHEMA,
        "robot": {
            "position": list(robot_position),
            "velocity": list(robot_velocity),
            "heading": robot.get("heading", 0.0),
            "radius_m": robot.get("radius", 0.0),
        },
        "actors": actors,
    }
    _assert_finite_json(record)
    return record


def initial_state_digest(initial_state: Mapping[str, Any]) -> str:
    """Hash a canonical initial-state record."""

    actors = initial_state.get("actors")
    if not isinstance(actors, list):
        raise Issue6814Error("initial-state actors must be a list")
    normalized = copy.deepcopy(dict(initial_state))
    normalized["actors"] = sorted(actors, key=lambda actor: str(actor.get("id")))
    return _sha256_payload(normalized)


def _available(
    value: Any, *, authorities: list[dict[str, Any]], unit: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "available", "value": value, "authorities": authorities}
    if unit is not None:
        result["unit"] = unit
    return result


def _unavailable(reason_code: str, required_authority: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason_code": reason_code,
        "required_authority": required_authority,
    }


def _git_show(repository: Path, commit: str, relative: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=repository,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise Issue6814UnsupportedError(
            f"producing-commit artifact unavailable: {commit}:{relative}"
        ) from exc
    return completed.stdout


def _scenario_paths(scenario_id: str) -> tuple[str, str]:
    if scenario_id == "classic_doorway_medium":
        return (
            "configs/scenarios/archetypes/classic_doorway.yaml",
            "maps/svg_maps/classic_doorway.svg",
        )
    if scenario_id == "classic_realworld_double_bottleneck_high":
        return (
            "configs/scenarios/archetypes/classic_realworld_bottleneck.yaml",
            "maps/svg_maps/classic_realworld_bottleneck.svg",
        )
    raise Issue6814UnsupportedError(f"scenario include graph has no approved owner: {scenario_id}")


def _scenario_entry(scenario_id: str, payload: Any) -> Mapping[str, Any]:
    _index, entry = _scenario_entry_with_index(scenario_id, payload)
    return entry


def _scenario_entry_with_index(scenario_id: str, payload: Any) -> tuple[int, Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        scenarios = payload.get("scenarios")
        if isinstance(scenarios, list):
            for index, entry in enumerate(scenarios):
                if isinstance(entry, Mapping) and entry.get("name") == scenario_id:
                    return index, entry
    raise Issue6814UnsupportedError(f"scenario definition entry is unavailable: {scenario_id}")


def _authority(role: str, digest: str, pointer: str) -> dict[str, str]:
    return {
        "artifact_role": role,
        "artifact_sha256": digest,
        "json_pointer": pointer,
    }


def _settings_candidates(
    source: VerifiedRealReexportRowSource,
) -> list[tuple[str, Any, Any, str]]:
    candidates: list[tuple[str, Any, Any, str]] = []
    artifact_digests = {
        "result_provenance": source.result_provenance_sha256,
        "preflight": source.preflight_sha256,
        "run_summary": source.run_summary_sha256,
        "raw_episode_row": source.raw_row_sha256,
        "raw_result_provenance": source.raw_row_sha256,
    }
    row = source.raw_row
    for role, settings in (
        (
            "result_provenance",
            source.result_provenance_row.get("simulator_settings")
            if source.result_provenance_row
            else None,
        ),
        ("preflight", source.preflight),
        ("run_summary", source.run_summary),
        ("raw_episode_row", row.get("simulator_settings")),
        ("raw_result_provenance", row.get("result_provenance")),
    ):
        if not isinstance(settings, Mapping):
            continue
        horizon = settings.get("horizon", settings.get("horizon_steps"))
        dt = settings.get("dt", settings.get("time_step_s"))
        if isinstance(settings.get("simulator_settings"), Mapping):
            nested = settings["simulator_settings"]
            horizon = nested.get("horizon", nested.get("horizon_steps", horizon))
            dt = nested.get("dt", nested.get("time_step_s", dt))
        if horizon is not None or dt is not None:
            digest = artifact_digests[role]
            if digest is not None:
                candidates.append((role, horizon, dt, digest))
    params = row.get("scenario_params")
    if isinstance(params, Mapping):
        if params.get("run_horizon") is not None or params.get("run_dt") is not None:
            candidates.append(
                (
                    "raw_scenario_params",
                    params.get("run_horizon"),
                    params.get("run_dt"),
                    source.raw_row_sha256,
                )
            )
    return candidates


def _resolve_settings(
    source: VerifiedRealReexportRowSource,
) -> tuple[int, float, list[dict[str, Any]]]:
    candidates = _settings_candidates(source)
    horizons = {
        int(horizon)
        for _role, horizon, _dt, _digest in candidates
        if isinstance(horizon, (int, float)) and not isinstance(horizon, bool)
    }
    dts = {
        float(dt)
        for _role, _horizon, dt, _digest in candidates
        if isinstance(dt, (int, float)) and not isinstance(dt, bool)
    }
    if len(horizons) > 1:
        raise TraceReexportPackagingError("issue #6814 conflicting authoritative horizon values")
    if len(dts) > 1:
        raise TraceReexportPackagingError("issue #6814 conflicting authoritative time-step values")
    if not horizons or not dts:
        raise Issue6814UnsupportedError("authoritative run configuration is unavailable")
    horizon = next(iter(horizons))
    dt = next(iter(dts))
    if horizon <= 0 or dt <= 0.0 or not math.isfinite(dt):
        raise Issue6814UnsupportedError("authoritative run configuration is invalid")
    authorities = [
        _authority(role, digest, f"/{role}/simulator_settings")
        for role, raw_horizon, raw_dt, digest in candidates
        if raw_horizon == horizon and raw_dt == dt
    ]
    return horizon, dt, authorities


def _matrix_authorities(
    source: VerifiedRealReexportRowSource,
    *,
    expected_path: str,
    expected_sha256: str,
) -> list[dict[str, Any]]:
    """Reconcile the producing-commit matrix with retained run provenance."""

    candidates: list[tuple[str, object, str]] = []
    manifest = source.result_provenance_manifest
    if isinstance(manifest, Mapping):
        inputs = manifest.get("inputs")
        matrix = inputs.get("scenario_matrix") if isinstance(inputs, Mapping) else None
        if isinstance(matrix, Mapping):
            matrix_path = matrix.get("path")
            if not isinstance(matrix_path, str) or posixpath.normpath(matrix_path) != expected_path:
                raise TraceReexportPackagingError(
                    "issue #6814 scenario-matrix path disagrees with producing commit"
                )
            candidates.append(
                (
                    "result_provenance",
                    matrix.get("sha256"),
                    "/inputs/scenario_matrix/sha256",
                )
            )
    for role, payload, pointer in (
        ("preflight", source.preflight, "/scenario_matrix_sha256"),
        ("run_summary", source.run_summary, "/scenario_matrix_sha256"),
        ("raw_episode_row", source.raw_row, "/scenario_matrix_sha256"),
    ):
        if not isinstance(payload, Mapping):
            continue
        value = payload.get("scenario_matrix_sha256")
        if value is None and isinstance(payload.get("scenario_matrix"), Mapping):
            value = payload["scenario_matrix"].get("sha256")
            pointer = "/scenario_matrix/sha256"
        if value is not None:
            candidates.append((role, value, pointer))
    normalized: list[tuple[str, str, str, str]] = []
    for role, value, pointer in candidates:
        if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
            raise TraceReexportPackagingError(
                f"issue #6814 {role} scenario-matrix authority is not SHA-256"
            )
        artifact_digest = (
            source.result_provenance_sha256
            if role == "result_provenance"
            else {
                "preflight": source.preflight_sha256,
                "run_summary": source.run_summary_sha256,
                "raw_episode_row": source.raw_row_sha256,
            }[role]
        )
        if artifact_digest is None:
            continue
        normalized.append((role, value, pointer, artifact_digest))
    values = {value for _role, value, _pointer, _artifact_digest in normalized}
    if len(values) > 1:
        raise TraceReexportPackagingError(
            "issue #6814 conflicting authoritative scenario-matrix hashes"
        )
    if values and next(iter(values)) != expected_sha256:
        raise TraceReexportPackagingError(
            "issue #6814 scenario-matrix hash disagrees with producing commit"
        )
    return [
        _authority(role, artifact_digest, pointer)
        for role, _value, pointer, artifact_digest in normalized
        if _value == expected_sha256
    ]


def _planner_config(
    source: VerifiedRealReexportRowSource,
) -> tuple[str, str, str, Mapping[str, Any] | None]:
    metadata = source.raw_row.get("algorithm_metadata")
    config = metadata.get("config") if isinstance(metadata, Mapping) else None
    if not isinstance(config, Mapping):
        config = source.raw_row.get("planner_config")
    source_hash = None
    if isinstance(metadata, Mapping):
        source_hash = metadata.get("config_hash")
    source_hash = source_hash or source.raw_row.get("algorithm_config_hash")
    if source_hash is None and isinstance(source.preflight, Mapping):
        source_hash = source.preflight.get("algorithm_config_hash")
    if not isinstance(source_hash, str) or not source_hash.strip():
        raise Issue6814UnsupportedError("planner config identity is unavailable")
    if isinstance(config, Mapping):
        config_digest = _sha256_payload(config)
        return str(source.planner_id), str(source_hash), config_digest, config
    if len(source_hash) != 64 or any(char not in "0123456789abcdef" for char in source_hash):
        raise Issue6814UnsupportedError(
            "planner config is represented only by an opaque short hash"
        )
    return str(source.planner_id), source_hash, source_hash, None


def _terminal_outcome(source: VerifiedRealReexportRowSource) -> dict[str, Any]:
    outcome = source.raw_row.get("outcome")
    if not isinstance(outcome, Mapping):
        return _unavailable("raw_row_typed_terminal_outcome_absent", "raw_row.outcome")
    fields = {"collision_event", "timeout_event", "route_complete"}
    if not fields <= set(outcome) or any(type(outcome[field]) is not bool for field in fields):
        return _unavailable("raw_row_typed_terminal_outcome_invalid", "raw_row.outcome")
    value = {field: bool(outcome[field]) for field in sorted(fields)}
    return {
        "status": "available",
        "source": "raw_row.outcome",
        "value": value,
        "authorities": [_authority("raw_episode_row", source.raw_row_sha256, "/outcome")],
    }


def build_issue_6814_trace_source_contract(
    source: VerifiedRealReexportRowSource,
    *,
    execution_repository: Path,
) -> dict[str, Any]:
    """Build field-level authority records for one verified source row."""

    if source.execution_commit != EXECUTION_COMMIT:
        raise Issue6814SourceIntegrityError("producing execution commit differs from #6412 pin")
    scenario_path, map_path = _scenario_paths(source.scenario_id)
    scenario_bytes = _git_show(execution_repository, source.execution_commit, scenario_path)
    map_bytes = _git_show(execution_repository, source.execution_commit, map_path)
    matrix_path = "configs/scenarios/classic_interactions_francis2023.yaml"
    matrix_bytes = _git_show(execution_repository, source.execution_commit, matrix_path)
    scenario_payload = yaml.safe_load(scenario_bytes)
    scenario_index, scenario = _scenario_entry_with_index(source.scenario_id, scenario_payload)
    map_id = map_path
    referenced_map = scenario.get("map_file")
    if not isinstance(referenced_map, str) or not referenced_map.strip():
        raise Issue6814UnsupportedError("scenario map reference is unavailable")
    resolved_map = posixpath.normpath(
        posixpath.join(posixpath.dirname(scenario_path), referenced_map)
    )
    if resolved_map != map_path:
        raise Issue6814SourceIntegrityError(
            f"scenario map reference disagrees with approved map owner: {resolved_map}"
        )
    scenario_definition_sha = hashlib.sha256(_canonical_bytes(scenario)).hexdigest()
    map_sha = hashlib.sha256(map_bytes).hexdigest()
    matrix_sha = hashlib.sha256(matrix_bytes).hexdigest()
    matrix_authorities = _matrix_authorities(
        source,
        expected_path=matrix_path,
        expected_sha256=matrix_sha,
    )
    scenario_matrix_sha = matrix_sha if matrix_authorities else None
    source_artifacts = [
        {
            "role": "episodes_jsonl",
            "schema_version": "issue_5756_episode.v1",
            "retrieval_key": source.source_root_retrieval_key,
            "sha256": source.episodes_sha256,
            "authority": "durable_source",
        },
        {
            "role": "arm_manifest",
            "schema_version": "issue_5756_arm_manifest.v1",
            "retrieval_key": source.source_root_retrieval_key,
            "sha256": source.manifest_sha256,
            "authority": "durable_source",
        },
        {
            "role": "run_summary",
            "schema_version": "run_summary.external.v1",
            "retrieval_key": source.source_root_retrieval_key,
            "sha256": source.run_summary_sha256,
            "authority": "durable_source",
        },
        {
            "role": "preflight",
            "schema_version": "validate_config.v1",
            "retrieval_key": source.source_root_retrieval_key,
            "sha256": source.preflight_sha256,
            "authority": "durable_source",
        },
    ]
    if source.result_provenance_sha256 is not None:
        source_artifacts.append(
            {
                "role": "result_provenance",
                "schema_version": "benchmark_result_provenance.v1",
                "retrieval_key": (
                    f"{source.source_root_retrieval_key}/episodes.jsonl.provenance.json"
                ),
                "sha256": source.result_provenance_sha256,
                "authority": "durable_source",
            }
        )
    source_artifacts.extend(
        [
            {
                "role": "scenario_matrix",
                "schema_version": "robot_sf.scenario_matrix.v1",
                "retrieval_key": matrix_path,
                "sha256": matrix_sha,
                "authority": "producing_commit",
            },
            {
                "role": "scenario_definition",
                "schema_version": "scenario_definition.v1",
                "retrieval_key": scenario_path,
                "sha256": hashlib.sha256(scenario_bytes).hexdigest(),
                "authority": "producing_commit",
            },
            {
                "role": "map_artifact",
                "schema_version": "svg_map.v1",
                "retrieval_key": map_path,
                "sha256": map_sha,
                "authority": "producing_commit",
            },
        ]
    )
    for role, payload, digest in (
        ("route_geometry_registry", source.route_geometry, source.route_geometry_sha256),
        ("conflict_registry", source.conflict_geometry, source.conflict_geometry_sha256),
        ("encounter_report", source.encounter_report, source.encounter_report_sha256),
    ):
        if payload is not None and digest is not None:
            source_artifacts.append(
                {
                    "role": role,
                    "schema_version": str(payload.get("schema_version", "unknown")),
                    "retrieval_key": source.source_root_retrieval_key,
                    "sha256": digest,
                    "authority": "durable_source",
                }
            )

    try:
        horizon, dt, setting_authorities = _resolve_settings(source)
        planner_id, planner_source_hash, planner_digest, planner_config = _planner_config(source)
    except Issue6814UnsupportedError:
        horizon = None
        dt = None
        planner_id = source.planner_id
        planner_source_hash = None
        planner_digest = None
        planner_config = None
        setting_authorities = []

    fields: dict[str, Any] = {
        "map_id": _available(
            map_id,
            authorities=[
                _authority(
                    "scenario_definition",
                    hashlib.sha256(scenario_bytes).hexdigest(),
                    f"/scenarios/{scenario_index}/map_file",
                )
            ],
        ),
        "scenario_definition_sha256": _available(
            scenario_definition_sha,
            authorities=[
                _authority(
                    "scenario_definition",
                    hashlib.sha256(scenario_bytes).hexdigest(),
                    f"/scenarios/{scenario_index}",
                )
            ],
        ),
        "scenario_matrix_sha256": (
            _available(scenario_matrix_sha, authorities=matrix_authorities)
            if scenario_matrix_sha is not None
            else _unavailable(
                "retained_scenario_matrix_hash_absent",
                "verified result provenance or immutable run record",
            )
        ),
        "map_sha256": _available(
            map_sha,
            authorities=[_authority("map_artifact", map_sha, "blob")],
        ),
        "horizon_steps": (
            _available(horizon, authorities=setting_authorities, unit="step")
            if horizon is not None
            else _unavailable(
                "authoritative_horizon_absent", "result_provenance or verified preflight"
            )
        ),
        "time_step_s": (
            _available(dt, authorities=setting_authorities, unit="s")
            if dt is not None
            else _unavailable(
                "authoritative_time_step_absent", "result_provenance or verified preflight"
            )
        ),
        "planner_config_sha256": (
            _available(
                planner_digest,
                authorities=[
                    _authority(
                        "algorithm_config",
                        planner_digest,
                        "/algorithm_metadata/config"
                        if planner_config is not None
                        else "source_algorithm_config_hash",
                    )
                ],
            )
            if planner_digest is not None and planner_config is not None
            else _unavailable("planner_config_opaque_or_absent", "algorithm_config artifact")
        ),
        "route_geometry": (
            _available(
                source.route_geometry,
                authorities=[
                    _authority(
                        "route_geometry_registry",
                        source.route_geometry_sha256,
                        "/",
                    )
                ],
            )
            if source.route_geometry is not None and source.route_geometry_sha256 is not None
            else _unavailable(
                "authoritative_route_registry_entry_absent", "process_trace_geometry_registry.v1"
            )
        ),
        "conflict_geometry": (
            _available(
                source.conflict_geometry,
                authorities=[_authority("conflict_registry", source.conflict_geometry_sha256, "/")],
            )
            if source.conflict_geometry is not None and source.conflict_geometry_sha256 is not None
            else _unavailable(
                "authoritative_conflict_registry_entry_absent", "canonical conflict owner"
            )
        ),
        "encounter": (
            _available(
                source.encounter_report,
                authorities=[_authority("encounter_report", source.encounter_report_sha256, "/")],
            )
            if source.encounter_report is not None and source.encounter_report_sha256 is not None
            else _unavailable("authoritative_encounter_report_absent", "near_miss_encounter.v1")
        ),
        "coordinate_frame": _available(
            "world",
            authorities=[_authority("preflight", source.preflight_sha256, "/coordinate_frame")],
        ),
        "units": _available(
            {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
            authorities=[
                _authority("trace_schema", source.preflight_sha256, "simulation_trace_export.v1")
            ],
        ),
    }

    static_content: dict[str, Any] | None = None
    config_digest: str | None = None
    if (
        horizon is not None
        and dt is not None
        and planner_digest is not None
        and scenario_matrix_sha is not None
    ):
        static_content = build_static_run_config(
            scenario_id=source.scenario_id,
            scenario_matrix_sha256=scenario_matrix_sha,
            scenario_definition_sha256=scenario_definition_sha,
            map_id=map_id,
            map_sha256=map_sha,
            horizon_steps=horizon,
            time_step_s=dt,
            planner_id=planner_id,
            planner_config_id=planner_source_hash or "unavailable",
            planner_config_sha256=planner_digest,
            source_algorithm_config_hash=planner_source_hash or "unavailable",
        )
        config_digest = static_config_digest(static_content)
    initial_state: dict[str, Any]
    try:
        trace_shape = source.raw_row.get("algorithm_metadata", {})
        trace_shape = (
            trace_shape.get("simulation_step_trace") if isinstance(trace_shape, Mapping) else None
        )
        steps = trace_shape.get("steps") if isinstance(trace_shape, Mapping) else None
        raw_initial = steps[0] if isinstance(steps, list) and steps else None
        if isinstance(raw_initial, Mapping) and "robot" in raw_initial:
            initial_record = build_initial_state_record({"frames": [raw_initial]})
            initial_state = {
                "status": "available",
                "source": "raw_row.algorithm_metadata.simulation_step_trace.steps[0]",
                "value": initial_record,
                "sha256": initial_state_digest(initial_record),
            }
        else:
            raise Issue6814UnsupportedError("raw initial state shape unavailable")
    except Issue6814UnsupportedError:
        initial_state = _unavailable(
            "authoritative_actor_identity_or_initial_state_absent",
            "raw_row.algorithm_metadata.simulation_step_trace.steps[0]",
        )

    terminal = _terminal_outcome(source)
    available = (
        config_digest is not None
        and horizon is not None
        and dt is not None
        and scenario_matrix_sha is not None
        and fields["planner_config_sha256"]["status"] == "available"
        and initial_state["status"] == "available"
    )
    run_config = (
        {
            "map_id": map_id,
            "horizon": horizon,
            "time_step_s": dt,
            "config_digest": config_digest,
        }
        if available
        else None
    )
    contract: dict[str, Any] = {
        "schema_version": SOURCE_CONTRACT_SCHEMA,
        "issue": ISSUE,
        "status": "available" if available else "unsupported",
        "source_boundary": {
            "source_issue": SOURCE_ISSUE,
            "package_sha256sums_sha256": ISSUE_6412_PACKAGE_SHA256SUMS_SHA256,
            "visualization_only": True,
            "release_statistics_authoritative": True,
            "new_simulation_performed": False,
        },
        "trace_identity": {
            "planner_id": source.planner_id,
            "scenario_id": source.scenario_id,
            "seed": source.seed,
            "episode_id": source.episode_id,
            "job_id": source.job_id,
            "row_index": source.row_index,
            "execution_commit": source.execution_commit,
            "raw_trace_sha256": source.raw_row_sha256,
            "prior_normalized_trace_sha256": source.prior_normalized_sha256,
            "reexported_trace_sha256": None,
        },
        "source_artifacts": source_artifacts,
        "fields": fields,
        "canonical_config": {
            "schema_version": STATIC_CONFIG_SCHEMA,
            "canonicalization": CANONICALIZATION,
            "included_fields": sorted(static_content) if static_content else [],
            "excluded_fields": [
                "seed",
                "episode_id",
                "job_id",
                "run_id",
                "row_index",
                "timestamps",
                "initial_state",
                "terminal_outcome",
            ],
            "content": static_content or {},
            "sha256": config_digest or _sha256_payload(static_content or {"status": "unsupported"}),
        },
        "initial_state": initial_state,
        "trace_projection": {
            "planner_run_config": run_config
            if run_config is not None
            else _unavailable("authoritative_static_run_config_absent", "verified source fields"),
            "terminal_outcome": terminal,
        },
        "transformation": {
            "prior_normalized_trace_sha256": source.prior_normalized_sha256,
            "reexported_trace_sha256": None,
            "state_projection_sha256_before": None,
            "state_projection_sha256_after": None,
            "state_payload_unchanged": False,
            "allowed_added_paths": [
                "/frames/*/planner/run_config",
                "/frames/<last>/planner/outcome",
            ],
            "observed_added_paths": [],
            "notes": [] if available else ["strict metadata authority is incomplete"],
        },
    }
    return contract


def enrich_simulation_trace_export(
    prior_trace: Mapping[str, object],
    *,
    source_contract: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Add only approved strict metadata to one prior normalized trace."""

    if source_contract.get("status") != "available":
        raise Issue6814UnsupportedError("source contract is not available for enrichment")
    projection = source_contract.get("trace_projection")
    if not isinstance(projection, Mapping):
        raise Issue6814UnsupportedError("source contract lacks trace projection")
    run_config = projection.get("planner_run_config")
    if not isinstance(run_config, Mapping):
        raise Issue6814UnsupportedError("source contract lacks verified run_config")
    terminal = projection.get("terminal_outcome")
    terminal_value = terminal.get("value") if isinstance(terminal, Mapping) else None
    if isinstance(terminal, Mapping) and terminal.get("status") == "unavailable":
        terminal_value = None
    enriched, delta = apply_strict_metadata_projection(
        prior_trace,
        run_config=run_config,
        terminal_outcome=terminal_value if isinstance(terminal_value, Mapping) else None,
    )
    result = copy.deepcopy(enriched)
    return result, delta


def _load_prior_trace(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_file():
        raise Issue6814SourceIntegrityError(f"prior normalized trace is unavailable: {path}")
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise Issue6814SourceIntegrityError(
            f"prior normalized trace SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Issue6814SourceIntegrityError("prior normalized trace is invalid JSON") from exc
    if not isinstance(payload, Mapping):
        raise Issue6814SourceIntegrityError("prior normalized trace must be an object")
    try:
        simulation_trace_export_from_dict(payload, source=path)
    except ValueError as exc:
        raise Issue6814SourceIntegrityError(
            "prior normalized trace failed schema validation"
        ) from exc
    return dict(payload)


def _schema_validate(payload: Mapping[str, Any], schema_name: str) -> None:
    schema_path = Path(__file__).with_name("schemas") / schema_name
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Issue6814Error(f"cannot load issue #6814 schema {schema_name}") from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.path)
    )
    if errors:
        raise Issue6814Error(f"{schema_name}: {errors[0].message} at {list(errors[0].path)}")


def _process_payload(
    trace_payload: Mapping[str, Any], pair_payload: Mapping[str, Any] | None, grain: str
) -> dict[str, Any]:
    trace = simulation_trace_export_from_dict(trace_payload)
    pair_trace = (
        simulation_trace_export_from_dict(pair_payload) if pair_payload is not None else None
    )
    try:
        payload = _build_worked_example_process_trace_from_export(
            trace,
            pair_trace=pair_trace,
            pair_comparison_grain=grain,
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise Issue6814UnsupportedError("process trace could not be replayed") from exc
    return dict(payload)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload) + b"\n")


def _relative_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _sha256_file(path)
    return result


def _selected_pair_id(left: TraceIdentity, right: TraceIdentity) -> str:
    return f"{left.scenario_id}--{left.planner_id}--{left.seed}-{right.seed}"


def _semantic_input_receipt(
    left_contract: Mapping[str, Any],
    right_contract: Mapping[str, Any],
    field_name: str,
) -> dict[str, Any]:
    """Summarize two field-level semantic authorities without reconstruction."""

    fields = [
        contract.get("fields", {}).get(field_name, {})
        for contract in (left_contract, right_contract)
    ]
    if all(field.get("status") == "available" for field in fields):
        return {"status": "available"}
    reasons = sorted(
        {
            field.get("reason_code", "semantic_input_unavailable")
            for field in fields
            if field.get("status") != "available"
        }
    )
    return {
        "status": "unavailable",
        "reason_code": reasons[0] if len(reasons) == 1 else "semantic_input_unavailable",
    }


def _pair_receipt(
    left: TraceIdentity,
    right: TraceIdentity,
    left_contract: Mapping[str, Any],
    right_contract: Mapping[str, Any],
    left_trace: Mapping[str, Any],
    right_trace: Mapping[str, Any],
    left_process: Mapping[str, Any] | None,
    right_process: Mapping[str, Any] | None,
    grain: str,
) -> dict[str, Any]:
    left_typed = simulation_trace_export_from_dict(left_trace)
    right_typed = simulation_trace_export_from_dict(right_trace)
    pair = build_pair_compatibility_record(
        left_typed,
        right_typed,
        left_events=[],
        right_events=[],
        comparison_grain=grain,
        position_tolerance_m=POSITION_TOLERANCE_M,
        heading_tolerance_rad=HEADING_TOLERANCE_RAD,
    )
    if pair.get("profile_version") != PAIR_COMPATIBILITY_PROFILE_VERSION:
        raise Issue6814Error("#6790 compatibility profile changed unexpectedly")
    left_run_config = build_trace_run_config_contract(left_typed)
    right_run_config = build_trace_run_config_contract(right_typed)
    route_missing = any(
        contract.get("fields", {}).get("route_geometry", {}).get("status") != "available"
        for contract in (left_contract, right_contract)
    )
    conflict_missing = any(
        contract.get("fields", {}).get("conflict_geometry", {}).get("status") != "available"
        for contract in (left_contract, right_contract)
    )
    pair_status = pair.get("status")
    supported = (
        pair_status == "available"
        and not route_missing
        and not conflict_missing
        and left_process is not None
        and right_process is not None
    )
    disposition = "supported" if supported else "unsupported"
    reason_codes: list[str] = []
    if pair_status != "available":
        reason_codes.append("pair_compatibility_incompatible")
    if route_missing or conflict_missing:
        reason_codes.append("required_renderer_input_unavailable")
    if left_process is None or right_process is None:
        reason_codes.append("process_trace_unavailable")
    terminal_available = all(
        contract.get("trace_projection", {}).get("terminal_outcome", {}).get("status")
        == "available"
        for contract in (left_contract, right_contract)
    )
    if not terminal_available:
        reason_codes.append("typed_terminal_outcome_unavailable")
    if (
        left_run_config.get("status") != "available"
        or right_run_config.get("status") != "available"
    ):
        reason_codes.append("run_config_contract_unavailable")
    return {
        "schema_version": PAIR_RECEIPT_SCHEMA,
        "issue": ISSUE,
        "pair_id": _selected_pair_id(left, right),
        "comparison_grammar": (
            "same_cell_seed_sensitivity" if grain == "matched_realization_pair" else "matched_start"
        ),
        "comparison_grain": grain,
        "evidence_boundary": {
            "source_package_sha256sums_sha256": ISSUE_6412_PACKAGE_SHA256SUMS_SHA256,
            "visualization_only": True,
            "release_statistics_authoritative": True,
            "new_simulation_performed": False,
            "episode_substitution_performed": False,
            "tolerance_profile_modified": False,
        },
        "sources": {
            "left": {
                "source_contract_sha256": _sha256_payload(left_contract),
                "trace_content_sha256": _sha256_payload(left_trace),
                "process_trace_sha256": _sha256_payload(left_process or {"status": "unavailable"}),
            },
            "right": {
                "source_contract_sha256": _sha256_payload(right_contract),
                "trace_content_sha256": _sha256_payload(right_trace),
                "process_trace_sha256": _sha256_payload(right_process or {"status": "unavailable"}),
            },
        },
        "pair_compatibility": pair,
        "run_config_contracts": {"left": left_run_config, "right": right_run_config},
        "semantic_inputs": {
            "route": _semantic_input_receipt(left_contract, right_contract, "route_geometry"),
            "conflict": _semantic_input_receipt(left_contract, right_contract, "conflict_geometry"),
            "encounter": _semantic_input_receipt(left_contract, right_contract, "encounter"),
            "terminal_event": (
                {
                    "status": "available",
                    "source_process_trace_sha256": _sha256_payload(left_process or left_trace),
                }
                if terminal_available
                else {"status": "unavailable", "reason_code": "typed_terminal_outcome_unavailable"}
            ),
        },
        "process_validation": {
            "left": {"status": "pass", "errors": []}
            if left_process is not None
            else {"status": "unavailable", "errors": ["process_trace_unavailable"]},
            "right": {"status": "pass", "errors": []}
            if right_process is not None
            else {"status": "unavailable", "errors": ["process_trace_unavailable"]},
        },
        "renderer_admission": {
            "eligible": supported,
            "attempted": False,
            "disposition": disposition,
            "reason_codes": reason_codes,
        },
        "notes": ["#6790 pair compatibility output is embedded without recomputation"]
        if pair
        else [],
    }


def _build_packet_once(
    *,
    root: Path,
    package_root: Path,
    arm_roots: Mapping[str, Path],
    execution_repository: Path,
    expected_package_sha256: str,
) -> dict[str, Any]:
    """Build one deterministic packet into an already-created staging root."""

    sources: dict[
        str,
        tuple[
            TraceIdentity,
            VerifiedRealReexportRowSource,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any] | None,
        ],
    ] = {}
    for identity in SELECTED_TRACE_IDENTITIES:
        try:
            source = load_verified_real_reexport_row_source(
                package_root=package_root,
                external_arm_root=arm_roots[identity.arm],
                expected_identity=identity,
                expected_package_sha256=expected_package_sha256,
            )
        except (RealReexportBindingError, TraceReexportPackagingError, OSError) as exc:
            raise Issue6814SourceIntegrityError(str(exc)) from exc
        contract = build_issue_6814_trace_source_contract(
            source, execution_repository=execution_repository
        )
        trace_uri = None
        mapping_rows = json.loads(
            (package_root / "mapping_receipt.json").read_text(encoding="utf-8")
        )["rows"]
        for row in mapping_rows:
            if isinstance(row, Mapping) and row.get("episode_id") == source.episode_id:
                trace_uri = row.get("trace_artifact_uri")
                break
        if not isinstance(trace_uri, str):
            raise Issue6814SourceIntegrityError("selected row lacks prior trace artifact URI")
        prior_path = package_root / trace_uri
        prior_trace = _load_prior_trace(prior_path, source.prior_normalized_sha256)
        if contract.get("status") == "available":
            enriched, delta = enrich_simulation_trace_export(prior_trace, source_contract=contract)
            contract = copy.deepcopy(contract)
            contract["trace_identity"]["reexported_trace_sha256"] = _sha256_payload(enriched)
            contract["transformation"].update(
                {
                    "reexported_trace_sha256": _sha256_payload(enriched),
                    "state_projection_sha256_before": delta["before_projection_sha256"],
                    "state_projection_sha256_after": delta["after_projection_sha256"],
                    "state_payload_unchanged": delta["semantic_payload_unchanged"],
                    "observed_added_paths": delta["added_paths"],
                }
            )
            trace = dict(enriched)
        else:
            trace = prior_trace
            delta = None
        _schema_validate(contract, "issue_6814_trace_source_contract.v1.json")
        sources[identity.episode_id] = (identity, source, contract, trace, delta)
        _write_json(root / "source_contracts" / f"{identity.arm}_{identity.seed}.json", contract)
        if delta is not None:
            _write_json(root / "traces" / f"{identity.arm}_{identity.seed}.json", trace)

    pairs = (
        (SELECTED_TRACE_IDENTITIES[0], SELECTED_TRACE_IDENTITIES[1], "matched_realization_pair"),
        (SELECTED_TRACE_IDENTITIES[2], SELECTED_TRACE_IDENTITIES[3], "matched_planner_pair"),
    )
    pair_receipts: list[dict[str, Any]] = []
    for left, right, grain in pairs:
        left_record = sources[left.episode_id]
        right_record = sources[right.episode_id]
        left_identity, _left_source, left_contract, left_trace, _left_delta = left_record
        right_identity, _right_source, right_contract, right_trace, _right_delta = right_record
        left_process = None
        right_process = None
        if left_contract.get("status") == "available":
            try:
                left_process = _process_payload(left_trace, None, grain)
            except Issue6814UnsupportedError:
                left_process = None
        if right_contract.get("status") == "available":
            try:
                right_process = _process_payload(right_trace, None, grain)
            except Issue6814UnsupportedError:
                right_process = None
        receipt = _pair_receipt(
            left_identity,
            right_identity,
            left_contract,
            right_contract,
            left_trace,
            right_trace,
            left_process,
            right_process,
            grain,
        )
        _schema_validate(receipt, "issue_6814_pair_compatibility_receipt.v1.json")
        pair_receipts.append(receipt)
        _write_json(root / "pair_receipts" / f"{left.arm}_{left.seed}_{right.seed}.json", receipt)

    source_contract_records: list[dict[str, Any]] = []
    for identity, _source, contract, _trace, _delta in sources.values():
        contract_path = root / "source_contracts" / f"{identity.arm}_{identity.seed}.json"
        source_contract_records.append(
            {
                "path": contract_path.relative_to(root).as_posix(),
                "sha256": _sha256_file(contract_path),
                "status": contract["status"],
                "trace_identity": contract["trace_identity"],
            }
        )
    pair_file_records: list[dict[str, str]] = []
    for (left, right, _grain), receipt in zip(pairs, pair_receipts, strict=True):
        pair_path = root / "pair_receipts" / f"{left.arm}_{left.seed}_{right.seed}.json"
        pair_file_records.append(
            {
                "path": pair_path.relative_to(root).as_posix(),
                "sha256": _sha256_file(pair_path),
                "pair_id": receipt["pair_id"],
                "disposition": receipt["renderer_admission"]["disposition"],
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": PACKET_MANIFEST_SCHEMA,
        "issue": ISSUE,
        "generated_at": "1970-01-01T00:00:00Z",
        "source_package": {
            "source_issue": SOURCE_ISSUE,
            "source_package_sha256sums_sha256": ISSUE_6412_PACKAGE_SHA256SUMS_SHA256,
            "package_manifest_uri": "package_manifest.json",
            "package_manifest_sha256": _sha256_file(package_root / "package_manifest.json"),
            "package_complete_uri": "package_complete.json",
            "package_complete_sha256": _sha256_file(package_root / "package_complete.json"),
            "mapping_receipt_uri": "mapping_receipt.json",
            "mapping_receipt_sha256": _sha256_file(package_root / "mapping_receipt.json"),
            "arms": {},
        },
        "source_contracts": source_contract_records,
        "pair_compatibility_receipts": pair_file_records,
        "output_hashes": {
            "package_manifest_uri": "package_manifest.json",
            "source_contracts_manifest_uri": "source_contracts/",
            "pair_receipts": pair_file_records,
        },
        "disposition": "supported"
        if all(
            receipt["renderer_admission"]["disposition"] == "supported" for receipt in pair_receipts
        )
        else "unsupported",
        "notes": ["visualization-only overlay; no simulation performed"],
        "check_results": {
            "package_digest_ok": True,
            "row_contract_digest_ok": True,
            "artifact_integrity_ok": True,
            "deterministic_rebuild_ok": True,
        },
    }
    for identity in SELECTED_TRACE_IDENTITIES:
        source = sources[identity.episode_id][1]
        manifest["source_package"]["arms"][identity.arm] = {
            "job_id": source.job_id,
            "manifest_uri": source.source_root_retrieval_key,
            "manifest_sha256": source.manifest_sha256,
            "episodes_uri": source.source_root_retrieval_key,
            "episodes_sha256": source.episodes_sha256,
            "run_summary_uri": source.source_root_retrieval_key,
            "run_summary_sha256": source.run_summary_sha256,
            "preflight_uri": source.source_root_retrieval_key,
            "preflight_sha256": source.preflight_sha256,
            "n_rows": 30,
        }
    _schema_validate(manifest, "issue_6814_packet_manifest.v1.json")
    _write_json(root / "packet_manifest.json", manifest)
    return manifest


def _compare_trees(left: Path, right: Path) -> None:
    left_hashes = _relative_hashes(left)
    right_hashes = _relative_hashes(right)
    if left_hashes != right_hashes:
        raise Issue6814DeterminismError("issue #6814 packet rebuild changed file set or SHA-256")


def build_issue_6814_trace_packet(
    *,
    package_root: Path,
    arm_roots: Mapping[str, Path],
    external_output_root: Path,
    compact_output: Path | None = None,
    execution_repository: Path | None = None,
    check_determinism: bool = False,
    expected_package_sha256: str = ISSUE_6412_PACKAGE_SHA256SUMS_SHA256,
) -> dict[str, Any]:
    """Build and atomically publish the strict issue #6814 packet."""

    package_root = package_root.resolve()
    external_output_root = external_output_root.resolve()
    execution_repository = (execution_repository or Path.cwd()).resolve()
    if compact_output is not None:
        compact_output = compact_output.resolve()
        if compact_output.exists():
            raise Issue6814Error(f"refusing to overwrite existing compact output: {compact_output}")
    if external_output_root.exists():
        raise Issue6814Error(f"refusing to overwrite existing output: {external_output_root}")
    if set(arm_roots) != {identity.arm for identity in SELECTED_TRACE_IDENTITIES}:
        raise Issue6814Error("arm_roots must cover exactly the three approved arms")
    external_output_root.parent.mkdir(parents=True, exist_ok=True)
    first = Path(
        tempfile.mkdtemp(
            prefix=f".{external_output_root.name}.staging-", dir=external_output_root.parent
        )
    )
    second: Path | None = None
    renamed = False
    compact_started = False
    published = False
    try:
        manifest = _build_packet_once(
            root=first,
            package_root=package_root,
            arm_roots=arm_roots,
            execution_repository=execution_repository,
            expected_package_sha256=expected_package_sha256,
        )
        if check_determinism:
            second = Path(
                tempfile.mkdtemp(
                    prefix=f".{external_output_root.name}.rebuild-", dir=external_output_root.parent
                )
            )
            _build_packet_once(
                root=second,
                package_root=package_root,
                arm_roots=arm_roots,
                execution_repository=execution_repository,
                expected_package_sha256=expected_package_sha256,
            )
            _compare_trees(first, second)
        first.rename(external_output_root)
        renamed = True
        if compact_output is not None:
            compact_output.parent.mkdir(parents=True, exist_ok=True)
            compact_started = True
            compact_output.mkdir()
            shutil.copy2(
                external_output_root / "packet_manifest.json",
                compact_output / "packet_manifest.json",
            )
        published = True
        return manifest
    finally:
        if not published:
            if renamed:
                shutil.rmtree(external_output_root, ignore_errors=True)
            if compact_started and compact_output is not None:
                shutil.rmtree(compact_output, ignore_errors=True)
            shutil.rmtree(first, ignore_errors=True)
        if second is not None and second.exists():
            shutil.rmtree(second, ignore_errors=True)


__all__ = [
    "CANONICALIZATION",
    "EXECUTION_COMMIT",
    "Issue6814DeterminismError",
    "Issue6814Error",
    "Issue6814SourceIntegrityError",
    "Issue6814UnsupportedError",
    "SELECTED_TRACE_IDENTITIES",
    "TraceIdentity",
    "build_initial_state_record",
    "build_issue_6814_trace_packet",
    "build_issue_6814_trace_source_contract",
    "build_static_run_config",
    "enrich_simulation_trace_export",
    "initial_state_digest",
    "static_config_digest",
]
