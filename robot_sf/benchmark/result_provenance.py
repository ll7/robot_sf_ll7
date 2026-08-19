"""Result-level provenance manifest for benchmark runs.

This module emits a structured JSON provenance manifest alongside
``episodes.jsonl`` that links every emitted row to config identity,
scenario ID, seed, repo commit, simulator settings, raw artifact paths,
and post-processing steps.

Schema version: ``benchmark_result_provenance.v1``. Strengthened input binding is
opted into explicitly with ``input_binding_schema_version``.
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import sys
import uuid
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numba
import numpy as np

from robot_sf._execution_context import build_execution_context, execution_context_digest
from robot_sf._numerical_thread_env import THREAD_ENV_VARS as _THREAD_ENV_VARS
from robot_sf.benchmark.utils import _config_hash, _git_hash_fallback

if TYPE_CHECKING:
    from collections.abc import Sequence
from robot_sf.errors import RobotSfError

SCHEMA_VERSION = "benchmark_result_provenance.v1"
INPUT_BINDING_SCHEMA_VERSION = "benchmark_result_provenance.input_binding.v2"
ROW_SCHEMA_VERSION = "benchmark_row_provenance.v1"

# Fields whose absence triggers a validation error.
_REQUIRED_TOP_LEVEL = ("schema_version", "run", "inputs", "campaign_identity", "completeness")
_REQUIRED_RUN = ("run_id", "repo_commit", "runner")
_REQUIRED_CAMPAIGN = (
    "scenario_matrix_hash",
    "input_bundle_sha256",
    "algorithm",
    "total_jobs",
    "written",
)
_STRENGTHENED_REQUIRED_TOP_LEVEL = _REQUIRED_TOP_LEVEL + ("input_binding_schema_version",)
_REQUIRED_ROW = ("episode_id", "scenario_id", "seed", "config_hash", "repo_commit")
_INPUT_ROLES = ("schema_path", "scenario_matrix", "algo_config")
_SHA256_HEX_CHARS = frozenset("0123456789abcdefABCDEF")


class ProvenanceValidationError(RobotSfError, ValueError):
    """Raised when a provenance manifest fails validation."""


class ProvenanceRequiredFieldError(ProvenanceValidationError):
    """Raised when a required field is missing or empty in a provenance manifest."""


class ProvenanceArtifactError(ProvenanceValidationError):
    """Raised when a required artifact is missing its SHA256."""


class ProvenanceRowLinkError(ProvenanceValidationError):
    """Raised when a row does not properly link to its raw artifact."""


def _require(
    condition: bool,
    message: str,
) -> None:
    """Fail-closed guard for required provenance fields."""
    if not condition:
        raise ProvenanceRequiredFieldError(message)


def _sha256_of_file(path: str | Path) -> str | None:
    """Return hex SHA-256 of a file, or ``None`` if the file cannot be read."""
    try:
        return sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, FileNotFoundError):
        return None


def _is_valid_sha256_hex(value: Any) -> bool:
    """Return whether ``value`` is a full SHA-256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _SHA256_HEX_CHARS for char in value)
    )


def _scenario_matrix_entry(
    scenario_path: Path,
    *,
    has_inline_scenarios: bool,
) -> dict[str, Any]:
    """Build the scenario input entry.

    A concrete scenario matrix/definition file is byte-bound when present. The
    only supported non-file case is an inline/generated scenario input, recorded
    as ``not_applicable`` with no path or digest. A directory or missing path is
    ``missing`` unless the caller supplied in-memory scenarios explicitly.

    Returns:
        Scenario matrix input entry.
    """
    if scenario_path.is_file():
        return {
            "path": str(scenario_path),
            "sha256": _sha256_of_file(scenario_path),
            "artifact_status": "available",
        }
    if has_inline_scenarios and not scenario_path.exists():
        return {
            "path": None,
            "sha256": None,
            "artifact_status": "not_applicable",
            "reason": "inline_or_generated_scenarios",
        }
    return {
        "path": str(scenario_path),
        "sha256": None,
        "artifact_status": "missing",
    }


def _canonical_input_bundle_sha256(
    *,
    inputs: Mapping[str, Any],
    algo: str,
    protocol_version: str,
    suite_key: str,
) -> str:
    """Return the byte-bound digest for benchmark inputs and suite identity."""
    roles: list[dict[str, Any]] = []
    for role in _INPUT_ROLES:
        raw_entry = inputs.get(role, {})
        entry = raw_entry if isinstance(raw_entry, Mapping) else {}
        status = entry.get("artifact_status")
        digest = (
            str(entry.get("sha256")).lower() if _is_valid_sha256_hex(entry.get("sha256")) else None
        )
        roles.append(
            {
                "role": role,
                "artifact_status": status,
                "sha256": digest if status == "available" else None,
            }
        )
    payload = {
        "algorithm": str(algo),
        "input_roles": roles,
        "protocol_version": str(protocol_version),
        "suite_key": str(suite_key),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _thread_env_snapshot() -> dict[str, str | None]:
    """Capture the numerical-thread environment used by BLAS/OpenMP.

    Returns:
        Mapping of thread-control env var name to its current value (or
        ``None`` when unset). These are the knobs that pinnable thread
        determinism for trace re-executions across nodes.
    """
    return {name: os.environ.get(name) for name in _THREAD_ENV_VARS}


def _cpu_model() -> str:
    """Best-effort CPU model string for the executing host.

    Returns:
        CPU model string, or ``"Unknown CPU"`` when not determinable.
    """
    try:
        with Path("/proc/cpuinfo").open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if line.lower().startswith("model name") and ":" in line:
                    model = line.split(":", 1)[1].strip()
                    if model:
                        return model
    except OSError:
        pass
    return platform.processor() or "Unknown CPU"


def build_execution_context_provenance() -> dict[str, Any]:
    """Capture the execution-context provenance of the current run.

    The benchmark path is per-context deterministic but diverges across
    execution contexts (CPU/BLAS/threading) at the ulp level, chaotically
    amplified in contact-rich scenarios. Recording hostname, CPU model, and
    thread-env alongside the commit makes cross-context trajectory comparisons
    detectable after the fact (see issue #5817 / #5816).

    Returns:
        Execution-context provenance dict.
    """
    context = build_execution_context(
        numpy_version=np.__version__,
        numba_version=str(numba.__version__),
    )
    return {
        "hostname": platform.node(),
        **context,
        "execution_context_sha256": execution_context_digest(context),
    }


def build_simulator_settings_provenance(
    *,
    horizon: int | None,
    dt: float | None,
    record_forces: bool,
    active_observation_mode: str | None,
    active_observation_level: str | None,
    noise_hash: str | None = None,
    tracking_precision_hash: str | None = None,
) -> dict[str, Any]:
    """Build simulator_settings block for a provenance manifest.

    Returns:
        Simulator settings dict.
    """
    settings: dict[str, Any] = {
        "horizon": horizon,
        "dt": dt,
    }
    if record_forces is not None:
        settings["record_forces"] = bool(record_forces)
    if active_observation_mode is not None:
        settings["observation_mode"] = str(active_observation_mode)
    if active_observation_level is not None:
        settings["observation_level"] = str(active_observation_level)
    if noise_hash is not None:
        settings["observation_noise_hash"] = str(noise_hash)
    if tracking_precision_hash is not None:
        settings["tracking_precision_hash"] = str(tracking_precision_hash)
    return settings


def build_row_result_provenance(  # noqa: PLR0913
    *,
    episode_id: str,
    scenario_id: str,
    seed: int,
    config_hash: str,
    repo_commit: str,
    raw_artifact_path: str,
    jsonl_line: int,
    dt: float | None,
    horizon: int | None,
    record_forces: bool,
    active_observation_mode: str | None,
    active_observation_level: str | None,
    noise_hash: str | None = None,
    tracking_precision_hash: str | None = None,
    postprocessing_steps: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build the provenance block for one benchmark row.

    Returns:
        Row provenance dict.
    """
    row: dict[str, Any] = {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": int(seed),
        "config_hash": config_hash,
        "repo_commit": repo_commit,
        "raw_artifact": str(raw_artifact_path),
        "jsonl_line": int(jsonl_line),
        "simulator_settings": build_simulator_settings_provenance(
            horizon=horizon,
            dt=dt,
            record_forces=record_forces,
            active_observation_mode=active_observation_mode,
            active_observation_level=active_observation_level,
            noise_hash=noise_hash,
            tracking_precision_hash=tracking_precision_hash,
        ),
        "postprocessing": (
            list(postprocessing_steps)
            if postprocessing_steps is not None
            else [
                {"step": "compute_all_metrics", "status": "completed"},
                {"step": "post_process_metrics", "status": "completed"},
            ]
        ),
    }
    return row


def _artifact_entry(
    *,
    kind: str,
    path: str | Path,
    artifact_status: str = "available",
) -> dict[str, Any]:
    """Build a single artifact entry with SHA256 when available.

    Returns:
        Artifact entry dict.
    """
    entry: dict[str, Any] = {
        "kind": kind,
        "path": str(path),
        "sha256": _sha256_of_file(path) if artifact_status == "available" else None,
        "artifact_status": artifact_status,
    }
    return entry


def _algo_config_entry(algo_config_path: str | Path | None) -> dict[str, Any]:
    """Build the algorithm config input entry.

    Returns:
        Dict that distinguishes *not provided* (None path)
        from *missing* (provided path does not exist).
    """
    if algo_config_path is None:
        return {
            "path": None,
            "sha256": None,
            "artifact_status": "not_provided",
        }
    resolved = Path(str(algo_config_path))
    if resolved.is_file():
        return {
            "path": str(resolved),
            "sha256": _sha256_of_file(resolved),
            "artifact_status": "available",
        }
    return {
        "path": str(resolved),
        "sha256": None,
        "artifact_status": "missing",
    }


def build_result_provenance_manifest(  # noqa: PLR0913
    *,
    out_path: Path,
    episode_records: list[dict[str, Any]],
    schema_path: str | Path,
    scenario_path: Path,
    scenarios: list[dict[str, Any]],
    algo: str,
    algo_config_path: str | Path | None,
    benchmark_profile: str,
    suite_key: str,
    total_jobs: int,
    written: int,
    horizon: int | None,
    dt: float | None,
    record_forces: bool,
    active_observation_mode: str | None,
    active_observation_level: str | None,
    noise_hash: str | None = None,
    tracking_precision_hash: str | None = None,
) -> dict[str, Any]:
    """Build the full ``benchmark_result_provenance.v1`` manifest.

    Returns:
        A JSON-serialisable dict representing the provenance manifest.
    """
    run_id = uuid.uuid4().hex
    repo_commit = _git_hash_fallback()
    raw_artifact_path = out_path

    # Lazy import to avoid circular dependency: release_protocol → camera_ready_campaign → runner → map_runner → result_provenance.
    from robot_sf.benchmark.release_protocol import (  # noqa: PLC0415
        BENCHMARK_PROTOCOL_VERSION,
    )

    # Build the invocation string.
    invocation = shlex.join(sys.argv) if hasattr(sys, "argv") and sys.argv else ""

    # Input entries.
    schema_path_obj = Path(schema_path)
    schema_entry: dict[str, Any] = {
        "path": str(schema_path_obj),
        "sha256": _sha256_of_file(schema_path_obj) if schema_path_obj.is_file() else None,
        "artifact_status": "available" if schema_path_obj.is_file() else "missing",
    }
    scenario_matrix_entry = _scenario_matrix_entry(
        Path(scenario_path),
        has_inline_scenarios=bool(scenarios),
    )
    algo_config_entry = _algo_config_entry(algo_config_path)
    inputs: dict[str, Any] = {
        "schema_path": schema_entry,
        "scenario_matrix": scenario_matrix_entry,
        "algo_config": algo_config_entry,
    }

    # Campaign identity.
    scenario_matrix_hash = _config_hash(scenarios)
    input_bundle_sha256 = _canonical_input_bundle_sha256(
        inputs=inputs,
        algo=algo,
        protocol_version=BENCHMARK_PROTOCOL_VERSION,
        suite_key=suite_key,
    )

    # Raw artifacts.
    raw_artifacts: list[dict[str, Any]] = []
    raw_artifact_status = "available" if written > 0 else "not_applicable"
    raw_artifacts.append(
        _artifact_entry(
            kind="episodes_jsonl",
            path=raw_artifact_path,
            artifact_status=raw_artifact_status,
        )
    )

    # Row-level provenance.
    rows: list[dict[str, Any]] = []
    for line_idx, rec in enumerate(episode_records):
        rows.append(
            build_row_result_provenance(
                episode_id=str(rec.get("episode_id", "")),
                scenario_id=str(rec.get("scenario_id", "")),
                seed=int(rec.get("seed", 0)),
                config_hash=str(rec.get("config_hash", "")),
                repo_commit=str(rec.get("git_hash", repo_commit)),
                raw_artifact_path=str(raw_artifact_path),
                jsonl_line=line_idx,
                dt=dt,
                horizon=horizon,
                record_forces=record_forces,
                active_observation_mode=active_observation_mode,
                active_observation_level=active_observation_level,
                noise_hash=noise_hash,
                tracking_precision_hash=tracking_precision_hash,
            )
        )

    # Completeness.
    is_complete = written > 0 and written >= total_jobs
    completeness: dict[str, Any]
    if is_complete:
        completeness = {
            "status": "complete",
            "required_fields_checked": sorted(
                _STRENGTHENED_REQUIRED_TOP_LEVEL
                + _REQUIRED_RUN
                + _REQUIRED_CAMPAIGN
                + _REQUIRED_ROW
            ),
        }
    else:
        completeness = {
            "status": "partial" if written > 0 else "not_applicable",
            "reason": "partial_batch_failure" if written > 0 else "preflight_skipped",
        }

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input_binding_schema_version": INPUT_BINDING_SCHEMA_VERSION,
        "run": {
            "run_id": run_id,
            "repo_commit": repo_commit,
            "python_version": platform.python_version(),
            "invocation": invocation,
            "benchmark_profile": str(benchmark_profile),
            "runner": "map_runner.run_map_batch",
            "protocol_version": BENCHMARK_PROTOCOL_VERSION,
            "execution_context": build_execution_context_provenance(),
        },
        "inputs": inputs,
        "campaign_identity": {
            "scenario_matrix_hash": scenario_matrix_hash,
            "input_bundle_sha256": input_bundle_sha256,
            "algorithm": str(algo),
            "config_hash": _config_hash(
                {
                    "schema_path": str(schema_path),
                    "algo": algo,
                    "algo_config_path": str(algo_config_path) if algo_config_path else None,
                }
            ),
            "suite_key": str(suite_key),
            "total_jobs": int(total_jobs),
            "written": int(written),
        },
        "raw_artifacts": raw_artifacts,
        "rows": rows,
        "derived_artifacts": [],
        "completeness": completeness,
    }
    return manifest


def _validate_input_artifact_entry(
    *,
    inputs: Mapping[str, Any],
    role: str,
    optional: bool = False,
    allow_not_applicable: bool = False,
) -> None:
    """Validate one manifest input artifact entry."""
    raw_entry = inputs.get(role)
    _require(isinstance(raw_entry, Mapping), f"inputs.{role} must be a dict")
    entry = raw_entry
    status = entry.get("artifact_status")
    path = entry.get("path")
    digest = entry.get("sha256")

    if status == "available":
        _require(isinstance(path, str) and bool(path.strip()), f"inputs.{role}.path is missing")
        _require(
            _is_valid_sha256_hex(digest),
            f"inputs.{role}.sha256 must be a 64-character SHA-256 hex digest",
        )
        resolved_path = Path(path)
        if resolved_path.exists():
            _require(resolved_path.is_file(), f"inputs.{role}.path must be a file")
        observed = _sha256_of_file(resolved_path)
        if observed is not None:
            _require(
                observed == str(digest).lower(),
                f"inputs.{role}.sha256 does not match current local file bytes",
            )
        return

    if status == "not_provided":
        _require(optional, f"inputs.{role} cannot be not_provided")
        _require(path is None, f"inputs.{role}.path must be null when not_provided")
        _require(digest is None, f"inputs.{role}.sha256 must be null when not_provided")
        return

    if status == "not_applicable":
        _require(allow_not_applicable, f"inputs.{role} cannot be not_applicable")
        _require(path is None, f"inputs.{role}.path must be null when not_applicable")
        _require(digest is None, f"inputs.{role}.sha256 must be null when not_applicable")
        _require(
            entry.get("reason") == "inline_or_generated_scenarios",
            f"inputs.{role}.reason must document inline/generated scenario input",
        )
        return

    if status == "missing":
        raise ProvenanceArtifactError(f"inputs.{role} is missing and cannot be complete evidence")

    raise ProvenanceArtifactError(f"inputs.{role}.artifact_status is invalid: {status!r}")


def _validate_campaign_identity(
    *,
    campaign: Mapping[str, Any],
    inputs: Mapping[str, Any],
    run: Mapping[str, Any],
    strengthened: bool,
) -> None:
    """Validate campaign identity fields and the strengthened input bundle."""
    if not strengthened:
        if "input_bundle_sha256" in campaign or "algorithm" in campaign:
            raise ProvenanceValidationError(
                f"schema_version {SCHEMA_VERSION!r} without "
                f"{INPUT_BINDING_SCHEMA_VERSION!r} cannot claim strengthened input-bundle fields"
            )
        required_campaign = ("scenario_matrix_hash", "total_jobs", "written")
    else:
        required_campaign = _REQUIRED_CAMPAIGN

    for field in required_campaign:
        _require(
            field in campaign,
            f"campaign_identity.{field} is missing",
        )
    if not strengthened:
        return

    _require(bool(campaign.get("algorithm")), "campaign_identity.algorithm is missing or empty")
    _require(bool(campaign.get("suite_key")), "campaign_identity.suite_key is missing or empty")
    _require(bool(run.get("protocol_version")), "run.protocol_version is missing or empty")
    _require(
        _is_valid_sha256_hex(campaign.get("input_bundle_sha256")),
        "campaign_identity.input_bundle_sha256 must be a 64-character SHA-256 hex digest",
    )
    expected_input_bundle = _canonical_input_bundle_sha256(
        inputs=inputs,
        algo=str(campaign.get("algorithm")),
        protocol_version=str(run.get("protocol_version")),
        suite_key=str(campaign.get("suite_key", "")),
    )
    _require(
        str(campaign.get("input_bundle_sha256")).lower() == expected_input_bundle,
        "campaign_identity.input_bundle_sha256 does not match inputs, algorithm identity, "
        "protocol version, and suite identity",
    )


def validate_result_provenance_manifest(payload: Mapping[str, Any]) -> None:
    """Validate a provenance manifest.

    Raises:
        ProvenanceRequiredFieldError: A required field is missing or empty.
        ProvenanceArtifactError: An available artifact has no SHA256.
        ProvenanceRowLinkError: A row does not properly link to its raw artifact.
    """
    schema_version = payload.get("schema_version")
    _require(
        schema_version == SCHEMA_VERSION,
        f"schema_version must be {SCHEMA_VERSION!r}",
    )

    for field in _REQUIRED_TOP_LEVEL:
        _require(field in payload, f"missing top-level field: {field!r}")

    run = payload.get("run", {})
    for field in _REQUIRED_RUN:
        _require(
            bool(run.get(field)),
            f"run.{field} is missing or empty",
        )

    inputs = payload.get("inputs", {})
    _require(isinstance(inputs, Mapping), "inputs must be dict")

    campaign = payload.get("campaign_identity", {})
    _require(isinstance(campaign, Mapping), "campaign_identity must be dict")
    binding_declared = "input_binding_schema_version" in payload
    binding_version = payload.get("input_binding_schema_version")
    strengthened = binding_declared
    if strengthened:
        _require(
            binding_version == INPUT_BINDING_SCHEMA_VERSION,
            f"input_binding_schema_version must be {INPUT_BINDING_SCHEMA_VERSION!r}",
        )
        _validate_input_artifact_entry(inputs=inputs, role="schema_path")
        _validate_input_artifact_entry(
            inputs=inputs,
            role="scenario_matrix",
            allow_not_applicable=True,
        )
        _validate_input_artifact_entry(
            inputs=inputs,
            role="algo_config",
            optional=True,
        )
    else:
        schema_input = inputs.get("schema_path", {})
        _require(isinstance(schema_input, Mapping), "inputs.schema_path must be a dict")
        _require(
            bool(schema_input.get("path")),
            "inputs.schema_path.path is missing or empty",
        )
    _validate_campaign_identity(
        campaign=campaign,
        inputs=inputs,
        run=run,
        strengthened=strengthened,
    )

    completeness = payload.get("completeness", {})
    _require(
        isinstance(completeness, Mapping),
        "completeness must be dict",
    )
    _require(
        completeness.get("status") != "partial",
        "completeness.status partial is not a valid complete provenance manifest",
    )

    raw_artifacts = payload.get("raw_artifacts", [])
    has_episodes = any(
        isinstance(a, dict) and a.get("kind") == "episodes_jsonl" for a in raw_artifacts
    )
    _require(has_episodes, "raw_artifacts must include an episodes_jsonl entry")

    for artifact in raw_artifacts:
        if isinstance(artifact, dict) and artifact.get("artifact_status") == "available":
            _require(
                bool(artifact.get("sha256")),
                f"available artifact {artifact.get('kind')!r} has no sha256",
            )

    rows = payload.get("rows", [])
    for row_idx, row in enumerate(rows):
        _require(isinstance(row, dict), f"rows[{row_idx}] must be a dict")
        for field in _REQUIRED_ROW:
            if field == "seed":
                _require(
                    row.get("seed") is not None,
                    f"rows[{row_idx}].{field} is missing or empty",
                )
            else:
                _require(
                    bool(row.get(field)),
                    f"rows[{row_idx}].{field} is missing or empty",
                )
        _require(
            bool(row.get("raw_artifact")),
            f"rows[{row_idx}].raw_artifact is missing",
        )
        _require(
            isinstance(row.get("simulator_settings"), dict),
            f"rows[{row_idx}].simulator_settings must be a dict",
        )
        postproc = row.get("postprocessing")
        _require(
            isinstance(postproc, list),
            f"rows[{row_idx}].postprocessing must be a list",
        )


def write_result_provenance_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a provenance manifest as pretty-printed JSON.

    Args:
        path: Output path for the JSON file.
        payload: The manifest dict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=str, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def manifest_path_for_result_jsonl(out_path: Path) -> Path:
    """Return the provenance manifest path for a given ``episodes.jsonl`` path.

    Args:
        out_path: Path to ``episodes.jsonl``.

    Returns:
        Path to ``episodes.jsonl.provenance.json``.
    """
    return out_path.with_suffix(out_path.suffix + ".provenance.json")


def load_result_provenance_manifest(path: str | Path) -> dict[str, Any]:
    """Load a provenance manifest from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed manifest dict.

    Raises:
        FileNotFoundError: The manifest file does not exist.
        json.JSONDecodeError: The file is not valid JSON.
    """
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "INPUT_BINDING_SCHEMA_VERSION",
    "ROW_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ProvenanceArtifactError",
    "ProvenanceRequiredFieldError",
    "ProvenanceRowLinkError",
    "ProvenanceValidationError",
    "build_execution_context_provenance",
    "build_result_provenance_manifest",
    "build_row_result_provenance",
    "build_simulator_settings_provenance",
    "load_result_provenance_manifest",
    "manifest_path_for_result_jsonl",
    "validate_result_provenance_manifest",
    "write_result_provenance_manifest",
]
