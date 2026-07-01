"""Result-level provenance manifest for benchmark runs.

This module emits a structured JSON provenance manifest alongside
``episodes.jsonl`` that links every emitted row to config identity,
scenario ID, seed, repo commit, simulator settings, raw artifact paths,
and post-processing steps.

Schema version: ``benchmark_result_provenance.v1``
"""

from __future__ import annotations

import json
import platform
import shlex
import sys
import uuid
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.utils import _config_hash, _git_hash_fallback

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMA_VERSION = "benchmark_result_provenance.v1"
ROW_SCHEMA_VERSION = "benchmark_row_provenance.v1"

# Fields whose absence triggers a validation error.
_REQUIRED_TOP_LEVEL = ("schema_version", "run", "inputs", "campaign_identity", "completeness")
_REQUIRED_RUN = ("run_id", "repo_commit", "runner")
_REQUIRED_CAMPAIGN = ("scenario_matrix_hash", "total_jobs", "written")
_REQUIRED_ROW = ("episode_id", "scenario_id", "seed", "config_hash", "repo_commit")


class ProvenanceValidationError(ValueError):
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
    schema_entry: dict[str, Any] = {
        "path": str(schema_path),
        "sha256": _sha256_of_file(schema_path),
        "artifact_status": "available",
    }
    scenario_matrix_path = str(scenario_path)
    scenario_matrix_entry: dict[str, Any] = {
        "path": scenario_matrix_path,
        "sha256": _sha256_of_file(scenario_path) if Path(scenario_path).is_file() else None,
        "artifact_status": "available" if Path(scenario_path).is_file() else "not_applicable",
    }
    algo_config_entry = _algo_config_entry(algo_config_path)

    # Campaign identity.
    scenario_matrix_hash = _config_hash(scenarios)

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
            "required_fields_checked": sorted(_REQUIRED_TOP_LEVEL + _REQUIRED_RUN + _REQUIRED_ROW),
        }
    else:
        completeness = {
            "status": "partial" if written > 0 else "not_applicable",
            "reason": "partial_batch_failure" if written > 0 else "preflight_skipped",
        }

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "run_id": run_id,
            "repo_commit": repo_commit,
            "python_version": platform.python_version(),
            "invocation": invocation,
            "benchmark_profile": str(benchmark_profile),
            "runner": "map_runner.run_map_batch",
            "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        },
        "inputs": {
            "schema_path": schema_entry,
            "scenario_matrix": scenario_matrix_entry,
            "algo_config": algo_config_entry,
        },
        "campaign_identity": {
            "scenario_matrix_hash": scenario_matrix_hash,
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


def validate_result_provenance_manifest(payload: Mapping[str, Any]) -> None:
    """Validate a provenance manifest.

    Raises:
        ProvenanceRequiredFieldError: A required field is missing or empty.
        ProvenanceArtifactError: An available artifact has no SHA256.
        ProvenanceRowLinkError: A row does not properly link to its raw artifact.
    """
    _require(
        payload.get("schema_version") == SCHEMA_VERSION,
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
    schema_input = inputs.get("schema_path", {})
    _require(
        bool(schema_input.get("path")),
        "inputs.schema_path.path is missing or empty",
    )

    campaign = payload.get("campaign_identity", {})
    for field in _REQUIRED_CAMPAIGN:
        _require(
            field in campaign,
            f"campaign_identity.{field} is missing",
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
    "ROW_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "ProvenanceArtifactError",
    "ProvenanceRequiredFieldError",
    "ProvenanceRowLinkError",
    "ProvenanceValidationError",
    "build_result_provenance_manifest",
    "build_row_result_provenance",
    "build_simulator_settings_provenance",
    "load_result_provenance_manifest",
    "manifest_path_for_result_jsonl",
    "validate_result_provenance_manifest",
    "write_result_provenance_manifest",
]
