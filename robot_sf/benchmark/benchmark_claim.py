"""Benchmark claim artifact generation.

The claim artifact is the compact, schema-checked boundary between benchmark
execution outputs and paper-facing statements.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.common.artifact_paths import get_repository_root

BENCHMARK_CLAIM_SCHEMA_VERSION = "benchmark_claim.v1"
BENCHMARK_CLAIM_SCHEMA_PATH = Path("robot_sf/benchmark/schemas/benchmark_claim.schema.v1.json")
_HEX_SHA256_LENGTH = 64


class BenchmarkClaimError(ValueError):
    """Raised when claim inputs cannot support a benchmark claim artifact."""


def _repo_relative(path: Path) -> str:
    """Return a repository-relative path when possible, otherwise the resolved path."""
    resolved = path.resolve()
    repo_root = get_repository_root().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(resolved)


def _require_file(path: Path, field_name: str) -> Path:
    """Resolve and validate a required input file path.

    Returns:
        Resolved input file path.
    """
    resolved = path.resolve()
    if not resolved.exists():
        raise BenchmarkClaimError(f"{field_name} not found: {resolved}")
    if not resolved.is_file():
        raise BenchmarkClaimError(f"{field_name} must be a file path: {resolved}")
    return resolved


def _require_sha256(value: str, field_name: str) -> str:
    """Validate a required SHA-256 digest string.

    Returns:
        Normalized lowercase SHA-256 digest.
    """
    digest = str(value).strip().lower()
    if len(digest) != _HEX_SHA256_LENGTH or any(ch not in "0123456789abcdef" for ch in digest):
        raise BenchmarkClaimError(f"{field_name} must be a 64-character SHA-256 digest")
    return digest


def _load_json_mapping(path: Path, field_name: str) -> dict[str, Any]:
    """Load a JSON object from ``path``.

    Returns:
        Parsed JSON mapping.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkClaimError(f"{field_name} must be valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkClaimError(f"{field_name} must be a JSON object: {path}")
    return payload


def _schema_version_from_payload(payload: dict[str, Any], field_name: str) -> str:
    """Extract a non-empty schema/version marker from a JSON payload.

    Returns:
        Schema/version marker string.
    """
    raw = payload.get("schema_version", payload.get("version"))
    version = str(raw or "").strip()
    if not version:
        raise BenchmarkClaimError(f"{field_name} is missing a schema version")
    return version


def _load_policy_metadata(path: Path) -> dict[str, Any]:
    """Load policy metadata and fail closed when policy hashes are missing.

    Returns:
        Normalized policy metadata claim block.
    """
    resolved = _require_file(path, "policy_metadata")
    payload = _load_json_mapping(resolved, "policy_metadata")
    schema_version = _schema_version_from_payload(payload, "policy_metadata")
    policies_raw = payload.get("policies")
    if not isinstance(policies_raw, list) or not policies_raw:
        raise BenchmarkClaimError("policy_metadata.policies must be a non-empty list")

    policies: list[dict[str, str | None]] = []
    for index, item in enumerate(policies_raw):
        if not isinstance(item, dict):
            raise BenchmarkClaimError(f"policy_metadata.policies[{index}] must be an object")
        policy_id = str(item.get("policy_id", item.get("id", ""))).strip()
        if not policy_id:
            raise BenchmarkClaimError(f"policy_metadata.policies[{index}].policy_id is required")
        digest = _require_sha256(
            str(item.get("sha256", item.get("policy_sha256", ""))),
            f"policy_metadata.policies[{index}].sha256",
        )
        artifact_path = item.get("artifact_path", item.get("path"))
        policies.append(
            {
                "policy_id": policy_id,
                "sha256": digest,
                "artifact_path": str(artifact_path) if artifact_path is not None else None,
            }
        )

    return {
        "path": _repo_relative(resolved),
        "sha256": sha256_file(resolved),
        "schema_version": schema_version,
        "policies": policies,
    }


def _episode_schema_version(record: dict[str, Any]) -> str:
    """Extract the schema/version marker from an episode record.

    Returns:
        Episode schema/version marker.
    """
    for key in ("schema_version", "version"):
        raw = record.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    integrity = record.get("integrity")
    if isinstance(integrity, dict):
        raw = integrity.get("schema_version")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    raise BenchmarkClaimError("episode record is missing a schema version")


def _episode_artifact(path: Path, field_name: str) -> dict[str, Any]:
    """Build a claim entry for one episode JSONL artifact.

    Returns:
        Normalized schema-backed episode artifact entry.
    """
    resolved = _require_file(path, field_name)
    schema_versions: set[str] = set()
    seeds: set[int] = set()
    count = 0
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkClaimError(
                    f"Line {line_number} in {field_name} must be valid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise BenchmarkClaimError(
                    f"Line {line_number} in {field_name} must be a JSON object"
                )
            try:
                schema_versions.add(_episode_schema_version(record))
            except BenchmarkClaimError as exc:
                raise BenchmarkClaimError(f"Line {line_number} in {field_name}: {exc}") from exc
            seed = record.get("seed")
            if isinstance(seed, int) and not isinstance(seed, bool):
                seeds.add(int(seed))
            count += 1
    if count == 0:
        raise BenchmarkClaimError(f"{field_name} must contain at least one episode record")
    return {
        "path": _repo_relative(resolved),
        "sha256": sha256_file(resolved),
        "schema_version": ",".join(sorted(schema_versions)),
        "episode_count": count,
        "seed_suite": sorted(seeds),
    }


def _episode_artifacts(paths: list[Path], group_name: str) -> list[dict[str, Any]]:
    """Build claim entries for an episode evidence group.

    Returns:
        Normalized episode artifact entries.
    """
    return [
        _episode_artifact(path, f"episode_groups.{group_name}[{index}]")
        for index, path in enumerate(paths)
    ]


def _aggregate_artifact(path: Path, index: int) -> dict[str, Any]:
    """Build a claim entry for one aggregate/statistical summary artifact.

    Returns:
        Normalized aggregate artifact entry.
    """
    resolved = _require_file(path, f"aggregate_reports[{index}]")
    payload = _load_json_mapping(resolved, f"aggregate_reports[{index}]")
    return {
        "path": _repo_relative(resolved),
        "sha256": sha256_file(resolved),
        "schema_version": _schema_version_from_payload(payload, f"aggregate_reports[{index}]"),
    }


def _git_sha() -> str:
    """Return the current Git commit SHA when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=get_repository_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def load_benchmark_claim_schema() -> dict[str, Any]:
    """Load the benchmark claim v1 JSON schema.

    Returns:
        Parsed benchmark claim JSON schema.
    """
    schema_path = get_repository_root() / BENCHMARK_CLAIM_SCHEMA_PATH
    return _load_json_mapping(schema_path, "benchmark_claim_schema")


def validate_benchmark_claim(claim: dict[str, Any]) -> None:
    """Validate a benchmark claim payload against the v1 schema."""
    Draft202012Validator(load_benchmark_claim_schema()).validate(claim)


def build_benchmark_claim(  # noqa: PLR0913
    *,
    claim_id: str,
    statement: str,
    scenario_matrix_path: Path,
    scenario_matrix_sha256: str,
    policy_metadata_path: Path,
    final_benchmark_episodes: list[Path],
    training_episodes: list[Path] | None = None,
    validation_episodes: list[Path] | None = None,
    aggregate_reports: list[Path] | None = None,
    dependency_group: str = "dev",
    container_image_digest: str | None = None,
) -> dict[str, Any]:
    """Build and validate a benchmark claim artifact payload.

    Returns:
        JSON-serializable benchmark claim payload.
    """
    normalized_claim_id = str(claim_id).strip()
    if not normalized_claim_id:
        raise BenchmarkClaimError("claim_id must be a non-empty string")
    normalized_statement = str(statement).strip()
    if not normalized_statement:
        raise BenchmarkClaimError("statement must be a non-empty string")

    matrix = _require_file(scenario_matrix_path, "scenario_matrix")
    expected_matrix_hash = _require_sha256(scenario_matrix_sha256, "scenario_matrix_sha256")
    actual_matrix_hash = sha256_file(matrix)
    if actual_matrix_hash != expected_matrix_hash:
        raise BenchmarkClaimError("scenario_matrix_sha256 does not match scenario_matrix")
    final_paths = list(final_benchmark_episodes)
    if not final_paths:
        raise BenchmarkClaimError("final_benchmark_episodes must contain at least one path")

    uv_lock_path = _require_file(get_repository_root() / "uv.lock", "uv.lock")
    claim = {
        "schema_version": BENCHMARK_CLAIM_SCHEMA_VERSION,
        "claim_id": normalized_claim_id,
        "statement": normalized_statement,
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "environment": {
            "git_sha": _git_sha(),
            "python_version": sys.version.split()[0],
            "dependency_group": str(dependency_group).strip() or "dev",
            "uv_lock_sha256": sha256_file(uv_lock_path),
            "container_image_digest": container_image_digest,
        },
        "evidence": {
            "scenario_matrix": {
                "path": _repo_relative(matrix),
                "sha256": actual_matrix_hash,
            },
            "policy_metadata": _load_policy_metadata(policy_metadata_path),
            "episode_groups": {
                "training": _episode_artifacts(list(training_episodes or []), "training"),
                "validation": _episode_artifacts(list(validation_episodes or []), "validation"),
                "final_benchmark": _episode_artifacts(final_paths, "final_benchmark"),
            },
            "aggregate_reports": [
                _aggregate_artifact(path, index)
                for index, path in enumerate(list(aggregate_reports or []))
            ],
        },
    }
    validate_benchmark_claim(claim)
    return claim


def write_benchmark_claim(path: Path, claim: dict[str, Any]) -> None:
    """Write a benchmark claim JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(claim, indent=2) + "\n", encoding="utf-8")


__all__ = [
    "BENCHMARK_CLAIM_SCHEMA_PATH",
    "BENCHMARK_CLAIM_SCHEMA_VERSION",
    "BenchmarkClaimError",
    "build_benchmark_claim",
    "load_benchmark_claim_schema",
    "validate_benchmark_claim",
    "write_benchmark_claim",
]
