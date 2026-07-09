"""Pure leaf helpers for the camera-ready campaign package.

Extracted from ``robot_sf.benchmark.camera_ready_campaign`` as the first slice
of the #3385 decomposition. No behavior change: the functions are verbatim
moves, and ``camera_ready_campaign`` re-exports them so the existing import
surface is unchanged.

Helpers hosted here:
- Hashers: ``_stable_json_bytes``, ``_hash_payload``, ``_sha256_payload``, ``_sha256_file``
- JSON converters: ``_jsonable``, ``_jsonable_repo_relative``
- Sanitizers: ``_sanitize_csv_cell``
- Path/time utilities: ``_repo_relative``, ``_utc_now``
- Kinematics matrix utility: ``_kinematics_matrix_or_default``
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from robot_sf.benchmark.latency_stress import LatencyStressProfile
    from robot_sf.benchmark.synthetic_actuation import SyntheticActuationProfile


def _repo_relative(path: Path) -> str:
    """Return a repo-relative POSIX string for *path*, falling back to the resolved absolute path.

    Returns:
        Repo-relative POSIX path string, or the resolved absolute path if *path*
        is outside the repository root.
    """
    path_resolved = path.resolve()
    repo_root = get_repository_root().resolve()
    try:
        return path_resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path_resolved)


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with trailing ``Z``."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _stable_json_bytes(payload: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for hashing payloads."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_payload(payload: Any) -> str:
    """Compute a deterministic SHA1 short hash for a JSON-serializable payload.

    Returns:
        Twelve-character SHA1 digest prefix.
    """
    return hashlib.sha1(_stable_json_bytes(payload)).hexdigest()[:12]


def _sha256_payload(payload: Any) -> str:
    """Return a stable SHA-256 digest for a JSON-serializable payload."""
    return hashlib.sha256(_stable_json_bytes(payload)).hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert nested values into JSON-serializable primitives.

    Returns:
        JSON-serializable value with ``Path`` objects converted to strings.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _jsonable_repo_relative(value: Any) -> Any:
    """Convert nested values into JSON-serializable primitives with repo-relative paths.

    Returns:
        JSON-serializable value with ``Path`` objects normalized to stable repo-relative strings.
    """
    if isinstance(value, Path):
        return _repo_relative(value)
    if isinstance(value, dict):
        return {str(key): _jsonable_repo_relative(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_repo_relative(item) for item in value]
    return value


def _sanitize_csv_cell(value: Any) -> Any:
    """Prevent spreadsheet formula execution for untrusted CSV cell values.

    Returns:
        Original value, or a quote-prefixed string for formula-like text cells.
    """
    if not isinstance(value, str):
        return value
    if value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def _kinematics_matrix_or_default(kinematics: tuple[str, ...]) -> tuple[str, ...]:
    """Return normalized kinematics labels, defaulting only for empty configured matrices.

    Returns:
        Lowercase non-empty kinematics tuple, defaulting to ``("differential_drive",)``.
    """
    return tuple(str(value).strip().lower() for value in kinematics if str(value).strip()) or (
        "differential_drive",
    )


_normalized_kinematics_matrix = _kinematics_matrix_or_default


def _synthetic_actuation_metadata(
    profile: SyntheticActuationProfile | None,
) -> dict[str, Any] | None:
    """Return a JSON-safe synthetic-actuation metadata payload when configured."""
    if profile is None:
        return None
    return profile.to_metadata()


def _latency_stress_metadata(
    profile: LatencyStressProfile | None,
    *,
    dt: float | None = None,
) -> dict[str, Any] | None:
    """Return a JSON-safe latency-stress metadata payload when configured."""
    if profile is None:
        return None
    return profile.to_metadata(dt=dt)
