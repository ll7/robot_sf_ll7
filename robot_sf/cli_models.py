"""User-facing ``robot-sf models`` UX glue.

Thin, beginner-facing surface over the existing model registry and checksum
machinery in :mod:`robot_sf.models.registry`. The functions here produce
structured payloads so the CLI dispatcher (:mod:`robot_sf.cli`) and tests can
share one implementation.

Plain-language summary: ``models list`` shows what is registered and whether the
local artifact is present; ``models verify`` checks the pinned SHA256 of each
artifact (pass/fail/missing per model); ``models download <id>`` resolves and
downloads a model artifact through the existing registry download path.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.models.registry import (
    DEFAULT_REGISTRY_PATH,
    load_registry,
    resolve_model_path,
    sha256_of_file,
)

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from collections.abc import Sequence

__all__ = [
    "download_model",
    "list_models",
    "verify_models",
]

# Status values reported by ``verify_models`` per artifact.
STATUS_OK = "ok"
STATUS_MISSING = "missing"
STATUS_MISMATCH = "mismatch"
STATUS_NO_CHECKSUM = "no_pinned_checksum"
STATUS_ERROR = "error"


def _claim_boundary(entry: Mapping[str, Any]) -> str | None:
    """Return the registry entry's benchmark-promotion claim boundary, if any."""
    promotion = entry.get("benchmark_promotion")
    if isinstance(promotion, Mapping):
        value = promotion.get("claim_boundary")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _entry_local_path(entry: Mapping[str, Any]) -> Path | None:
    """Return the resolved local artifact path for a registry entry, or None."""
    raw = entry.get("local_path")
    if not isinstance(raw, str) or not raw.strip():
        return None
    resolved = Path(raw)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def _entry_release_sha256(entry: Mapping[str, Any]) -> str | None:
    """Return the pinned GitHub-release SHA256 for an entry, normalized, if any."""
    release = entry.get("github_release")
    if not isinstance(release, Mapping):
        return None
    value = release.get("sha256")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().lower()


def _entry_summary(entry: Mapping[str, Any], *, present: bool) -> dict[str, Any]:
    """Build one ``list_models`` entry from a registry row plus a presence flag.

    Returns:
        dict[str, Any]: Per-model summary row.
    """
    return {
        "model_id": entry.get("model_id"),
        "display_name": entry.get("display_name"),
        "tags": list(entry.get("tags") or []),
        "claim_boundary": _claim_boundary(entry),
        "local_path": str(_entry_local_path(entry)) if _entry_local_path(entry) else None,
        "present_locally": present,
        "downloadable": bool(entry.get("github_release")) or bool(entry.get("wandb_run_path")),
        "local_only": bool(entry.get("local_only")),
    }


def list_models(
    *,
    registry_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return one summary row per registry entry, with local presence status.

    Returns:
        list[dict[str, Any]]: Per-model summary rows in registry order.
    """
    registry = load_registry(registry_path)
    rows: list[dict[str, Any]] = []
    for entry in registry.values():
        local = _entry_local_path(entry)
        present = bool(local and local.exists())
        rows.append(_entry_summary(entry, present=present))
    return rows


def verify_models(
    *,
    registry_path: str | Path | None = None,
    model_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Verify each artifact's checksum and return a pass/fail report.

    For entries with a pinned ``github_release.sha256``, the local file's SHA256
    is compared against the pin and reported as ``ok`` / ``mismatch`` / ``missing``.
    Entries without a pinned checksum are reported as ``no_pinned_checksum`` once
    their local file presence is confirmed (or ``missing`` if absent). Local-only
    entries without a pin therefore report presence rather than a checksum verdict.

    Returns:
        dict[str, Any]: Report with ``ok`` (all pinned checksums passed),
        per-artifact ``results``, and aggregate counts.
    """
    registry = load_registry(registry_path)
    if model_ids is not None:
        requested = list(model_ids)
        missing_ids = [mid for mid in requested if mid not in registry]
        if missing_ids:
            raise KeyError(f"Unknown model_id(s) in registry: {', '.join(missing_ids)}")
        targets = {mid: registry[mid] for mid in requested}
    else:
        targets = registry

    results: list[dict[str, Any]] = []
    pass_count = 0
    pinned_total = 0
    for model_id, entry in targets.items():
        result = _verify_one(model_id, entry)
        results.append(result)
        if result["pinned"]:
            pinned_total += 1
            if result["status"] == STATUS_OK:
                pass_count += 1

    return {
        "schema": "robot_sf_models_verify.v1",
        "ok": pinned_total > 0 and pass_count == pinned_total,
        "checked": len(results),
        "pinned_checksums": pinned_total,
        "passed": pass_count,
        "results": results,
    }


def _verify_one(model_id: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Build the verify verdict for a single model artifact.

    Returns:
        dict[str, Any]: Per-model verify result with checksum status.
    """
    local = _entry_local_path(entry)
    expected = _entry_release_sha256(entry)
    result: dict[str, Any] = {
        "model_id": model_id,
        "display_name": entry.get("display_name"),
        "local_path": str(local) if local else None,
        "expected_sha256": expected,
        "observed_sha256": None,
        "pinned": bool(expected),
        "status": STATUS_MISSING,
    }
    if local is None or not local.exists():
        result["status"] = STATUS_MISSING
        return result
    if not expected:
        # No pinned checksum to compare against; report presence only.
        result["status"] = STATUS_NO_CHECKSUM
        return result
    try:
        observed = sha256_of_file(local)
    except OSError as exc:
        result["status"] = STATUS_ERROR
        result["error"] = str(exc)
        return result
    result["observed_sha256"] = observed
    result["status"] = STATUS_OK if observed == expected else STATUS_MISMATCH
    return result


def download_model(
    model_id: str,
    *,
    registry_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve (and download if needed) a model artifact and return a report.

    Delegates to :func:`robot_sf.models.registry.resolve_model_path` so the
    existing download/cache/checksum-validation path is reused unchanged.

    Returns:
        dict[str, Any]: Report with ``model_id``, resolved ``path``, and ``ok``.
    """
    path = resolve_model_path(
        model_id,
        registry_path=registry_path,
        allow_download=True,
        cache_dir=cache_dir,
    )
    return {
        "model_id": model_id,
        "path": str(path),
        "ok": True,
    }


def _default_registry_path_or_none() -> Path | None:
    """Return the default registry path (used by the CLI help text)."""
    return DEFAULT_REGISTRY_PATH
