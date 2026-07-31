"""Reusable model-asset preflight for benchmark campaigns and CI (issue #6189).

The exact-repeat determinism failure (issue #6188) is caused by each forked
worker downloading a required model checkpoint (for example
``predictive_proxy_selected_v2_full``) from a GitHub release *inside* the timed
repeat/worker loop. A transient network failure there makes the predictive
planner silently fall back to constant velocity, so ostensibly identical repeats
diverge.

This module resolves every model asset a campaign/config requires **before** the
worker loop starts. It builds directly on the registry's existing
checksum-verified download path -- it does **not** reimplement checksum logic.
The registry already:

* verifies the registry-pinned SHA-256 for both freshly downloaded and cached
  artifacts (:func:`robot_sf.models.registry._download_from_github_release` /
  ``_cached_release_path_is_valid`` / ``_verify_download_checksum``), and
* writes downloads atomically (temp file + ``os.replace``) so a partial file is
  never left at the cache path.

On top of that primitive this module adds **bounded retries with backoff** and a
clear, fail-closed setup error. It is intentionally shared: scientific campaigns
call it before ``sbatch``/worker creation, and CI calls it as a setup step
before the timed exact-repeat test (see ``scripts/models/preflight_models.py``
and the ``exact-repeat-model-preflight`` job in ``.github/workflows/ci.yml``).

Claim boundary: this is a provisioning / fail-closed preflight. It does not
change the planner fallback metadata or evidence-eligibility contract (that is
issue #6190). It only guarantees the required assets are present, verified, and
cached before execution so the timed loop performs no network download.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.errors import RobotSfError
from robot_sf.models.registry import resolve_model_path

if TYPE_CHECKING:
    from pathlib import Path

# Bounded-retry defaults. Persistent failure after these attempts is a hard setup
# error, never a silent fallback.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 2.0

# Exceptions ``resolve_model_path`` can raise for a transient/resolvable failure.
# ``RuntimeError`` wraps download (HTTP/URL/timeout) errors; ``ValueError`` is a
# checksum mismatch; the rest are missing-entry / filesystem faults.
_RESOLVE_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    ValueError,
    FileNotFoundError,
    KeyError,
    OSError,
    TimeoutError,
)

# Config keys (at any nesting depth of a planner/algo config) that name a
# registry ``model_id`` the timed loop would otherwise resolve on-demand. This
# mirrors the checkpoint-reference keys used by the campaign checkpoint preflight
# and adds the predictive-foresight model, which is gated behind an explicit
# ``predictive_foresight_enabled`` flag.
_MODEL_ID_KEYS: tuple[str, ...] = (
    "model_id",
    "sacadrl_model_id",
    "predictive_model_id",
)
_GATED_MODEL_ID_KEYS: tuple[tuple[str, str], ...] = (
    # (gate flag key, model id key): the model id is only required when the gate
    # flag is truthy.
    ("predictive_foresight_enabled", "predictive_foresight_model_id"),
)


class ModelPreflightError(RobotSfError):
    """Raised when a required model asset cannot be staged before execution.

    This is a fail-closed setup error: callers must treat it as a reason to abort
    the campaign/CI job *before* the worker loop, not to proceed and let a forked
    worker silently fall back.
    """


def preflight_model(
    model_id: str,
    *,
    registry_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Resolve one model asset to a verified local cache path with bounded retries.

    Delegates the download + registry-pinned SHA-256 verification + atomic cache
    replacement to :func:`robot_sf.models.registry.resolve_model_path` (the
    existing checksum path -- not reimplemented here) and wraps it in a bounded
    retry loop with linear backoff.

    Args:
        model_id: Registry model id to stage.
        registry_path: Optional model-registry path override (tests/fixtures).
        cache_dir: Optional cache directory override; defaults to the registry's
            ``output/model_cache``.
        max_attempts: Total attempts before failing (>= 1). Persistent failure
            raises :class:`ModelPreflightError`.
        backoff_seconds: Base backoff; attempt ``n`` sleeps ``n * backoff_seconds``.
        sleep: Injectable sleep for deterministic tests.

    Returns:
        Path: Local filesystem path to the verified, cached model artifact.

    Raises:
        ValueError: If ``max_attempts`` is less than 1.
        ModelPreflightError: If the asset cannot be staged after ``max_attempts``.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            path = resolve_model_path(
                model_id,
                registry_path=registry_path,
                allow_download=True,
                cache_dir=cache_dir,
            )
        except _RESOLVE_ERRORS as exc:
            last_exc = exc
            logger.warning(
                "Model preflight attempt {}/{} for '{}' failed: {}: {}",
                attempt,
                max_attempts,
                model_id,
                type(exc).__name__,
                exc,
            )
        else:
            if path.is_file():
                logger.info(
                    "Model preflight staged '{}' at {} (attempt {}/{}).",
                    model_id,
                    path,
                    attempt,
                    max_attempts,
                )
                return path
            last_exc = ModelPreflightError(
                f"Model '{model_id}' resolved to {path} but no file is present."
            )
            logger.warning(
                "Model preflight attempt {}/{} for '{}' resolved a missing file: {}",
                attempt,
                max_attempts,
                model_id,
                path,
            )

        if attempt < max_attempts:
            sleep(backoff_seconds * attempt)

    raise ModelPreflightError(
        f"Could not stage required model '{model_id}' after {max_attempts} attempt(s). "
        "Refusing to start execution: a forked worker would otherwise download the asset "
        "inside the timed loop and could silently fall back on a transient failure. "
        f"Last error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def preflight_models(
    model_ids: Iterable[str],
    *,
    registry_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Path]:
    """Stage every required model asset before the worker loop starts.

    De-duplicates ``model_ids`` (preserving first-seen order) and resolves each to
    a verified local cache path via :func:`preflight_model`. Returns the mapping
    of model id to cached path so callers can assert every asset is present before
    creating workers.

    Returns:
        dict[str, Path]: Mapping of each de-duplicated model id to its verified,
        cached local path.

    Raises:
        ModelPreflightError: If any asset cannot be staged after bounded retries.
    """
    resolved: dict[str, Path] = {}
    for model_id in _dedupe(model_ids):
        resolved[model_id] = preflight_model(
            model_id,
            registry_path=registry_path,
            cache_dir=cache_dir,
            max_attempts=max_attempts,
            backoff_seconds=backoff_seconds,
            sleep=sleep,
        )
    return resolved


def required_model_ids_for_config(config: Mapping[str, Any] | Sequence[Any] | Any) -> list[str]:
    """Collect the registry model ids a planner/algo config would resolve at runtime.

    Walks an arbitrarily nested config mapping/sequence and collects:

    * every value under a direct model-id key (``model_id``, ``sacadrl_model_id``,
      ``predictive_model_id``), and
    * the predictive-foresight model id, but only within a mapping level whose
      ``predictive_foresight_enabled`` gate is truthy.

    Returns:
        list[str]: De-duplicated model ids, in first-seen order.
    """
    collected: list[str] = []
    _walk_config(config, collected)
    return _dedupe(collected)


def _walk_config(node: Any, collected: list[str]) -> None:
    """Recursively collect required model ids from a config node."""
    if isinstance(node, Mapping):
        for key in _MODEL_ID_KEYS:
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                collected.append(value.strip())
        for gate_key, model_key in _GATED_MODEL_ID_KEYS:
            if bool(node.get(gate_key)):
                value = node.get(model_key)
                if isinstance(value, str) and value.strip():
                    collected.append(value.strip())
        for value in node.values():
            _walk_config(value, collected)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _walk_config(value, collected)


def _dedupe(values: Iterable[str]) -> list[str]:
    """Return ``values`` with duplicates removed, preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


__all__ = [
    "DEFAULT_BACKOFF_SECONDS",
    "DEFAULT_MAX_ATTEMPTS",
    "ModelPreflightError",
    "preflight_model",
    "preflight_models",
    "required_model_ids_for_config",
]
