"""Legacy compatibility and deprecation mapping for environment factories.

T007: Provides DeprecationMap and apply_legacy_kwargs used to normalize legacy
keyword arguments into the new ergonomic factory parameter surface.

Design goals:
- Centralize legacy → new parameter mapping logic.
- Allow strict (default) vs permissive (env var) handling of unknown legacy kwargs.
- Emit structured warnings via Loguru (no prints) for every mapped or ignored kw.

This module is intentionally internal (leading underscore). Public factories
import and use `apply_legacy_kwargs` prior to constructing option objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterable

LEGACY_PERMISSIVE_ENV = "ROBOT_SF_FACTORY_LEGACY"


@dataclass(frozen=True)
class DeprecationEntry:
    """DeprecationEntry class."""

    legacy: str
    new: str | None  # None means removed with no replacement (ignored permissively)
    note: str = ""


# Initial mapping (extend as repository evolves)
_DEPRECATION_TABLE: tuple[DeprecationEntry, ...] = (
    DeprecationEntry(
        "record_video",
        "recording_options.record",
        "Boolean convenience retained; prefer RecordingOptions",
    ),
    DeprecationEntry("video_output_path", "recording_options.video_path", "Renamed for clarity"),
    DeprecationEntry("fps", "render_options.max_fps_override", "Scoped inside RenderOptions"),
)

# Fast lookup structures
_MAP = {e.legacy: e for e in _DEPRECATION_TABLE}

# Whitelisted keys that are part of the current public (or transitional) factory
# signature surface and should NOT be treated as unknown legacy parameters.
_ALLOWED_PASSTHROUGH = {
    # Core factory params (current + transitional convenience flags)
    "config",
    "debug",
    "seed",
    "record_video",
    "video_path",
    "reward_func",
    "max_episode_steps",
    "render_options",
    "recording_options",
    "robot_model",
    # Mapping targets (flattened keys consumed later by normalization logic)
    "recording_options.record",
    "recording_options.video_path",
    "render_options.max_fps_override",
}


class UnknownLegacyParameterError(ValueError):
    """Raised when an unknown legacy keyword is encountered in strict mode."""


def is_permissive_mode() -> bool:
    """Is permissive mode.

    Returns:
        bool: Boolean flag.
    """
    return os.getenv(LEGACY_PERMISSIVE_ENV, "0") not in ("", "0", "false", "False", "NO", "no")


def iter_deprecation_entries() -> Iterable[DeprecationEntry]:
    """Iter deprecation entries.

    Returns:
        Iterable[DeprecationEntry]: Iterable of cataloged deprecation entries.
    """
    return iter(_DEPRECATION_TABLE)


def apply_legacy_kwargs(
    kwargs: dict[str, Any],
    strict: bool = True,
) -> tuple[dict[str, Any], list[str]]:
    """Normalize legacy kwargs into new structure.

    Parameters
    ----------
    kwargs : dict
        Raw kwargs passed to a factory prior to signature filtering.
    strict : bool, default True
        If True, unknown legacy parameters raise; if False (or permissive mode env
        var set) they are ignored with warnings.

    Returns
    -------
    (normalized_kwargs, warnings)
        normalized_kwargs: dict without legacy keys (they are removed)
        warnings: list of warning message strings emitted (also logged)
    """
    warnings: list[str] = []
    normalized = dict(kwargs)  # shallow copy

    effective_strict = strict and not is_permissive_mode()

    for key in list(normalized.keys()):
        if key not in _MAP:
            # Not a known legacy parameter; skip
            continue
        entry = _MAP[key]
        value = normalized.pop(key)
        if entry.new is None:
            msg = f"Deprecated parameter '{key}' removed with no replacement; value ignored"
            logger.warning(msg)
            warnings.append(msg)
            continue
        msg = f"Deprecated parameter '{key}' mapped to '{entry.new}'"
        logger.warning(msg)
        warnings.append(msg)
        # The mapping target can be nested like 'recording_options.video_path'. We encode
        # into flattened keys for later processing by factory normalization logic.
        normalized[entry.new] = value

    # Detect unknown parameters: any key not recognized as legacy mapping source
    # or allowed passthrough. In strict mode we raise; otherwise warn & drop.
    for key in list(normalized.keys()):
        if key in _MAP or key in _ALLOWED_PASSTHROUGH:
            continue
        if effective_strict:
            raise UnknownLegacyParameterError(
                f"Unknown parameter '{key}'. Set {LEGACY_PERMISSIVE_ENV}=1 to ignore or update to supported signature.",
            )
        msg = f"Unknown parameter '{key}' ignored (permissive mode)"
        logger.warning(msg)
        warnings.append(msg)
        normalized.pop(key, None)

    return normalized, warnings


__all__ = [
    "LEGACY_PERMISSIVE_ENV",
    "DeprecationEntry",
    "UnknownLegacyParameterError",
    "apply_legacy_kwargs",
    "iter_deprecation_entries",
]
