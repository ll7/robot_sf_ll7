"""Render helper catalog for reusable rendering and frame capture utilities.

This module provides helper functions for managing output directories, capturing
frames from environments, handling recording-related tasks, and deriving
standardized naming metadata for JSONL recordings.
"""

from hashlib import sha1
from pathlib import Path

import numpy as np
from loguru import logger


def ensure_output_dir(path: Path) -> Path:
    """Create directories with exist_ok=True and return normalized path.

    Args:
        path: Path to the output directory

    Returns:
        Normalized absolute path to the created directory

    Raises:
        OSError: If directory creation fails due to permissions or other issues
    """
    try:
        path = Path(path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {path}")
        return path
    except OSError as e:
        logger.error(f"Failed to create output directory {path}: {e}")
        raise


def capture_frames(env, stride: int = 1) -> list[np.ndarray]:
    """Provide reusable frame sampling logic for recording helpers.

    Args:
        env: Environment instance with render method
        stride: Sample every N frames (1 = capture all frames)

    Returns:
        List of captured frame arrays

    Raises:
        AttributeError: If environment doesn't support rendering
        ValueError: If stride is less than 1
    """
    if stride < 1:
        msg = f"stride must be >= 1, got {stride}"
        raise ValueError(msg)

    if not hasattr(env, "render"):
        msg = f"Environment {type(env).__name__} does not support rendering"
        raise AttributeError(msg)

    frames = []

    try:
        # This is a placeholder implementation - actual frame capture would
        # depend on the specific environment interface
        logger.debug(f"Starting frame capture with stride={stride}")

        # For now, return empty list as implementation will be filled
        # when integrating with actual environment rendering
        logger.warning("capture_frames is not yet fully implemented")
        return frames

    except Exception as e:
        logger.error(f"Frame capture failed: {e}")
        raise


def derive_recording_tags(source: str | Path) -> tuple[str, str, str]:
    """Derive suite, scenario, and algorithm tags from a path or stem.

    Args:
        source: Path or string used to derive recording identifiers. Typically a
            filename stem like ``suite_scenario_algorithm``.

    Returns:
        Tuple of (suite, scenario, algorithm) with sensible defaults when
        segments are missing.
    """

    stem = Path(source).stem
    parts = [part for part in stem.split("_") if part]

    suite = parts[0] if len(parts) > 0 else "converted"
    scenario = parts[1] if len(parts) > 1 else "legacy"
    algorithm = parts[2] if len(parts) > 2 else "unknown"

    logger.debug(
        "Derived recording tags suite=%s scenario=%s algorithm=%s from stem=%s",
        suite,
        scenario,
        algorithm,
        stem,
    )

    return suite, scenario, algorithm


def deterministic_seed_from_name(name: str | Path) -> int:
    """Return a deterministic 31-bit seed derived from a file name or path.

    Args:
        name: File name or path to hash for deterministic seeding.

    Returns:
        Non-negative integer seed stable across runs and platforms.
    """

    normalized = str(name)
    digest = sha1(normalized.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16) & 0x7FFFFFFF
    logger.debug("Generated deterministic seed=%s from name=%s", seed, normalized)
    return seed
