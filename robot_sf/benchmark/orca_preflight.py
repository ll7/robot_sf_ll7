"""Preflight guard that checks rvo2 importability when a campaign config includes ORCA planners.

Camera-ready benchmarks must not silently fall back to heuristic ORCA when rvo2 is
unavailable. This module provides a lightweight check that inspects a campaign config for
ORCA planners and fails fast with an actionable error before an expensive campaign runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

    from robot_sf.benchmark.camera_ready_campaign import CampaignConfig, PlannerSpec
from robot_sf.errors import RobotSfError

_SYNC_COMMAND_EXTRA = "uv sync --extra orca"
_SYNC_COMMAND_ALL = "uv sync --all-extras"


class OrcaRvo2PreflightError(RobotSfError, RuntimeError):
    """Typed ORCA preflight failure for library-facing campaign callers."""

    def __init__(
        self,
        message: str,
        *,
        planner_keys: tuple[str, ...] = (),
        error: Exception | None = None,
    ) -> None:
        """Store the actionable message plus optional planner/import context."""
        super().__init__(message)
        self.planner_keys = planner_keys
        self.error = error


def _missing_rvo2_message(
    *, planner_keys: list[str] | None = None, error: Exception | None = None
) -> str:
    """Return the actionable fail-fast message for a missing ORCA runtime."""
    planner_line = ""
    if planner_keys:
        planner_line = (
            f"Camera-ready benchmark config includes ORCA planner(s): {', '.join(planner_keys)}.\n"
        )
    error_line = ""
    if error is not None:
        error_line = f"Import failure: {type(error).__name__}: {error}\n"
    return (
        f"{planner_line}"
        "The required optional dependency 'rvo2' is not importable.\n"
        f"{error_line}"
        "Install it with:\n"
        f"  {_SYNC_COMMAND_EXTRA}\n"
        "  or\n"
        f"  {_SYNC_COMMAND_ALL}\n"
        "Aborting before starting the benchmark campaign."
    )


def _has_orca_algo(algo: str) -> bool:
    """Return True when the algo string identifies an ORCA-based planner."""
    return "orca" in str(algo).lower()


def _is_orca_dependent_planner(planner_spec: PlannerSpec) -> bool:
    """Return True when a planner spec represents an ORCA-dependent planner.

    A planner is ORCA-dependent when its algo field contains ``orca``
    (case-insensitive) and it is not disabled.

    Args:
        planner_spec: A planner spec object with ``algo`` and ``enabled`` attributes.

    Returns:
        bool: True when the planner is enabled and ORCA-dependent.
    """
    algo = planner_spec.algo
    enabled = planner_spec.enabled
    return enabled and _has_orca_algo(algo)


def _ensure_rvo2_importable(*, planner_keys: list[str] | None = None) -> None:
    """Raise a typed preflight error when ``rvo2`` is unavailable."""
    try:
        import rvo2  # type: ignore[import-untyped,unused-ignore]  # noqa: F401, PLC0415
    except (ImportError, OSError) as exc:
        message = _missing_rvo2_message(planner_keys=planner_keys, error=exc)
        logger.error(message)
        raise OrcaRvo2PreflightError(
            message,
            planner_keys=tuple(planner_keys or ()),
            error=exc,
        ) from None


def check_rvo2_importable() -> None:
    """Check whether ``rvo2`` can be imported in the current interpreter.

    Raises:
        OrcaRvo2PreflightError: When ``rvo2`` is not importable, with a message naming the
            missing dependency and the exact sync commands.
    """
    _ensure_rvo2_importable()


def check_orca_rvo2_preflight(cfg: CampaignConfig) -> None:
    """Validate that rvo2 is importable when a campaign config includes ORCA planners.

    When the config contains enabled ORCA planners and rvo2 is not importable, this
    function raises ``OrcaRvo2PreflightError`` with an actionable error message naming the
    ORCA planner keys, the missing dependency, and the exact sync commands.

    Args:
        cfg: A loaded camera-ready campaign config.

    Raises:
        OrcaRvo2PreflightError: When ORCA planners are present and ``rvo2`` is not importable.
    """
    orca_specs = [p for p in cfg.planners if _is_orca_dependent_planner(p)]
    if not orca_specs:
        logger.debug("No ORCA-dependent planners in config; rvo2 preflight skipped.")
        return

    orca_keys = sorted({p.key for p in orca_specs})
    orca_keys_str = ", ".join(orca_keys)
    logger.info(
        f"ORCA-dependent planner(s) detected: {orca_keys_str}. Checking rvo2 importability..."
    )

    _ensure_rvo2_importable(planner_keys=orca_keys)

    logger.info("rvo2 is importable; ORCA preflight passed.")


def check_orca_rvo2_preflight_from_config(config_path: Path) -> None:
    """Load a campaign config from *config_path* and run the ORCA-rvo2 preflight check.

    This entry point is suitable for scripting without importing the full
    campaign machinery in callers.

    Args:
        config_path: Path to a camera-ready campaign config YAML.

    Raises:
        OrcaRvo2PreflightError: When ORCA planners are present and ``rvo2`` is not importable.
    """
    from robot_sf.benchmark.camera_ready_campaign import (  # noqa: PLC0415
        load_campaign_config,
    )

    cfg = load_campaign_config(config_path)
    check_orca_rvo2_preflight(cfg)


__all__ = [
    "OrcaRvo2PreflightError",
    "check_orca_rvo2_preflight",
    "check_orca_rvo2_preflight_from_config",
    "check_rvo2_importable",
]
