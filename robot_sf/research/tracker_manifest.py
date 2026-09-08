"""Shared validation for tracker manifests consumed by research workflows."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from robot_sf.research.exceptions import ValidationError

if TYPE_CHECKING:
    from pathlib import Path


def coerce_tracker_int(value: object, field: str) -> int:
    """Coerce a tracker field to an integer or raise a validation error.

    Returns:
        The coerced integer value.
    """
    try:
        return int(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValidationError(f"Tracker manifest {field} must contain an integer") from exc


def coerce_tracker_float(value: object, field: str) -> float:
    """Coerce a tracker field to a finite float or raise a validation error.

    Returns:
        The coerced finite float value.
    """
    try:
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValidationError(f"Tracker manifest {field} must contain a finite number") from exc
    if not math.isfinite(converted):
        raise ValidationError(f"Tracker manifest {field} must contain a finite number")
    return converted


def validate_tracker_payload(
    payload: object, path: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], list[Any], dict[str, Any], list[Any]]:
    """Validate and normalize the JSON shapes consumed by tracker readers.

    Returns:
        Tuple of the validated payload, step records, enabled-step identifiers,
        summary mapping, and seed values.
    """
    if not isinstance(payload, dict):
        raise ValidationError(f"Tracker manifest must contain a JSON object: {path}")

    raw_steps = payload.get("steps")
    if "steps" in payload and (raw_steps is None or not isinstance(raw_steps, list)):
        raise ValidationError(f"Tracker manifest steps must be a list: {path}")
    steps = raw_steps or []
    if any(not isinstance(step, dict) for step in steps):
        raise ValidationError(f"Tracker manifest steps must contain objects: {path}")

    raw_enabled_steps = payload.get("enabled_steps")
    if "enabled_steps" in payload and (
        raw_enabled_steps is None or not isinstance(raw_enabled_steps, list)
    ):
        raise ValidationError(f"Tracker manifest enabled_steps must be a list: {path}")
    enabled_steps = raw_enabled_steps or [
        step.get("step_id") for step in steps if step.get("step_id")
    ]

    raw_summary = payload.get("summary")
    if "summary" in payload and (raw_summary is None or not isinstance(raw_summary, dict)):
        raise ValidationError(f"Tracker manifest summary must be an object: {path}")
    summary = raw_summary or {}
    if "seeds" in summary and (summary["seeds"] is None or not isinstance(summary["seeds"], list)):
        raise ValidationError(f"Tracker manifest summary seeds must be a list: {path}")
    raw_metrics = payload.get("metrics")
    if "metrics" in payload and (raw_metrics is None or not isinstance(raw_metrics, dict)):
        raise ValidationError(f"Tracker manifest metrics must be an object: {path}")
    if "metrics" in summary and (
        summary["metrics"] is None or not isinstance(summary["metrics"], dict)
    ):
        raise ValidationError(f"Tracker manifest summary metrics must be an object: {path}")
    raw_seeds = payload.get("seeds")
    if "seeds" in payload and (raw_seeds is None or not isinstance(raw_seeds, list)):
        raise ValidationError(f"Tracker manifest seeds must be a list: {path}")
    raw_seeds = raw_seeds or summary.get("seeds", [])
    return payload, steps, enabled_steps, summary, raw_seeds
