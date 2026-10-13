"""Base-owned complete outcome contract shared by collection and admission.

Canonical raw fields come from benchmark episode rows; summary fields come from
run_empty_world_sweep.summarize_row. Aliases are consistency claims, never
substitutes. Readers validate without rewriting sealed source objects.
"""

from __future__ import annotations

from typing import Any

from scripts.ci.receipt_provenance import HardwareIdentityError

# Every outcome interpreted by collection/classification, plus the executed-step
# count required by comparison, has mandatory paths in each producer format.
OUTCOME_PATHS = {
    "raw": {
        "success": (("metrics", "success"),),
        "collisions": (("metrics", "collisions"), ("metrics", "total_collision_count")),
        "steps": (("steps",),),
    },
    "summary": {
        "success": (("success",),),
        "collisions": (("collisions",),),
        "steps": (("steps",),),
    },
}
OUTCOME_ALIASES = {
    "success": (("success",), ("metrics", "success"), ("outcome", "route_complete")),
    "collisions": (
        ("collisions",),
        ("metrics", "collisions"),
        ("metrics", "total_collision_count"),
    ),
    "steps": (("steps",),),
}
_MISSING = object()


def _at(row: dict[str, Any], path: tuple[str, ...]) -> Any:
    value = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _refuse(reason: str) -> None:
    raise HardwareIdentityError(f"outcome schema refused: {reason}")


def _typed(value: Any, field: str, location: str) -> None:
    valid = (
        type(value) is bool
        if field == "success"
        else (type(value) is int and value >= (1 if field == "steps" else 0))
    )
    if not valid:
        _refuse(
            f"invalid {location}: expected {'boolean' if field == 'success' else 'positive integer' if field == 'steps' else 'nonnegative integer'}"
        )


def read_outcomes(row: dict[str, Any], *, form: str) -> dict[str, Any]:
    """Require every canonical field and validate all explicitly provided aliases.

    Returns:
        Complete validated outcome values keyed by their canonical field names.
    """
    values = {}
    for field, paths in OUTCOME_PATHS[form].items():
        location = f"{form} {'.'.join(paths[0])}"
        value = _MISSING
        for path in paths:
            required_location = f"{form} {'.'.join(path)}"
            supplied = _at(row, path)
            if supplied is _MISSING:
                _refuse(f"missing {required_location}")
            _typed(supplied, field, required_location)
            if value is _MISSING:
                value = supplied
            elif supplied != value:
                _refuse(f"contradictory {required_location} and {location}")
        for alias in OUTCOME_ALIASES[field]:
            supplied = _at(row, alias)
            if supplied is _MISSING or alias in paths:
                continue
            alias_location = f"{form} {'.'.join(alias)}"
            _typed(supplied, field, alias_location)
            if supplied != value:
                _refuse(f"contradictory {alias_location} and {location}")
        values[field] = value
    return values


def raw_outcomes(row: dict[str, Any]) -> dict[str, Any]:
    """Read canonical raw outcomes without defaults, coercion, or mutation.

    Returns:
        The required success, collision count and executed-step count.
    """
    return read_outcomes(row, form="raw")


def summary_outcomes(row: dict[str, Any]) -> dict[str, Any]:
    """Read the mandatory summary mirrors without substituting raw aliases.

    Returns:
        The required success, collision count and executed-step count.
    """
    return read_outcomes(row, form="summary")
