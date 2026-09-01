"""Opt-in, dependency-light force-component diagnostics for PySocialForce.

The regular simulator path only needs the aggregate force.  This module defines
the diagnostic result used when a caller explicitly requests a typed force
decomposition.  It intentionally has no dependency on Robot SF or benchmark
types so the physics package can remain reusable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class ForceComponentOperation(StrEnum):
    """Semantic role of one force in the registered force list."""

    BASE_COMPONENT = "base_component"
    ADDITIVE_DELTA = "additive_delta"
    REPLACEMENT_TOTAL = "replacement_total"
    TRANSFORM_TOTAL = "transform_total"
    POST_PROCESSING_DELTA = "post_processing_delta"
    DEDICATED_INTEGRATOR = "dedicated_integrator"
    UNAVAILABLE = "unavailable"


def _json_safe(value: Any) -> Any:
    """Convert configuration values into deterministic JSON-compatible values.

    Returns:
        A JSON-compatible representation of ``value``.
    """
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def stable_config_hash(config: Any) -> str:
    """Return a stable SHA-256 hash for a force configuration object."""
    payload = json.dumps(_json_safe(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _snake_case(value: str) -> str:
    """Convert a class name to a stable lower-case identifier.

    Returns:
        The normalized identifier.
    """
    chars: list[str] = []
    for index, character in enumerate(value):
        if character.isupper() and index:
            chars.append("_")
        chars.append(character.lower())
    return "".join(chars)


_KNOWN_COMPONENT_TYPES = {
    "DesiredForce": "desired",
    "SocialForce": "social",
    "ObstacleForce": "obstacle",
    "GroupCoherenceForceAlt": "group_coherence",
    "GroupRepulsiveForce": "group_repulsive",
    "GroupGazeForceAlt": "group_gaze",
}


def annotate_force_component(
    force: Any,
    *,
    component_id: str,
    component_type: str,
    source_entity: str | None = None,
    actor_observable: bool = False,
) -> Any:
    """Attach explicit stable metadata to a force instance and return it.

    Robot-specific force instances are created outside the fast-pysf package.
    Keeping this tiny annotation helper here avoids coupling the low-level result
    type to Robot SF classes while still giving multiplicity a canonical identity.

    Returns:
        The annotated force instance.
    """
    force.component_id = component_id
    force.component_type = component_type
    force.source_entity = source_entity
    force.actor_observable = actor_observable
    return force


def _component_metadata(
    force: Any,
    evaluation_order: int,
    occurrence_counts: dict[str, int],
) -> dict[str, Any]:
    """Resolve deterministic metadata for one registered force object.

    Returns:
        Metadata used to construct a typed component result.
    """
    class_name = type(force).__name__
    component_type = str(
        getattr(
            force,
            "component_type",
            _KNOWN_COMPONENT_TYPES.get(class_name, _snake_case(class_name)),
        )
    )
    requested_id = getattr(force, "component_id", None)
    base_id = component_type if requested_id is None else str(requested_id)
    occurrence = occurrence_counts.get(base_id, 0)
    component_id = base_id if occurrence == 0 else f"{base_id}:{occurrence}"
    while component_id in occurrence_counts:
        occurrence += 1
        component_id = f"{base_id}:{occurrence}"
    occurrence_counts[base_id] = occurrence + 1
    occurrence_counts[component_id] = 1

    config = getattr(force, "config", {})
    return {
        "component_id": component_id,
        "component_type": component_type,
        "implementation_module": type(force).__module__,
        "implementation_class": type(force).__qualname__,
        "source_entity": getattr(force, "source_entity", None),
        "enabled": True,
        "config_hash": stable_config_hash(config),
        "evaluation_order": evaluation_order,
        "operation": getattr(
            force,
            "component_operation",
            ForceComponentOperation.BASE_COMPONENT,
        ),
        "actor_observable": bool(getattr(force, "actor_observable", False)),
    }


@dataclass(frozen=True, slots=True)
class ForceComponentResult:
    """One evaluated force component and its provenance."""

    component_id: str
    component_type: str
    implementation_module: str
    implementation_class: str
    source_entity: str | None
    values: np.ndarray | None
    enabled: bool
    config_hash: str
    evaluation_order: int
    operation: ForceComponentOperation = ForceComponentOperation.BASE_COMPONENT
    actor_observable: bool = False
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate and freeze the diagnostic array."""
        for field_name in (
            "component_id",
            "component_type",
            "implementation_module",
            "implementation_class",
            "config_hash",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be text")
            if not value.strip():
                raise ValueError(f"{field_name} must be non-empty")
        if self.source_entity is not None and not isinstance(self.source_entity, str):
            raise TypeError("source_entity must be text or None")
        if type(self.enabled) is not bool or type(self.actor_observable) is not bool:
            raise TypeError("enabled and actor_observable must be bool")
        if type(self.evaluation_order) is not int or self.evaluation_order < 0:
            raise ValueError("evaluation_order must be a non-negative integer")
        if not isinstance(self.operation, ForceComponentOperation):
            raise TypeError("operation must be ForceComponentOperation")
        if self.values is None:
            if self.operation is not ForceComponentOperation.UNAVAILABLE:
                raise ValueError("unavailable components must use operation=unavailable")
            if not isinstance(self.unavailable_reason, str) or not self.unavailable_reason.strip():
                raise ValueError("unavailable components must name unavailable_reason")
        else:
            values = np.asarray(self.values)
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError("force component values must have shape (N, 2)")
            if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
                raise ValueError("force component values must be finite")
            if self.unavailable_reason is not None:
                raise ValueError("available components must omit unavailable_reason")
            frozen_values = np.array(values, copy=True)
            frozen_values.setflags(write=False)
            object.__setattr__(self, "values", frozen_values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe component record."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "implementation_module": self.implementation_module,
            "implementation_class": self.implementation_class,
            "source_entity": self.source_entity,
            "values": self.values.tolist() if self.values is not None else None,
            "enabled": self.enabled,
            "config_hash": self.config_hash,
            "evaluation_order": self.evaluation_order,
            "operation": self.operation.value,
            "actor_observable": self.actor_observable,
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class ForceComputationResult:
    """All registered force outputs and the exact aggregate used by integration."""

    components: tuple[ForceComponentResult, ...]
    base_total: np.ndarray

    def __post_init__(self) -> None:
        """Validate roster order, shapes, and the exact component sum."""
        components = tuple(self.components)
        if any(type(component) is not ForceComponentResult for component in components):
            raise TypeError("components must contain ForceComponentResult values")
        ids = [component.component_id for component in components]
        if len(ids) != len(set(ids)):
            raise ValueError("force component identifiers must be unique")
        orders = [component.evaluation_order for component in components]
        if orders != list(range(len(components))):
            raise ValueError("force component evaluation_order must be contiguous")
        total = np.asarray(self.base_total)
        if total.ndim != 2 or total.shape[1] != 2:
            raise ValueError("base_total must have shape (N, 2)")
        if not np.issubdtype(total.dtype, np.number) or not np.all(np.isfinite(total)):
            raise ValueError("base_total must be finite")
        expected_shape = total.shape
        running = np.zeros_like(total)
        for component in components:
            if component.values is None:
                raise ValueError("force computation results cannot contain unavailable values")
            if component.values.shape != expected_shape:
                raise ValueError("force component shapes must match base_total")
            running += component.values
        if not np.array_equal(running, total):
            raise ValueError("base_total must equal the exact registered component sum")
        frozen_total = np.array(total, copy=True)
        frozen_total.setflags(write=False)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "base_total", frozen_total)

    @property
    def component_sum(self) -> np.ndarray:
        """Return a fresh aggregate recomputed from the typed component arrays."""
        total = np.zeros_like(self.base_total)
        for component in self.components:
            assert component.values is not None
            total += component.values
        return total

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe diagnostic result."""
        return {
            "components": [component.to_dict() for component in self.components],
            "base_total": self.base_total.tolist(),
        }


def compute_force_components(
    force_list: list[Any],
    ped_state: Any,
) -> ForceComputationResult:
    """Evaluate every registered force exactly once and return its exact sum.

    Returns:
        The ordered component results and exact aggregate used by integration.
    """
    occurrence_counts: dict[str, int] = {}
    components: list[ForceComponentResult] = []
    combined = np.zeros_like(np.asarray(ped_state.pos(), dtype=float))
    for evaluation_order, force in enumerate(force_list):
        values = np.asarray(force())
        metadata = _component_metadata(force, evaluation_order, occurrence_counts)
        component = ForceComponentResult(values=values, **metadata)
        components.append(component)
        assert component.values is not None
        combined += component.values
    return ForceComputationResult(tuple(components), combined)


__all__ = [
    "ForceComponentOperation",
    "ForceComponentResult",
    "ForceComputationResult",
    "annotate_force_component",
    "compute_force_components",
    "stable_config_hash",
]
