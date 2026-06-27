"""Dry-run Robot SF -> external-benchmark scenario converter (intermediate representation).

This module implements the *agent-executable, local-only slice* of issue #3285: a
deterministic, schema-validated **intermediate representation (IR)** for Robot SF
benchmark scenarios that downstream SocNavBench / HuNavSim adapters can build on.

Scope and claim boundary
------------------------
- This is a **dry-run** converter. It reads an in-repo Robot SF scenario-matrix
  entry (see ``robot_sf/benchmark/schema/scenarios.schema.json``) and emits a
  target-neutral IR plus an explicit **unsupported-field report**.
- It requires **no external assets** and emits **no** SocNavBench or HuNavSim file.
  It makes **no** cross-benchmark validity or score-parity claim. Producing actual
  SocNavBench/HuNavSim assets is blocked on staged external assets
  (issues #1456 / #1498 / #2414 / #1134) and is intentionally out of scope here.

Determinism contract
---------------------
For a given scenario dict the IR is a pure function of the input: section order is
fixed by construction, ``provenance.source_fields`` is sorted, and
``unsupported_fields`` is sorted by field name. ``convert_scenario_to_ir`` therefore
returns byte-identical JSON across runs for identical input.

Unsupported-field contract
---------------------------
Every top-level source key is classified exactly once. Recognized keys are mapped
into the IR; every other key is recorded in ``unsupported_fields`` with a reason
(either a known simulator-specific field or an unrecognized field). Nothing is
silently dropped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.common.json_pointer import json_pointer

IR_SCHEMA_VERSION = "robot_sf.scenario_interop_ir.v1"
CONVERTER_NAME = "robot_sf.scenario_interop"
CONVERTER_VERSION = "1"
IR_SCHEMA_FILE = Path(__file__).with_name("schemas") / "scenario_interop_ir.v1.json"

# Source keys that map directly into IR sections. Anything not in this set is
# reported in ``unsupported_fields`` rather than silently dropped.
_ID_KEYS: tuple[str, ...] = ("id", "scenario_id", "name")
_ENVIRONMENT_KEYS: tuple[str, ...] = (
    "density",
    "flow",
    "groups",
    "speed_var",
    "goal_topology",
    "robot_context",
)
_GEOMETRY_KEYS: tuple[str, ...] = ("obstacle", "map_file")
_TIMING_KEYS: tuple[str, ...] = ("repeats", "seeds")
_AGENT_KEYS: tuple[str, ...] = ("single_pedestrians",)
_METADATA_KEYS: tuple[str, ...] = ("metadata",)

# Keys that are recognized but intentionally have no target-neutral IR mapping.
# Each is reported with a reason so the omission is explicit, not silent.
_KNOWN_UNSUPPORTED: dict[str, str] = {
    "simulation_config": (
        "Robot SF simulator configuration has no target-neutral scenario "
        "representation in the interop IR"
    ),
    "robot_config": (
        "robot-platform configuration is simulator-specific and has no "
        "target-neutral representation in the interop IR"
    ),
    "route_overrides_file": (
        "external route-override file reference is not resolved by the dry-run converter"
    ),
    "amv": "automated-mobility-vehicle extension has no SocNavBench/HuNavSim analogue",
    "multi_amv": "multi-AMV extension has no SocNavBench/HuNavSim analogue",
    "supported": ("internal benchmark gating flag, not a transferable scenario-geometry field"),
}

_UNRECOGNIZED_REASON = "no intermediate-representation mapping is defined for this field"

# Obstacle topology -> coarse, target-neutral environment-type label.
_ENVIRONMENT_TYPE_BY_OBSTACLE: dict[str, str] = {
    "open": "open_space",
    "bottleneck": "constrained_passage",
    "maze": "cluttered",
}


@dataclass(frozen=True, slots=True)
class ScenarioInteropResult:
    """Result of a dry-run scenario conversion.

    Attributes:
        ir: The target-neutral intermediate representation (JSON-safe primitives).
        unsupported_fields: Explicit report of source fields that were not mapped
            into the IR, each as ``{"field": str, "reason": str}``.
        schema_errors: JSON-Schema validation errors for the emitted IR. Empty when
            the IR is valid.
    """

    ir: dict[str, Any]
    unsupported_fields: list[dict[str, str]]
    schema_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when the emitted IR satisfies its JSON Schema."""

        return not self.schema_errors


@lru_cache(maxsize=1)
def load_interop_ir_schema() -> dict[str, Any]:
    """Load the ``scenario_interop_ir.v1`` JSON Schema from disk.

    Returns:
        Parsed JSON Schema dictionary.
    """

    return json.loads(IR_SCHEMA_FILE.read_text(encoding="utf-8"))


def validate_interop_ir(ir: Mapping[str, Any]) -> list[str]:
    """Validate an IR payload against the interop IR schema.

    Args:
        ir: Candidate intermediate representation.

    Returns:
        Sorted JSON-pointer-prefixed validation error strings; empty when valid.
    """

    validator = Draft202012Validator(load_interop_ir_schema())
    return [
        f"{json_pointer(error.absolute_path)}: {error.message}"
        for error in sorted(validator.iter_errors(ir), key=lambda err: list(err.absolute_path))
    ]


def convert_scenario_to_ir(
    scenario: Mapping[str, Any],
    *,
    source_file: str | None = None,
) -> ScenarioInteropResult:
    """Convert a Robot SF scenario-matrix entry into the dry-run interop IR.

    The conversion is deterministic: for identical input the returned IR is
    byte-identical when serialized with stable key handling (see
    :func:`dump_ir`).

    Args:
        scenario: A single Robot SF scenario dict (one scenario-matrix entry).
        source_file: Optional path the scenario was loaded from, recorded in
            provenance for traceability.

    Returns:
        A :class:`ScenarioInteropResult` with the IR, the unsupported-field
        report, and any IR schema-validation errors.

    Raises:
        TypeError: If ``scenario`` is not a mapping.
        ValueError: If the scenario has no usable identifier field.
    """

    if not isinstance(scenario, Mapping):
        raise TypeError(f"scenario must be a mapping, got {type(scenario).__name__}")

    source_id = _resolve_source_id(scenario)
    source_kind = "explicit_map" if "map_file" in scenario else "axis"

    unsupported = _collect_unsupported(scenario)

    ir: dict[str, Any] = {
        "schema_version": IR_SCHEMA_VERSION,
        "provenance": {
            "source_scenario_id": source_id,
            "source_kind": source_kind,
            "source_file": source_file,
            "converter": CONVERTER_NAME,
            "converter_version": CONVERTER_VERSION,
            "source_fields": sorted(str(key) for key in scenario),
        },
        "geometry": _build_geometry(scenario),
        "environment": _build_environment(scenario),
        "agents": _build_agents(scenario),
        "timing": _build_timing(scenario),
        "unsupported_fields": unsupported,
    }

    metadata = scenario.get("metadata")
    if isinstance(metadata, Mapping):
        ir["provenance"]["source_metadata"] = dict(metadata)

    schema_errors = validate_interop_ir(ir)
    return ScenarioInteropResult(
        ir=ir,
        unsupported_fields=unsupported,
        schema_errors=schema_errors,
    )


def dump_ir(ir: Mapping[str, Any]) -> str:
    """Serialize an IR to a deterministic, human-readable JSON string.

    Args:
        ir: Intermediate representation produced by :func:`convert_scenario_to_ir`.

    Returns:
        Pretty-printed JSON ending with a trailing newline. Section/key order is
        preserved from construction so output is stable across runs.
    """

    return json.dumps(ir, indent=2, ensure_ascii=False) + "\n"


def _resolve_source_id(scenario: Mapping[str, Any]) -> str:
    """Resolve the source scenario identifier with a fixed precedence.

    Returns:
        The first non-empty string among ``id``, ``scenario_id``, ``name``.

    Raises:
        ValueError: When no usable identifier is present.
    """

    for key in _ID_KEYS:
        value = scenario.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(
        f"scenario is missing a usable identifier (expected one of {', '.join(_ID_KEYS)})"
    )


def _build_geometry(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Build the IR geometry section from obstacle topology and map reference.

    Returns:
        Geometry section dict.
    """

    obstacle = scenario.get("obstacle")
    obstacle_topology = obstacle if isinstance(obstacle, str) else None
    environment_type = (
        _ENVIRONMENT_TYPE_BY_OBSTACLE.get(obstacle_topology)
        if obstacle_topology is not None
        else None
    )
    map_file = scenario.get("map_file")
    return {
        "environment_type": environment_type,
        "obstacle_topology": obstacle_topology,
        "map_file": map_file if isinstance(map_file, str) else None,
    }


def _build_environment(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Build the IR environment-semantics section.

    Returns:
        Environment section dict with ``None`` for absent fields.
    """

    groups = scenario.get("groups")
    return {
        "density": _opt_str(scenario.get("density")),
        "flow": _opt_str(scenario.get("flow")),
        "groups_fraction": float(groups) if isinstance(groups, (int, float)) else None,
        "speed_variation": _opt_str(scenario.get("speed_var")),
        "goal_topology": _opt_str(scenario.get("goal_topology")),
        "robot_context": _opt_str(scenario.get("robot_context")),
    }


def _build_agents(scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the IR agent list from authored single-pedestrian specs.

    Source order is preserved; agents without an explicit ``id`` are assigned a
    deterministic positional id (``agent_<index>``).

    Returns:
        List of agent dicts. Empty when no single-pedestrian specs are present.
    """

    raw_agents = scenario.get("single_pedestrians")
    if not isinstance(raw_agents, Sequence) or isinstance(raw_agents, (str, bytes)):
        return []

    agents: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_agents):
        if not isinstance(raw, Mapping):
            continue
        raw_id = raw.get("id")
        agent_id = raw_id if isinstance(raw_id, str) and raw_id else f"agent_{index}"
        speed = raw.get("speed_m_s")
        wait_at = raw.get("wait_at")
        agents.append(
            {
                "id": agent_id,
                "start": _opt_str(raw.get("start_poi")),
                "goal": _opt_str(raw.get("goal_poi")),
                "preferred_speed_mps": (float(speed) if isinstance(speed, (int, float)) else None),
                "role": _opt_str(raw.get("role")),
                "role_target_id": _opt_str(raw.get("role_target_id")),
                "wait_points": list(wait_at)
                if isinstance(wait_at, Sequence) and not isinstance(wait_at, (str, bytes))
                else None,
            }
        )
    return agents


def _build_timing(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Build the IR timing section from repeats and seeds.

    Returns:
        Timing section dict with ``None`` for absent fields.
    """

    repeats = scenario.get("repeats")
    seeds = scenario.get("seeds")
    return {
        "repeats": int(repeats) if isinstance(repeats, int) else None,
        "seeds": (
            [int(seed) for seed in seeds if isinstance(seed, int)]
            if isinstance(seeds, Sequence) and not isinstance(seeds, (str, bytes))
            else None
        ),
    }


def _collect_unsupported(scenario: Mapping[str, Any]) -> list[dict[str, str]]:
    """Classify every unmapped top-level key as an explicit unsupported field.

    Returns:
        Unsupported-field reports sorted by field name for determinism.
    """

    mapped = set(
        _ID_KEYS + _ENVIRONMENT_KEYS + _GEOMETRY_KEYS + _TIMING_KEYS + _AGENT_KEYS + _METADATA_KEYS
    )
    reports: list[dict[str, str]] = []
    for key in scenario:
        name = str(key)
        if name in mapped:
            continue
        reason = _KNOWN_UNSUPPORTED.get(name, _UNRECOGNIZED_REASON)
        reports.append({"field": name, "reason": reason})
    return sorted(reports, key=lambda item: item["field"])


def _opt_str(value: Any) -> str | None:
    """Return ``value`` when it is a non-empty string, else ``None``.

    Returns:
        Normalized optional string.
    """

    return value if isinstance(value, str) and value else None


__all__ = [
    "CONVERTER_NAME",
    "CONVERTER_VERSION",
    "IR_SCHEMA_FILE",
    "IR_SCHEMA_VERSION",
    "ScenarioInteropResult",
    "convert_scenario_to_ir",
    "dump_ir",
    "load_interop_ir_schema",
    "validate_interop_ir",
]
