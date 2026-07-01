"""AMMV benchmark protocol manifest loader and validator.

This module provides a typed loader and fail-closed validator for the
versioned AMMV benchmark protocol declared in ``benchmarks/ammv_benchmark_v0.yaml``.

The protocol is declarative: it declares scenario classes, planner panels,
metric layers, and claim rules.  It does **not** execute scenarios,
instantiate planners, or enforce CI/release gates.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.common.artifact_paths import get_repository_root

#: Default path to the AMMV benchmark protocol manifest, relative to the
#: repository root.
AMMV_BENCHMARK_PROTOCOL_PATH = Path("benchmarks/ammv_benchmark_v0.yaml")

#: Required top-level sections in the protocol manifest.
_REQUIRED_TOP_LEVEL = frozenset(
    {"scenario_classes", "planner_panel", "metric_layers", "claim_rules"}
)

#: Required keys inside ``claim_rules``.
_REQUIRED_CLAIM_RULES = frozenset(
    {"global_superiority_forbidden", "report_tradeoffs", "safety_gate_precedes_efficiency"},
)


class BenchmarkProtocolError(ValueError):
    """Raised when a benchmark protocol manifest is missing or invalid."""


@dataclass(frozen=True)
class ClaimRules:
    """Claim rules that constrain what a benchmark run may assert."""

    global_superiority_forbidden: bool
    report_tradeoffs: bool
    safety_gate_precedes_efficiency: bool


@dataclass(frozen=True)
class BenchmarkProtocolManifest:
    """Typed representation of a versioned benchmark protocol manifest."""

    protocol_id: str
    source_path: Path
    scenario_classes: tuple[str, ...]
    planner_panel: tuple[str, ...]
    metric_layers: tuple[str, ...]
    claim_rules: ClaimRules


def _protocol_id(source_path: Path) -> str:
    """Derive a protocol id from the manifest file stem.

    Returns:
        The file stem as a string.
    """
    return source_path.stem


def _ensure_mapping(payload: Any, *, label: str = "root") -> None:
    """Fail-closed: verify the value is a Mapping (dict-like)."""
    if not isinstance(payload, Mapping):
        raise BenchmarkProtocolError(
            f"{label}: expected a mapping, got {type(payload).__name__}",
        )


def _ensure_non_empty_string_list(
    value: Any,
    *,
    section: str,
) -> tuple[str, ...]:
    """Fail-closed: verify the value is a non-empty sequence of non-empty strings.

    Returns:
        A tuple of the validated strings.
    """
    if not isinstance(value, (list, tuple)):
        raise BenchmarkProtocolError(
            f"{section}: expected a list, got {type(value).__name__}",
        )
    if not value:
        raise BenchmarkProtocolError(f"{section}: must not be empty")
    result: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise BenchmarkProtocolError(
                f"{section}[{idx}]: expected a non-empty string, got {type(item).__name__}",
            )
        result.append(item.strip())
    return tuple(result)


def validate_benchmark_protocol_payload(
    payload: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> BenchmarkProtocolManifest:
    """Validate a parsed benchmark protocol payload and return a typed manifest.

    Args:
        payload: Parsed YAML content as a mapping.
        source_path: Path to the source file (used for protocol id derivation).

    Returns:
        A ``BenchmarkProtocolManifest`` with validated fields.

    Raises:
        BenchmarkProtocolError: When any required section or key is missing or
            has an invalid type.
    """
    _ensure_mapping(payload, label="root")

    missing = _REQUIRED_TOP_LEVEL - set(payload.keys())
    if missing:
        raise BenchmarkProtocolError(
            f"missing required top-level section(s): {', '.join(sorted(missing))}",
        )

    scenario_classes = _ensure_non_empty_string_list(
        payload.get("scenario_classes"),
        section="scenario_classes",
    )
    planner_panel = _ensure_non_empty_string_list(
        payload.get("planner_panel"),
        section="planner_panel",
    )
    metric_layers = _ensure_non_empty_string_list(
        payload.get("metric_layers"),
        section="metric_layers",
    )

    claim_rules_raw = payload.get("claim_rules")
    _ensure_mapping(claim_rules_raw, label="claim_rules")

    claim_keys = set(claim_rules_raw.keys())
    missing_rules = _REQUIRED_CLAIM_RULES - claim_keys
    if missing_rules:
        raise BenchmarkProtocolError(
            f"missing required claim rule(s): {', '.join(sorted(missing_rules))}",
        )

    claim_values: dict[str, bool] = {}
    for key in sorted(_REQUIRED_CLAIM_RULES):
        raw = claim_rules_raw[key]
        if not isinstance(raw, bool):
            raise BenchmarkProtocolError(
                f"claim_rules.{key}: expected bool, got {type(raw).__name__}",
            )
        claim_values[key] = raw

    resolved_path = source_path or AMMV_BENCHMARK_PROTOCOL_PATH
    protocol_id = _protocol_id(resolved_path)

    return BenchmarkProtocolManifest(
        protocol_id=protocol_id,
        source_path=resolved_path,
        scenario_classes=scenario_classes,
        planner_panel=planner_panel,
        metric_layers=metric_layers,
        claim_rules=ClaimRules(
            global_superiority_forbidden=claim_values["global_superiority_forbidden"],
            report_tradeoffs=claim_values["report_tradeoffs"],
            safety_gate_precedes_efficiency=claim_values["safety_gate_precedes_efficiency"],
        ),
    )


def load_benchmark_protocol(
    path: str | Path = AMMV_BENCHMARK_PROTOCOL_PATH,
) -> BenchmarkProtocolManifest:
    """Load and validate a benchmark protocol manifest from a YAML file.

    Args:
        path: Path to the protocol YAML file.  Relative paths are resolved
            against the repository root.

    Returns:
        A ``BenchmarkProtocolManifest`` with validated fields.

    Raises:
        BenchmarkProtocolError: When the file cannot be read, YAML is invalid,
            or validation fails.
    """
    import yaml  # noqa: PLC0415

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = get_repository_root() / resolved

    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise BenchmarkProtocolError(f"cannot read protocol file {resolved}: {exc}") from exc

    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise BenchmarkProtocolError(f"invalid YAML in {resolved}: {exc}") from exc

    return validate_benchmark_protocol_payload(payload, source_path=resolved)


__all__ = [
    "AMMV_BENCHMARK_PROTOCOL_PATH",
    "BenchmarkProtocolError",
    "BenchmarkProtocolManifest",
    "ClaimRules",
    "load_benchmark_protocol",
    "validate_benchmark_protocol_payload",
]
