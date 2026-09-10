"""Preparation-only relevance-window selection for retained parent traces.

The selector is intentionally deterministic and conservative.  It creates an
excerpt proposal over complete parent rows; it does not remove actors, change
simulation physics, or turn a short excerpt into benchmark evidence.  Missing
signals remain explicit unknowns rather than benign zeroes, and precursor rows
are retained (or a manually shortened manifest is rejected).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

RELEVANCE_SCHEMA = "scenario_relevance_windows.v1"
MANIFEST_STATUS = "proposal_only"
EVIDENCE_GRADE = "synthetic_preparation"

_SIGNAL_SPECS: dict[str, tuple[str, str, str]] = {
    "clearance_m": ("m", "trace.geometry", "measured robot/actor clearance"),
    "closing_velocity_m_s": ("m/s", "trace.kinematics", "relative closing velocity"),
    "ttc_s": ("s", "trace.derived", "prediction model and response mode must be declared"),
    "closest_approach_m": ("m", "trace.derived", "observed or predicted closest approach"),
    "braking_margin_m": ("m", "trace.derived", "declared braking proxy, not a safety certificate"),
    "visibility_latency_s": ("s", "trace.observation", "first-observation or visibility latency"),
    "path_conflict": ("bool", "trace.geometry", "declared path-conflict region"),
    "fallback_or_saturation": (
        "bool",
        "trace.controller",
        "fallback, constraint, or actuator saturation transition",
    ),
    "progress_m": ("m", "trace.metrics", "robot progress along the declared route"),
    "stall_s": ("s", "trace.metrics", "progress-stall duration"),
    "discomfort": ("unitless", "trace.metrics", "declared discomfort exposure"),
    "collision": ("bool", "trace.events", "event label from the retained parent trace"),
}

_LOWER_IS_RISK = frozenset({"clearance_m", "ttc_s", "closest_approach_m", "braking_margin_m"})
_UPPER_IS_RISK = frozenset(
    {
        "closing_velocity_m_s",
        "visibility_latency_s",
        "stall_s",
        "discomfort",
    }
)


class RelevanceContractError(ValueError):
    """Base error for malformed or unsafe relevance-window proposals."""


class ExcerptContractError(RelevanceContractError):
    """Raised when a manifest cannot be linked to the complete parent rows."""


def _require_digest(value: str, name: str) -> str:
    """Normalize one lowercase SHA-256 digest.

    Returns:
        The normalized digest.
    """
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise RelevanceContractError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _finite(value: Any, name: str) -> float:
    """Return a finite float or fail closed."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RelevanceContractError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise RelevanceContractError(f"{name} must be finite")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    """Return a JSON boolean without coercing truthy strings or numbers."""
    if not isinstance(value, bool):
        raise RelevanceContractError(f"{name} must be boolean")
    return value


def _event_id(value: Any, name: str) -> str | None:
    """Normalize an optional event identity and reject ambiguous text.

    Returns:
        The stripped event identity, or ``None`` when no event is declared.
    """
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RelevanceContractError(f"{name} must be a non-empty string or null")
    return value.strip()


@dataclass(frozen=True, slots=True)
class RelevanceSignal:
    """One typed signal with units, provenance, availability, and hindsight."""

    name: str
    value: float | bool | None
    unit: str
    provenance: str
    available_at_step: int | None
    prediction_assumptions: str
    missingness: str = "observed"
    hindsight: bool = False

    def __post_init__(self) -> None:
        """Validate signal value and explicit missingness semantics."""
        if not str(self.name).strip() or not str(self.unit).strip():
            raise RelevanceContractError("signal name and unit must be non-empty")
        if self.missingness not in {"observed", "not_available", "invalid"}:
            raise RelevanceContractError(f"unsupported signal missingness {self.missingness!r}")
        if self.value is None and self.missingness == "observed":
            raise RelevanceContractError(f"missing signal {self.name!r} cannot be observed")
        if self.value is not None and self.missingness != "observed":
            raise RelevanceContractError(f"signal {self.name!r} has a value and missingness")
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise RelevanceContractError(f"signal {self.name!r} contains a non-finite value")
        if self.available_at_step is not None and (
            not isinstance(self.available_at_step, int) or self.available_at_step < 0
        ):
            raise RelevanceContractError("signal available_at_step must be a non-negative integer")
        _strict_bool(self.hindsight, "signal hindsight")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe signal record."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "provenance": self.provenance,
            "available_at_step": self.available_at_step,
            "prediction_assumptions": self.prediction_assumptions,
            "missingness": self.missingness,
            "hindsight": self.hindsight,
        }


@dataclass(frozen=True, slots=True)
class RelevanceThresholds:
    """Proposed numeric threshold and interval rules for offline preparation."""

    clearance_m: float = 1.0
    closing_velocity_m_s: float = 0.2
    ttc_s: float = 3.0
    closest_approach_m: float = 1.0
    braking_margin_m: float = 0.0
    visibility_latency_s: float = 0.5
    stall_s: float = 2.0
    discomfort: float = 0.5
    pre_roll_steps: int = 2
    post_roll_steps: int = 2
    merge_gap_steps: int = 1
    hysteresis_steps: int = 1
    approval_status: str = "proposed"
    approved_thresholds: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate proposed threshold magnitudes and approval boundary."""
        for name in (
            "clearance_m",
            "closing_velocity_m_s",
            "ttc_s",
            "closest_approach_m",
            "visibility_latency_s",
            "stall_s",
            "discomfort",
        ):
            if _finite(getattr(self, name), name) < 0.0:
                raise RelevanceContractError(f"{name} must be >= 0")
        _finite(self.braking_margin_m, "braking_margin_m")
        for name in ("pre_roll_steps", "post_roll_steps", "merge_gap_steps", "hysteresis_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise RelevanceContractError(f"{name} must be a non-negative integer")
        if not isinstance(self.approval_status, str) or self.approval_status not in {
            "proposed",
            "approved",
        }:
            raise RelevanceContractError("approval_status must be proposed or approved")
        if self.approved_thresholds is not None and not isinstance(
            self.approved_thresholds, Mapping
        ):
            raise RelevanceContractError("approved_thresholds must be an object or null")
        if self.approval_status == "approved" and not self.approved_thresholds:
            raise RelevanceContractError(
                "approved approval_status requires a nonempty approved_thresholds object"
            )

    def numeric_proposed(self) -> dict[str, float]:
        """Return only the proposed numeric signal thresholds."""
        return {
            name: float(getattr(self, name))
            for name in (
                "clearance_m",
                "closing_velocity_m_s",
                "ttc_s",
                "closest_approach_m",
                "braking_margin_m",
                "visibility_latency_s",
                "stall_s",
                "discomfort",
            )
        }

    def to_dict(self) -> dict[str, Any]:
        """Return proposed and approved rules in separate fields."""
        return {
            "approval_status": self.approval_status,
            "proposed": {
                **self.numeric_proposed(),
                "pre_roll_steps": self.pre_roll_steps,
                "post_roll_steps": self.post_roll_steps,
                "merge_gap_steps": self.merge_gap_steps,
                "hysteresis_steps": self.hysteresis_steps,
            },
            "approved": None
            if self.approved_thresholds is None
            else deepcopy(dict(self.approved_thresholds)),
        }


@dataclass(frozen=True, slots=True)
class RelevanceVector:
    """All candidate signals and threshold decisions at one retained row."""

    step: int
    time_s: float
    actor_ids: tuple[str, ...]
    signals: tuple[RelevanceSignal, ...]
    active_reasons: tuple[str, ...]
    unknown_signals: tuple[str, ...]
    precursor: bool = False
    event_id: str | None = None
    hindsight: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe vector record."""
        return {
            "step": self.step,
            "time_s": self.time_s,
            "actor_ids": list(self.actor_ids),
            "signals": [signal.to_dict() for signal in self.signals],
            "active_reasons": list(self.active_reasons),
            "unknown_signals": list(self.unknown_signals),
            "precursor": self.precursor,
            "event_id": self.event_id,
            "hindsight": self.hindsight,
        }


@dataclass(frozen=True, slots=True)
class RelevanceWindow:
    """One expanded, possibly merged interval in the parent trace."""

    start_step: int
    end_step: int
    row_indices: tuple[int, ...]
    original_step_indices: tuple[int, ...]
    trigger_steps: tuple[int, ...]
    precursor_steps: tuple[int, ...]
    reasons: tuple[str, ...]
    actor_ids: tuple[str, ...]
    missing_signals: tuple[str, ...]
    hindsight: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe interval record."""
        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "row_indices": list(self.row_indices),
            "original_step_indices": list(self.original_step_indices),
            "trigger_steps": list(self.trigger_steps),
            "precursor_steps": list(self.precursor_steps),
            "reasons": list(self.reasons),
            "actor_ids": list(self.actor_ids),
            "missing_signals": list(self.missing_signals),
            "hindsight": self.hindsight,
        }


@dataclass(frozen=True, slots=True)
class ExcerptManifest:
    """Immutable proposal linking selected intervals to complete parent rows."""

    parent_digest: str
    parent_rows_sha256: str
    source_row_count: int
    actor_ids: tuple[str, ...]
    selected_step_indices: tuple[int, ...]
    windows: tuple[RelevanceWindow, ...]
    selector_config: Mapping[str, Any]
    missing_signals: tuple[str, ...]
    hindsight: bool
    required_precursor_steps: tuple[int, ...] = ()
    full_parent_retained: bool = True
    status: str = MANIFEST_STATUS
    evidence_grade: str = EVIDENCE_GRADE

    def __post_init__(self) -> None:
        """Validate manifest identity and retention guarantees."""
        object.__setattr__(
            self, "parent_digest", _require_digest(self.parent_digest, "parent_digest")
        )
        object.__setattr__(
            self,
            "parent_rows_sha256",
            _require_digest(self.parent_rows_sha256, "parent_rows_sha256"),
        )
        if not isinstance(self.source_row_count, int) or self.source_row_count < 0:
            raise RelevanceContractError("source_row_count must be a non-negative integer")
        if not isinstance(self.selector_config, Mapping):
            raise RelevanceContractError("selector_config must be an object")
        object.__setattr__(
            self, "selector_config", MappingProxyType(deepcopy(dict(self.selector_config)))
        )
        if tuple(sorted(self.selected_step_indices)) != self.selected_step_indices:
            raise RelevanceContractError("selected_step_indices must be sorted")
        if len(set(self.selected_step_indices)) != len(self.selected_step_indices):
            raise RelevanceContractError("selected_step_indices must not contain duplicates")
        if not self.full_parent_retained:
            raise RelevanceContractError("full parent rows must remain retained")
        if self.status != MANIFEST_STATUS:
            raise RelevanceContractError("only proposal_only manifests are admitted in preparation")

    def to_dict(self) -> dict[str, Any]:
        """Return a machine-readable immutable-manifest payload."""
        return {
            "schema_version": RELEVANCE_SCHEMA,
            "parent_digest": self.parent_digest,
            "parent_rows_sha256": self.parent_rows_sha256,
            "source_row_count": self.source_row_count,
            "actor_ids": list(self.actor_ids),
            "selected_step_indices": list(self.selected_step_indices),
            "windows": [window.to_dict() for window in self.windows],
            "selector_config": deepcopy(dict(self.selector_config)),
            "missing_signals": list(self.missing_signals),
            "hindsight": self.hindsight,
            "required_precursor_steps": list(self.required_precursor_steps),
            "full_parent_retained": self.full_parent_retained,
            "status": self.status,
            "evidence_grade": self.evidence_grade,
        }


@dataclass(frozen=True, slots=True)
class RelevanceSelection:
    """Vectors, windows, and a copied complete parent for one proposal."""

    parent_rows: tuple[Mapping[str, Any], ...]
    vectors: tuple[RelevanceVector, ...]
    windows: tuple[RelevanceWindow, ...]
    manifest: ExcerptManifest

    def __post_init__(self) -> None:
        """Copy parent rows so later caller mutation cannot alter the proposal."""
        copied = tuple(deepcopy(dict(row)) for row in self.parent_rows)
        object.__setattr__(self, "parent_rows", copied)

    @property
    def selected_rows(self) -> tuple[Mapping[str, Any], ...]:
        """Return copied parent rows referenced by the selected step indices."""
        selected = set(self.manifest.selected_step_indices)
        return tuple(deepcopy(row) for row in self.parent_rows if int(row["step"]) in selected)

    def to_dict(self) -> dict[str, Any]:
        """Return manifest and typed vectors without dropping the parent."""
        return {
            "manifest": self.manifest.to_dict(),
            "vectors": [vector.to_dict() for vector in self.vectors],
            "parent_rows": [deepcopy(dict(row)) for row in self.parent_rows],
        }


def compute_parent_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash complete parent rows using canonical JSON encoding.

    Returns:
        Lowercase SHA-256 digest of the ordered parent rows.
    """
    try:
        encoded = json.dumps(
            [dict(row) for row in rows],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RelevanceContractError(f"parent rows are not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _normalize_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy and validate ordered parent rows.

    Returns:
        Owned, validated row dictionaries.
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise RelevanceContractError("parent rows must be a sequence of mappings")
    normalized: list[dict[str, Any]] = []
    previous_step: int | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise RelevanceContractError(f"parent row {index} must be an object")
        copied = deepcopy(dict(row))
        step = copied.get("step")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise RelevanceContractError(f"parent row {index}.step must be a non-negative integer")
        if previous_step is not None and step <= previous_step:
            raise RelevanceContractError("parent rows must have strictly increasing step values")
        _strict_bool(copied.get("precursor", False), f"parent row {index}.precursor")
        _strict_bool(copied.get("hindsight", False), f"parent row {index}.hindsight")
        _event_id(copied.get("event_id"), f"parent row {index}.event_id")
        previous_step = step
        normalized.append(copied)
    return normalized


def _actor_ids(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return stable actor IDs declared by one row."""
    value = row.get("actor_ids", row.get("actors", ()))
    if isinstance(value, Mapping):
        values = value.keys()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = value
    elif value in (None, ""):
        values = ()
    else:
        values = (value,)
    normalized: set[str] = set()
    for item in values:
        if isinstance(item, bool):
            raise RelevanceContractError("actor IDs must not be boolean values")
        text = str(item).strip()
        if not text:
            raise RelevanceContractError("actor IDs must be non-empty")
        normalized.add(text)
    return tuple(sorted(normalized))


def _row_signal_value(row: Mapping[str, Any], name: str) -> tuple[Any, Mapping[str, Any]]:
    """Extract a signal value plus optional row-provided metadata.

    Returns:
        ``(value, metadata)``; absent values are represented by ``None``.
    """
    signals = row.get("signals")
    raw: Any = None
    metadata: Mapping[str, Any] = {}
    found = False
    if isinstance(signals, Mapping) and name in signals:
        raw = signals[name]
        found = True
    elif name in row:
        raw = row[name]
        found = True
    if isinstance(raw, Mapping) and "value" in raw:
        metadata = raw
        raw = raw["value"]
    if not found or name in set(row.get("missing_signals", ()) or ()):
        return None, metadata
    return raw, metadata


def _signal_from_row(row: Mapping[str, Any], name: str) -> RelevanceSignal:
    """Build a typed signal, preserving invalid and unavailable states.

    Returns:
        A typed signal record.
    """
    unit, provenance, assumptions = _SIGNAL_SPECS[name]
    raw, metadata = _row_signal_value(row, name)
    row_metadata = row.get("signal_metadata")
    if isinstance(row_metadata, Mapping) and isinstance(row_metadata.get(name), Mapping):
        metadata = {**metadata, **row_metadata[name]}
    if raw is None:
        missingness = "not_available"
        value: float | bool | None = None
    elif unit == "bool":
        if not isinstance(raw, bool):
            missingness = "invalid"
            value = None
        else:
            missingness = "observed"
            value = raw
    else:
        try:
            value = _finite(raw, f"signals.{name}")
        except RelevanceContractError:
            missingness = "invalid"
            value = None
        else:
            missingness = "observed"
    availability = metadata.get("available_at_step", row.get("available_at_step"))
    if availability is not None:
        if not isinstance(availability, int) or isinstance(availability, bool):
            raise RelevanceContractError(f"signals.{name}.available_at_step must be integer")
        # A retained row may carry a value computed later in the parent trace.
        # It must remain unknown until the declared availability step; otherwise
        # offline selection can use future information as if it were causal.
        if availability > int(row["step"]):
            raw = None
            missingness = "not_available"
            value = None
    signal_hindsight = metadata.get("hindsight", row.get("hindsight", False))
    _strict_bool(signal_hindsight, f"signals.{name}.hindsight")
    return RelevanceSignal(
        name=name,
        value=value,
        unit=str(metadata.get("unit", unit)),
        provenance=str(metadata.get("provenance", provenance)),
        available_at_step=availability,
        prediction_assumptions=str(metadata.get("prediction_assumptions", assumptions)),
        missingness=missingness,
        hindsight=signal_hindsight,
    )


def _threshold(thresholds: RelevanceThresholds, name: str) -> float:
    """Get the numeric threshold for a risk-bearing signal.

    Returns:
        The proposed numeric threshold.
    """
    return float(getattr(thresholds, name))


def _on_trigger(name: str, value: float | bool, thresholds: RelevanceThresholds) -> bool:
    """Return whether one observed signal crosses its proposed threshold."""
    if name in {"path_conflict", "fallback_or_saturation", "collision"}:
        return bool(value)
    if name not in _LOWER_IS_RISK and name not in _UPPER_IS_RISK:
        return False
    numeric = float(value)
    limit = _threshold(thresholds, name)
    return numeric <= limit if name in _LOWER_IS_RISK else numeric >= limit


def _vector_for_row(
    row: Mapping[str, Any],
    thresholds: RelevanceThresholds,
    active_reasons: set[str],
    safe_counts: dict[str, int],
) -> RelevanceVector:
    """Evaluate one row while carrying threshold hysteresis state.

    Returns:
        A vector containing signals, active reasons, and unknown fields.
    """
    signals = tuple(_signal_from_row(row, name) for name in _SIGNAL_SPECS)
    unknown = {signal.name for signal in signals if signal.missingness != "observed"}
    current: set[str] = set()
    for signal in signals:
        name = signal.name
        if signal.missingness != "observed":
            if name in active_reasons:
                current.add(name)
            safe_counts[name] = 0
            continue
        triggered = _on_trigger(name, signal.value, thresholds)  # type: ignore[arg-type]
        if triggered:
            current.add(name)
            safe_counts[name] = 0
        elif name in active_reasons:
            count = safe_counts.get(name, 0) + 1
            safe_counts[name] = count
            if count <= thresholds.hysteresis_steps:
                current.add(name)
    active_reasons.clear()
    active_reasons.update(current)
    time_s = _finite(row.get("time_s", row["step"]), f"row[{row['step']}].time_s")
    event_id = _event_id(row.get("event_id"), f"row[{row['step']}].event_id")
    row_hindsight = _strict_bool(row.get("hindsight", False), f"row[{row['step']}].hindsight")
    row_precursor = _strict_bool(row.get("precursor", False), f"row[{row['step']}].precursor")
    return RelevanceVector(
        step=int(row["step"]),
        time_s=time_s,
        actor_ids=_actor_ids(row),
        signals=signals,
        active_reasons=tuple(sorted(current)),
        unknown_signals=tuple(sorted(unknown)),
        precursor=row_precursor,
        event_id=event_id,
        hindsight=row_hindsight or any(signal.hindsight for signal in signals),
    )


def _required_precursor_indices(
    rows: Sequence[Mapping[str, Any]], trigger_indices: Sequence[int]
) -> set[int]:
    """Find earlier explicitly marked precursor rows for triggered event IDs.

    Returns:
        Source-row indices that must remain in a safe excerpt.
    """
    required: set[int] = set()
    for trigger_index in trigger_indices:
        event_id = _event_id(
            rows[trigger_index].get("event_id"),
            f"row[{rows[trigger_index]['step']}].event_id",
        )
        if event_id is None:
            continue
        required.update(
            index
            for index, row in enumerate(rows[: trigger_index + 1])
            if _event_id(row.get("event_id"), f"row[{row['step']}].event_id") == event_id
            and _strict_bool(row.get("precursor", False), f"row[{row['step']}].precursor")
        )
    return required


def _build_windows(
    rows: Sequence[Mapping[str, Any]],
    vectors: Sequence[RelevanceVector],
    thresholds: RelevanceThresholds,
) -> tuple[tuple[RelevanceWindow, ...], set[int]]:
    """Build expanded/merged windows and include declared precursors.

    Returns:
        ``(windows, required_precursor_row_indices)``.
    """
    _validate_precursor_rows(rows)
    trigger_indices = [index for index, vector in enumerate(vectors) if vector.active_reasons]
    if not trigger_indices:
        return (), set()
    runs: list[list[int]] = [[trigger_indices[0]]]
    for index in trigger_indices[1:]:
        if index - runs[-1][-1] - 1 <= thresholds.merge_gap_steps:
            runs[-1].append(index)
        else:
            runs.append([index])
    required_precursors = _required_precursor_indices(rows, trigger_indices)
    expanded: list[tuple[int, int, list[int], set[str], set[int]]] = []
    for run in runs:
        start = max(0, run[0] - thresholds.pre_roll_steps)
        end = min(len(rows) - 1, run[-1] + thresholds.post_roll_steps)
        run_event_ids = _run_event_ids(rows, run)
        relevant_precursors = _precursors_for_run(rows, required_precursors, run, run_event_ids)
        if relevant_precursors:
            start = min([start, *relevant_precursors])
        expanded.append((start, end, run, run_event_ids, set(relevant_precursors)))
    merged: list[tuple[int, int, list[int], set[str], set[int]]] = []
    for start, end, run, event_ids, precursor_indices in expanded:
        same_event = bool(event_ids) and bool(merged) and bool(event_ids & merged[-1][3])
        untyped_event = not event_ids or (bool(merged) and not merged[-1][3])
        if merged and start <= merged[-1][1] + 1 and (same_event or untyped_event):
            old_start, old_end, old_run, old_event_ids, old_precursors = merged[-1]
            merged[-1] = (
                old_start,
                max(old_end, end),
                old_run + run,
                old_event_ids | event_ids,
                old_precursors | precursor_indices,
            )
        else:
            merged.append((start, end, list(run), set(event_ids), precursor_indices))
    windows: list[RelevanceWindow] = []
    for start, end, run, _event_ids, precursor_indices in merged:
        indices = tuple(range(start, end + 1))
        trigger_steps = tuple(sorted(rows[index]["step"] for index in run))
        window_precursors = tuple(sorted(index for index in precursor_indices if index in indices))
        windows.append(
            RelevanceWindow(
                start_step=int(rows[start]["step"]),
                end_step=int(rows[end]["step"]),
                row_indices=indices,
                original_step_indices=tuple(int(rows[index]["step"]) for index in indices),
                trigger_steps=trigger_steps,
                precursor_steps=tuple(int(rows[index]["step"]) for index in window_precursors),
                reasons=tuple(
                    sorted({reason for index in run for reason in vectors[index].active_reasons})
                ),
                actor_ids=tuple(
                    sorted({actor for index in indices for actor in vectors[index].actor_ids})
                ),
                missing_signals=tuple(
                    sorted({name for index in indices for name in vectors[index].unknown_signals})
                ),
                hindsight=any(vectors[index].hindsight for index in indices),
            )
        )
    return tuple(windows), required_precursors


def _validate_precursor_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Reject precursor markers that cannot be associated with an event."""
    for row in rows:
        if _strict_bool(row.get("precursor", False), f"row[{row['step']}].precursor"):
            if _event_id(row.get("event_id"), f"row[{row['step']}].event_id") is None:
                raise ExcerptContractError(f"row[{row['step']}] precursor must declare an event_id")


def _run_event_ids(rows: Sequence[Mapping[str, Any]], run: Sequence[int]) -> set[str]:
    """Return normalized event identities carried by one trigger run."""
    event_ids: set[str] = set()
    for index in run:
        event_id = _event_id(rows[index].get("event_id"), f"row[{rows[index]['step']}].event_id")
        if event_id is not None:
            event_ids.add(event_id)
    return event_ids


def _precursors_for_run(
    rows: Sequence[Mapping[str, Any]],
    required_precursors: set[int],
    run: Sequence[int],
    run_event_ids: set[str],
) -> list[int]:
    """Return precursor row indices owned by one trigger run."""
    relevant: list[int] = []
    for index in required_precursors:
        if index > run[-1]:
            continue
        event_id = _event_id(rows[index].get("event_id"), f"row[{rows[index]['step']}].event_id")
        if not run_event_ids or event_id in run_event_ids:
            relevant.append(index)
    return relevant


def _evaluate_rows(
    rows: Sequence[Mapping[str, Any]], thresholds: RelevanceThresholds
) -> tuple[tuple[RelevanceVector, ...], tuple[RelevanceWindow, ...], set[int]]:
    """Recompute vectors, windows, and precursor ownership deterministically.

    Returns:
        Vectors, deterministic windows, and required precursor row indices.
    """
    vectors: list[RelevanceVector] = []
    active_reasons: set[str] = set()
    safe_counts: dict[str, int] = {}
    for row in rows:
        vectors.append(_vector_for_row(row, thresholds, active_reasons, safe_counts))
    windows, required_precursors = _build_windows(rows, vectors, thresholds)
    return tuple(vectors), windows, required_precursors


def _thresholds_from_selector_config(
    selector_config: Mapping[str, Any],
) -> tuple[RelevanceThresholds, bool]:
    """Parse and canonicalize the selector contract stored in a manifest.

    Returns:
        The declared thresholds and the selector-level hindsight flag.
    """
    expected_keys = {"schema_version", "thresholds", "hindsight", "actor_policy", "parent_rows"}
    config = dict(selector_config)
    if set(config) != expected_keys:
        raise ExcerptContractError("selector_config has unexpected or missing fields")
    if config["schema_version"] != RELEVANCE_SCHEMA:
        raise ExcerptContractError("selector_config schema_version does not match manifest schema")
    selector_hindsight = _strict_bool(config["hindsight"], "selector_config.hindsight")
    if config["actor_policy"] != "all_parent_actors":
        raise ExcerptContractError("selector_config actor_policy is not supported")
    if config["parent_rows"] != "complete_parent_retained":
        raise ExcerptContractError("selector_config parent_rows policy is not supported")
    return _thresholds_from_payload(config["thresholds"]), selector_hindsight


def _thresholds_from_payload(value: Any) -> RelevanceThresholds:
    """Parse the canonical nested threshold payload from a selector config.

    Returns:
        The validated threshold configuration.
    """
    threshold_payload = value
    if not isinstance(threshold_payload, Mapping):
        raise ExcerptContractError("selector_config.thresholds must be an object")
    threshold_payload = dict(threshold_payload)
    if set(threshold_payload) != {"approval_status", "proposed", "approved"}:
        raise ExcerptContractError("selector_config.thresholds has unexpected or missing fields")
    proposed = threshold_payload["proposed"]
    if not isinstance(proposed, Mapping):
        raise ExcerptContractError("selector_config.thresholds.proposed must be an object")
    proposed = dict(proposed)
    required_proposed = {
        "clearance_m",
        "closing_velocity_m_s",
        "ttc_s",
        "closest_approach_m",
        "braking_margin_m",
        "visibility_latency_s",
        "stall_s",
        "discomfort",
        "pre_roll_steps",
        "post_roll_steps",
        "merge_gap_steps",
        "hysteresis_steps",
    }
    if set(proposed) != required_proposed:
        raise ExcerptContractError(
            "selector_config.thresholds.proposed has unexpected or missing fields"
        )
    approved = threshold_payload["approved"]
    if approved is not None and not isinstance(approved, Mapping):
        raise ExcerptContractError("selector_config.thresholds.approved must be an object or null")
    try:
        thresholds = RelevanceThresholds(
            **proposed,
            approval_status=threshold_payload["approval_status"],
            approved_thresholds=approved,
        )
    except (RelevanceContractError, TypeError) as exc:
        raise ExcerptContractError(f"invalid selector thresholds: {exc}") from exc
    if thresholds.to_dict() != threshold_payload:
        raise ExcerptContractError("selector_config.thresholds is not canonical")
    return thresholds


def select_relevance_windows(
    rows: Sequence[Mapping[str, Any]],
    *,
    parent_digest: str,
    thresholds: RelevanceThresholds | None = None,
    actor_ids: Sequence[str] | None = None,
    hindsight: bool = False,
) -> RelevanceSelection:
    """Select deterministic relevance windows while retaining the full parent.

    Args:
        rows: Complete, ordered parent trace rows.
        parent_digest: External parent artifact identity (SHA-256).
        thresholds: Proposed offline threshold/hysteresis rules.
        actor_ids: Optional explicit actor identity set; row identities are used otherwise.
        hindsight: Mark an offline selector that can use future rows.

    Returns:
        Vectors, immutable window manifest, and a copied complete parent.
    """
    normalized = _normalize_rows(rows)
    effective = thresholds or RelevanceThresholds()
    selector_hindsight = _strict_bool(hindsight, "hindsight")
    vectors, windows, required_precursors = _evaluate_rows(normalized, effective)
    row_actor_ids = tuple(sorted({actor for vector in vectors for actor in vector.actor_ids}))
    effective_actor_ids = (
        tuple(sorted({str(item) for item in actor_ids})) if actor_ids else row_actor_ids
    )
    if actor_ids is not None and row_actor_ids and effective_actor_ids != row_actor_ids:
        raise ExcerptContractError("explicit actor_ids must retain every parent actor")
    selected_steps = tuple(
        sorted({step for window in windows for step in window.original_step_indices})
    )
    missing_signals = tuple(sorted({name for vector in vectors for name in vector.unknown_signals}))
    selector_config = {
        "schema_version": RELEVANCE_SCHEMA,
        "thresholds": effective.to_dict(),
        "hindsight": selector_hindsight,
        "actor_policy": "all_parent_actors",
        "parent_rows": "complete_parent_retained",
    }
    manifest = ExcerptManifest(
        parent_digest=_require_digest(parent_digest, "parent_digest"),
        parent_rows_sha256=compute_parent_rows_sha256(normalized),
        source_row_count=len(normalized),
        actor_ids=effective_actor_ids,
        selected_step_indices=selected_steps,
        windows=windows,
        selector_config=selector_config,
        missing_signals=missing_signals,
        hindsight=selector_hindsight or any(vector.hindsight for vector in vectors),
        required_precursor_steps=tuple(
            sorted(normalized[index]["step"] for index in required_precursors)
        ),
    )
    selection = RelevanceSelection(tuple(normalized), tuple(vectors), windows, manifest)
    validate_excerpt_manifest(selection.manifest, selection.parent_rows)
    return selection


def _validate_manifest_parent(
    manifest: ExcerptManifest, normalized: Sequence[Mapping[str, Any]]
) -> None:
    """Validate parent row count, digest, and actor retention."""
    if len(normalized) != manifest.source_row_count:
        raise ExcerptContractError("manifest source row count does not match parent")
    if compute_parent_rows_sha256(normalized) != manifest.parent_rows_sha256:
        raise ExcerptContractError("manifest parent rows digest does not match complete parent")
    actual_actor_ids = tuple(sorted({actor for row in normalized for actor in _actor_ids(row)}))
    if actual_actor_ids and actual_actor_ids != tuple(manifest.actor_ids):
        raise ExcerptContractError("manifest actor IDs do not retain the complete parent actor set")


def _validate_manifest_selection(
    manifest: ExcerptManifest, normalized: Sequence[Mapping[str, Any]]
) -> tuple[set[int], dict[int, int]]:
    """Validate selected and required precursor step membership.

    Returns:
        ``(selected_steps, step_by_row_index)`` for window checks.
    """
    available_steps = {int(row["step"]) for row in normalized}
    step_by_index = {index: int(row["step"]) for index, row in enumerate(normalized)}
    selected = set(manifest.selected_step_indices)
    if not selected <= available_steps:
        raise ExcerptContractError("manifest selects a step not present in the parent")
    required = set(manifest.required_precursor_steps)
    missing_precursors = sorted(required - selected)
    if missing_precursors:
        raise ExcerptContractError(
            f"unsafe crop: selected excerpt omits required precursor steps {missing_precursors}"
        )

    return selected, step_by_index


def _validate_manifest_semantics(
    manifest: ExcerptManifest, normalized: Sequence[Mapping[str, Any]]
) -> None:
    """Recompute the selector contract so a forged manifest cannot redefine it."""
    thresholds, selector_hindsight = _thresholds_from_selector_config(manifest.selector_config)
    vectors, expected_windows, required_precursors = _evaluate_rows(normalized, thresholds)
    expected_selected = tuple(
        sorted({step for window in expected_windows for step in window.original_step_indices})
    )
    expected_required = tuple(sorted(normalized[index]["step"] for index in required_precursors))
    expected_missing = tuple(
        sorted({name for vector in vectors for name in vector.unknown_signals})
    )
    expected_hindsight = selector_hindsight or any(vector.hindsight for vector in vectors)
    if tuple(manifest.selected_step_indices) != expected_selected:
        raise ExcerptContractError("manifest selected steps do not match deterministic selector")
    if tuple(manifest.required_precursor_steps) != expected_required:
        raise ExcerptContractError("manifest precursor steps do not match parent events")
    if tuple(manifest.windows) != expected_windows:
        raise ExcerptContractError("manifest windows do not match deterministic selector")
    if tuple(manifest.missing_signals) != expected_missing:
        raise ExcerptContractError("manifest missing signals do not match parent rows")
    if manifest.hindsight != expected_hindsight:
        raise ExcerptContractError("manifest hindsight does not match selector and parent rows")


def _validate_manifest_windows(
    manifest: ExcerptManifest, selected: set[int], step_by_index: Mapping[int, int]
) -> None:
    """Validate each window's row indices, boundaries, and precursor coverage."""
    for window in manifest.windows:
        if any(index not in step_by_index for index in window.row_indices):
            raise ExcerptContractError("window contains a row index absent from the parent")
        expected_steps = tuple(step_by_index[index] for index in window.row_indices)
        if expected_steps != window.original_step_indices:
            raise ExcerptContractError("window row indices do not match original parent steps")
        if expected_steps and (
            window.start_step != expected_steps[0] or window.end_step != expected_steps[-1]
        ):
            raise ExcerptContractError("window boundaries do not match original parent steps")
        if expected_steps and expected_steps != tuple(
            step_by_index[index]
            for index in range(window.row_indices[0], window.row_indices[-1] + 1)
        ):
            raise ExcerptContractError("window row indices must form a contiguous parent interval")
        if not set(window.original_step_indices) <= selected:
            raise ExcerptContractError("window contains a step outside the manifest selection")
        if not set(window.precursor_steps) <= selected:
            raise ExcerptContractError("unsafe crop: window omits a declared precursor")
        if not set(window.original_step_indices) <= set(step_by_index.values()):
            raise ExcerptContractError("window contains a step absent from the parent")


def validate_excerpt_manifest(manifest: ExcerptManifest, rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate parent identity, actor retention, and precursor coverage.

    Raises:
        ExcerptContractError: If rows, indices, actor identities, or precursors drift.
    """
    normalized = _normalize_rows(rows)
    _validate_manifest_parent(manifest, normalized)
    selected, step_by_index = _validate_manifest_selection(manifest, normalized)
    _validate_manifest_windows(manifest, selected, step_by_index)
    _validate_manifest_semantics(manifest, normalized)


def write_selection_manifest(selection: RelevanceSelection, path: str | Path) -> None:
    """Write a deterministic JSON proposal manifest and retain parent linkage."""
    validate_excerpt_manifest(selection.manifest, selection.parent_rows)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = [
    "EVIDENCE_GRADE",
    "MANIFEST_STATUS",
    "RELEVANCE_SCHEMA",
    "ExcerptContractError",
    "ExcerptManifest",
    "RelevanceContractError",
    "RelevanceSelection",
    "RelevanceSignal",
    "RelevanceThresholds",
    "RelevanceVector",
    "RelevanceWindow",
    "compute_parent_rows_sha256",
    "select_relevance_windows",
    "validate_excerpt_manifest",
    "write_selection_manifest",
]
