"""Prospective machine-verifiable tuning-run provenance (issue #6595).

This module deliberately separates prospective run records from the frozen
historical tuning-effort registry.  A launch record can be minimal for a
``debug`` run, contributes only machine-supported counters for a ``tuning``
run, and is explicitly excluded from tuning totals for an ``evidence`` run.
Unknown values remain ``None`` in serialized records; no counter is inferred
from commits, elapsed training artifacts, or person-hours.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

TUNING_RUN_RECORD_SCHEMA = "tuning-run-record.v1"
TUNING_LEDGER_SCHEMA = "tuning-ledger.v1"
TUNING_RUN_CLASS_DEBUG = "debug"
TUNING_RUN_CLASS_TUNING = "tuning"
TUNING_RUN_CLASS_EVIDENCE = "evidence"
TUNING_RUN_CLASSES = (
    TUNING_RUN_CLASS_DEBUG,
    TUNING_RUN_CLASS_TUNING,
    TUNING_RUN_CLASS_EVIDENCE,
)

_COUNTER_FIELDS = (
    "attempted_configurations",
    "simulator_episodes",
    "simulator_calls",
    "wall_clock_seconds",
    "person_hours",
)


def _optional_text(value: Any) -> str | None:
    """Normalize an optional text field without inventing a placeholder.

    Returns:
        Stripped text, or ``None`` when the value is absent or blank.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_text_tuple(value: Any, *, field_name: str) -> tuple[str, ...] | None:
    """Normalize an optional list-like text field.

    Returns:
        A tuple of non-empty stripped values, or ``None`` when absent.
    """
    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray, dict)):
        raise TypeError(f"{field_name} must be a string or list of strings when provided")
    normalized: list[str] = []
    for item in values:
        if item is None:
            raise TypeError(f"{field_name} entries must be strings")
        text = str(item).strip()
        if text:
            normalized.append(text)
    return tuple(normalized)


def _optional_non_negative_int(value: Any, *, field_name: str) -> int | None:
    """Normalize an optional non-negative integer counter.

    Returns:
        The validated integer, or ``None`` when absent.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a non-negative integer when provided")
    if value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer when provided")
    return value


def _optional_non_negative_float(value: Any, *, field_name: str) -> float | None:
    """Normalize an optional non-negative finite numeric counter.

    Returns:
        The validated finite float, or ``None`` when absent.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a non-negative number when provided")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a non-negative number when provided") from exc
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative finite number when provided")
    return normalized


@dataclass(frozen=True)
class TuningRunSpec:
    """Configurable capture fields for one prospective launch or ingestion run."""

    run_class: str = TUNING_RUN_CLASS_DEBUG
    run_id: str | None = None
    parameters_changed: tuple[str, ...] | None = None
    objective: str | None = None
    development_scenario_ids: tuple[str, ...] | None = None
    development_split: str | None = None
    eval_set_disjoint: bool | None = None
    attempted_configurations: int | None = None
    simulator_episodes: int | None = None
    simulator_calls: int | None = None
    wall_clock_seconds: float | None = None
    compute_resource: str | None = None
    stopping_rule: str | None = None
    parent_run_id: str | None = None
    person_hours: float | None = None


def parse_tuning_run_spec(
    raw: Any,
    *,
    source: str = "tuning_run_provenance",
) -> TuningRunSpec | None:
    """Parse the optional campaign-level prospective tuning-run block.

    Returns:
        Parsed run specification, or ``None`` when no block was supplied.
    """
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError(f"{source} must be a mapping when provided")
    run_class = str(raw.get("run_class", TUNING_RUN_CLASS_DEBUG)).strip().lower()
    if run_class not in TUNING_RUN_CLASSES:
        known = ", ".join(TUNING_RUN_CLASSES)
        raise ValueError(f"{source}.run_class '{run_class}' is not one of: {known}")
    eval_set_disjoint = raw.get("eval_set_disjoint")
    if eval_set_disjoint is not None and not isinstance(eval_set_disjoint, bool):
        raise TypeError(f"{source}.eval_set_disjoint must be a boolean when provided")
    return TuningRunSpec(
        run_class=run_class,
        run_id=_optional_text(raw.get("run_id")),
        parameters_changed=_optional_text_tuple(
            raw.get("parameters_changed"),
            field_name=f"{source}.parameters_changed",
        ),
        objective=_optional_text(raw.get("objective")),
        development_scenario_ids=_optional_text_tuple(
            raw.get("development_scenario_ids"),
            field_name=f"{source}.development_scenario_ids",
        ),
        development_split=_optional_text(raw.get("development_split")),
        eval_set_disjoint=eval_set_disjoint,
        attempted_configurations=_optional_non_negative_int(
            raw.get("attempted_configurations"),
            field_name=f"{source}.attempted_configurations",
        ),
        simulator_episodes=_optional_non_negative_int(
            raw.get("simulator_episodes"),
            field_name=f"{source}.simulator_episodes",
        ),
        simulator_calls=_optional_non_negative_int(
            raw.get("simulator_calls"),
            field_name=f"{source}.simulator_calls",
        ),
        wall_clock_seconds=_optional_non_negative_float(
            raw.get("wall_clock_seconds"),
            field_name=f"{source}.wall_clock_seconds",
        ),
        compute_resource=_optional_text(raw.get("compute_resource")),
        stopping_rule=_optional_text(raw.get("stopping_rule")),
        parent_run_id=_optional_text(raw.get("parent_run_id")),
        person_hours=_optional_non_negative_float(
            raw.get("person_hours"),
            field_name=f"{source}.person_hours",
        ),
    )


def validate_tuning_run_spec(  # noqa: C901
    spec: TuningRunSpec | None,
    *,
    strict: bool = False,
    parameter_sources: Mapping[str, Sequence[str] | None] | None = None,
) -> None:
    """Validate a prospective spec, optionally applying the publication gate.

    Strict validation is intentionally narrow: it is for the existing
    ``tuning_effort_enforcement='error'`` publication-style gate.  Debug and
    smoke configs remain free to omit the block or provide only ``run_class``.
    """
    if spec is None:
        if strict:
            raise ValueError(
                "tuning_effort_enforcement='error' requires a complete "
                "'tuning_run_provenance' block"
            )
        return
    if spec.run_class not in TUNING_RUN_CLASSES:
        known = ", ".join(TUNING_RUN_CLASSES)
        raise ValueError(f"Unsupported tuning run class '{spec.run_class}'; expected: {known}")
    if not strict:
        return
    if spec.run_class == TUNING_RUN_CLASS_DEBUG:
        raise ValueError(
            "tuning_effort_enforcement='error' cannot use run_class='debug'; "
            "use 'tuning' or 'evidence'"
        )
    missing: list[str] = []
    if spec.run_id is None:
        missing.append("run_id")
    if spec.objective is None:
        missing.append("objective")
    if spec.stopping_rule is None:
        missing.append("stopping_rule")
    if not spec.development_scenario_ids and spec.development_split is None:
        missing.append("development_scenario_ids or development_split")
    if spec.eval_set_disjoint is None:
        missing.append("eval_set_disjoint")
    if spec.run_class == TUNING_RUN_CLASS_TUNING and not spec.parameters_changed:
        sources = parameter_sources or {}
        missing_planners = sorted(
            planner_id for planner_id, values in sources.items() if not values
        )
        if missing_planners:
            missing.append("parameters_changed for: " + ", ".join(missing_planners))
    if missing:
        raise ValueError(
            "strict tuning-run provenance is incomplete; missing: " + ", ".join(missing)
        )


@dataclass(frozen=True)
class TuningRunRecord:
    """One serialized prospective run record.

    ``None`` means unknown/not captured.  In particular, it is not equivalent
    to zero for any counter.
    """

    run_id: str
    run_class: str
    planner_id: str | None = None
    source_commit: str | None = None
    config_hash: str | None = None
    parameters_changed: tuple[str, ...] | None = None
    objective: str | None = None
    development_scenario_ids: tuple[str, ...] | None = None
    development_split: str | None = None
    eval_set_disjoint: bool | None = None
    attempted_configurations: int | None = None
    simulator_episodes: int | None = None
    simulator_calls: int | None = None
    wall_clock_seconds: float | None = None
    compute_resource: str | None = None
    stopping_rule: str | None = None
    parent_run_id: str | None = None
    person_hours: float | None = None
    campaign_id: str | None = None
    recorded_at_utc: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible record mapping."""
        return {
            "schema_version": TUNING_RUN_RECORD_SCHEMA,
            "run_id": self.run_id,
            "run_class": self.run_class,
            "planner_id": self.planner_id,
            "source_commit": self.source_commit,
            "config_hash": self.config_hash,
            "parameters_changed": (
                list(self.parameters_changed) if self.parameters_changed is not None else None
            ),
            "objective": self.objective,
            "development_scenario_ids": (
                list(self.development_scenario_ids)
                if self.development_scenario_ids is not None
                else None
            ),
            "development_split": self.development_split,
            "eval_set_disjoint": self.eval_set_disjoint,
            "attempted_configurations": self.attempted_configurations,
            "simulator_episodes": self.simulator_episodes,
            "simulator_calls": self.simulator_calls,
            "wall_clock_seconds": self.wall_clock_seconds,
            "compute_resource": self.compute_resource,
            "stopping_rule": self.stopping_rule,
            "parent_run_id": self.parent_run_id,
            "person_hours": self.person_hours,
            "campaign_id": self.campaign_id,
            "recorded_at_utc": self.recorded_at_utc,
            "counts_toward_tuning": self.run_class == TUNING_RUN_CLASS_TUNING,
            "provenance": dict(self.provenance),
        }


def _validate_record(  # noqa: C901
    record: TuningRunRecord, *, strict: bool = False
) -> None:
    """Validate record identity, class semantics, and numeric counters."""
    if not record.run_id.strip():
        raise ValueError("tuning run record requires a non-empty run_id")
    if record.run_class not in TUNING_RUN_CLASSES:
        known = ", ".join(TUNING_RUN_CLASSES)
        raise ValueError(f"Unsupported tuning run class '{record.run_class}'; expected: {known}")
    if record.run_class == TUNING_RUN_CLASS_TUNING and not record.planner_id:
        raise ValueError("tuning run records require planner_id")
    if strict and record.run_class != TUNING_RUN_CLASS_DEBUG:
        required = {
            "source_commit": record.source_commit,
            "config_hash": record.config_hash,
            "objective": record.objective,
            "stopping_rule": record.stopping_rule,
        }
        missing = sorted(
            name
            for name, value in required.items()
            if not value or str(value).strip().lower() in {"unknown", "null"}
        )
        if not record.development_scenario_ids and not record.development_split:
            missing.append("development_scenario_ids or development_split")
        if record.eval_set_disjoint is None:
            missing.append("eval_set_disjoint")
        if missing:
            raise ValueError("strict tuning run record is incomplete: " + ", ".join(missing))
    for field_name in _COUNTER_FIELDS:
        value = getattr(record, field_name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{field_name} must be numeric or null")
        if not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"{field_name} must be finite and non-negative or null")
    if not isinstance(record.provenance, Mapping):
        raise TypeError("provenance must be a mapping")


def record_from_mapping(raw: Mapping[str, Any], *, strict: bool = False) -> TuningRunRecord:
    """Parse and validate one serialized record for ingestion.

    Returns:
        Validated typed record.
    """
    if raw.get("schema_version") != TUNING_RUN_RECORD_SCHEMA:
        raise ValueError(
            "unsupported tuning-run record schema: "
            f"{raw.get('schema_version')!r}; expected {TUNING_RUN_RECORD_SCHEMA!r}"
        )
    run_class = str(raw.get("run_class") or "").strip().lower()
    counts_toward_tuning = raw.get("counts_toward_tuning")
    if not isinstance(counts_toward_tuning, bool):
        raise TypeError("counts_toward_tuning must be a boolean")
    if counts_toward_tuning != (run_class == TUNING_RUN_CLASS_TUNING):
        raise ValueError("counts_toward_tuning does not match run_class")
    raw_provenance = raw.get("provenance")
    if raw_provenance is not None and not isinstance(raw_provenance, Mapping):
        raise TypeError("provenance must be a mapping when provided")
    record = TuningRunRecord(
        run_id=str(raw.get("run_id") or "").strip(),
        run_class=run_class,
        planner_id=_optional_text(raw.get("planner_id")),
        source_commit=_optional_text(raw.get("source_commit")),
        config_hash=_optional_text(raw.get("config_hash")),
        parameters_changed=_optional_text_tuple(
            raw.get("parameters_changed"),
            field_name="parameters_changed",
        ),
        objective=_optional_text(raw.get("objective")),
        development_scenario_ids=_optional_text_tuple(
            raw.get("development_scenario_ids"),
            field_name="development_scenario_ids",
        ),
        development_split=_optional_text(raw.get("development_split")),
        eval_set_disjoint=raw.get("eval_set_disjoint"),
        attempted_configurations=_optional_non_negative_int(
            raw.get("attempted_configurations"),
            field_name="attempted_configurations",
        ),
        simulator_episodes=_optional_non_negative_int(
            raw.get("simulator_episodes"),
            field_name="simulator_episodes",
        ),
        simulator_calls=_optional_non_negative_int(
            raw.get("simulator_calls"),
            field_name="simulator_calls",
        ),
        wall_clock_seconds=_optional_non_negative_float(
            raw.get("wall_clock_seconds"),
            field_name="wall_clock_seconds",
        ),
        compute_resource=_optional_text(raw.get("compute_resource")),
        stopping_rule=_optional_text(raw.get("stopping_rule")),
        parent_run_id=_optional_text(raw.get("parent_run_id")),
        person_hours=_optional_non_negative_float(
            raw.get("person_hours"),
            field_name="person_hours",
        ),
        campaign_id=_optional_text(raw.get("campaign_id")),
        recorded_at_utc=_optional_text(raw.get("recorded_at_utc")),
        provenance=dict(raw_provenance or {}),
    )
    if record.eval_set_disjoint is not None and not isinstance(record.eval_set_disjoint, bool):
        raise TypeError("eval_set_disjoint must be a boolean or null")
    _validate_record(record, strict=strict)
    return record


def build_launch_records(
    spec: TuningRunSpec | None,
    *,
    campaign_id: str,
    source_commit: str | None,
    config_hash: str | None,
    planner_parameters: Mapping[str, Sequence[str] | None],
    recorded_at_utc: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> tuple[TuningRunRecord, ...]:
    """Emit one automatic launch record per enabled planner arm.

    Returns:
        One record for each planner id in ``planner_parameters``.
    """
    effective = spec or TuningRunSpec()
    base_run_id = effective.run_id or campaign_id
    records: list[TuningRunRecord] = []
    for planner_id, planner_values in planner_parameters.items():
        parameters = effective.parameters_changed
        if not parameters and planner_values is not None:
            parameters = tuple(str(value).strip() for value in planner_values if str(value).strip())
        record = TuningRunRecord(
            run_id=f"{base_run_id}:{planner_id}",
            run_class=effective.run_class,
            planner_id=planner_id,
            source_commit=source_commit,
            config_hash=config_hash,
            parameters_changed=parameters,
            objective=effective.objective,
            development_scenario_ids=effective.development_scenario_ids,
            development_split=effective.development_split,
            eval_set_disjoint=effective.eval_set_disjoint,
            attempted_configurations=effective.attempted_configurations,
            simulator_episodes=effective.simulator_episodes,
            simulator_calls=effective.simulator_calls,
            wall_clock_seconds=effective.wall_clock_seconds,
            compute_resource=effective.compute_resource,
            stopping_rule=effective.stopping_rule,
            parent_run_id=effective.parent_run_id,
            person_hours=effective.person_hours,
            campaign_id=campaign_id,
            recorded_at_utc=recorded_at_utc,
            provenance={
                "capture_mode": "camera_ready_preflight",
                **dict(provenance or {}),
            },
        )
        _validate_record(record, strict=strict)
        records.append(record)
    return tuple(records)


def _canonical_json(payload: Any) -> str:
    """Return deterministic JSON for ledger sorting and hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _aggregate_counter(records: Sequence[TuningRunRecord], field_name: str) -> int | float | None:
    """Sum a counter only when every eligible record supplied it.

    Returns:
        Sum of the counter, or ``None`` when no eligible record exists or one
        eligible record omitted the counter.
    """
    if not records:
        return None
    values = [getattr(record, field_name) for record in records]
    if any(value is None for value in values):
        return None
    return sum(values)


def _counter_unknown_count(records: Sequence[TuningRunRecord], field_name: str) -> int:
    """Count eligible records with an unknown counter.

    Returns:
        Number of records whose counter is ``None``.
    """
    return sum(getattr(record, field_name) is None for record in records)


def _aggregate_planner_records(records: Sequence[TuningRunRecord]) -> dict[str, Any]:
    """Build a null-preserving aggregate for one planner.

    Returns:
        JSON-compatible aggregate with class counts and supported counters.
    """
    tuning_records = [record for record in records if record.run_class == TUNING_RUN_CLASS_TUNING]
    summary: dict[str, Any] = {
        "record_count": len(records),
        "tuning_record_count": len(tuning_records),
        "debug_record_count": sum(record.run_class == TUNING_RUN_CLASS_DEBUG for record in records),
        "evidence_record_count": sum(
            record.run_class == TUNING_RUN_CLASS_EVIDENCE for record in records
        ),
        "counts_toward_tuning": bool(tuning_records),
    }
    for field_name in _COUNTER_FIELDS:
        summary[field_name] = _aggregate_counter(tuning_records, field_name)
        summary[f"unknown_{field_name}_count"] = _counter_unknown_count(tuning_records, field_name)
    return summary


def aggregate_tuning_records(records: Iterable[TuningRunRecord]) -> dict[str, Any]:
    """Return a deterministic per-arm tuning ledger from validated records."""
    normalized = list(records)
    for record in normalized:
        _validate_record(record)
    normalized.sort(key=lambda record: _canonical_json(record.to_mapping()))
    by_planner: dict[str, list[TuningRunRecord]] = {}
    for record in normalized:
        planner_id = record.planner_id or "<unknown>"
        by_planner.setdefault(planner_id, []).append(record)
    record_mappings = [record.to_mapping() for record in normalized]
    digest = hashlib.sha256(_canonical_json(record_mappings).encode("utf-8")).hexdigest()
    return {
        "schema_version": TUNING_LEDGER_SCHEMA,
        "record_schema_version": TUNING_RUN_RECORD_SCHEMA,
        "ledger_sha256": digest,
        "records": record_mappings,
        "summary": _aggregate_planner_records(normalized),
        "by_planner": {
            planner_id: _aggregate_planner_records(planner_records)
            for planner_id, planner_records in sorted(by_planner.items())
        },
        "policy": {
            "debug_counts_toward_tuning": False,
            "tuning_counts_toward_tuning": True,
            "evidence_counts_toward_tuning": False,
            "unknown_counter_value": None,
            "person_hours_required": False,
            "commit_counts_as_trial": False,
        },
    }


def load_tuning_records(path: Path) -> tuple[TuningRunRecord, ...]:
    """Load records from a record list or a generated tuning ledger.

    Returns:
        Validated records in the order serialized by the input file.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_records = payload.get("records") if isinstance(payload, Mapping) else payload
    if not isinstance(raw_records, list):
        raise ValueError(f"Tuning provenance file requires a records list: {path}")
    if any(not isinstance(item, Mapping) for item in raw_records):
        raise TypeError(f"Tuning provenance records must be mappings: {path}")
    return tuple(record_from_mapping(item) for item in raw_records)


def write_tuning_ledger(path: Path, records: Iterable[TuningRunRecord]) -> dict[str, Any]:
    """Aggregate and write a machine-readable ledger, returning its payload.

    Returns:
        The JSON-compatible ledger written to ``path``.
    """
    ledger = aggregate_tuning_records(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ledger
