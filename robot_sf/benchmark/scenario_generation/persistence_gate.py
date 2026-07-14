"""Evidence gate between generated candidates and reusable critical scenarios.

This module adds the ``generated_scenario_persistence.v1`` evidence record that
the stage-1 generation pipeline does not yet produce.  A candidate is promoted
only when three independent status checks all pass:

1. ``exact_replay`` - byte/config-equivalent replay of the source episode;
2. ``critical_event_reproduced`` - the critical event under the source planner;
3. ``perturbation_persistence`` - the event survives a preregistered perturbation grid.

The three statuses remain separate.  A ``pass`` in one does not imply a ``pass``
in another, and the promotion verdict fails closed on any ``fail``/``unknown``
required status, on missing trace fields, on replay divergence, and on an
unregistered (non-frozen) configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from robot_sf.errors import RobotSfError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

PERSISTENCE_SCHEMA_VERSION = "generated_scenario_persistence.v1"
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1] / "schemas" / "generated_scenario_persistence.v1.json"
)
_SCHEMA_VALIDATOR: Draft202012Validator | None = None
_REQUIRED_EPISODE_FIELDS = ("episode_id", "source_seed", "source_map")

PASS = "pass"
FAIL = "fail"
UNKNOWN = "unknown"

REQUIRED_STATUSES = ("exact_replay", "critical_event_reproduced", "perturbation_persistence")


class ScenarioPersistenceValidationError(RobotSfError, ValueError):
    """Raised when a persistence record is incomplete or internally inconsistent."""


def load_persistence_schema() -> dict[str, Any]:
    """Return the versioned JSON Schema for persistence records."""

    with _SCHEMA_PATH.open(encoding="utf-8") as schema_file:
        return json.load(schema_file)


def _stable_digest(payload: Any) -> str:
    """Return a short, order-independent SHA-256 digest for a JSON-safe payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_schema(record: Mapping[str, Any]) -> None:
    global _SCHEMA_VALIDATOR
    if _SCHEMA_VALIDATOR is None:
        _SCHEMA_VALIDATOR = Draft202012Validator(load_persistence_schema())
    errors = sorted(
        _SCHEMA_VALIDATOR.iter_errors(dict(record)),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        formatted = "; ".join(
            f"/{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
            for error in errors
        )
        raise ScenarioPersistenceValidationError(formatted)


def compute_persistence_record(
    *,
    scenario_id: str,
    source_episode: Mapping[str, Any],
    generated_scenario: Mapping[str, Any],
    planner: str,
    seed: int,
    config: Mapping[str, Any],
    commit_hashes: Mapping[str, Any],
    exact_replay: Mapping[str, Any],
    critical_event_reproduced: Mapping[str, Any],
    perturbation_grid: Mapping[str, Any],
    cell_verdicts: Sequence[Mapping[str, Any]],
    missing_cell_reasons: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble a schema-valid ``generated_scenario_persistence.v1`` record.

    The three status blocks are supplied independently.  This function validates
    the JSON Schema contract, derives the persistence summary from the supplied
    cells, and computes the fail-closed promotion verdict.

    Args:
        scenario_id: Reusable-scenario identifier.
        source_episode: episode_id, source_seed, source_map, replay_digest.
        generated_scenario: catalog_schema_version, scenario_id, optional
            catalog_entry_digest.
        planner: Source planner name used for reproduction.
        seed: Episode/run seed.
        config: Frozen perturbation/promotion configuration (config_id, frozen,
            optional config_hash).
        commit_hashes: code and config commit hashes.
        exact_replay: status, divergence_reason, replay_digest.
        critical_event_reproduced: status, event_type, tolerances, observed deltas.
        perturbation_grid: timing_offsets_s and speed_deltas_m_s lists.
        cell_verdicts: One verdict per preregistered grid cell.
        missing_cell_reasons: Optional per-cell exclusion reasons.

    Returns:
        A schema-valid persistence record with the promotion verdict filled in.

    Raises:
        ScenarioPersistenceValidationError: If the record cannot be built or
            would violate the fail-closed contract.
    """

    missing_cell_reasons = list(missing_cell_reasons or [])
    if not config.get("frozen"):
        raise ScenarioPersistenceValidationError(
            "config.frozen must be true: unfrozen perturbation config cannot gate promotion"
        )

    pass_count = sum(1 for cell in cell_verdicts if cell.get("verdict") == PASS)
    total_cells = len(cell_verdicts)
    persistence_rate = (pass_count / total_cells) if total_cells else 0.0
    if total_cells:
        rates = [pass_count / total_cells]
    else:
        rates = []
    persistence_interval: list[float] = [min(rates), max(rates)] if rates else [0.0, 0.0]

    record: dict[str, Any] = {
        "schema_version": PERSISTENCE_SCHEMA_VERSION,
        "scenario_id": scenario_id,
        "source_episode": dict(source_episode),
        "generated_scenario": dict(generated_scenario),
        "planner": planner,
        "seed": int(seed),
        "config": dict(config),
        "commit_hashes": dict(commit_hashes),
        "exact_replay": dict(exact_replay),
        "critical_event_reproduced": dict(critical_event_reproduced),
        "perturbation_persistence": {
            "grid": dict(perturbation_grid),
            "cells": [dict(cell) for cell in cell_verdicts],
            "persistence_rate": persistence_rate,
            "persistence_interval": persistence_interval,
            "missing_cell_reasons": [dict(reason) for reason in missing_cell_reasons],
        },
        "promotion": _promotion_verdict(
            exact_replay=exact_replay,
            critical_event_reproduced=critical_event_reproduced,
            cell_verdicts=cell_verdicts,
            missing_cell_reasons=missing_cell_reasons,
        ),
    }
    _verify_schema(record)
    return record


def _promotion_verdict(
    *,
    exact_replay: Mapping[str, Any],
    critical_event_reproduced: Mapping[str, Any],
    cell_verdicts: Sequence[Mapping[str, Any]],
    missing_cell_reasons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the fail-closed promotion block from the three independent checks."""

    reasons: list[str] = []

    if exact_replay.get("status") != PASS:
        reasons.append(
            f"exact_replay:{exact_replay.get('status')}:{exact_replay.get('divergence_reason')}"
        )
    if critical_event_reproduced.get("status") != PASS:
        reasons.append(
            f"critical_event_reproduced:{critical_event_reproduced.get('status')}:"
            f"event_type={critical_event_reproduced.get('event_type')}"
        )

    cell_verdicts = list(cell_verdicts)
    if not cell_verdicts:
        reasons.append("perturbation_persistence:no_cells:empty preregistered grid")
    else:
        for cell in cell_verdicts:
            if cell.get("verdict") != PASS:
                reasons.append(
                    f"perturbation_cell:{cell.get('timing_offset_s')}:"
                    f"{cell.get('speed_delta_m_s')}:{cell.get('verdict')}"
                )
    if missing_cell_reasons:
        reasons.append(f"perturbation_persistence:missing_cells:{len(missing_cell_reasons)}")

    if reasons:
        return {
            "verdict": "reject",
            "exclusion_reason": "; ".join(reasons),
            "required_statuses": list(REQUIRED_STATUSES),
        }
    return {
        "verdict": "promote",
        "exclusion_reason": "all three independent status checks passed",
        "required_statuses": list(REQUIRED_STATUSES),
    }


def assess_exact_replay(
    source_episode: Mapping[str, Any],
    *,
    replayed_episode: Mapping[str, Any] | None = None,
    replay_error: str | None = None,
) -> dict[str, Any]:
    """Compare a replayed episode to its source and return an exact-replay block.

    ``replayed_episode`` is the byte/config-equivalent replay of
    ``source_episode``.  When it is missing or diverges, the block records the
    concrete reason and a ``fail``/``unknown`` status.  The digest is computed
    over the deterministic identity of the source episode so identical inputs
    produce identical digests.

    Returns:
        An ``exact_replay`` block with status, divergence_reason, and digest.
    """

    source_missing = _missing_episode_fields(source_episode)
    source_digest = _episode_digest(source_episode)
    if source_missing:
        return {
            "status": FAIL,
            "divergence_reason": f"source episode missing required fields: {source_missing}",
            "replay_digest": source_digest,
        }
    if replayed_episode is None:
        status = UNKNOWN if replay_error is None else FAIL
        reason = replay_error or "replay not executed"
        return {
            "status": status,
            "divergence_reason": reason,
            "replay_digest": source_digest,
        }
    replay_missing = _missing_episode_fields(replayed_episode)
    replay_digest = _episode_digest(replayed_episode)
    if replay_missing:
        return {
            "status": FAIL,
            "divergence_reason": f"replayed episode missing required fields: {replay_missing}",
            "replay_digest": source_digest,
        }
    if replay_digest != source_digest:
        return {
            "status": FAIL,
            "divergence_reason": (
                f"replay digest mismatch: source={source_digest} replay={replay_digest}"
            ),
            "replay_digest": source_digest,
        }
    return {
        "status": PASS,
        "divergence_reason": "byte/config-equivalent replay matched source episode",
        "replay_digest": source_digest,
    }


def _missing_episode_fields(episode: Mapping[str, Any]) -> list[str]:
    """Return required episode identity fields that are absent or empty."""
    return [
        field
        for field in _REQUIRED_EPISODE_FIELDS
        if field not in episode
        or episode[field] is None
        or (isinstance(episode[field], str) and not episode[field].strip())
    ]


def _episode_digest(episode: Mapping[str, Any]) -> str:
    """Digest only the exact-replay identity fields.

    Returns:
        The deterministic identity digest.
    """
    return _stable_digest({field: episode.get(field) for field in _REQUIRED_EPISODE_FIELDS})


def assess_critical_event_reproduction(
    *,
    event_type: str,
    source_event_time_s: float,
    source_event_location: Sequence[float],
    replayed_event_time_s: float | None = None,
    replayed_event_location: Sequence[float] | None = None,
    time_tolerance_s: float = 0.0,
    location_tolerance_m: float = 0.0,
    not_observed_reason: str | None = None,
) -> dict[str, Any]:
    """Return a critical-event-reproduction block from source vs replayed event.

    The event is reproduced when an event of ``event_type`` is observed in the
    replayed episode within the declared time and location tolerances.  Missing
    replay fields or an out-of-tolerance observation fail closed.

    Returns:
        A ``critical_event_reproduced`` block with status and observed deltas.
    """

    if replayed_event_time_s is None or replayed_event_location is None:
        return {
            "status": UNKNOWN if not_observed_reason is None else FAIL,
            "event_type": event_type,
            "time_tolerance_s": float(time_tolerance_s),
            "location_tolerance_m": float(location_tolerance_m),
            "observed_time_delta_s": None,
            "observed_location_delta_m": None,
        }

    time_delta = abs(float(replayed_event_time_s) - float(source_event_time_s))
    location_delta = float(
        math.dist(
            [float(value) for value in replayed_event_location],
            [float(value) for value in source_event_location],
        )
    )
    reproduced = time_delta <= float(time_tolerance_s) and location_delta <= float(
        location_tolerance_m
    )
    return {
        "status": PASS if reproduced else FAIL,
        "event_type": event_type,
        "time_tolerance_s": float(time_tolerance_s),
        "location_tolerance_m": float(location_tolerance_m),
        "observed_time_delta_s": time_delta,
        "observed_location_delta_m": location_delta,
    }


def evaluate_perturbation_grid(
    *,
    timing_offsets_s: Sequence[float],
    speed_deltas_m_s: Sequence[float],
    cell_verdict_fn: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run a preregistered grid and return per-cell verdicts and missing reasons.

    ``cell_verdict_fn`` receives ``(timing_offset_s, speed_delta_m_s)`` and must
    return either ``{"verdict": "pass"|"fail", "reason": <str>}`` or
    ``None`` when the cell cannot be evaluated (recorded as a missing-cell
    reason rather than a silent pass).

    Returns:
        The cells list and the missing-cell-reasons list.
    """

    cells: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for timing_offset_s in timing_offsets_s:
        for speed_delta_m_s in speed_deltas_m_s:
            result = cell_verdict_fn(
                timing_offset_s=float(timing_offset_s),
                speed_delta_m_s=float(speed_delta_m_s),
            )
            if result is None:
                missing.append(
                    {
                        "timing_offset_s": float(timing_offset_s),
                        "speed_delta_m_s": float(speed_delta_m_s),
                        "reason": "cell_not_evaluable",
                    }
                )
                continue
            verdict = result.get("verdict", FAIL)
            cells.append(
                {
                    "timing_offset_s": float(timing_offset_s),
                    "speed_delta_m_s": float(speed_delta_m_s),
                    "verdict": verdict,
                    "reason": str(result.get("reason", "")),
                }
            )
    return cells, missing


def validate_persistence_record(record: Mapping[str, Any]) -> None:
    """Fail closed unless *record* is a valid, internally consistent persistence entry.

    Raises:
        ScenarioPersistenceValidationError: If JSON Schema or the fail-closed
            promotion invariants reject the record.
    """

    _verify_schema(record)
    if not record["config"].get("frozen"):
        raise ScenarioPersistenceValidationError("config.frozen must be true")
    promotion = record["promotion"]
    expected_statuses = set(REQUIRED_STATUSES)
    if set(promotion.get("required_statuses", [])) != expected_statuses:
        raise ScenarioPersistenceValidationError(
            "promotion.required_statuses must be the three independent checks"
        )

    all_pass = (
        record["exact_replay"]["status"] == PASS
        and record["critical_event_reproduced"]["status"] == PASS
        and all(cell.get("verdict") == PASS for cell in record["perturbation_persistence"]["cells"])
        and not record["perturbation_persistence"]["missing_cell_reasons"]
    )
    if all_pass and promotion["verdict"] != "promote":
        raise ScenarioPersistenceValidationError(
            "all checks passed but promotion verdict is not 'promote'"
        )
    if not all_pass and promotion["verdict"] == "promote":
        raise ScenarioPersistenceValidationError(
            "promotion verdict is 'promote' but a required check did not pass"
        )


__all__ = [
    "FAIL",
    "PASS",
    "PERSISTENCE_SCHEMA_VERSION",
    "REQUIRED_STATUSES",
    "UNKNOWN",
    "ScenarioPersistenceValidationError",
    "assess_critical_event_reproduction",
    "assess_exact_replay",
    "compute_persistence_record",
    "evaluate_perturbation_grid",
    "load_persistence_schema",
    "validate_persistence_record",
]
