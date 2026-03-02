#!/usr/bin/env python3
"""Migrate episode JSONL records to include v1 schema defaults.

Usage:
  uv run python scripts/tools/migrate_episode_schema_v1.py \
    --input output/benchmarks/episodes.jsonl \
    --output output/benchmarks/episodes_v1.jsonl

This script adds missing fields that the v1 schema expects or documents:
- version: "v1" when missing
- episode_id/scenario_id/seed: synthesized when missing
- metrics: minimally populated when missing
- termination_reason/outcome/integrity: synthesized when missing
- scenario_params: {} when missing
- algorithm_metadata: {"status": "unknown"} when missing
- timestamps: inferred from created_at when possible
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robot_sf.benchmark.termination_reason import (
    outcome_contradictions,
    resolve_termination_reason,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input JSONL path")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL path (will be overwritten)",
    )
    return parser.parse_args()


def _to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def _to_int(value: Any, default: int = 0) -> int:
    """Return ``value`` coerced to int, falling back to ``default`` on invalid input."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _metric_scalar(metrics: Any, *keys: str, default: float = 0.0) -> float:
    """Return the first parseable numeric metric among candidate keys."""
    if not isinstance(metrics, dict):
        return float(default)
    for key in keys:
        if key not in metrics:
            continue
        try:
            return float(metrics.get(key))
        except (TypeError, ValueError):
            continue
    return float(default)


def _scenario_id(record: dict[str, Any]) -> str:
    """Resolve a non-empty scenario id from legacy record fields."""
    direct = record.get("scenario_id")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    params = record.get("scenario_params")
    if isinstance(params, dict):
        for key in ("id", "scenario_id", "name"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "legacy_unknown_scenario"


def _infer_outcome(record: dict[str, Any], termination_reason: str) -> dict[str, bool]:
    """Infer outcome payload from legacy status/metrics and termination reason."""
    existing = record.get("outcome")
    if isinstance(existing, dict):
        if {"route_complete", "collision_event", "timeout_event"} <= set(existing.keys()):
            return {
                "route_complete": bool(existing.get("route_complete")),
                "collision_event": bool(existing.get("collision_event")),
                "timeout_event": bool(existing.get("timeout_event")),
            }

    metrics = record.get("metrics")
    collision_metric = (
        _metric_scalar(
            metrics,
            "collisions",
            "collision_rate",
            default=0.0,
        )
        > 0.0
    )
    success_metric = (
        _metric_scalar(
            metrics,
            "success",
            "success_rate",
            default=0.0,
        )
        > 0.0
    )

    status = str(record.get("status", "")).strip().lower()
    collision = bool(termination_reason == "collision" or status == "collision" or collision_metric)
    route_complete = bool(
        not collision and (termination_reason == "success" or status == "success" or success_metric)
    )
    timeout = bool(termination_reason in {"max_steps", "truncated"})
    return {
        "route_complete": route_complete,
        "collision_event": collision,
        "timeout_event": timeout,
    }


def _infer_termination_reason(record: dict[str, Any], outcome: dict[str, bool]) -> str:
    """Infer a schema-compliant termination reason for migrated legacy records."""
    existing = record.get("termination_reason")
    if isinstance(existing, str):
        normalized = existing.strip().lower()
        if normalized in {
            "success",
            "collision",
            "terminated",
            "truncated",
            "max_steps",
            "error",
        }:
            return normalized

    status = str(record.get("status", "")).strip().lower()
    if status == "collision":
        return "collision"
    if status == "success":
        return "success"
    return resolve_termination_reason(
        terminated=False,
        truncated=bool(outcome.get("timeout_event")),
        success=bool(outcome.get("route_complete")),
        collision=bool(outcome.get("collision_event")),
        reached_max_steps=bool(outcome.get("timeout_event")),
    )


def _migrate_record(record: dict[str, Any]) -> dict[str, Any]:
    updated = dict(record)
    if "version" not in updated:
        updated["version"] = "v1"
    scenario_id = _scenario_id(updated)
    updated["scenario_id"] = scenario_id
    updated["seed"] = _to_int(updated.get("seed"), default=0)
    if "episode_id" not in updated or not str(updated.get("episode_id", "")).strip():
        updated["episode_id"] = f"{scenario_id}::seed={updated['seed']}"
    metrics = updated.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        metrics = {"success": 0.0}
        updated["metrics"] = metrics
    if "scenario_params" not in updated:
        updated["scenario_params"] = {}
    if "algorithm_metadata" not in updated:
        updated["algorithm_metadata"] = {"status": "unknown"}

    outcome = _infer_outcome(updated, termination_reason="")
    termination_reason = _infer_termination_reason(updated, outcome)
    outcome = _infer_outcome(updated, termination_reason=termination_reason)
    updated["termination_reason"] = termination_reason
    updated["outcome"] = outcome
    integrity = updated.get("integrity")
    if isinstance(integrity, dict) and isinstance(integrity.get("contradictions"), list):
        updated["integrity"] = integrity
    else:
        updated["integrity"] = {
            "contradictions": outcome_contradictions(
                termination_reason=termination_reason,
                outcome=outcome,
                metrics=updated.get("metrics"),
            )
        }

    if "timestamps" not in updated:
        created_at = updated.get("created_at")
        if isinstance(created_at, (int, float)):
            iso = _to_iso(float(created_at))
            updated["timestamps"] = {"start": iso, "end": iso}
    return updated


def main() -> int:
    """Run the schema v1 migration CLI.

    Returns:
        Exit code (0 for success).
    """
    args = _parse_args()
    input_path: Path = args.input
    output_path: Path = args.output

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open("r", encoding="utf-8") as src,
        output_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            migrated = _migrate_record(record)
            dst.write(json.dumps(migrated) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
