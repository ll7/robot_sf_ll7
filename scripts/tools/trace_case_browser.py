#!/usr/bin/env python3
"""Find edge-case benchmark episodes and critical trace windows.

The browser reads one or more ``episodes.jsonl`` files, or recursively discovers
those files below a run directory. It only reads inputs and writes nothing.

Episode field assumptions are centralized in :data:`FIELD_PATHS`:

* arm: ``arm``, ``planner_key``, ``algo``,
  ``algorithm_metadata.canonical_algorithm``, or
  ``algorithm_metadata.algorithm``;
* scenario: ``scenario_id``, ``scenario``, ``scenario_name``,
  ``scenario_params.id``, or ``scenario_params.name``;
* seed: ``seed``, ``scenario_seed``, ``episode_seed``, or
  ``scenario_params.simulation_config.route_spawn_seed``;
* outcome: canonical top-level ``status``/``termination_reason`` first, then
  ``outcome`` flags and ``metrics.success``/collision/timeout fields;
* steps: top-level ``steps``/``num_steps``/``episode_length``, then the nested
  simulation trace length;
* near misses: ``metrics.near_misses`` and common count aliases;
* minimum pedestrian clearance: ``metrics.min_clearance`` and explicit
  pedestrian-clearance aliases (center distance is not mislabeled as clearance);
* trace: ``algorithm_metadata.simulation_step_trace`` or top-level
  ``simulation_step_trace``.

The ``critical`` command adapts current ``simulation-step-trace.v1`` data via
``robot_sf.benchmark.critical_intervals.adapt_simulation_step_trace``. That
bounded adapter requires stable pedestrian IDs across recorded steps. Critical
trace distances remain center-to-center because this trace schema does not carry
actor radii; list/summary clearance values come from the episode metrics.

Pair output includes execution-commit, scenario-contract, and runtime-contract
compatibility. Missing or mismatched provenance remains visible as a caveat;
such a pair is a discovery lead, not comparison-ready evidence.

Executable figure commands are emitted for any pair the generic fail-closed
adapter (``scripts/repro/trace_series_adapter.py``, Issue #5883) can convert:
each row is selected exactly by source path plus episode identity
(``episode_id``, or the ``scenario``/``planner``/``seed`` triple) from its own
source JSONL, then adapted into the ``trace_series.json`` /
``metadata.json`` contract consumed by the figure renderer. Pairs that fall
outside the adapter's fail-closed contract (unknown trace schema, missing
provenance, actor-set changes, zero-pedestrian frames, ambiguous same-seed
rows) return ``command_hint.status=adapter_required`` and no commands instead
of emitting a command that would fail at runtime.

Examples:
    uv run python scripts/tools/trace_case_browser.py list RUNS_DIR --sort=-near
    uv run python scripts/tools/trace_case_browser.py list EPISODES --filter outcome=collision
    uv run python scripts/tools/trace_case_browser.py pairs RUNS_DIR --json
    uv run python scripts/tools/trace_case_browser.py pairs GOAL.jsonl PPO.jsonl --json
    uv run python scripts/tools/trace_case_browser.py critical EPISODES --seed 114
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.critical_intervals import (
    DEFAULT_NEAR_MISS_DIST,
    CriticalInterval,
    IntervalMetrics,
    adapt_simulation_step_trace,
    extract_critical_intervals,
    summarize_interval_metrics,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

FieldPath = tuple[str, ...]
SortValue = str | int | float | bool | None

FIELD_PATHS: dict[str, tuple[FieldPath, ...]] = {
    "arm": (
        ("arm",),
        ("planner_key",),
        ("algo",),
        ("algorithm_metadata", "canonical_algorithm"),
        ("algorithm_metadata", "algorithm"),
        ("scenario_params", "algo"),
    ),
    "scenario": (
        ("scenario_id",),
        ("scenario",),
        ("scenario_name",),
        ("scenario_params", "id"),
        ("scenario_params", "name"),
    ),
    "seed": (
        ("seed",),
        ("scenario_seed",),
        ("episode_seed",),
        ("scenario_params", "simulation_config", "route_spawn_seed"),
    ),
    "status": (("status",), ("termination_reason",)),
    "steps": (("steps",), ("num_steps",), ("episode_length",)),
    "near_misses": (
        ("metrics", "near_misses"),
        ("metrics", "near_miss_count"),
        ("metrics", "near_miss"),
        ("near_misses",),
        ("near_miss_count",),
    ),
    "min_clearance": (
        ("metrics", "min_clearance"),
        ("metrics", "min_clearance_m"),
        ("metrics", "min_ped_clearance"),
        ("metrics", "min_pedestrian_clearance_m"),
        ("min_clearance",),
        ("min_pedestrian_clearance_m",),
    ),
    "episode_id": (("episode_id",), ("id",)),
    "trace": (
        ("algorithm_metadata", "simulation_step_trace"),
        ("simulation_step_trace",),
    ),
}

_OUTCOME_ALIASES = {
    "success": "success",
    "succeeded": "success",
    "goal": "success",
    "goal_reached": "success",
    "route_complete": "success",
    "completed": "success",
    "collision": "collision",
    "collided": "collision",
    "agent_collision": "collision",
    "ped_collision": "collision",
    "timeout": "timeout",
    "timed_out": "timeout",
    "time_limit": "timeout",
    "truncated": "timeout",
}
_OPPOSITE_FAILURES = frozenset({"collision", "timeout"})
_FILTER_RE = re.compile(
    r"^(outcome|arm|scenario|seed|near|near_misses|trace|has_trace)"
    r"\s*(>=|<=|!=|=|>|<)\s*(.+)$",
    re.IGNORECASE,
)
_SEED_RANGE_RE = re.compile(r"^(-?\d+)\s*(?:\.\.|:|-)\s*(-?\d+)$")

DEFAULT_CRITICAL_CONFIG: dict[str, Any] = {
    "critical_intervals": {
        "closest_approach": {
            "enabled": True,
            "before_s": 1.0,
            "after_s": 1.0,
        },
        "ttc_threshold_crossing": {
            "enabled": True,
            "threshold_s": 1.5,
            "before_s": 1.0,
            "after_s": 2.0,
        },
        "first_braking_event": {
            "enabled": True,
            "deceleration_threshold_mps2": 0.75,
            "before_s": 1.0,
            "after_s": 2.0,
        },
        "collision_or_near_miss": {
            "enabled": True,
            "before_s": 2.0,
            "after_s": 2.0,
        },
    }
}


class CaseBrowserError(ValueError):
    """User-facing input or selection error."""


@dataclass(frozen=True)
class Episode:
    """Normalized episode row plus its source location."""

    source: Path
    line_number: int
    arm: str
    scenario: str
    seed: int | None
    outcome: str
    steps: int | None
    near_misses: float | None
    min_clearance: float | None
    has_trace: bool
    episode_id: str
    git_commit: str | None
    config_hash: str | None
    scenario_fingerprint: str | None
    runtime_fingerprint: str | None
    record: dict[str, Any] = field(repr=False, compare=False)


def _nested_value(mapping: dict[str, Any], path: FieldPath) -> Any:
    """Return one nested value, or ``None`` when a path is absent."""
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first_value(mapping: dict[str, Any], field_name: str) -> Any:
    """Return the first non-empty configured field alias."""
    for path in FIELD_PATHS[field_name]:
        value = _nested_value(mapping, path)
        if value is not None and value != "":
            return value
    return None


def _finite_float(value: Any) -> float | None:
    """Coerce one finite number without treating booleans as numbers."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    """Coerce one exact finite integer."""
    numeric = _finite_float(value)
    if numeric is None or not numeric.is_integer():
        return None
    return int(numeric)


def _canonical_outcome_token(value: Any) -> str | None:
    """Normalize a recognized outcome spelling."""
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"[\s-]+", "_", value.strip().lower())
    return _OUTCOME_ALIASES.get(normalized)


def _flag_is_true(mapping: dict[str, Any], *paths: FieldPath) -> bool:
    """Return whether any named boolean/numeric flag is truthy."""
    for path in paths:
        value = _nested_value(mapping, path)
        if value is True:
            return True
        numeric = _finite_float(value)
        if numeric is not None and numeric > 0.0:
            return True
    return False


def _episode_outcome(record: dict[str, Any]) -> str:
    """Resolve a canonical success/collision/timeout label defensively."""
    for path in FIELD_PATHS["status"]:
        outcome = _canonical_outcome_token(_nested_value(record, path))
        if outcome is not None:
            return outcome

    if _flag_is_true(
        record,
        ("outcome", "collision_event"),
        ("metrics", "collisions"),
        ("metrics", "collision_count"),
        ("metrics", "total_collision_count"),
        ("collision",),
    ):
        return "collision"
    if _flag_is_true(
        record,
        ("outcome", "timeout_event"),
        ("metrics", "timeout"),
        ("timeout",),
    ):
        return "timeout"
    if _flag_is_true(
        record,
        ("outcome", "route_complete"),
        ("metrics", "success"),
        ("success",),
    ):
        return "success"
    return "other"


def _trace_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return a supported nested or columnar trace mapping from an episode row."""
    trace = _first_value(record, "trace")
    if isinstance(trace, dict):
        return trace
    if "robot_pos" in record:
        return record
    return None


def _canonical_fingerprint(value: Any) -> str | None:
    """Hash a JSON-compatible provenance contract deterministically."""
    if value is None:
        return None
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(encoded).hexdigest()


def _scenario_fingerprint(record: dict[str, Any]) -> str | None:
    """Hash scenario semantics while excluding planner and sampled seed fields."""
    scenario_params = record.get("scenario_params")
    if not isinstance(scenario_params, dict):
        return None
    comparison_params = {
        key: value
        for key, value in scenario_params.items()
        if key not in {"algo", "algo_config_hash", "observation_level", "observation_mode"}
    }
    simulation_config = comparison_params.get("simulation_config")
    if isinstance(simulation_config, dict):
        comparison_params["simulation_config"] = {
            key: value for key, value in simulation_config.items() if key != "route_spawn_seed"
        }
    return _canonical_fingerprint(comparison_params)


def _runtime_fingerprint(record: dict[str, Any]) -> str | None:
    """Hash simulator settings shared across planner arms."""
    provenance = record.get("result_provenance")
    if not isinstance(provenance, dict):
        return None
    settings = provenance.get("simulator_settings")
    if not isinstance(settings, dict):
        return None
    comparison_settings = {
        key: value
        for key, value in settings.items()
        if key not in {"observation_level", "observation_mode"}
    }
    return _canonical_fingerprint(comparison_settings)


def _normalize_episode(record: dict[str, Any], source: Path, line_number: int) -> Episode:
    """Normalize one schema-variable JSONL row."""
    arm_value = _first_value(record, "arm")
    scenario_value = _first_value(record, "scenario")
    episode_id_value = _first_value(record, "episode_id")
    trace = _trace_from_record(record)
    trace_steps = trace.get("steps") if isinstance(trace, dict) else None
    steps = _integer(_first_value(record, "steps"))
    if steps is None and isinstance(trace_steps, list):
        steps = len(trace_steps)
    provenance = record.get("result_provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    git_commit = provenance.get("repo_commit", record.get("git_hash"))
    config_hash = provenance.get("config_hash", record.get("config_hash"))

    return Episode(
        source=source,
        line_number=line_number,
        arm=str(arm_value) if arm_value is not None else source.parent.name,
        scenario=str(scenario_value) if scenario_value is not None else "unknown",
        seed=_integer(_first_value(record, "seed")),
        outcome=_episode_outcome(record),
        steps=steps,
        near_misses=_finite_float(_first_value(record, "near_misses")),
        min_clearance=_finite_float(_first_value(record, "min_clearance")),
        has_trace=trace is not None,
        episode_id=(
            str(episode_id_value)
            if episode_id_value is not None
            else f"{source.name}:{line_number}"
        ),
        git_commit=str(git_commit) if git_commit is not None else None,
        config_hash=str(config_hash) if config_hash is not None else None,
        scenario_fingerprint=_scenario_fingerprint(record),
        runtime_fingerprint=_runtime_fingerprint(record),
        record=record,
    )


def _discover_episode_files(inputs: Sequence[Path]) -> list[Path]:
    """Resolve explicit JSONL files and recursively discover run-directory files."""
    discovered: list[Path] = []
    seen: set[Path] = set()
    for input_path in inputs:
        path = input_path.expanduser()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            candidates = sorted(path.rglob("episodes.jsonl"))
            if not candidates:
                raise CaseBrowserError(f"no episodes.jsonl files found under {path}")
        else:
            raise CaseBrowserError(f"input path does not exist: {path}")
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(resolved)
    if not discovered:
        raise CaseBrowserError("no episode files were supplied")
    return discovered


def load_episodes(inputs: Sequence[Path]) -> tuple[list[Episode], list[Path]]:
    """Load and normalize all non-empty JSONL rows from the requested inputs."""
    files = _discover_episode_files(inputs)
    episodes: list[Episode] = []
    for source in files:
        try:
            with source.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise CaseBrowserError(
                            f"{source}:{line_number}: invalid JSON: {exc.msg}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise CaseBrowserError(
                            f"{source}:{line_number}: episode row must be a JSON object"
                        )
                    episodes.append(_normalize_episode(record, source, line_number))
        except OSError as exc:
            raise CaseBrowserError(f"could not read {source}: {exc}") from exc
    return episodes, files


def _episode_dict(episode: Episode) -> dict[str, Any]:
    """Return the stable JSON-facing episode summary."""
    return {
        "arm": episode.arm,
        "scenario": episode.scenario,
        "seed": episode.seed,
        "outcome": episode.outcome,
        "steps": episode.steps,
        "near_misses": episode.near_misses,
        "min_clearance": episode.min_clearance,
        "has_trace": episode.has_trace,
        "episode_id": episode.episode_id,
        "git_commit": episode.git_commit,
        "config_hash": episode.config_hash,
        "scenario_fingerprint": episode.scenario_fingerprint,
        "runtime_fingerprint": episode.runtime_fingerprint,
        "source": str(episode.source),
        "line_number": episode.line_number,
    }


def _numeric_compare(actual: float | int | None, operator: str, expected: float) -> bool:
    """Apply a numeric filter while making missing values non-matches."""
    if actual is None:
        return False
    if operator == "=":
        return float(actual) == expected
    if operator == "!=":
        return float(actual) != expected
    if operator == ">=":
        return float(actual) >= expected
    if operator == "<=":
        return float(actual) <= expected
    if operator == ">":
        return float(actual) > expected
    if operator == "<":
        return float(actual) < expected
    raise CaseBrowserError(f"unsupported numeric operator {operator!r}")


def _parse_seed_ranges(specification: str) -> set[int]:
    """Expand comma-separated integer/range selectors such as ``111-120,130``."""
    values: set[int] = set()
    for token in specification.split(","):
        token = token.strip()
        if not token:
            raise CaseBrowserError(f"empty seed selector in {specification!r}")
        range_match = _SEED_RANGE_RE.fullmatch(token)
        if range_match:
            start, end = (int(value) for value in range_match.groups())
            low, high = sorted((start, end))
            values.update(range(low, high + 1))
            continue
        try:
            values.add(int(token))
        except ValueError as exc:
            raise CaseBrowserError(f"invalid seed selector {token!r}") from exc
    return values


def _parse_bool(value: str) -> bool:
    """Parse a human boolean filter value."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise CaseBrowserError(f"invalid trace boolean {value!r}; use true or false")


def _compile_filter(expression: str) -> Callable[[Episode], bool]:  # noqa: C901
    """Parse one filter once and return its episode predicate."""
    match = _FILTER_RE.fullmatch(expression.strip())
    if match is None:
        raise CaseBrowserError(
            f"invalid filter {expression!r}; expected outcome=..., near>=N, or seed=111-120"
        )
    field_name, operator, raw_expected = match.groups()
    field_name = field_name.lower()
    raw_expected = raw_expected.strip()

    if field_name == "outcome":
        if operator not in {"=", "!="}:
            raise CaseBrowserError("outcome filters support only = and !=")
        expected = frozenset(
            _canonical_outcome_token(token) or token.strip().lower()
            for token in raw_expected.split(",")
        )
        return lambda episode: (episode.outcome in expected) == (operator == "=")

    if field_name in {"arm", "scenario"}:
        if operator not in {"=", "!="}:
            raise CaseBrowserError(f"{field_name} filters support only = and !=")
        expected_text = raw_expected.lower()
        return lambda episode: (
            (getattr(episode, field_name).lower() == expected_text) == (operator == "=")
        )

    if field_name == "seed" and operator in {"=", "!="}:
        expected_seeds = frozenset(_parse_seed_ranges(raw_expected))
        return lambda episode: (episode.seed in expected_seeds) == (operator == "=")

    if field_name in {"trace", "has_trace"}:
        if operator not in {"=", "!="}:
            raise CaseBrowserError("trace filters support only = and !=")
        expected_trace = _parse_bool(raw_expected)
        return lambda episode: (episode.has_trace == expected_trace) == (operator == "=")

    if field_name not in {"seed", "near", "near_misses"}:
        raise CaseBrowserError(f"unsupported numeric filter field {field_name!r}")
    try:
        expected_number = float(raw_expected)
    except ValueError as exc:
        raise CaseBrowserError(f"filter value must be numeric: {raw_expected!r}") from exc
    if not math.isfinite(expected_number):
        raise CaseBrowserError(f"filter value must be finite: {raw_expected!r}")

    def matches_numeric(episode: Episode) -> bool:
        actual = episode.seed if field_name == "seed" else episode.near_misses
        return _numeric_compare(actual, operator, expected_number)

    return matches_numeric


def _matches_filter(episode: Episode, expression: str) -> bool:
    """Evaluate one filter outside the compiled multi-episode path."""
    return _compile_filter(expression)(episode)


def filter_episodes(episodes: Sequence[Episode], expressions: Sequence[str]) -> list[Episode]:
    """Apply all repeated filters with AND semantics."""
    predicates = [_compile_filter(expression) for expression in expressions]
    return [episode for episode in episodes if all(predicate(episode) for predicate in predicates)]


_SORT_ACCESSORS: dict[str, Callable[[Episode], SortValue]] = {
    "arm": lambda episode: episode.arm.lower(),
    "scenario": lambda episode: episode.scenario.lower(),
    "seed": lambda episode: episode.seed,
    "outcome": lambda episode: episode.outcome,
    "steps": lambda episode: episode.steps,
    "near": lambda episode: episode.near_misses,
    "near_misses": lambda episode: episode.near_misses,
    "clearance": lambda episode: episode.min_clearance,
    "min_clearance": lambda episode: episode.min_clearance,
    "trace": lambda episode: episode.has_trace,
}


def sort_episodes(episodes: Sequence[Episode], specification: str) -> list[Episode]:
    """Sort by comma-separated fields; prefix descending fields with ``-``."""
    parsed: list[tuple[Callable[[Episode], SortValue], bool]] = []
    for raw_field in specification.split(","):
        token = raw_field.strip()
        if not token:
            continue
        descending = token.startswith("-") or token.lower().endswith(":desc")
        if token.startswith(("-", "+")):
            token = token[1:]
        token = re.sub(r":(?:asc|desc)$", "", token, flags=re.IGNORECASE).lower()
        accessor = _SORT_ACCESSORS.get(token)
        if accessor is None:
            raise CaseBrowserError(
                f"unknown sort field {token!r}; choose from {sorted(_SORT_ACCESSORS)}"
            )
        parsed.append((accessor, descending))
    if not parsed:
        raise CaseBrowserError("sort specification must name at least one field")

    result = list(episodes)
    for accessor, descending in reversed(parsed):
        present = [episode for episode in result if accessor(episode) is not None]
        missing = [episode for episode in result if accessor(episode) is None]
        present.sort(key=accessor, reverse=descending)
        result = [*present, *missing]
    return result


def _display_number(value: float | int | None, *, digits: int = 3) -> str:
    """Format compact table numbers while preserving explicit missing values."""
    if value is None:
        return "-"
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.{digits}f}"


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render a dependency-free aligned text table."""
    rendered_rows = [[str(value) for value in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    header_line = "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    separator = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rendered_rows
    ]
    return "\n".join([header_line, separator, *body])


def _print_json(payload: dict[str, Any]) -> None:
    """Write stable, strict JSON to stdout."""
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _run_list(args: argparse.Namespace, episodes: Sequence[Episode], files: Sequence[Path]) -> int:
    """Render normalized episodes after requested list filters and sorting."""
    selected = filter_episodes(episodes, args.filters)
    selected = sort_episodes(selected, args.sort)
    if args.json_output:
        _print_json(
            {
                "count": len(selected),
                "episodes": [_episode_dict(episode) for episode in selected],
                "sources": [str(path) for path in files],
            }
        )
        return 0

    rows = [
        (
            episode.arm,
            episode.scenario,
            _display_number(episode.seed),
            episode.outcome,
            _display_number(episode.steps),
            _display_number(episode.near_misses),
            _display_number(episode.min_clearance),
            "yes" if episode.has_trace else "no",
        )
        for episode in selected
    ]
    print(
        _format_table(
            (
                "ARM",
                "SCENARIO",
                "SEED",
                "OUTCOME",
                "STEPS",
                "NEAR",
                "MIN PED CLEARANCE",
                "TRACE",
            ),
            rows,
        )
    )
    print(f"\n{len(selected)} episode(s) from {len(files)} file(s)")
    return 0


def _summary_cells(episodes: Sequence[Episode]) -> list[dict[str, Any]]:
    """Aggregate the requested per-arm and per-scenario summary cells."""
    groups: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for episode in episodes:
        groups[(episode.arm, episode.scenario)].append(episode)

    cells: list[dict[str, Any]] = []
    for (arm, scenario), rows in sorted(groups.items()):
        near_values = [row.near_misses for row in rows if row.near_misses is not None]
        clearance_values = [row.min_clearance for row in rows if row.min_clearance is not None]
        cells.append(
            {
                "arm": arm,
                "scenario": scenario,
                "n": len(rows),
                "success": sum(row.outcome == "success" for row in rows),
                "collision": sum(row.outcome == "collision" for row in rows),
                "timeout": sum(row.outcome == "timeout" for row in rows),
                "near_miss_mean": (sum(near_values) / len(near_values) if near_values else None),
                "near_miss_max": max(near_values) if near_values else None,
                "min_clearance_min": min(clearance_values) if clearance_values else None,
            }
        )
    return cells


def _run_summary(
    args: argparse.Namespace,
    episodes: Sequence[Episode],
    files: Sequence[Path],
) -> int:
    """Render per-arm and per-scenario aggregate cells."""
    cells = _summary_cells(episodes)
    if args.json_output:
        _print_json(
            {
                "cells": cells,
                "episode_count": len(episodes),
                "sources": [str(path) for path in files],
            }
        )
        return 0
    rows = [
        (
            cell["arm"],
            cell["scenario"],
            cell["n"],
            cell["success"],
            cell["collision"],
            cell["timeout"],
            _display_number(cell["near_miss_mean"], digits=2),
            _display_number(cell["near_miss_max"]),
            _display_number(cell["min_clearance_min"]),
        )
        for cell in cells
    ]
    print(
        _format_table(
            (
                "ARM",
                "SCENARIO",
                "N",
                "SUCCESS",
                "COLLISION",
                "TIMEOUT",
                "NEAR MEAN",
                "NEAR MAX",
                "MIN CLEARANCE",
            ),
            rows,
        )
    )
    return 0


def _opposite_outcomes(first: Episode, second: Episode) -> bool:
    """Return whether exactly one episode succeeded and the other failed."""
    return (first.outcome == "success" and second.outcome in _OPPOSITE_FAILURES) or (
        second.outcome == "success" and first.outcome in _OPPOSITE_FAILURES
    )


def _ordered_pair(first: Episode, second: Episode) -> tuple[Episode, Episode]:
    """Order a pair as success A and failure B for the hinge-figure contract."""
    return (first, second) if first.outcome == "success" else (second, first)


def _pair_scores(
    first: Episode,
    second: Episode,
    *,
    seed_distance: int,
) -> dict[str, float]:
    """Compute transparent diagnostic ranking components."""
    near_values = [value for value in (first.near_misses, second.near_misses) if value is not None]
    clearances = [
        value for value in (first.min_clearance, second.min_clearance) if value is not None
    ]
    near_extremity = max(near_values, default=0.0)
    clearance_contrast = abs(clearances[0] - clearances[1]) if len(clearances) == 2 else 0.0
    marginality = 1.0 / (1.0 + min(map(abs, clearances))) if clearances else 0.0
    seed_proximity = 1.0 / (1.0 + seed_distance)
    extremity = near_extremity + clearance_contrast
    return {
        "marginality_score": round(marginality, 6),
        "extremity_score": round(extremity, 6),
        "seed_proximity_score": round(seed_proximity, 6),
        "rank_score": round(marginality + extremity + seed_proximity, 6),
    }


def _slug(value: str) -> str:
    """Make a safe, readable output-path component."""
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return normalized or "case"


def _adapter_gaps(episode: Episode) -> list[str]:
    """Return reasons the generic fail-closed adapter would reject an episode.

    The generic adapter (Issue #5883) requires a supported ``simulation-step-trace``
    and an exact identity (``episode_id``, or ``scenario``/``planner``/``seed``),
    which the normalized :class:`Episode` already carries; this only checks the
    checks the browser can perform without re-reading the source file.
    """
    gaps: list[str] = []
    if not episode.has_trace:
        gaps.append("no simulation trace")
    if episode.seed is None:
        gaps.append("missing seed")
    if episode.git_commit is None:
        gaps.append("missing execution commit")
    return gaps


def _command_hint(kind: str, episode_a: Episode, episode_b: Episode) -> dict[str, Any]:
    """Build bundle-preparation and hinge-figure commands for one matched pair.

    Uses the generic fail-closed adapter (Issue #5883) when both rows carry a
    convertible trace and exact identity. Otherwise emits ``adapter_required`` with
    an actionable reason rather than a command that would fail at runtime.
    """
    gaps_a = _adapter_gaps(episode_a)
    gaps_b = _adapter_gaps(episode_b)
    if gaps_a or gaps_b:
        reasons = []
        if gaps_a:
            reasons.append("episode A: " + ", ".join(gaps_a))
        if gaps_b:
            reasons.append("episode B: " + ", ".join(gaps_b))
        return {
            "status": "adapter_required",
            "working_directory": "repository root",
            "commands": [],
            "note": (
                "No executable command is emitted: the generic fail-closed adapter "
                "(Issue #5883) cannot convert this pair ("
                + "; ".join(reasons)
                + "). The adapter selects each row exactly by source identity and rejects "
                "unknown trace schemas, missing provenance, actor-set changes, "
                "zero-pedestrian frames, and ambiguous same-seed rows."
            ),
        }

    case_name = (
        f"{kind}__{_slug(episode_a.scenario)}"
        f"__{_slug(episode_a.arm)}-seed{episode_a.seed}"
        f"__vs__{_slug(episode_b.arm)}-seed{episode_b.seed}"
    )
    case_root = Path("output") / "trace_case_browser" / case_name
    bundle_a = case_root / "episode_a"
    bundle_b = case_root / "episode_b"
    figure_dir = case_root / "figure"
    adapter = "scripts/repro/trace_series_adapter.py"
    figure = "scripts/repro/butterfly_hinge_figure_proto.py"

    def prepare_command(episode: Episode, output_dir: Path) -> str:
        identity = (
            f"--episode-id {shlex.quote(str(episode.episode_id))}"
            if episode.episode_id
            and episode.episode_id != f"{episode.source.name}:{episode.line_number}"
            else (
                f"--scenario {shlex.quote(episode.scenario)}"
                f" --planner {shlex.quote(episode.arm)}"
                f" --seed {episode.seed}"
            )
        )
        return (
            f"uv run python {adapter} build-bundle"
            f" --episodes-jsonl {shlex.quote(str(episode.source))}"
            f" {identity}"
            f" --out-dir {shlex.quote(str(output_dir))}"
        )

    label_a = f"{episode_a.arm} seed {episode_a.seed} ({episode_a.outcome})"
    label_b = f"{episode_b.arm} seed {episode_b.seed} ({episode_b.outcome})"
    commands = [
        prepare_command(episode_a, bundle_a),
        prepare_command(episode_b, bundle_b),
        (
            f"uv run python {figure}"
            f" --episode-a {shlex.quote(str(bundle_a))}"
            f" --episode-b {shlex.quote(str(bundle_b))}"
            f" --label-a {shlex.quote(label_a)}"
            f" --label-b {shlex.quote(label_b)}"
            f" --out-dir {shlex.quote(str(figure_dir))}"
        ),
    ]
    return {
        "status": "ready",
        "working_directory": "repository root",
        "commands": commands,
        "note": (
            "The generic adapter selects each row exactly by source identity; use "
            "separate per-arm source files for same-seed planner upsets."
        ),
    }


def _provenance_compatibility(episode_a: Episode, episode_b: Episode) -> dict[str, Any]:
    """Report whether a pair shares the contracts needed for comparison."""
    fields = {
        "execution_commit": (episode_a.git_commit, episode_b.git_commit),
        "scenario_contract": (
            episode_a.scenario_fingerprint,
            episode_b.scenario_fingerprint,
        ),
        "runtime_contract": (
            episode_a.runtime_fingerprint,
            episode_b.runtime_fingerprint,
        ),
    }
    checks: dict[str, bool | None] = {}
    caveats: list[str] = []
    values: dict[str, dict[str, str | None]] = {}
    for name, (value_a, value_b) in fields.items():
        values[name] = {"episode_a": value_a, "episode_b": value_b}
        if value_a is None or value_b is None:
            checks[f"same_{name}"] = None
            caveats.append(f"{name} missing for one or both episodes")
        elif value_a != value_b:
            checks[f"same_{name}"] = False
            caveats.append(f"{name} mismatch")
        else:
            checks[f"same_{name}"] = True
    return {
        **values,
        **checks,
        "comparison_ready": all(value is True for value in checks.values()),
        "caveats": caveats,
    }


def _pair_payload(kind: str, first: Episode, second: Episode) -> dict[str, Any]:
    """Build one ranked pair with rationale and ready-to-paste commands."""
    episode_a, episode_b = _ordered_pair(first, second)
    seed_distance = abs((episode_a.seed or 0) - (episode_b.seed or 0))
    scores = _pair_scores(episode_a, episode_b, seed_distance=seed_distance)
    near_max = max(
        (value for value in (episode_a.near_misses, episode_b.near_misses) if value is not None),
        default=None,
    )
    closest_boundary = min(
        (
            abs(value)
            for value in (episode_a.min_clearance, episode_b.min_clearance)
            if value is not None
        ),
        default=None,
    )
    why = (
        f"opposite outcomes; seed gap {seed_distance}; "
        f"max near misses {_display_number(near_max)}; "
        f"closest |clearance| {_display_number(closest_boundary)} m"
    )
    provenance_compatibility = _provenance_compatibility(episode_a, episode_b)
    command_hint = _command_hint(kind, episode_a, episode_b)
    if not provenance_compatibility["comparison_ready"]:
        command_hint["note"] += (
            " Verify provenance before interpreting the pair: "
            + "; ".join(provenance_compatibility["caveats"])
            + "."
        )
    return {
        "kind": kind,
        "episode_a": _episode_dict(episode_a),
        "episode_b": _episode_dict(episode_b),
        "seed_distance": seed_distance,
        **scores,
        "why": why,
        "provenance_compatibility": provenance_compatibility,
        "command_hint": command_hint,
    }


def _seed_flip_pairs(episodes: Sequence[Episode]) -> list[dict[str, Any]]:
    """Match each episode to its nearest opposite-outcome seed in the same cell."""
    groups: dict[tuple[str, str], list[Episode]] = defaultdict(list)
    for episode in episodes:
        if episode.seed is not None and episode.outcome in {"success", *_OPPOSITE_FAILURES}:
            groups[(episode.arm, episode.scenario)].append(episode)

    pairs: dict[tuple[tuple[str, int], tuple[str, int]], dict[str, Any]] = {}
    for rows in groups.values():
        for episode in rows:
            candidates = [candidate for candidate in rows if _opposite_outcomes(episode, candidate)]
            if not candidates:
                continue
            counterpart = min(
                candidates,
                key=lambda candidate: (
                    abs((candidate.seed or 0) - (episode.seed or 0)),
                    candidate.seed or 0,
                    str(candidate.source),
                    candidate.line_number,
                ),
            )
            identifiers = sorted(
                (
                    (str(episode.source), episode.line_number),
                    (str(counterpart.source), counterpart.line_number),
                )
            )
            key = (identifiers[0], identifiers[1])
            pairs[key] = _pair_payload("seed_flip", episode, counterpart)
    return sorted(
        pairs.values(),
        key=lambda pair: (
            -pair["rank_score"],
            pair["seed_distance"],
            pair["episode_a"]["seed"],
            pair["episode_b"]["seed"],
        ),
    )


def _planner_upset_pairs(episodes: Sequence[Episode]) -> list[dict[str, Any]]:
    """Find same-scenario, same-seed opposite outcomes across planner arms."""
    groups: dict[tuple[str, int], list[Episode]] = defaultdict(list)
    for episode in episodes:
        if episode.seed is not None:
            groups[(episode.scenario, episode.seed)].append(episode)

    pairs: list[dict[str, Any]] = []
    for rows in groups.values():
        for index, first in enumerate(rows):
            for second in rows[index + 1 :]:
                if first.arm == second.arm or not _opposite_outcomes(first, second):
                    continue
                pairs.append(_pair_payload("planner_upset", first, second))
    return sorted(
        pairs,
        key=lambda pair: (
            -pair["rank_score"],
            pair["episode_a"]["scenario"],
            pair["episode_a"]["seed"],
            pair["episode_a"]["arm"],
        ),
    )


def _pair_table(pairs: Sequence[dict[str, Any]]) -> str:
    """Render one compact matched-pair table."""
    rows = [
        (
            rank,
            pair["episode_a"]["scenario"],
            pair["episode_a"]["arm"],
            pair["episode_a"]["seed"],
            pair["episode_a"]["outcome"],
            pair["episode_b"]["arm"],
            pair["episode_b"]["seed"],
            pair["episode_b"]["outcome"],
            f"{pair['rank_score']:.3f}",
        )
        for rank, pair in enumerate(pairs, start=1)
    ]
    return _format_table(
        (
            "RANK",
            "SCENARIO",
            "ARM A",
            "SEED A",
            "OUTCOME A",
            "ARM B",
            "SEED B",
            "OUTCOME B",
            "SCORE",
        ),
        rows,
    )


def _run_pairs(args: argparse.Namespace, episodes: Sequence[Episode]) -> int:
    """Render ranked seed flips and planner upsets."""
    if args.top_k < 1:
        raise CaseBrowserError("--top-k must be at least 1")
    seed_flips = _seed_flip_pairs(episodes)[: args.top_k]
    planner_upsets = _planner_upset_pairs(episodes)[: args.top_k]
    if args.json_output:
        _print_json(
            {
                "seed_flips": seed_flips,
                "planner_upsets": planner_upsets,
                "ranking": (
                    "rank_score = near-miss/clearance extremity + clearance marginality "
                    "+ seed proximity; diagnostic discovery ranking only"
                ),
            }
        )
        return 0

    for title, pairs in (("SEED FLIPS", seed_flips), ("PLANNER UPSETS", planner_upsets)):
        print(title)
        print(_pair_table(pairs))
        for rank, pair in enumerate(pairs, start=1):
            print(f"\n{title[:-1].title()} {rank}: {pair['why']}")
            for command in pair["command_hint"]["commands"]:
                print(f"  {command}")
        print()
    return 0


def _select_episode(episodes: Sequence[Episode], args: argparse.Namespace) -> Episode:
    """Resolve the critical command's selectors to exactly one episode."""
    if args.seed is None and args.episode_id is None:
        raise CaseBrowserError("critical requires --seed or --episode-id")
    selected = list(episodes)
    if args.seed is not None:
        selected = [episode for episode in selected if episode.seed == args.seed]
    if args.arm is not None:
        selected = [episode for episode in selected if episode.arm == args.arm]
    if args.scenario is not None:
        selected = [episode for episode in selected if episode.scenario == args.scenario]
    if args.episode_id is not None:
        selected = [episode for episode in selected if episode.episode_id == args.episode_id]
    if not selected:
        raise CaseBrowserError("no episode matches the critical selectors")
    if len(selected) > 1:
        choices = ", ".join(
            f"{episode.arm}/{episode.scenario}/seed={episode.seed}" for episode in selected[:8]
        )
        raise CaseBrowserError(
            f"critical selectors match {len(selected)} episodes ({choices}); add --arm, "
            "--scenario, or --episode-id"
        )
    return selected[0]


def _interval_metrics_dict(metrics: IntervalMetrics | None) -> dict[str, Any]:
    """Select finite, interpretable interval metrics for browser output."""
    if metrics is None:
        return {}
    return {
        "n_steps": metrics.n_steps,
        "min_center_distance_m": metrics.min_distance_m,
        "min_ttc_s": metrics.min_ttc_s,
        "mean_speed_m_s": metrics.mean_speed_ms,
        "max_speed_m_s": metrics.max_speed_ms,
        "max_deceleration_m_s2": metrics.max_deceleration_mps2,
        "near_miss_count_at_center_distance_threshold": metrics.near_miss_count,
    }


def _interval_why(interval: CriticalInterval, metrics: IntervalMetrics | None) -> str:
    """Explain why one available library interval is interesting."""
    values = _interval_metrics_dict(metrics)
    if interval.anchor == "closest_approach":
        return (
            "minimum robot-pedestrian center distance"
            f" ({_display_number(values.get('min_center_distance_m'))} m in window)"
        )
    if interval.anchor == "ttc_threshold_crossing":
        return (
            "first line-of-sight time-to-collision below 1.5 s"
            f" (window min {_display_number(values.get('min_ttc_s'))} s)"
        )
    if interval.anchor == "first_braking_event":
        return (
            "first braking deceleration above 0.75 m/s^2"
            f" (window max {_display_number(values.get('max_deceleration_m_s2'))} m/s^2)"
        )
    if interval.anchor == "collision_or_near_miss":
        return (
            "first center-distance crossing below "
            f"{DEFAULT_NEAR_MISS_DIST:g} m (experimental trace threshold)"
        )
    return interval.anchor.replace("_", " ")


def _critical_windows(episode: Episode) -> dict[str, Any]:
    """Extract and rank library intervals plus the recorded terminal-outcome window."""
    trace = _trace_from_record(episode.record)
    if trace is None:
        raise CaseBrowserError(f"selected episode {episode.episode_id} has no simulation trace")
    try:
        adapted = adapt_simulation_step_trace(trace)
        intervals = extract_critical_intervals(trace, DEFAULT_CRITICAL_CONFIG)
        report = summarize_interval_metrics(trace, intervals)
    except (TypeError, ValueError) as exc:
        raise CaseBrowserError(f"could not adapt selected simulation trace: {exc}") from exc

    robot_positions = adapted.get("robot_pos", [])
    trace_step_count = len(robot_positions) if isinstance(robot_positions, list) else 0
    metrics_by_anchor = {metrics.anchor: metrics for metrics in report.interval_metrics}
    priorities = {
        "collision_or_near_miss": 850.0,
        "ttc_threshold_crossing": 750.0,
        "closest_approach": 650.0,
        "first_braking_event": 550.0,
    }
    windows: list[dict[str, Any]] = []
    for interval in intervals:
        if (
            interval.status != "available"
            or interval.start_step is None
            or interval.end_step is None
            or interval.anchor_step is None
        ):
            continue
        metrics = metrics_by_anchor.get(interval.anchor)
        windows.append(
            {
                "anchor": interval.anchor,
                "start_step": interval.start_step,
                "end_step_exclusive": interval.end_step,
                "anchor_step": interval.anchor_step,
                "rank_score": priorities.get(interval.anchor, 100.0),
                "why": _interval_why(interval, metrics),
                "metrics": _interval_metrics_dict(metrics),
            }
        )

    if trace_step_count and episode.outcome in {"success", *_OPPOSITE_FAILURES}:
        dt = _finite_float(adapted.get("dt")) or 0.1
        lookback_steps = max(1, round(2.0 / dt))
        anchor_step = trace_step_count - 1
        terminal_priority = {
            "collision": 1000.0,
            "timeout": 900.0,
            "success": 150.0,
        }[episode.outcome]
        windows.append(
            {
                "anchor": f"{episode.outcome}_termination",
                "start_step": max(0, anchor_step - lookback_steps),
                "end_step_exclusive": trace_step_count,
                "anchor_step": anchor_step,
                "rank_score": terminal_priority,
                "why": (
                    f"episode {episode.outcome} at final sample {trace_step_count} "
                    f"(zero-based trace step {anchor_step})"
                ),
                "metrics": {},
            }
        )

    windows.sort(key=lambda window: (-window["rank_score"], window["anchor_step"]))
    return {
        "episode": _episode_dict(episode),
        "trace_schema": trace.get("schema_version", "legacy-columnar"),
        "trace_step_count": trace_step_count,
        "windows": windows,
        "missing_anchors": report.missing_anchors,
        "boundary": (
            "Critical-window distances are center-to-center trace diagnostics; "
            "episode min_clearance is the radius-adjusted benchmark metric."
        ),
    }


def _run_critical(args: argparse.Namespace, episodes: Sequence[Episode]) -> int:
    """Render top-ranked critical windows for one selected episode."""
    if args.top_k < 1:
        raise CaseBrowserError("--top-k must be at least 1")
    episode = _select_episode(episodes, args)
    payload = _critical_windows(episode)
    payload["windows"] = payload["windows"][: args.top_k]
    if args.json_output:
        _print_json(payload)
        return 0

    print(
        f"{episode.arm} / {episode.scenario} / seed {episode.seed} / "
        f"{episode.outcome} / {payload['trace_step_count']} trace steps"
    )
    rows = [
        (
            rank,
            window["anchor"],
            f"{window['start_step']}..{window['end_step_exclusive'] - 1}",
            window["anchor_step"],
            window["why"],
        )
        for rank, window in enumerate(payload["windows"], start=1)
    ]
    print(_format_table(("RANK", "ANCHOR", "STEP RANGE", "ANCHOR STEP", "WHY"), rows))
    if payload["missing_anchors"]:
        print("\nMissing anchors:")
        for missing in payload["missing_anchors"]:
            print(f"  {missing['anchor']}: {missing['reason']}")
    print(f"\nBoundary: {payload['boundary']}")
    return 0


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the common episode-source positional argument."""
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more episodes.jsonl files or directories containing them.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the case-browser command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List normalized episode rows.")
    _add_input_arguments(list_parser)
    list_parser.add_argument(
        "--sort",
        default="arm,scenario,seed",
        help="Comma-separated fields; prefix descending fields with '-' (for example --sort=-near).",
    )
    list_parser.add_argument(
        "--filter",
        dest="filters",
        action="append",
        default=[],
        help="Repeatable AND filter: outcome=collision, near>=10, seed=111-120.",
    )
    list_parser.add_argument("--json", dest="json_output", action="store_true")

    pairs_parser = subparsers.add_parser(
        "pairs",
        help="Find opposite-outcome seed flips and planner upsets.",
    )
    _add_input_arguments(pairs_parser)
    pairs_parser.add_argument("--top-k", type=int, default=10, help="Pairs per match type.")
    pairs_parser.add_argument("--json", dest="json_output", action="store_true")

    summary_parser = subparsers.add_parser(
        "summary",
        help="Aggregate arm-by-scenario outcome and safety cells.",
    )
    _add_input_arguments(summary_parser)
    summary_parser.add_argument("--json", dest="json_output", action="store_true")

    critical_parser = subparsers.add_parser(
        "critical",
        help="Show top critical windows for one selected traced episode.",
    )
    _add_input_arguments(critical_parser)
    critical_parser.add_argument("--seed", type=int)
    critical_parser.add_argument("--arm")
    critical_parser.add_argument("--scenario")
    critical_parser.add_argument("--episode-id")
    critical_parser.add_argument("--top-k", type=int, default=5)
    critical_parser.add_argument("--json", dest="json_output", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested case-browser subcommand."""
    args = build_parser().parse_args(argv)
    try:
        episodes, files = load_episodes(args.inputs)
        if args.command == "list":
            return _run_list(args, episodes, files)
        if args.command == "summary":
            return _run_summary(args, episodes, files)
        if args.command == "pairs":
            return _run_pairs(args, episodes)
        if args.command == "critical":
            return _run_critical(args, episodes)
        raise CaseBrowserError(f"unsupported command {args.command!r}")
    except CaseBrowserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
