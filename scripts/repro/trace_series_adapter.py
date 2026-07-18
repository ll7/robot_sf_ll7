"""Generic fail-closed adapter: one ``simulation-step-trace.v1`` episode row to a
``trace_series.json`` / ``metadata.json`` bundle.

This module generalizes the PPO-doorway-only converter
(``scripts/repro/butterfly_reexport_to_trace_series.py``) so any supported
benchmark episode -- any planner, scenario, commit, or trace contract -- can be
turned into the bundle schema consumed by
``robot_sf.benchmark.trace_scene_figure.load_episode`` and
``scripts/repro/butterfly_hinge_figure_proto.py``.

Selection is exact, not seed-only: a row is selected by source path plus
episode identity. When a stable ``episode_id`` is present it is the primary key
and must be unique within the source file. When it is absent, the row is
selected by the ``(scenario, planner, seed)`` triple and must likewise be
unique -- ambiguous matches fail closed rather than silently picking one.

The adapter rejects unsupported inputs instead of papering over them:

* unknown ``simulation-step-trace`` schema versions;
* missing execution provenance (no ``git_commit`` / ``result_provenance``);
* actor-set changes -- the pedestrian-id set must be identical across every
  recorded step (the renderer assumes stable IDs);
* zero-pedestrian frames, which would produce a non-finite nearest-pedestrian
  distance that the downstream schema validator rejects;
* non-finite or missing robot state.

The derived-row computation mirrors the one the renderer independently
re-derives from the emitted ``frames`` (the same strict-less-than nearest
neighbour scan over ``frame["pedestrians"]`` in list order), so the two agree
by construction and the bundle is self-consistent. Source provenance is copied
into the emitted metadata, and generated timestamps are used only when the
source row supplies one, keeping repeated conversion deterministic.

Usage::

    uv run python scripts/repro/trace_series_adapter.py build-bundle \\
        --episodes-jsonl RUNS/ppo__differential_drive/episodes.jsonl \\
        --scenario classic_doorway_medium --planner ppo --seed 113 \\
        --out-dir output/cases/seed113

    uv run python scripts/repro/trace_series_adapter.py build-bundle \\
        --episodes-jsonl RUNS/orca__differential_drive/episodes.jsonl \\
        --episode-id doorway--orca--113 \\
        --out-dir output/cases/orca-113
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# Schema contract this adapter understands. It is intentionally narrow: the
# issue asks for a fail-closed adapter, not a permissive one.
SUPPORTED_TRACE_SCHEMA: str = "simulation-step-trace.v1"

# Top-level bundle container tag. Distinct from the issue-4891 reference tag and
# from the PPO-doorway re-export tag; this is a different generation pipeline.
TRACE_SERIES_SCHEMA_VERSION: str = "generic-episode-trace-series.v1"
METADATA_SCHEMA_VERSION: str = "generic-episode-trace.v1"

# Machine-generated artifact marker carried into both bundle files.
REVIEW_MARKER: str = "AI-GENERATED"


class TraceSeriesAdapterError(ValueError):
    """Fail-closed input, selection, or schema error."""


def _nested(mapping: dict[str, Any], *path: str) -> Any:
    """Return one nested value, or ``None`` when a path is absent."""
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first(mapping: dict[str, Any], *aliases: tuple[str, ...]) -> Any:
    """Return the first non-empty configured field alias."""
    for alias in aliases:
        value = _nested(mapping, *alias)
        if value is not None and value != "":
            return value
    return None


def _coerce_int(value: Any) -> int | None:
    """Coerce one exact finite integer, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if numeric.is_integer() else None


def _coerce_float(value: Any) -> float | None:
    """Coerce one finite number (bools excluded), or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _episode_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Resolve the canonical identity fields used for exact selection."""
    provenance = row.get("result_provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    git_commit = provenance.get("repo_commit") or row.get("git_hash")
    if not isinstance(git_commit, str) or not git_commit.strip():
        git_commit = None
    else:
        git_commit = git_commit.strip()
    return {
        "episode_id": _first(row, ("episode_id",), ("id",)),
        "scenario": _first(
            row,
            ("scenario_id",),
            ("scenario",),
            ("scenario_name",),
            ("scenario_params", "id"),
            ("scenario_params", "name"),
        ),
        "planner": _first(
            row,
            ("algo",),
            ("arm",),
            ("planner_key",),
            ("algorithm_metadata", "canonical_algorithm"),
            ("algorithm_metadata", "algorithm"),
        ),
        "seed": _coerce_int(_first(row, ("seed",), ("scenario_seed",), ("episode_seed",))),
        "git_commit": str(git_commit) if git_commit is not None else None,
    }


def _trace_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the supported nested trace mapping, or ``None``."""
    trace = _first(
        row,
        ("algorithm_metadata", "simulation_step_trace"),
        ("simulation_step_trace",),
    )
    return trace if isinstance(trace, dict) else None


def _row_matches(
    identity: dict[str, Any],
    *,
    episode_id: str | None,
    scenario: str | None,
    planner: str | None,
    seed: int | None,
) -> bool:
    """Return whether one resolved identity matches the active selection key."""
    if episode_id is not None:
        return identity["episode_id"] is not None and str(identity["episode_id"]) == str(episode_id)
    return (
        (scenario is None or str(identity["scenario"]) == scenario)
        and (planner is None or str(identity["planner"]) == planner)
        and (seed is None or identity["seed"] == seed)
    )


def select_row(
    episodes_jsonl: Path,
    *,
    episode_id: str | None = None,
    scenario: str | None = None,
    planner: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Select exactly one episode row matching the supplied identity keys.

    Either ``episode_id`` (primary key) or the
    ``(scenario, planner, seed)`` triple must be supplied. Ambiguous or missing
    matches fail closed.

    Returns:
        The selected row dict.

    Raises:
        TraceSeriesAdapterError: on no match, ambiguous match, or inconsistent
            selector usage.
    """
    if episode_id is None and (scenario is None or planner is None or seed is None):
        raise TraceSeriesAdapterError("supply --episode-id, or all of --scenario/--planner/--seed")
    if episode_id is not None and (scenario is not None or planner is not None or seed is not None):
        raise TraceSeriesAdapterError(
            "--episode-id is mutually exclusive with --scenario/--planner/--seed"
        )

    matches: list[tuple[dict[str, Any], int]] = []
    with episodes_jsonl.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TraceSeriesAdapterError(
                    f"{episodes_jsonl}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                continue
            if _row_matches(
                _episode_identity(row),
                episode_id=episode_id,
                scenario=scenario,
                planner=planner,
                seed=seed,
            ):
                matches.append((row, line_number))

    if not matches:
        selector = (
            f"episode_id={episode_id!r}"
            if episode_id is not None
            else f"scenario={scenario!r}, planner={planner!r}, seed={seed!r}"
        )
        raise TraceSeriesAdapterError(f"no episode matching {selector} in {episodes_jsonl}")
    if len(matches) > 1:
        lines = ", ".join(str(line_number) for _, line_number in matches)
        raise TraceSeriesAdapterError(
            f"{len(matches)} ambiguous rows match in {episodes_jsonl} (lines {lines}); "
            "exact selection requires a unique identity"
        )
    return matches[0][0]


def _validate_provenance(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Fail closed when execution provenance is missing.

    Returns:
        The resolved execution commit and source provenance mapping.
    """
    raw_provenance = row.get("result_provenance")
    if raw_provenance is not None and not isinstance(raw_provenance, dict):
        raise TraceSeriesAdapterError("episode row result_provenance must be an object")
    provenance = dict(raw_provenance) if isinstance(raw_provenance, dict) else {}
    identity = _episode_identity(row)
    git_commit = identity["git_commit"]
    if not git_commit:
        raise TraceSeriesAdapterError(
            "episode row has no execution provenance (missing git_hash/result_provenance.repo_commit)"
        )
    return git_commit, provenance


def _finite_number(value: Any, context: str) -> float:
    """Return one finite JSON number or raise an actionable adapter error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceSeriesAdapterError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TraceSeriesAdapterError(f"{context} must be a finite number")
    return result


def _integer(value: Any, context: str) -> int:
    """Return one JSON integer or raise an actionable adapter error."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceSeriesAdapterError(f"{context} must be an integer")
    return value


def _xy_vector(value: Any, context: str) -> list[float]:
    """Return one finite two-dimensional JSON vector."""
    if not isinstance(value, list) or len(value) != 2:
        raise TraceSeriesAdapterError(f"{context} must be a finite [x, y] vector")
    return [_finite_number(value[0], f"{context}[0]"), _finite_number(value[1], f"{context}[1]")]


def _validate_pedestrians(pedestrians: Any, context: str) -> None:
    """Validate one non-empty, duplicate-free pedestrian array."""
    if not isinstance(pedestrians, list) or not pedestrians:
        raise TraceSeriesAdapterError(
            f"{context}.pedestrians has zero pedestrians; a finite nearest distance is required"
        )
    seen_ids: set[object] = set()
    for pedestrian_index, pedestrian in enumerate(pedestrians):
        ped_context = f"{context}.pedestrians[{pedestrian_index}]"
        if not isinstance(pedestrian, dict):
            raise TraceSeriesAdapterError(f"{ped_context} must be an object")
        pedestrian_id = pedestrian.get("id")
        if isinstance(pedestrian_id, bool) or not isinstance(pedestrian_id, (int, str)):
            raise TraceSeriesAdapterError(f"{ped_context}.id must be an integer or string")
        if pedestrian_id in seen_ids:
            raise TraceSeriesAdapterError(
                f"{context} contains duplicate pedestrian id {pedestrian_id!r}"
            )
        seen_ids.add(pedestrian_id)
        _xy_vector(pedestrian.get("position"), f"{ped_context}.position")


def _validate_frame(frame: Any, frame_index: int) -> None:
    """Validate fields consumed by one emitted frame and its derived row."""
    context = f"simulation_step_trace.steps[{frame_index}]"
    if not isinstance(frame, dict):
        raise TraceSeriesAdapterError(f"{context} must be an object")
    _integer(frame.get("step"), f"{context}.step")
    _finite_number(frame.get("time_s"), f"{context}.time_s")
    robot = frame.get("robot")
    if not isinstance(robot, dict):
        raise TraceSeriesAdapterError(f"{context}.robot must be an object")
    _xy_vector(robot.get("position"), f"{context}.robot.position")
    _xy_vector(robot.get("velocity"), f"{context}.robot.velocity")
    _finite_number(robot.get("heading"), f"{context}.robot.heading")
    _validate_pedestrians(frame.get("pedestrians"), context)
    planner = frame.get("planner")
    if planner is not None and not isinstance(planner, dict):
        raise TraceSeriesAdapterError(f"{context}.planner must be an object")
    if isinstance(planner, dict) and planner.get("selected_action") is not None:
        if not isinstance(planner["selected_action"], dict):
            raise TraceSeriesAdapterError(f"{context}.planner.selected_action must be an object")


def _validate_frames(frames: list[Any]) -> None:
    """Validate all fields consumed by the emitted bundle and derived rows."""
    for frame_index, frame in enumerate(frames):
        _validate_frame(frame, frame_index)


def _nearest_pedestrian(frame: dict[str, Any]) -> tuple[float, int | str | None]:
    """Center-to-center distance and id of the nearest pedestrian in one frame.

    Uses the strict-less-than nearest-neighbour scan (in ``frame["pedestrians"]``
    list order, ties resolved identically) that ``trace_scene_figure`` /
    ``butterfly_hinge_figure_proto.compute_trace_metrics`` independently re-derive
    from the emitted ``frames`` -- so this adapter's derived numbers agree with the
    renderer's re-derivation by construction.
    """
    rx, ry = frame["robot"]["position"]
    best_dist = math.inf
    best_id: int | str | None = None
    for ped in frame.get("pedestrians", []):
        px, py = ped["position"]
        dist = math.hypot(rx - px, ry - py)
        if dist < best_dist:
            best_dist = dist
            best_id = ped.get("id")
    return best_dist, best_id


def _build_derived_rows(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the ``derived_rows`` array required by
    ``trace_scene_figure._parse_derived_rows``, one row per frame.

    Raises:
        TraceSeriesAdapterError: on a zero-pedestrian frame (would yield a
            non-finite distance the schema validator rejects) or a frame missing
            finite robot state.
    """
    rows: list[dict[str, Any]] = []
    for frame in frames:
        robot = frame["robot"]
        rx, ry = robot["position"]
        rvx, rvy = robot["velocity"]
        speed = math.hypot(rvx, rvy)
        dist, ped_id = _nearest_pedestrian(frame)
        if not math.isfinite(dist):
            raise TraceSeriesAdapterError(
                f"frame step={frame.get('step')} has zero pedestrians -- cannot compute a "
                "finite nearest-pedestrian distance for derived_rows"
            )
        planner = frame.get("planner") or {}
        action = planner.get("selected_action") or {}
        peds = frame["pedestrians"]
        rows.append(
            {
                "step": frame["step"],
                "time_s": float(frame["time_s"]),
                "robot_x_m": rx,
                "robot_y_m": ry,
                "robot_heading_rad": robot["heading"],
                "executed_speed_m_s": speed,
                "executed_vx_m_s": rvx,
                "executed_vy_m_s": rvy,
                "commanded_linear_velocity_m_s": action.get("linear_velocity"),
                "commanded_angular_velocity_rad_s": action.get("angular_velocity"),
                "min_robot_ped_distance_m": dist,
                "nearest_pedestrian_id": str(ped_id) if ped_id is not None else None,
                "pedestrian_count": len(peds),
                "pedestrian_positions_json": json.dumps(
                    [
                        {"id": str(p["id"]), "x_m": p["position"][0], "y_m": p["position"][1]}
                        for p in peds
                    ]
                ),
            }
        )
    return rows


def _validate_actor_set(frames: list[dict[str, Any]]) -> None:
    """Fail closed when the pedestrian-id set changes across recorded steps.

    The renderer assumes stable actor IDs across a trace; a changing set would
    misalign pedestrian tracks.
    """
    expected: frozenset[int] | None = None
    for frame in frames:
        ids = frozenset(p["id"] for p in frame.get("pedestrians", []))
        if expected is None:
            expected = ids
        elif ids != expected:
            raise TraceSeriesAdapterError(
                f"actor-set change at step={frame.get('step')}: pedestrian ids "
                f"{sorted(ids)} differ from earlier {sorted(expected)}"
            )


def _build_metadata(
    row: dict[str, Any],
    identity: dict[str, Any],
    provenance: dict[str, Any],
    derived_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the ``metadata`` dict shared by ``trace_series.json`` and
    ``metadata.json`` (minus the ``review_marker`` key ``metadata.json`` adds)."""
    for field_name in ("planner", "scenario"):
        if not isinstance(identity[field_name], str) or not identity[field_name].strip():
            raise TraceSeriesAdapterError(
                f"episode identity {field_name} must be a non-empty string"
            )
    if identity["seed"] is None:
        raise TraceSeriesAdapterError("episode identity seed is missing or invalid")
    for field_name in ("status", "termination_reason"):
        value = row.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise TraceSeriesAdapterError(f"episode row {field_name} must be a non-empty string")

    dists = [r["min_robot_ped_distance_m"] for r in derived_rows]
    global_min_idx = min(range(len(dists)), key=lambda i: dists[i])
    metadata = {
        "planner": identity["planner"],
        "scenario_id": identity["scenario"],
        "seed": identity["seed"],
        "episode_id": identity["episode_id"],
        "episode_status": row.get("status"),
        "termination_reason": row.get("termination_reason"),
        "git_commit": identity["git_commit"],
        "source_provenance": provenance,
        "source_file": None,  # filled by caller once the source path is known
        "schema_version": METADATA_SCHEMA_VERSION,
        "review_marker": REVIEW_MARKER,
        "summary": {
            "episode_status": row.get("status"),
            "planner": identity["planner"],
            "scenario_id": identity["scenario"],
            "seed": identity["seed"],
            "step_count": len(derived_rows),
            "global_min_robot_ped_distance_m": dists[global_min_idx],
            "global_min_distance_step": derived_rows[global_min_idx]["step"],
            "termination_reason": row.get("termination_reason"),
        },
    }
    generated_at = row.get("generated_at_utc")
    if generated_at is not None:
        if not isinstance(generated_at, str) or not generated_at.strip():
            raise TraceSeriesAdapterError("episode row generated_at_utc must be a non-empty string")
        metadata["generated_at_utc"] = generated_at
    return metadata


def build_bundle(
    episodes_jsonl: Path,
    out_dir: Path,
    *,
    episode_id: str | None = None,
    scenario: str | None = None,
    planner: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Convert one exact episode row into a ``trace_series.json`` + ``metadata.json`` bundle.

    Returns:
        Small summary dict (paths written, step count, outcome) for CLI reporting.
    """
    row = select_row(
        episodes_jsonl,
        episode_id=episode_id,
        scenario=scenario,
        planner=planner,
        seed=seed,
    )
    git_commit, provenance = _validate_provenance(row)
    identity = _episode_identity(row)
    identity["git_commit"] = git_commit

    trace = _trace_from_row(row)
    if trace is None:
        raise TraceSeriesAdapterError("episode row has no simulation_step_trace to adapt")
    if trace.get("schema_version") != SUPPORTED_TRACE_SCHEMA:
        raise TraceSeriesAdapterError(
            f"unsupported simulation_step_trace schema_version: "
            f"{trace.get('schema_version')!r} (expected {SUPPORTED_TRACE_SCHEMA!r})"
        )
    frames = trace.get("steps")
    if not isinstance(frames, list) or not frames:
        raise TraceSeriesAdapterError("simulation_step_trace.steps must be a non-empty array")
    _validate_frames(frames)
    _validate_actor_set(frames)
    derived_rows = _build_derived_rows(frames)
    metadata = _build_metadata(row, identity, provenance, derived_rows)
    metadata["source_file"] = str(episodes_jsonl)

    trace_series = {
        "schema_version": TRACE_SERIES_SCHEMA_VERSION,
        "metadata": metadata,
        "frames": frames,
        "derived_rows": derived_rows,
        "review_marker": REVIEW_MARKER,
    }
    metadata_json = dict(metadata)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trace_series.json").write_text(json.dumps(trace_series, indent=2), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(metadata_json, indent=2), encoding="utf-8")

    return {
        "episode_id": identity["episode_id"],
        "scenario_id": identity["scenario"],
        "planner": identity["planner"],
        "seed": identity["seed"],
        "git_commit": git_commit,
        "episode_status": row.get("status"),
        "n_steps": len(frames),
        "global_min_distance_step": metadata["summary"]["global_min_distance_step"],
        "global_min_robot_ped_distance_m": metadata["summary"]["global_min_robot_ped_distance_m"],
        "out_dir": str(out_dir),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_p = sub.add_parser("build-bundle", help="Convert one episode row into a trace bundle.")
    build_p.add_argument("--episodes-jsonl", type=Path, required=True)
    build_p.add_argument("--out-dir", type=Path, required=True)
    selector = build_p.add_mutually_exclusive_group(required=True)
    selector.add_argument("--episode-id", dest="episode_id")
    selector.add_argument(
        "--scenario",
        help="With --planner and --seed, selects the row by identity triple.",
    )
    build_p.add_argument("--planner")
    build_p.add_argument("--seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    triple = (args.scenario, args.planner, args.seed)
    if args.episode_id is None and any(value is None for value in triple):
        raise SystemExit(
            "either --episode-id or the full --scenario/--planner/--seed triple is required"
        )
    try:
        summary = build_bundle(
            args.episodes_jsonl,
            args.out_dir,
            episode_id=args.episode_id,
            scenario=args.scenario,
            planner=args.planner,
            seed=args.seed,
        )
    except (OSError, TraceSeriesAdapterError) as exc:
        print(f"trace-series adapter failed closed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
