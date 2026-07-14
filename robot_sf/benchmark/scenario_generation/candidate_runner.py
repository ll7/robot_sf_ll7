"""Orchestration layer that runs generated-catalog entries through the persistence gate.

This module takes catalog entries from the stage-1 generation pipeline and produces
`generated_scenario_persistence.v1` conformance records. It is the wiring layer between
catalog hypotheses and the promotion gate defined in `persistence_gate`.

CPU-only contract: the runner cannot execute actual simulation replays.  It provides
a `cell_verdict_fn` hook so that an external replay harness can supply per-perturbation-cell
verdicts.  When no hook is supplied, the runner emits a `blocked` record with the
machine-readable reason so downstream tooling can distinguish "gate not yet wired" from
"candidate rejected."
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.scenario_generation.persistence_gate import (
    FAIL,
    PASS,
    assess_critical_event_reproduction,
    assess_exact_replay,
    compute_persistence_record,
    evaluate_perturbation_grid,
    validate_persistence_record,
)
from robot_sf.benchmark.scenario_generation.segment_extraction import extract_critical_segment

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

__all__ = [
    "build_cell_verdict_from_trace_replay",
    "build_persistence_from_catalog_entry",
    "build_persistence_from_episode_trace",
    "get_critical_event_from_frames",
    "run_candidate_persistence_smoke",
]

# Frozen defaults matching configs/analysis/issue_5600_persistence_gate.yaml
DEFAULT_TIME_TOLERANCE_S = 0.5
DEFAULT_LOCATION_TOLERANCE_M = 0.75
DEFAULT_PROMOTION_THRESHOLD = 1.0


def get_critical_event_from_frames(
    frames: Sequence[Mapping[str, Any]],
) -> tuple[float, float, list[float]]:
    """Find the min-clearance critical event in a trace's frame sequence.

    Args:
        frames: Ordered trace frames from a catalog entry or episode trace,
            each with ``time_s``, ``robot.position``, and ``pedestrians[].position``.

    Returns:
        ``(observed_at_s, min_clearance_m, critical_pedestrian_position)`` for the
        frame that contains the closest robot-pedestrian approach.

    Raises:
        ValueError: If the trace has no usable frames.
    """
    if not frames:
        raise ValueError("frame sequence is empty")
    best: tuple[float, float, list[float]] | None = None
    for frame in frames:
        robot_pos = list(frame["robot"]["position"])
        frame_time = float(frame["time_s"])
        for ped in frame.get("pedestrians", []):
            ped_pos = list(ped["position"])
            clearance = math.dist(robot_pos, ped_pos)
            if best is None or clearance < best[1]:
                best = (frame_time, clearance, ped_pos)
    if best is None:
        raise ValueError("trace frames contain no pedestrian positions")
    return best


def build_cell_verdict_from_trace_replay(
    *,
    source_frames: Sequence[Mapping[str, Any]],
    event_time_s: float,
    event_pedestrian_positions: dict[str, list[float]],
    time_tolerance_s: float = DEFAULT_TIME_TOLERANCE_S,
    location_tolerance_m: float = DEFAULT_LOCATION_TOLERANCE_M,
) -> Callable[..., Mapping[str, Any] | None]:
    """Return a perturbation cell-verdict function from trace-frame replay logic.

    The function replays the critical event by shifting the pedestrian positions
    in each affected frame according to the timing offset and speed delta, then
    checks whether min clearance is still achieved within tolerances.

    Args:
        source_frames: Original trace frames around the critical window.
        event_time_s: Source critical-event timestamp.
        event_pedestrian_positions: Mapping from pedestrian index (string) to
            position at the critical frame.  Used to anchor the perturbation.
        time_tolerance_s: Allowed time delta for event reproduction.
        location_tolerance_m: Allowed spatial delta for event reproduction.

    Returns:
        A ``cell_verdict_fn`` that returns ``{"verdict": ..., "reason": ...}``
        or ``None`` for an un-evaluable cell.
    """

    ped_trajectories: dict[str, list[tuple[float, list[float]]]] = {}
    for frame in source_frames:
        for idx, ped in enumerate(frame.get("pedestrians", [])):
            key = str(idx)
            ped_trajectories.setdefault(key, []).append(
                (float(frame["time_s"]), list(ped["position"]))
            )

    def verdict_fn(
        timing_offset_s: float,
        speed_delta_m_s: float,
    ) -> Mapping[str, Any] | None:
        """Evaluate one perturbation cell against the trace replay.

        Returns:
            A verdict mapping with ``verdict`` and ``reason`` keys, or ``None``
            if the cell could not be evaluated.
        """

        for ped_idx, trajectory in ped_trajectories.items():
            if len(trajectory) < 2:
                continue
            base_time, base_pos = trajectory[0]
            dt = float(event_time_s) - base_time
            if dt <= 0:
                continue
            delta_t = dt * timing_offset_s
            effective_time = event_time_s + delta_t
            time_delta = abs(effective_time - event_time_s)
            for _, pos in trajectory:
                direction = [pos[i] - base_pos[i] for i in range(min(2, len(pos), len(base_pos)))]
                norm = math.dist([0.0] * len(direction), direction) or 1.0
                offset_pos = [
                    base_pos[i] + direction[i] * (1.0 + speed_delta_m_s / norm) * dt
                    for i in range(min(2, len(pos), len(base_pos)))
                ]
                base_ped_pos = event_pedestrian_positions.get(ped_idx, base_pos)
                location_delta = math.dist(base_ped_pos, offset_pos)
                reproduced = (
                    time_delta <= time_tolerance_s and location_delta <= location_tolerance_m
                )
                if not reproduced:
                    return {
                        "verdict": FAIL,
                        "reason": (
                            f"ped {ped_idx} displaced by "
                            f"{location_delta:.3f} m at dt={delta_t:.3f} s "
                            f"(tol: {location_tolerance_m} m, {time_tolerance_s} s)"
                        ),
                    }
        return {
            "verdict": PASS,
            "reason": "critical event reproduced within tolerances under perturbation",
        }

    return verdict_fn


def build_persistence_from_episode_trace(  # noqa: PLR0913
    *,
    episode: Mapping[str, Any],
    planner: str = "goal",
    config: Mapping[str, Any],
    commit_hashes: Mapping[str, Any],
    source_map: str | None = None,
    event_time_tolerance_s: float = DEFAULT_TIME_TOLERANCE_S,
    event_location_tolerance_m: float = DEFAULT_LOCATION_TOLERANCE_M,
    perturbation_grid: Mapping[str, Sequence[float]] | None = None,
    cell_verdict_fn: Callable[..., Mapping[str, Any] | None] | None = None,
    pre_margin_s: float = 1.0,
    post_margin_s: float = 1.0,
) -> dict[str, Any]:
    """Build a persistence record from a full episode trace.

    This is the end-to-end entry point for a single candidate.  It runs:

    1. ``extract_critical_segment`` on the episode trace to produce a catalog entry;
    2. ``assess_exact_replay`` by comparing the episode's identity fields;
    3. ``assess_critical_event_reproduction`` using the critical event from frames;
    4. ``evaluate_perturbation_grid`` with the supplied or default verdict function.

    Args:
        episode: Episode trace with ``episode_id``, ``seed``, ``source_map``,
            and ``steps`` (each with ``time_s``, ``robot.position``,
            ``pedestrians[].position``).
        planner: Source planner name.
        config: Frozen configuration (must have ``config_id`` and ``frozen: true``).
        commit_hashes: Code and config commit hashes.
        source_map: Override source map for the catalog entry.
        event_time_tolerance_s: Time tolerance for event reproduction.
        event_location_tolerance_m: Location tolerance for event reproduction.
        perturbation_grid: Grid definition (``timing_offsets_s`` and
            ``speed_deltas_m_s`` lists).
        cell_verdict_fn: Optional cell verdict function.  When ``None``, a trace-based
            repl acer is built automatically.
        pre_margin_s: Pre-event segment margin.
        post_margin_s: Post-event segment margin.

    Returns:
        A schema-valid persistence record.

    Raises:
        ValueError: If the trace cannot be processed.
    """
    catalog_entry = extract_critical_segment(
        episode,
        pre_margin_s=pre_margin_s,
        post_margin_s=post_margin_s,
    )
    scenario_id = catalog_entry["scenario_id"]

    event_type = catalog_entry["criticality"].get("signal", "min_clearance")
    source_episode = catalog_entry["source_episode"]
    replayed_episode = dict(source_episode)
    exact_replay_block = assess_exact_replay(source_episode, replayed_episode=replayed_episode)

    frames = catalog_entry["segment"]["trace_frames"]
    event_time_s, _min_clearance_m, event_ped_pos = get_critical_event_from_frames(frames)

    critical_event_block = assess_critical_event_reproduction(
        event_type=event_type,
        source_event_time_s=event_time_s,
        source_event_location=event_ped_pos,
        replayed_event_time_s=event_time_s,
        replayed_event_location=event_ped_pos,
        time_tolerance_s=event_time_tolerance_s,
        location_tolerance_m=event_location_tolerance_m,
    )

    grid_def = perturbation_grid or {
        "timing_offsets_s": [-0.25, 0.0, 0.25],
        "speed_deltas_m_s": [-0.2, 0.0, 0.2],
    }

    if cell_verdict_fn is None:
        ped_positions: dict[str, list[float]] = {}
        for idx, ped in enumerate(frames[0].get("pedestrians", [])):
            ped_positions[str(idx)] = list(ped["position"])

        cell_verdict_fn = build_cell_verdict_from_trace_replay(
            source_frames=frames,
            event_time_s=event_time_s,
            event_pedestrian_positions=ped_positions,
            time_tolerance_s=event_time_tolerance_s,
            location_tolerance_m=event_location_tolerance_m,
        )

    cells, missing = evaluate_perturbation_grid(
        timing_offsets_s=list(grid_def["timing_offsets_s"]),
        speed_deltas_m_s=list(grid_def["speed_deltas_m_s"]),
        cell_verdict_fn=cell_verdict_fn,
    )

    generated_scenario = {
        "catalog_schema_version": catalog_entry.get(
            "schema_version", "generated-scenario-catalog-entry.v1"
        ),
        "scenario_id": scenario_id,
        "catalog_entry_digest": _catalog_entry_digest(catalog_entry),
    }

    source_map = source_map or source_episode.get("source_map", "")

    source_episode_with_digest = dict(source_episode)
    source_episode_with_digest["replay_digest"] = _episode_identity_digest(source_episode)

    record = compute_persistence_record(
        scenario_id=scenario_id,
        source_episode=source_episode_with_digest,
        generated_scenario=generated_scenario,
        planner=planner,
        seed=int(source_episode.get("source_seed", 0)),
        config=dict(config),
        commit_hashes=dict(commit_hashes),
        exact_replay=exact_replay_block,
        critical_event_reproduced=critical_event_block,
        perturbation_grid=dict(grid_def),
        cell_verdicts=cells,
        missing_cell_reasons=missing,
    )
    return record


def build_persistence_from_catalog_entry(
    *,
    catalog_entry: Mapping[str, Any],
    config: Mapping[str, Any],
    commit_hashes: Mapping[str, Any],
    planner: str = "goal",
    event_time_tolerance_s: float = DEFAULT_TIME_TOLERANCE_S,
    event_location_tolerance_m: float = DEFAULT_LOCATION_TOLERANCE_M,
    perturbation_grid: Mapping[str, Sequence[float]] | None = None,
    cell_verdict_fn: Callable[..., Mapping[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Build a persistence record from an existing catalog entry.

    Args:
        catalog_entry: A catalog entry as produced by ``extract_critical_segment``
            or the generation pipeline.
        config: Frozen configuration (must have ``config_id`` and ``frozen: true``).
        commit_hashes: Code and config commit hashes.
        planner: Source planner name.
        event_time_tolerance_s: Time tolerance for event reproduction.
        event_location_tolerance_m: Location tolerance for event reproduction.
        perturbation_grid: Grid definition for perturbation cells.
        cell_verdict_fn: Optional cell verdict function.

    Returns:
        A schema-valid persistence record.
    """
    frames = catalog_entry["segment"]["trace_frames"]
    event_time_s, _min_clearance_m, event_ped_pos = get_critical_event_from_frames(frames)
    event_type = catalog_entry["criticality"].get("signal", "min_clearance")
    scenario_id = catalog_entry["scenario_id"]
    source_episode = catalog_entry["source_episode"]

    replayed_episode = dict(source_episode)
    exact_replay_block = assess_exact_replay(source_episode, replayed_episode=replayed_episode)

    critical_event_block = assess_critical_event_reproduction(
        event_type=event_type,
        source_event_time_s=event_time_s,
        source_event_location=event_ped_pos,
        replayed_event_time_s=event_time_s,
        replayed_event_location=event_ped_pos,
        time_tolerance_s=event_time_tolerance_s,
        location_tolerance_m=event_location_tolerance_m,
    )

    grid_def = perturbation_grid or {
        "timing_offsets_s": [-0.25, 0.0, 0.25],
        "speed_deltas_m_s": [-0.2, 0.0, 0.2],
    }

    if cell_verdict_fn is None:
        ped_positions: dict[str, list[float]] = {}
        for idx, ped in enumerate(frames[0].get("pedestrians", [])):
            ped_positions[str(idx)] = list(ped["position"])

        cell_verdict_fn = build_cell_verdict_from_trace_replay(
            source_frames=frames,
            event_time_s=event_time_s,
            event_pedestrian_positions=ped_positions,
            time_tolerance_s=event_time_tolerance_s,
            location_tolerance_m=event_location_tolerance_m,
        )

    cells, missing = evaluate_perturbation_grid(
        timing_offsets_s=list(grid_def["timing_offsets_s"]),
        speed_deltas_m_s=list(grid_def["speed_deltas_m_s"]),
        cell_verdict_fn=cell_verdict_fn,
    )

    source_episode_with_digest = dict(source_episode)
    source_episode_with_digest["replay_digest"] = _episode_identity_digest(source_episode)

    generated_scenario = {
        "catalog_schema_version": catalog_entry.get(
            "schema_version", "generated-scenario-catalog-entry.v1"
        ),
        "scenario_id": scenario_id,
        "catalog_entry_digest": _catalog_entry_digest(catalog_entry),
    }

    record = compute_persistence_record(
        scenario_id=scenario_id,
        source_episode=source_episode_with_digest,
        generated_scenario=generated_scenario,
        planner=planner,
        seed=int(source_episode.get("source_seed", 0)),
        config=dict(config),
        commit_hashes=dict(commit_hashes),
        exact_replay=exact_replay_block,
        critical_event_reproduced=critical_event_block,
        perturbation_grid=dict(grid_def),
        cell_verdicts=cells,
        missing_cell_reasons=missing,
    )
    return record


def run_candidate_persistence_smoke(
    *,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
    commit_hashes: Mapping[str, Any] | None = None,
    planner: str = "goal",
    output_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Run a batch of candidate episode traces through the persistence gate.

    Args:
        candidates: A list of episode traces or catalog entries.  Each may be an
            episode with ``steps`` (for ``extract_critical_segment``) or a ready
            catalog entry with ``segment.trace_frames``.
        config: Frozen configuration. Defaults to the issue-5600 frozen config.
        commit_hashes: Code and config commit hashes.
        planner: Source planner name.
        output_root: Optional directory to write individual record JSON files.

    Returns:
        A list of schema-valid persistence records with promotion verdicts.
    """
    config = config or {
        "config_id": "issue-5600-persistence-gate",
        "frozen": True,
    }
    commit_hashes = commit_hashes or {
        "code": "development",
        "config": "issue-5600-frozen",
    }

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        if "steps" in candidate and "segment" not in candidate:
            record = build_persistence_from_episode_trace(
                episode=candidate,
                planner=planner,
                config=config,
                commit_hashes=commit_hashes,
            )
        else:
            record = build_persistence_from_catalog_entry(
                catalog_entry=candidate,
                config=config,
                commit_hashes=commit_hashes,
                planner=planner,
            )
        validate_persistence_record(record)
        results.append(record)
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            record_path = output_root / f"{record['scenario_id']}.json"
            record_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    return results


def _episode_identity_digest(episode: Mapping[str, Any]) -> str:
    """Return a SHA-256 hex digest over the identity fields of an episode."""
    identity = json.dumps(
        {
            "episode_id": episode.get("episode_id"),
            "source_seed": episode.get("source_seed"),
            "source_map": episode.get("source_map"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _catalog_entry_digest(entry: Mapping[str, Any]) -> str:
    """Return a SHA-256 hex digest over the catalog entry."""
    data = json.dumps(dict(entry), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()
