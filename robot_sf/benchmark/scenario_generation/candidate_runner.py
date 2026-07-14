"""Orchestration layer that runs generated-catalog entries through the persistence gate.

This module takes catalog entries from the stage-1 generation pipeline and produces
`generated_scenario_persistence.v1` conformance records. It is the wiring layer between
catalog hypotheses and the promotion gate defined in `persistence_gate`.

CPU-only contract: the runner cannot execute actual simulation replays.  Callers must provide
replayed episode/frame evidence and a `cell_verdict_fn` from an external replay harness before
the record can promote.  When that evidence is absent, the runner emits a fail-closed record with
unknown replay/event status and missing perturbation cells.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from robot_sf.benchmark.scenario_generation.persistence_gate import (
    FAIL,
    PASS,
    assess_critical_event_reproduction,
    compute_persistence_record,
    evaluate_perturbation_grid,
    validate_persistence_record,
)
from robot_sf.benchmark.scenario_generation.segment_extraction import extract_critical_segment

if TYPE_CHECKING:
    from collections.abc import Callable
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
    event_time_s, clearance_m, pedestrian_positions = _critical_event_context(frames)
    return event_time_s, clearance_m, next(iter(pedestrian_positions.values()))


def _critical_event_context(
    frames: Sequence[Mapping[str, Any]],
) -> tuple[float, float, dict[str, list[float]]]:
    """Return the closest-event time, clearance, and all positions at that frame."""

    if not frames:
        raise ValueError("frame sequence is empty")
    best: tuple[float, float, dict[str, list[float]]] | None = None
    for frame in frames:
        robot_pos = list(frame["robot"]["position"])
        frame_time = float(frame["time_s"])
        frame_positions = {
            str(index): list(pedestrian["position"])
            for index, pedestrian in enumerate(frame.get("pedestrians", []))
        }
        for ped_pos in frame_positions.values():
            clearance = math.dist(robot_pos, ped_pos)
            if best is None or clearance < best[1]:
                best = (frame_time, clearance, frame_positions)
    if best is None:
        raise ValueError("trace frames contain no pedestrian positions")
    return best


def _missing_cell_verdict(**_: Any) -> None:
    """Mark a perturbation cell missing when no replay harness supplied evidence."""

    return None


def _replay_identity(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize source or replay episode identity fields for the record contract.

    Returns:
        The normalized episode identity mapping.
    """

    seed = episode.get("source_seed", episode.get("seed"))
    return {
        "episode_id": episode.get("episode_id"),
        "source_seed": seed,
        "source_map": episode.get("source_map"),
    }


def _replay_frames(
    replayed_episode: Mapping[str, Any] | None,
    replayed_frames: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]] | None:
    """Resolve explicit replay frames or the ``steps`` field from a replay episode.

    Returns:
        Replay frames, or ``None`` when the replay harness supplied no trace.
    """

    if replayed_frames is not None:
        return list(replayed_frames)
    if replayed_episode is None:
        return None
    steps = replayed_episode.get("steps")
    if isinstance(steps, Sequence) and not isinstance(steps, str | bytes):
        return list(steps)
    return None


def _select_replay_window(
    replayed_frames: list[Mapping[str, Any]] | None,
    *,
    window_start_s: float,
    window_end_s: float,
) -> list[Mapping[str, Any]] | None:
    """Restrict a full replay trace to the source catalog segment window.

    Returns:
        The selected replay frames, or ``None`` when no replay was supplied.
    """

    if replayed_frames is None:
        return None
    selected: list[Mapping[str, Any]] = []
    for frame in replayed_frames:
        try:
            frame_time_s = float(frame["time_s"])
        except (KeyError, TypeError, ValueError):
            return replayed_frames
        if window_start_s <= frame_time_s <= window_end_s:
            selected.append(frame)
    return selected or replayed_frames


def _assess_replay(
    *,
    source_episode: Mapping[str, Any],
    source_frames: Sequence[Mapping[str, Any]],
    replayed_episode: Mapping[str, Any] | None,
    replayed_frames: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Assess replay identity and the full trace payload, failing closed without evidence.

    Returns:
        An exact-replay status block suitable for the persistence schema.
    """

    source_identity = _replay_identity(source_episode)
    source_digest = _episode_identity_digest(source_identity, source_frames)
    if replayed_episode is None or replayed_frames is None:
        return {
            "status": "unknown",
            "divergence_reason": "replay evidence was not supplied by a replay harness",
            "replay_digest": source_digest,
        }

    replay_identity = _replay_identity(replayed_episode)
    missing_identity = [
        field
        for field, value in replay_identity.items()
        if value is None or (isinstance(value, str) and not value.strip())
    ]
    if missing_identity:
        return {
            "status": FAIL,
            "divergence_reason": f"replayed episode missing required fields: {missing_identity}",
            "replay_digest": source_digest,
        }

    replay_digest = _episode_identity_digest(replay_identity, replayed_frames)
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
        "divergence_reason": "identity and replay trace payload matched source episode",
        "replay_digest": source_digest,
    }


def _assess_replayed_event(
    *,
    event_type: str,
    source_event_time_s: float,
    source_event_location: Sequence[float],
    replayed_frames: Sequence[Mapping[str, Any]] | None,
    time_tolerance_s: float,
    location_tolerance_m: float,
) -> dict[str, Any]:
    """Compare the source critical event with an explicitly supplied replay trace.

    Returns:
        A critical-event reproduction status block.
    """

    if replayed_frames is None:
        return assess_critical_event_reproduction(
            event_type=event_type,
            source_event_time_s=source_event_time_s,
            source_event_location=source_event_location,
            time_tolerance_s=time_tolerance_s,
            location_tolerance_m=location_tolerance_m,
            not_observed_reason=None,
        )
    try:
        replayed_event_time_s, _clearance_m, replayed_event_location = get_critical_event_from_frames(
            replayed_frames
        )
    except (KeyError, TypeError, ValueError) as exc:
        return assess_critical_event_reproduction(
            event_type=event_type,
            source_event_time_s=source_event_time_s,
            source_event_location=source_event_location,
            time_tolerance_s=time_tolerance_s,
            location_tolerance_m=location_tolerance_m,
            not_observed_reason=f"replay event could not be extracted: {exc}",
        )
    return assess_critical_event_reproduction(
        event_type=event_type,
        source_event_time_s=source_event_time_s,
        source_event_location=source_event_location,
        replayed_event_time_s=replayed_event_time_s,
        replayed_event_location=replayed_event_location,
        time_tolerance_s=time_tolerance_s,
        location_tolerance_m=location_tolerance_m,
    )


def _resolve_grid(
    perturbation_grid: Mapping[str, Sequence[float]] | None,
) -> dict[str, list[float]]:
    """Normalize and validate the perturbation grid before evaluating any cells.

    Returns:
        A grid with finite floating-point timing and speed values.
    """

    raw_grid = perturbation_grid or {
        "timing_offsets_s": [-0.25, 0.0, 0.25],
        "speed_deltas_m_s": [-0.2, 0.0, 0.2],
    }
    normalized: dict[str, list[float]] = {}
    for name in ("timing_offsets_s", "speed_deltas_m_s"):
        values = raw_grid.get(name)
        if values is None or isinstance(values, str | bytes) or not values:
            raise ValueError(f"perturbation_grid.{name} must be a non-empty sequence")
        converted: list[float] = []
        for value in values:
            converted_value = float(value)
            if not math.isfinite(converted_value):
                raise ValueError(f"perturbation_grid.{name} must contain finite numbers")
            converted.append(converted_value)
        normalized[name] = converted
    return normalized


def build_cell_verdict_from_trace_replay(  # noqa: C901
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

    source_positions = list(event_pedestrian_positions.values())
    if not source_positions:
        raise ValueError("event_pedestrian_positions must contain at least one position")

    def interpolate(
        trajectory: list[tuple[float, list[float]]],
        target_time_s: float,
    ) -> list[float] | None:
        """Linearly interpolate a pedestrian trajectory at one event time.

        Returns:
            Interpolated position, or ``None`` when the time is outside the trace.
        """

        if target_time_s < trajectory[0][0] or target_time_s > trajectory[-1][0]:
            return None
        for left, right in pairwise(trajectory):
            left_time, left_pos = left
            right_time, right_pos = right
            if left_time <= target_time_s <= right_time:
                span = right_time - left_time
                fraction = 0.0 if span == 0.0 else (target_time_s - left_time) / span
                return [
                    left_pos[index] + fraction * (right_pos[index] - left_pos[index])
                    for index in range(min(len(left_pos), len(right_pos)))
                ]
        return list(trajectory[-1][1])

    def average_speed(trajectory: list[tuple[float, list[float]]]) -> float:
        """Estimate the source pedestrian speed over the available trace window.

        Returns:
            Average speed in metres per second.
        """

        elapsed = trajectory[-1][0] - trajectory[0][0]
        if elapsed <= 0.0:
            return 0.0
        return math.dist(trajectory[0][1], trajectory[-1][1]) / elapsed

    def verdict_fn(
        timing_offset_s: float,
        speed_delta_m_s: float,
    ) -> Mapping[str, Any] | None:
        """Evaluate one perturbation cell against the trace replay.

        Returns:
            A verdict mapping with ``verdict`` and ``reason`` keys, or ``None``
            if the cell could not be evaluated.
        """

        effective_time = float(event_time_s) + float(timing_offset_s)
        if abs(float(timing_offset_s)) > float(time_tolerance_s):
            return {
                "verdict": FAIL,
                "reason": (
                    f"event time shifted by {abs(float(timing_offset_s)):.3f} s "
                    f"(tol: {time_tolerance_s} s)"
                ),
            }

        best_location_delta: float | None = None
        best_pedestrian: str | None = None
        for ped_idx, trajectory in ped_trajectories.items():
            if len(trajectory) < 2:
                continue
            position = interpolate(trajectory, effective_time)
            if position is None:
                continue
            base_pos = trajectory[0][1]
            source_speed = average_speed(trajectory)
            if source_speed > 0.0:
                speed_scale = max(0.0, (source_speed + float(speed_delta_m_s)) / source_speed)
                adjusted_position = [
                    base_pos[index] + (position[index] - base_pos[index]) * speed_scale
                    for index in range(min(len(base_pos), len(position)))
                ]
            else:
                adjusted_position = list(position)
            location_delta = min(
                math.dist(source_position, adjusted_position) for source_position in source_positions
            )
            if best_location_delta is None or location_delta < best_location_delta:
                best_location_delta = location_delta
                best_pedestrian = ped_idx

        if best_location_delta is None:
            return None
        if best_location_delta > float(location_tolerance_m):
            return {
                "verdict": FAIL,
                "reason": (
                    f"ped {best_pedestrian} displaced by {best_location_delta:.3f} m "
                    f"at t={effective_time:.3f} s "
                    f"(tol: {location_tolerance_m} m)"
                ),
            }
        return {
            "verdict": PASS,
            "reason": (
                f"critical event reproduced at t={effective_time:.3f} s with "
                f"location delta {best_location_delta:.3f} m"
            ),
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
    replayed_episode: Mapping[str, Any] | None = None,
    replayed_frames: Sequence[Mapping[str, Any]] | None = None,
    pre_margin_s: float = 1.0,
    post_margin_s: float = 1.0,
) -> dict[str, Any]:
    """Build a persistence record from a full episode trace.

    This is the end-to-end entry point for a single candidate.  It runs:

    1. ``extract_critical_segment`` on the episode trace to produce a catalog entry;
    2. replay evidence by comparing the source and replay trace payloads;
    3. ``assess_critical_event_reproduction`` using the source and replay events;
    4. ``evaluate_perturbation_grid`` using explicit replay-harness cell verdicts.

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
        cell_verdict_fn: Optional per-perturbation replay verdict function.  When ``None``,
            every cell is recorded as missing and promotion fails closed.
        replayed_episode: Identity or full episode payload returned by a replay harness.
        replayed_frames: Replay trace frames.  If omitted, ``replayed_episode.steps`` is used.
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

    frames = catalog_entry["segment"]["trace_frames"]
    source_episode = dict(catalog_entry["source_episode"])
    if source_map is not None:
        source_episode["source_map"] = source_map
    replayed_frames = _replay_frames(replayed_episode, replayed_frames)
    replayed_frames = _select_replay_window(
        replayed_frames,
        window_start_s=float(catalog_entry["segment"]["window_start_s"]),
        window_end_s=float(catalog_entry["segment"]["window_end_s"]),
    )
    exact_replay_block = _assess_replay(
        source_episode=source_episode,
        source_frames=frames,
        replayed_episode=replayed_episode,
        replayed_frames=replayed_frames,
    )

    event_type = catalog_entry["criticality"].get("signal", "min_clearance")
    event_time_s, _min_clearance_m, event_pedestrian_positions = _critical_event_context(frames)
    event_ped_pos = next(iter(event_pedestrian_positions.values()))
    critical_event_block = _assess_replayed_event(
        event_type=event_type,
        source_event_time_s=event_time_s,
        source_event_location=event_ped_pos,
        replayed_frames=replayed_frames,
        time_tolerance_s=event_time_tolerance_s,
        location_tolerance_m=event_location_tolerance_m,
    )

    grid_def = _resolve_grid(perturbation_grid)
    cell_verdict_fn = cell_verdict_fn or _missing_cell_verdict

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

    source_episode_with_digest = dict(source_episode)
    source_episode_with_digest["replay_digest"] = _episode_identity_digest(source_episode, frames)

    record = compute_persistence_record(
        scenario_id=scenario_id,
        source_episode=source_episode_with_digest,
        generated_scenario=generated_scenario,
        planner=planner,
        seed=int(source_episode["source_seed"]),
        config=dict(config),
        commit_hashes=dict(commit_hashes),
        exact_replay=exact_replay_block,
        critical_event_reproduced=critical_event_block,
        perturbation_grid=dict(grid_def),
        cell_verdicts=cells,
        missing_cell_reasons=missing,
    )
    return record


def build_persistence_from_catalog_entry(  # noqa: PLR0913
    *,
    catalog_entry: Mapping[str, Any],
    config: Mapping[str, Any],
    commit_hashes: Mapping[str, Any],
    planner: str = "goal",
    event_time_tolerance_s: float = DEFAULT_TIME_TOLERANCE_S,
    event_location_tolerance_m: float = DEFAULT_LOCATION_TOLERANCE_M,
    perturbation_grid: Mapping[str, Sequence[float]] | None = None,
    cell_verdict_fn: Callable[..., Mapping[str, Any] | None] | None = None,
    replayed_episode: Mapping[str, Any] | None = None,
    replayed_frames: Sequence[Mapping[str, Any]] | None = None,
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
        cell_verdict_fn: Optional per-perturbation replay verdict function.  When ``None``,
            every cell is recorded as missing and promotion fails closed.
        replayed_episode: Identity or full episode payload returned by a replay harness.
        replayed_frames: Replay trace frames.  If omitted, ``replayed_episode.steps`` is used.

    Returns:
        A schema-valid persistence record.
    """
    frames = catalog_entry["segment"]["trace_frames"]
    source_episode = dict(catalog_entry["source_episode"])
    replayed_frames = _replay_frames(replayed_episode, replayed_frames)
    replayed_frames = _select_replay_window(
        replayed_frames,
        window_start_s=float(catalog_entry["segment"]["window_start_s"]),
        window_end_s=float(catalog_entry["segment"]["window_end_s"]),
    )
    exact_replay_block = _assess_replay(
        source_episode=source_episode,
        source_frames=frames,
        replayed_episode=replayed_episode,
        replayed_frames=replayed_frames,
    )

    event_time_s, _min_clearance_m, event_pedestrian_positions = _critical_event_context(frames)
    event_ped_pos = next(iter(event_pedestrian_positions.values()))
    event_type = catalog_entry["criticality"].get("signal", "min_clearance")
    scenario_id = catalog_entry["scenario_id"]
    critical_event_block = _assess_replayed_event(
        event_type=event_type,
        source_event_time_s=event_time_s,
        source_event_location=event_ped_pos,
        replayed_frames=replayed_frames,
        time_tolerance_s=event_time_tolerance_s,
        location_tolerance_m=event_location_tolerance_m,
    )

    grid_def = _resolve_grid(perturbation_grid)
    cell_verdict_fn = cell_verdict_fn or _missing_cell_verdict

    cells, missing = evaluate_perturbation_grid(
        timing_offsets_s=list(grid_def["timing_offsets_s"]),
        speed_deltas_m_s=list(grid_def["speed_deltas_m_s"]),
        cell_verdict_fn=cell_verdict_fn,
    )

    source_episode_with_digest = dict(source_episode)
    source_episode_with_digest["replay_digest"] = _episode_identity_digest(source_episode, frames)

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
        seed=int(source_episode["source_seed"]),
        config=dict(config),
        commit_hashes=dict(commit_hashes),
        exact_replay=exact_replay_block,
        critical_event_reproduced=critical_event_block,
        perturbation_grid=dict(grid_def),
        cell_verdicts=cells,
        missing_cell_reasons=missing,
    )
    return record


def run_candidate_persistence_smoke(  # noqa: PLR0913
    *,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
    commit_hashes: Mapping[str, Any] | None = None,
    planner: str = "goal",
    output_root: Path | None = None,
    event_time_tolerance_s: float = DEFAULT_TIME_TOLERANCE_S,
    event_location_tolerance_m: float = DEFAULT_LOCATION_TOLERANCE_M,
    perturbation_grid: Mapping[str, Sequence[float]] | None = None,
    replay_evidence_fn: Callable[[Mapping[str, Any]], Mapping[str, Any] | None] | None = None,
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
        event_time_tolerance_s: Frozen event-time tolerance.
        event_location_tolerance_m: Frozen event-location tolerance.
        perturbation_grid: Frozen timing/speed grid.
        replay_evidence_fn: Optional function returning ``replayed_episode``,
            ``replayed_frames``, and ``cell_verdict_fn`` for each candidate.  Without it,
            replay and perturbation evidence remain unknown/missing.

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
        evidence = replay_evidence_fn(candidate) if replay_evidence_fn is not None else {}
        evidence = evidence or {}
        replayed_episode = evidence.get("replayed_episode")
        replayed_frames = evidence.get("replayed_frames")
        cell_verdict_fn = evidence.get("cell_verdict_fn")
        if "steps" in candidate and "segment" not in candidate:
            record = build_persistence_from_episode_trace(
                episode=candidate,
                planner=planner,
                config=config,
                commit_hashes=commit_hashes,
                event_time_tolerance_s=event_time_tolerance_s,
                event_location_tolerance_m=event_location_tolerance_m,
                perturbation_grid=perturbation_grid,
                cell_verdict_fn=cell_verdict_fn,
                replayed_episode=replayed_episode,
                replayed_frames=replayed_frames,
            )
        else:
            record = build_persistence_from_catalog_entry(
                catalog_entry=candidate,
                config=config,
                commit_hashes=commit_hashes,
                planner=planner,
                event_time_tolerance_s=event_time_tolerance_s,
                event_location_tolerance_m=event_location_tolerance_m,
                perturbation_grid=perturbation_grid,
                cell_verdict_fn=cell_verdict_fn,
                replayed_episode=replayed_episode,
                replayed_frames=replayed_frames,
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


def _episode_identity_digest(
    episode: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Return a SHA-256 digest over identity and, when supplied, the full trace payload."""

    identity = json.dumps(
        {
            "episode_id": episode.get("episode_id"),
            "source_seed": episode.get("source_seed"),
            "source_map": episode.get("source_map"),
            "trace_frames": list(frames) if frames is not None else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _catalog_entry_digest(entry: Mapping[str, Any]) -> str:
    """Return a SHA-256 hex digest over the catalog entry."""
    data = json.dumps(dict(entry), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()
