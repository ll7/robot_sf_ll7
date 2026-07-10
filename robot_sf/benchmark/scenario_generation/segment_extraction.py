"""Extract a conservative, replay-pending critical segment from one episode trace."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from robot_sf.benchmark.scenario_generation.catalog_schema import validate_catalog_entry

_DISTILLER_ID = "critical_segment_min_clearance.v1"
_CLAIM_BOUNDARY = "generated scenario hypotheses only"


def extract_critical_segment(
    episode: Mapping[str, Any],
    *,
    pre_margin_s: float | None = None,
    post_margin_s: float | None = None,
) -> dict[str, Any]:
    """Extract the closest robot--pedestrian window into a catalog entry.

    The input is a narrow, JSON-compatible trace contract: ``episode_id``,
    ``seed``, ``source_map``, and ordered ``steps`` with ``time_s``,
    ``robot.position``, and ``pedestrians[].position``.  The result preserves
    sampled states but reports ``not_representable_yet`` because this slice does
    not claim that trace states can already be replayed as scenario YAML.

    Returns:
        A JSON-compatible generated catalog entry with a pinned source seed and
        the closest robot--pedestrian trace segment.

    Raises:
        ValueError: If the trace lacks a finite, ordered robot/pedestrian record.
    """

    pre_margin_s = 1.0 if pre_margin_s is None else pre_margin_s
    post_margin_s = 1.0 if post_margin_s is None else post_margin_s
    _validate_margin("pre_margin_s", pre_margin_s)
    _validate_margin("post_margin_s", post_margin_s)
    episode_id = _required_string(episode, "episode_id")
    source_map = _required_string(episode, "source_map")
    seed = _required_seed(episode)
    frames = _normalize_frames(episode.get("steps"))

    critical_index, min_clearance_m = _critical_frame(frames)
    critical_time_s = frames[critical_index]["time_s"]
    start_index = _first_frame_at_or_after(frames, critical_time_s - pre_margin_s)
    end_index = _last_frame_at_or_before(frames, critical_time_s + post_margin_s)
    selected_frames = frames[start_index : end_index + 1]
    start_s = selected_frames[0]["time_s"]
    end_s = selected_frames[-1]["time_s"]

    scenario_id = _stable_scenario_id(episode_id, seed, start_s, end_s, critical_time_s)
    entry = {
        "schema_version": "generated-scenario-catalog-entry.v1",
        "scenario_id": scenario_id,
        "metadata": {
            "source": "auto_generated",
            "generated_by": "robot_sf.benchmark.scenario_generation.segment_extraction",
            "required_manual_review": True,
            "benchmark_evidence": False,
        },
        "source_episode": {
            "episode_id": episode_id,
            "source_seed": seed,
            "source_map": source_map,
        },
        "criticality": {
            "signal": "min_clearance",
            "observed_at_s": critical_time_s,
            "source_metrics": {"min_clearance_m": min_clearance_m},
        },
        "segment": {
            "window_start_s": start_s,
            "window_end_s": end_s,
            "initial_robot_state": selected_frames[0]["robot"],
            "trace_frames": selected_frames,
        },
        "replay": {
            "schema_version": "generated-scenario-replay.v1",
            "source_seed": seed,
            "replay_contract": "source_episode_seed_pinned.v1",
            "status": "not_representable_yet",
            "warnings": [
                "replay_gap: sampled trace states are not yet representable as standalone scenario YAML"
            ],
        },
        "provenance": {
            "schema_version": "generated-scenario-provenance.v1",
            "source_issue": "#4932",
            "distiller": _DISTILLER_ID,
            "claim_boundary": _CLAIM_BOUNDARY,
        },
    }
    validate_catalog_entry(entry)
    return entry


def _validate_margin(name: str, value: float) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")


def _required_string(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"episode.{name} must be a non-empty string")
    return value.strip()


def _required_seed(episode: Mapping[str, Any]) -> int:
    seed = episode.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("episode.seed must be an integer")
    return seed


def _normalize_frames(raw_steps: object) -> list[dict[str, Any]]:
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, str | bytes) or not raw_steps:
        raise ValueError("episode.steps must be a non-empty sequence")

    frames: list[dict[str, Any]] = []
    previous_time_s = -math.inf
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise ValueError(f"episode.steps[{index}] must be a mapping")
        time_s = _finite_number(raw_step.get("time_s"), f"episode.steps[{index}].time_s")
        if time_s <= previous_time_s:
            raise ValueError("episode.steps time_s must be strictly increasing")
        previous_time_s = time_s
        robot = _state(raw_step.get("robot"), f"episode.steps[{index}].robot")
        raw_pedestrians = raw_step.get("pedestrians")
        if not isinstance(raw_pedestrians, Sequence) or isinstance(raw_pedestrians, str | bytes):
            raise ValueError(f"episode.steps[{index}].pedestrians must be a sequence")
        pedestrians = [
            _state(pedestrian, f"episode.steps[{index}].pedestrians[{pedestrian_index}]")
            for pedestrian_index, pedestrian in enumerate(raw_pedestrians)
        ]
        frames.append({"time_s": time_s, "robot": robot, "pedestrians": pedestrians})
    return frames


def _state(raw_state: object, path: str) -> dict[str, list[float]]:
    if not isinstance(raw_state, Mapping):
        raise ValueError(f"{path} must be a mapping")
    raw_position = raw_state.get("position")
    if (
        not isinstance(raw_position, Sequence)
        or isinstance(raw_position, str | bytes)
        or len(raw_position) != 2
    ):
        raise ValueError(f"{path}.position must contain exactly two numbers")
    return {
        "position": [
            _finite_number(raw_position[0], f"{path}.position[0]"),
            _finite_number(raw_position[1], f"{path}.position[1]"),
        ]
    }


def _finite_number(value: object, path: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def _critical_frame(frames: list[dict[str, Any]]) -> tuple[int, float]:
    best: tuple[int, float] | None = None
    for frame_index, frame in enumerate(frames):
        robot_position = frame["robot"]["position"]
        for pedestrian in frame["pedestrians"]:
            pedestrian_position = pedestrian["position"]
            clearance_m = math.dist(robot_position, pedestrian_position)
            if best is None or clearance_m < best[1]:
                best = (frame_index, clearance_m)
    if best is None:
        raise ValueError(
            "episode trace contains no pedestrian positions for min_clearance extraction"
        )
    return best


def _first_frame_at_or_after(frames: list[dict[str, Any]], threshold_s: float) -> int:
    return next(index for index, frame in enumerate(frames) if frame["time_s"] >= threshold_s)


def _last_frame_at_or_before(frames: list[dict[str, Any]], threshold_s: float) -> int:
    return next(
        index for index in reversed(range(len(frames))) if frames[index]["time_s"] <= threshold_s
    )


def _stable_scenario_id(
    episode_id: str,
    seed: int,
    start_s: float,
    end_s: float,
    observed_at_s: float,
) -> str:
    identity = json.dumps(
        [episode_id, seed, start_s, end_s, observed_at_s], separators=(",", ":"), ensure_ascii=True
    )
    return f"generated-{hashlib.sha256(identity.encode()).hexdigest()[:16]}"


__all__ = ["extract_critical_segment"]
