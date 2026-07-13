"""Score trace-exemplar episodes for figure-selection interest.

This module implements a figure-selection heuristic for choosing illustrative
episodes. It is NOT a benchmark metric and carries no evaluation claim.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

DEFAULT_WEIGHTS: dict[str, float] = {
    "min_dist_severity": 0.20,
    "collapse_rate": 0.15,
    "time_below_2p5m": 0.10,
    "outcome_salience": 0.20,
    "speed_modulation": 0.05,
    "heading_activity": 0.05,
    "detour_ratio": 0.10,
    "frozen_robot": 0.15,
}

COLLISION_LIKE_TERMINATIONS: frozenset[str] = frozenset(
    {
        "collision",
        "collided",
        "crash",
        "contact",
        "pedestrian_collision",
        "robot_collision",
    }
)


@dataclass(frozen=True, slots=True)
class EpisodeInterest:
    """Normalized trace-exemplar interest score for one episode.

    Feature values are normalized to the inclusive range ``[0, 1]``:

    - ``min_dist_severity``: ``1 - global_min_robot_ped_distance_m / 3.0``
      clamped to ``[0, 1]`` where 3 m is comfortable clearance.
    - ``collapse_rate``: steepest decline of minimum robot-pedestrian distance
      over any one-second trailing window, divided by 2.0 m/s and clamped.
    - ``time_below_2p5m``: fraction of trace steps with minimum distance below
      2.5 m.
    - ``outcome_salience``: 1.0 for non-success collision-like termination,
      0.8 for other non-success outcomes, and 0.0 for success.
    - ``speed_modulation``: standard deviation of executed speed divided by
      the maximum commanded linear-velocity cap seen, clamped.
    - ``heading_activity``: mean absolute robot heading change per second,
      divided by 1.0 rad/s and clamped.
    - ``detour_ratio``: excess path length over start-to-end straight-line
      distance, divided by 1.5 and clamped.
    - ``frozen_robot``: 1.0 when a failed episode is effectively stationary
      under the available displacement proxy or the long-episode fallback.
    """

    episode_dir: Path
    episode_id: str
    episode_status: str
    planner: str
    scenario_id: str
    seed: int | None
    features: dict[str, float]
    composite_score: float


@dataclass(frozen=True, slots=True)
class ComparisonPair:
    """Pairwise same-scenario trace comparison across planners."""

    scenario_id: str
    seed: int | None
    left_episode_id: str
    left_planner: str
    right_episode_id: str
    right_planner: str
    outcome_divergence: float
    trajectory_divergence: float
    pair_score: float


@dataclass(frozen=True, slots=True)
class InterestReport:
    """Complete interest report for one or more trace-exemplar bundle roots."""

    roots: list[Path]
    weights: dict[str, float]
    episodes: list[EpisodeInterest]
    comparison_pairs: list[ComparisonPair]


def discover_episode_dirs(roots: Sequence[Path]) -> list[Path]:
    """Return all directories below roots that contain episode trace inputs.

    Args:
        roots: Bundle roots or bare episode directories to scan.

    Returns:
        Deterministically sorted directories containing both ``metadata.json``
        and ``trace_series.json``.
    """

    discovered: set[Path] = set()
    for root in roots:
        root_path = Path(root)
        if _is_episode_dir(root_path):
            discovered.add(root_path)
            continue
        if not root_path.exists():
            continue
        for metadata_path in root_path.rglob("metadata.json"):
            episode_dir = metadata_path.parent
            if _is_episode_dir(episode_dir):
                discovered.add(episode_dir)
    return sorted(discovered, key=lambda path: path.as_posix())


def score_episode(episode_dir: Path) -> EpisodeInterest:
    """Score a single episode directory.

    Args:
        episode_dir: Directory containing ``metadata.json`` and
            ``trace_series.json``.

    Returns:
        Frozen dataclass containing normalized feature values and the default
        weighted composite score.

    Raises:
        FileNotFoundError: If required JSON files are missing.
        ValueError: If required trace rows are absent.
    """

    metadata, trace = _load_episode(episode_dir)
    rows = trace.get("derived_rows", [])
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"trace_series.json has no derived_rows: {episode_dir}")

    summary = metadata.get("summary", {})
    summary = summary if isinstance(summary, dict) else {}
    episode_status = str(
        metadata.get("episode_status")
        or summary.get("episode_status")
        or trace.get("metadata", {}).get("episode_status")
        or "unknown"
    )
    planner = str(metadata.get("planner") or summary.get("planner") or "unknown_planner")
    scenario_id = str(
        metadata.get("scenario_id") or summary.get("scenario_id") or "unknown_scenario"
    )
    seed = _optional_int(metadata.get("seed", summary.get("seed")))
    episode_id = str(metadata.get("episode_id") or episode_dir.name)

    features = _compute_features(rows, metadata, episode_status)
    return EpisodeInterest(
        episode_dir=episode_dir,
        episode_id=episode_id,
        episode_status=episode_status,
        planner=planner,
        scenario_id=scenario_id,
        seed=seed,
        features=features,
        composite_score=_weighted_score(features, DEFAULT_WEIGHTS),
    )


def score_bundles(
    roots: Sequence[Path],
    weights: Mapping[str, float] | None = None,
) -> InterestReport:
    """Score every episode in bundle roots and compute planner comparison pairs.

    Args:
        roots: Bundle roots or bare episode directories.
        weights: Optional feature-weight overrides. Unspecified features use
            ``DEFAULT_WEIGHTS``.

    Returns:
        Interest report sorted by composite score descending, then episode ID.
    """

    effective_weights = _effective_weights(weights)
    episodes = [
        _replace_score(score_episode(episode_dir), effective_weights)
        for episode_dir in discover_episode_dirs(roots)
    ]
    episodes = sorted(episodes, key=lambda item: (-item.composite_score, item.episode_id))
    return InterestReport(
        roots=[Path(root) for root in roots],
        weights=effective_weights,
        episodes=episodes,
        comparison_pairs=_comparison_pairs(episodes),
    )


def write_report_json(report: InterestReport, output_path: Path, top_n: int | None = None) -> None:
    """Write a deterministic JSON interest report.

    Args:
        report: Report returned by ``score_bundles``.
        output_path: Destination JSON path.
        top_n: Optional number of top-ranked episodes to include.
    """

    payload = _report_payload(report, top_n)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_report_markdown(
    report: InterestReport, output_path: Path, top_n: int | None = None
) -> None:
    """Write a Markdown interest report with ranked episode and pair tables.

    Args:
        report: Report returned by ``score_bundles``.
        output_path: Destination Markdown path.
        top_n: Optional number of top-ranked episodes to include.
    """

    episodes = _top_episodes(report, top_n)
    lines = [
        "# Trace-Exemplar Interest Report",
        "",
        "Figure-selection heuristic only; not a benchmark metric or evaluation claim.",
        "",
        "## Episode Ranking",
        "",
        "| rank | score | episode_id | planner | scenario_id | seed | status |",
        "| ---: | ---: | --- | --- | --- | ---: | --- |",
    ]
    for rank, episode in enumerate(episodes, start=1):
        lines.append(
            "| "
            f"{rank} | {_fmt(episode.composite_score)} | {episode.episode_id} | "
            f"{episode.planner} | {episode.scenario_id} | {_seed_text(episode.seed)} | "
            f"{episode.episode_status} |"
        )

    lines.extend(
        [
            "",
            "## Feature Values",
            "",
            "| episode_id | min_dist_severity | collapse_rate | time_below_2p5m | "
            "outcome_salience | speed_modulation | heading_activity | detour_ratio | "
            "frozen_robot |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for episode in episodes:
        lines.append(
            "| "
            f"{episode.episode_id} | "
            f"{_fmt(episode.features['min_dist_severity'])} | "
            f"{_fmt(episode.features['collapse_rate'])} | "
            f"{_fmt(episode.features['time_below_2p5m'])} | "
            f"{_fmt(episode.features['outcome_salience'])} | "
            f"{_fmt(episode.features['speed_modulation'])} | "
            f"{_fmt(episode.features['heading_activity'])} | "
            f"{_fmt(episode.features['detour_ratio'])} | "
            f"{_fmt(episode.features['frozen_robot'])} |"
        )

    lines.extend(
        [
            "",
            "## Comparison Pairs",
            "",
            "| rank | score | scenario_id | seed | left_planner | right_planner | "
            "outcome_divergence | trajectory_divergence |",
            "| ---: | ---: | --- | ---: | --- | --- | ---: | ---: |",
        ]
    )
    for rank, pair in enumerate(report.comparison_pairs, start=1):
        lines.append(
            "| "
            f"{rank} | {_fmt(pair.pair_score)} | {pair.scenario_id} | {_seed_text(pair.seed)} | "
            f"{pair.left_planner} | {pair.right_planner} | "
            f"{_fmt(pair.outcome_divergence)} | {_fmt(pair.trajectory_divergence)} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_episode_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "metadata.json").is_file()
        and (path / "trace_series.json").is_file()
    )


def _load_episode(episode_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata_path = episode_dir / "metadata.json"
    trace_path = episode_dir / "trace_series.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata.json is not an object: {metadata_path}")
    if not isinstance(trace, dict):
        raise ValueError(f"trace_series.json is not an object: {trace_path}")
    return metadata, trace


def _compute_features(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    episode_status: str,
) -> dict[str, float]:
    distances = [_float(row.get("min_robot_ped_distance_m")) for row in rows]
    distance_values = [value for value in distances if value is not None]
    global_min = _metadata_global_min(metadata)
    if global_min is None and distance_values:
        global_min = min(distance_values)

    positions = _positions(rows)
    times = _times(rows)
    path_length = _path_length(positions)
    straight_line = _straight_line(positions)
    duration = max(times[-1] - times[0], 0.0) if len(times) >= 2 else 0.0

    executed_speeds = [_float(row.get("executed_speed_m_s")) for row in rows]
    executed_speed_values = [value for value in executed_speeds if value is not None]
    commanded_caps = [_float(row.get("commanded_linear_velocity_m_s")) for row in rows]
    commanded_cap = max([abs(value) for value in commanded_caps if value is not None], default=0.0)
    heading_values = [_float(row.get("robot_heading_rad")) for row in rows]

    return {
        "min_dist_severity": _clamp(1.0 - (global_min or 0.0) / 3.0),
        "collapse_rate": _collapse_rate(rows),
        "time_below_2p5m": _time_below(distance_values, threshold=2.5, total_steps=len(rows)),
        "outcome_salience": _outcome_salience(episode_status, metadata),
        "speed_modulation": _clamp(_stddev(executed_speed_values) / max(commanded_cap, 0.1)),
        "heading_activity": _heading_activity(heading_values, times),
        "detour_ratio": _clamp((path_length / max(straight_line, 1e-6) - 1.0) / 1.5),
        "frozen_robot": _frozen_robot(
            episode_status=episode_status,
            displacement=straight_line,
            proxy_distance=_spawn_goal_proxy(metadata, commanded_cap, duration),
            step_count=len(rows),
        ),
    }


def _metadata_global_min(metadata: Mapping[str, Any]) -> float | None:
    summary = metadata.get("summary")
    if isinstance(summary, Mapping):
        value = _float(summary.get("global_min_robot_ped_distance_m"))
        if value is not None:
            return value
    return _float(metadata.get("global_min_robot_ped_distance_m"))


def _collapse_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    samples: list[tuple[float, float]] = []
    for index, row in enumerate(rows):
        distance = _float(row.get("min_robot_ped_distance_m"))
        if distance is None:
            continue
        time_s = _float(row.get("time_s"))
        samples.append((float(index) if time_s is None else time_s, distance))

    steepest = 0.0
    for end_index, (end_time, end_distance) in enumerate(samples):
        for start_time, start_distance in samples[: end_index + 1]:
            elapsed = end_time - start_time
            if elapsed <= 0.0 or elapsed > 1.0:
                continue
            decline = start_distance - end_distance
            if decline > 0.0:
                steepest = max(steepest, decline / elapsed)
    return _clamp(steepest / 2.0)


def _time_below(values: Sequence[float], threshold: float, total_steps: int) -> float:
    if total_steps <= 0:
        return 0.0
    return _clamp(sum(1 for value in values if value < threshold) / total_steps)


def _outcome_salience(episode_status: str, metadata: Mapping[str, Any]) -> float:
    if episode_status == "success":
        return 0.0
    termination_reason = ""
    summary = metadata.get("summary")
    if isinstance(summary, Mapping):
        termination_reason = str(summary.get("termination_reason") or "")
    termination_reason = str(metadata.get("termination_reason") or termination_reason).lower()
    if any(token in termination_reason for token in COLLISION_LIKE_TERMINATIONS):
        return 1.0
    return 0.8


def _heading_activity(values: Sequence[float | None], times: Sequence[float]) -> float:
    changes: list[float] = []
    for previous, current in pairwise(values):
        if previous is None or current is None:
            continue
        changes.append(abs(_angle_delta(current, previous)))
    if not changes:
        return 0.0
    if len(times) >= 2:
        elapsed = max(times[-1] - times[0], 1e-6)
    else:
        elapsed = float(len(changes))
    return _clamp((sum(changes) / elapsed) / 1.0)


def _frozen_robot(
    *,
    episode_status: str,
    displacement: float,
    proxy_distance: float | None,
    step_count: int,
) -> float:
    if episode_status == "success":
        return 0.0
    if proxy_distance is not None and proxy_distance > 0.0 and displacement < 0.2 * proxy_distance:
        return 1.0
    if step_count >= 300 and displacement < 2.0:
        return 1.0
    return 0.0


def _spawn_goal_proxy(
    metadata: Mapping[str, Any],
    commanded_cap: float,
    duration: float,
) -> float | None:
    start = _point_from_any(metadata.get("start") or metadata.get("spawn"))
    goal = _point_from_any(metadata.get("goal") or metadata.get("target"))
    if start is not None and goal is not None:
        return _distance(start, goal)
    if commanded_cap > 0.0 and duration > 0.0:
        return commanded_cap * duration
    return None


def _comparison_pairs(episodes: Sequence[EpisodeInterest]) -> list[ComparisonPair]:
    groups: dict[tuple[str, int | None], list[EpisodeInterest]] = {}
    for episode in episodes:
        groups.setdefault((episode.scenario_id, episode.seed), []).append(episode)

    pairs: list[ComparisonPair] = []
    for (scenario_id, seed), group in groups.items():
        ordered = sorted(group, key=lambda item: (item.planner, item.episode_id))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                if left.planner == right.planner:
                    continue
                outcome_divergence = 1.0 if left.episode_status != right.episode_status else 0.0
                trajectory_divergence = _trajectory_divergence(left.episode_dir, right.episode_dir)
                pairs.append(
                    ComparisonPair(
                        scenario_id=scenario_id,
                        seed=seed,
                        left_episode_id=left.episode_id,
                        left_planner=left.planner,
                        right_episode_id=right.episode_id,
                        right_planner=right.planner,
                        outcome_divergence=outcome_divergence,
                        trajectory_divergence=trajectory_divergence,
                        pair_score=_clamp(0.6 * outcome_divergence + 0.4 * trajectory_divergence),
                    )
                )
    return sorted(
        pairs,
        key=lambda item: (
            -item.pair_score,
            item.scenario_id,
            -1 if item.seed is None else item.seed,
            item.left_planner,
            item.right_planner,
        ),
    )


def _trajectory_divergence(left_dir: Path, right_dir: Path) -> float:
    _, left_trace = _load_episode(left_dir)
    _, right_trace = _load_episode(right_dir)
    left_positions = _positions_by_step(left_trace.get("derived_rows", []))
    right_positions = _positions_by_step(right_trace.get("derived_rows", []))
    aligned_steps = sorted(set(left_positions).intersection(right_positions))
    if not aligned_steps:
        return 0.0
    mean_distance = sum(
        _distance(left_positions[step], right_positions[step]) for step in aligned_steps
    ) / len(aligned_steps)
    return _clamp(mean_distance / 10.0)


def _positions_by_step(raw_rows: Any) -> dict[int, tuple[float, float]]:
    if not isinstance(raw_rows, list):
        return {}
    out: dict[int, tuple[float, float]] = {}
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            continue
        point = _row_position(row)
        if point is None:
            continue
        step = _optional_int(row.get("step"))
        out[index if step is None else step] = point
    return out


def _positions(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    return [point for row in rows if (point := _row_position(row)) is not None]


def _row_position(row: Mapping[str, Any]) -> tuple[float, float] | None:
    x = _float(row.get("robot_x_m"))
    y = _float(row.get("robot_y_m"))
    if x is None or y is None:
        return None
    return (x, y)


def _times(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    times: list[float] = []
    for index, row in enumerate(rows):
        value = _float(row.get("time_s"))
        times.append(float(index) if value is None else value)
    return times


def _path_length(positions: Sequence[tuple[float, float]]) -> float:
    return sum(_distance(left, right) for left, right in pairwise(positions))


def _straight_line(positions: Sequence[tuple[float, float]]) -> float:
    if len(positions) < 2:
        return 0.0
    return _distance(positions[0], positions[-1])


def _distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(right[0] - left[0], right[1] - left[1])


def _point_from_any(raw: Any) -> tuple[float, float] | None:
    if isinstance(raw, Mapping):
        x = _float(raw.get("x") or raw.get("x_m"))
        y = _float(raw.get("y") or raw.get("y_m"))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
        x = _float(raw[0])
        y = _float(raw[1])
    else:
        return None
    if x is None or y is None:
        return None
    return (x, y)


def _stddev(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _angle_delta(current: float, previous: float) -> float:
    return (current - previous + math.pi) % (2.0 * math.pi) - math.pi


def _effective_weights(overrides: Mapping[str, float] | None) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    if overrides:
        for key, value in overrides.items():
            if key not in DEFAULT_WEIGHTS:
                raise ValueError(f"unknown feature weight: {key}")
            weights[key] = float(value)
    return weights


def _weighted_score(features: Mapping[str, float], weights: Mapping[str, float]) -> float:
    weight_total = sum(max(0.0, value) for value in weights.values())
    if weight_total <= 0.0:
        raise ValueError("at least one weight must be positive")
    raw_score = sum(features[key] * max(0.0, weights[key]) for key in DEFAULT_WEIGHTS)
    return _clamp(raw_score / weight_total)


def _replace_score(episode: EpisodeInterest, weights: Mapping[str, float]) -> EpisodeInterest:
    return EpisodeInterest(
        episode_dir=episode.episode_dir,
        episode_id=episode.episode_id,
        episode_status=episode.episode_status,
        planner=episode.planner,
        scenario_id=episode.scenario_id,
        seed=episode.seed,
        features=episode.features,
        composite_score=_weighted_score(episode.features, weights),
    )


def _report_payload(report: InterestReport, top_n: int | None) -> dict[str, Any]:
    return {
        "roots": [root.as_posix() for root in report.roots],
        "weights": report.weights,
        "episodes": [_episode_payload(episode) for episode in _top_episodes(report, top_n)],
        "comparison_pairs": [asdict(pair) for pair in report.comparison_pairs],
    }


def _episode_payload(episode: EpisodeInterest) -> dict[str, Any]:
    payload = asdict(episode)
    payload["episode_dir"] = episode.episode_dir.as_posix()
    return payload


def _top_episodes(report: InterestReport, top_n: int | None) -> list[EpisodeInterest]:
    if top_n is None:
        return list(report.episodes)
    return list(report.episodes[: max(0, top_n)])


def _float(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _optional_int(raw: Any) -> int | None:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _seed_text(seed: int | None) -> str:
    return "" if seed is None else str(seed)
