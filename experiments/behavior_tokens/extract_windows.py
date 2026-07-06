"""Offline extraction of interaction-trace windows into feature vectors.

Reads existing benchmark episode JSONL rows that contain a
``algorithm_metadata.simulation_step_trace.steps`` list, slides fixed-size windows
over each trace, and converts every window into a compact, interpretable feature
vector (see :data:`experiments.behavior_tokens.schemas.FEATURE_NAMES`).

This script is offline and non-invasive: it only reads saved traces and writes
diagnostic artifacts under ``output/`` by default. It never launches a new
simulation campaign. Rows without a usable step trace are skipped with an explicit
reason (reported in the run summary), not silently dropped.

Example::

    uv run python experiments/behavior_tokens/extract_windows.py \\
        --episode-jsonl 'output/**/episodes.jsonl' \\
        --window-steps 10 --stride-steps 5 \\
        --output-jsonl output/experiments/behavior_tokens/windows.jsonl \\
        --output-csv output/experiments/behavior_tokens/windows.csv

Claim boundary: exploratory diagnostic tooling only; not validated metric or
benchmark evidence.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# Allow ``python experiments/behavior_tokens/extract_windows.py`` (script-style
# execution puts the script directory on sys.path, not the repo root) as well as
# namespace-package imports used by the test suite.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.behavior_tokens.schemas import (  # noqa: E402
    CLAIM_BOUNDARY,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    NEAR_PEDESTRIAN_THRESHOLD_M,
    STOP_SPEED_THRESHOLD_M_S,
    WindowRecord,
)


def _finite(value: Any) -> float | None:
    """Return ``value`` as a finite float, or ``None`` when not numeric/finite."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _vec2(value: Any) -> np.ndarray | None:
    """Return a length-2 float array from a ``[x, y]`` payload, or ``None``."""
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        x = _finite(value[0])
        y = _finite(value[1])
        if x is not None and y is not None:
            return np.array([x, y], dtype=float)
    return None


def _slope(times: list[float], values: list[float]) -> float | None:
    """Return the least-squares slope of ``values`` vs ``times`` (per second)."""
    if len(times) < 2 or len(set(times)) < 2:
        return None
    slope = float(np.polyfit(np.asarray(times), np.asarray(values), 1)[0])
    return slope if math.isfinite(slope) else None


def _rms(values: list[float]) -> float | None:
    """Return the root-mean-square of ``values``, or ``None`` when empty."""
    if not values:
        return None
    return float(np.sqrt(np.mean(np.square(np.asarray(values)))))


def _command_linear_angular(planner: Any) -> tuple[float | None, float | None]:
    """Extract (linear_speed, angular_velocity) proxies from a step planner payload.

    Supports both differential ``{linear_velocity, angular_velocity}`` and holonomic
    ``{command_kind: holonomic_vxy_world, vx, vy}`` actions. Angular velocity is
    ``None`` for holonomic commands (no scalar yaw command available).
    """
    if not isinstance(planner, dict):
        return None, None
    action = planner.get("selected_action")
    if not isinstance(action, dict):
        return None, None
    if action.get("command_kind") == "holonomic_vxy_world":
        vx = _finite(action.get("vx"))
        vy = _finite(action.get("vy"))
        if vx is None or vy is None:
            return None, None
        return float(math.hypot(vx, vy)), None
    linear = _finite(action.get("linear_velocity"))
    angular = _finite(action.get("angular_velocity"))
    return linear, angular


def _robot_speed(
    robot: Any, prev_pos: np.ndarray | None, pos: np.ndarray | None, dt: float | None
) -> float | None:
    """Return robot speed from an explicit velocity, else a finite-difference proxy."""
    if isinstance(robot, dict):
        velocity = _vec2(robot.get("velocity"))
        if velocity is not None:
            return float(np.linalg.norm(velocity))
    if prev_pos is not None and pos is not None and dt and dt > 0.0:
        return float(np.linalg.norm(pos - prev_pos) / dt)
    return None


def _nearest_pedestrian(
    robot_pos: np.ndarray | None, pedestrians: Any, robot_vel: np.ndarray | None
) -> dict[str, Any] | None:
    """Return nearest-pedestrian distance/speed info for one step, or ``None``.

    ``rel_speed`` is only populated when the pedestrian carries a velocity; otherwise
    it stays ``None`` so downstream features never fabricate a zero relative speed.
    """
    if robot_pos is None or not isinstance(pedestrians, list) or not pedestrians:
        return None
    best: dict[str, Any] | None = None
    for ped in pedestrians:
        if not isinstance(ped, dict):
            continue
        ped_pos = _vec2(ped.get("position"))
        if ped_pos is None:
            continue
        distance = float(np.linalg.norm(ped_pos - robot_pos))
        if best is None or distance < best["distance"]:
            ped_vel = _vec2(ped.get("velocity"))
            rel_speed = None
            if ped_vel is not None and robot_vel is not None:
                rel_speed = float(np.linalg.norm(ped_vel - robot_vel))
            best = {
                "distance": distance,
                "ped_id": str(ped.get("id")) if ped.get("id") is not None else None,
                "ped_speed": float(np.linalg.norm(ped_vel)) if ped_vel is not None else None,
                "rel_speed": rel_speed,
            }
    return best


def _step_time(step: dict[str, Any], index: int) -> float:
    """Return a step time in seconds, falling back to the step index when absent."""
    time_s = _finite(step.get("time_s"))
    if time_s is not None:
        return time_s
    step_idx_val = _finite(step.get("step"))
    return step_idx_val if step_idx_val is not None else float(index)


class _ParsedWindow:
    """Per-step series parsed from a window, aligned to make features easy to derive.

    Keeping the parsing separate from feature math keeps each feature helper small and
    individually testable, and ensures the missing-vs-zero distinction is applied once.
    """

    def __init__(self, steps: list[dict[str, Any]]) -> None:
        self.times: list[float] = [_step_time(step, idx) for idx, step in enumerate(steps)]
        # nearest_dists is aligned to steps (None when no pedestrian that step).
        self.nearest_dists: list[float | None] = []
        self.nearest_dist_times: list[float] = []
        self.rel_speeds: list[float] = []
        self.ped_speed_series: list[tuple[float, float]] = []  # (time, ped speed) while near
        self.robot_speed_series: list[tuple[float, float]] = []  # (time, robot speed)
        self.step_speed_dist: list[tuple[float | None, float | None]] = []  # per-step
        self.linear_cmds: list[float] = []
        self.angular_cmds: list[float] = []
        self.headings: list[float] = []
        self._parse(steps)

    def _parse(self, steps: list[dict[str, Any]]) -> None:
        prev_pos: np.ndarray | None = None
        for idx, step in enumerate(steps):
            robot = step.get("robot") if isinstance(step.get("robot"), dict) else {}
            pos = _vec2(robot.get("position"))
            robot_vel = _vec2(robot.get("velocity"))
            dt = (self.times[idx] - self.times[idx - 1]) if idx > 0 else None
            speed = _robot_speed(robot, prev_pos, pos, dt)
            if speed is not None:
                self.robot_speed_series.append((self.times[idx], speed))
            heading = _finite(robot.get("heading"))
            if heading is not None:
                self.headings.append(heading)

            nearest = _nearest_pedestrian(pos, step.get("pedestrians"), robot_vel)
            distance = nearest["distance"] if nearest is not None else None
            self.nearest_dists.append(distance)
            self.step_speed_dist.append((speed, distance))
            if nearest is not None:
                self.nearest_dist_times.append(self.times[idx])
                if nearest["rel_speed"] is not None:
                    self.rel_speeds.append(nearest["rel_speed"])
                if (
                    nearest["ped_speed"] is not None
                    and nearest["distance"] < NEAR_PEDESTRIAN_THRESHOLD_M
                ):
                    self.ped_speed_series.append((self.times[idx], nearest["ped_speed"]))

            linear, angular = _command_linear_angular(step.get("planner"))
            if linear is not None:
                self.linear_cmds.append(linear)
            if angular is not None:
                self.angular_cmds.append(angular)
            prev_pos = pos


def _clearance_features(parsed: _ParsedWindow) -> dict[str, float | None]:
    """Clearance min/mean/slope and near-conflict recovery from the distance series."""
    valid = [d for d in parsed.nearest_dists if d is not None]
    out: dict[str, float | None] = {}
    if not valid:
        return out
    out["clearance_min_m"] = float(min(valid))
    out["clearance_mean_m"] = float(np.mean(valid))
    out["clearance_slope_m_per_s"] = _slope(parsed.nearest_dist_times, valid)
    min_idx = min(range(len(valid)), key=lambda i: valid[i])
    if min_idx < len(valid) - 1:
        out["near_conflict_recovery_m"] = float(valid[-1] - valid[min_idx])
    return out


def _ttc_features(parsed: _ParsedWindow) -> dict[str, float | None]:
    """Time-to-contact proxy min/slope from the closing-distance series."""
    dists, times = parsed.nearest_dists, parsed.times
    values: list[float] = []
    value_times: list[float] = []
    for i in range(1, len(dists)):
        d_prev, d_cur = dists[i - 1], dists[i]
        if d_prev is None or d_cur is None:
            continue
        dt = times[i] - times[i - 1]
        if dt <= 0.0:
            continue
        closing = (d_prev - d_cur) / dt
        if closing > 1e-9:
            values.append(d_cur / closing)
            value_times.append(times[i])
    out: dict[str, float | None] = {}
    if values:
        out["ttc_proxy_min_s"] = float(min(values))
        out["ttc_proxy_slope_s_per_s"] = _slope(value_times, values)
    return out


def _robot_speed_features(parsed: _ParsedWindow) -> dict[str, float | None]:
    """Robot speed statistics plus an acceleration RMS proxy."""
    series = parsed.robot_speed_series
    out: dict[str, float | None] = {}
    if not series:
        return out
    speeds = [s for _, s in series]
    out["robot_speed_mean_m_s"] = float(np.mean(speeds))
    out["robot_speed_min_m_s"] = float(min(speeds))
    out["robot_speed_max_m_s"] = float(max(speeds))
    accel = [
        (series[i][1] - series[i - 1][1]) / (series[i][0] - series[i - 1][0])
        for i in range(1, len(series))
        if series[i][0] - series[i - 1][0] > 0.0
    ]
    out["robot_accel_rms_m_s2"] = _rms(accel)
    return out


def _command_features(parsed: _ParsedWindow) -> dict[str, float | None]:
    """Command-change RMS and an oscillation (steering sign-change) rate proxy."""
    out: dict[str, float | None] = {}
    if len(parsed.linear_cmds) >= 2:
        out["command_change_rms"] = _rms(
            [
                parsed.linear_cmds[i] - parsed.linear_cmds[i - 1]
                for i in range(1, len(parsed.linear_cmds))
            ]
        )
    # Prefer the angular command; fall back to heading deltas when commands are absent.
    signal = parsed.angular_cmds if len(parsed.angular_cmds) >= 2 else None
    if signal is None and len(parsed.headings) >= 3:
        signal = [
            parsed.headings[i] - parsed.headings[i - 1] for i in range(1, len(parsed.headings))
        ]
    if signal is not None and len(signal) >= 2:
        signs = [1 if v > 1e-6 else (-1 if v < -1e-6 else 0) for v in signal]
        nonzero = [s for s in signs if s != 0]
        changes = sum(1 for i in range(1, len(nonzero)) if nonzero[i] != nonzero[i - 1])
        out["oscillation_rate"] = float(changes / max(len(signal) - 1, 1))
    return out


def _interaction_features(parsed: _ParsedWindow) -> dict[str, float | None]:
    """Relative-speed, pedestrian-speed-change, and stop/yield proxies."""
    out: dict[str, float | None] = {}
    if parsed.rel_speeds:
        out["rel_speed_to_nearest_mean_m_s"] = float(np.mean(parsed.rel_speeds))
    if len(parsed.ped_speed_series) >= 2:
        changes = [
            abs(parsed.ped_speed_series[i][1] - parsed.ped_speed_series[i - 1][1])
            for i in range(1, len(parsed.ped_speed_series))
        ]
        out["ped_speed_change_near_robot_m_s"] = float(np.mean(changes))
    # Stop/yield proxy: fraction of steps (with a known robot speed) that are slow
    # while a pedestrian is near. Denominator is steps with a measured robot speed.
    measured = [(speed, dist) for speed, dist in parsed.step_speed_dist if speed is not None]
    if measured:
        near = sum(
            1
            for speed, dist in measured
            if speed < STOP_SPEED_THRESHOLD_M_S
            and dist is not None
            and dist < NEAR_PEDESTRIAN_THRESHOLD_M
        )
        out["stop_yield_fraction"] = float(near / len(measured))
    return out


def compute_window_features(
    steps: list[dict[str, Any]],
) -> tuple[dict[str, float | None], list[str]]:
    """Compute the feature vector for one window of trace steps.

    Returns a ``(features, missing_features)`` tuple where ``features`` maps every
    name in :data:`FEATURE_NAMES` to a float or ``None``. Non-derivable features are
    ``None`` and listed in ``missing_features``; genuine zero measurements stay
    ``0.0`` and are not treated as missing. ``route_progress_delta_m`` is always
    ``None`` because route progress is not present in the step trace.
    """
    parsed = _ParsedWindow(steps)
    features: dict[str, float | None] = dict.fromkeys(FEATURE_NAMES)
    for helper in (
        _clearance_features,
        _ttc_features,
        _robot_speed_features,
        _command_features,
        _interaction_features,
    ):
        features.update(helper(parsed))
    missing = [name for name in FEATURE_NAMES if features[name] is None]
    return features, missing


def _resolve_planner_key(row: dict[str, Any]) -> str:
    """Resolve a planner identifier from the common episode-row fields."""
    if row.get("planner_key"):
        return str(row["planner_key"])
    params = row.get("scenario_params")
    if isinstance(params, dict) and params.get("algo"):
        return str(params["algo"])
    algo_meta = row.get("algorithm_metadata")
    if isinstance(algo_meta, dict) and algo_meta.get("algorithm"):
        return str(algo_meta["algorithm"])
    return "unknown"


def _trace_steps(row: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Return the simulation step-trace list for a row, or ``None`` when absent."""
    algo_meta = row.get("algorithm_metadata")
    if not isinstance(algo_meta, dict):
        return None
    trace = algo_meta.get("simulation_step_trace")
    if not isinstance(trace, dict):
        return None
    steps = trace.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    return [step for step in steps if isinstance(step, dict)]


def iter_episode_rows(paths: list[str]) -> Any:
    """Yield ``(source_path, row_index, row)`` for every JSONL row in ``paths``.

    ``row`` is ``None`` for malformed (non-dict / unparseable) lines so the caller can
    count them as skips.
    """
    for path in paths:
        try:
            handle = open(path, encoding="utf-8")
        except OSError:
            continue
        with handle:
            for row_index, raw_line in enumerate(handle):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    yield path, row_index, None
                    continue
                yield path, row_index, row if isinstance(row, dict) else None


def extract_windows_from_row(
    source_path: str,
    row_index: int,
    row: dict[str, Any],
    *,
    window_steps: int,
    stride_steps: int,
) -> tuple[list[WindowRecord], str | None]:
    """Extract window records from one episode row.

    Returns ``(records, skip_reason)``. ``skip_reason`` is a short string when no
    windows could be produced (for example ``"no_simulation_step_trace"``), else
    ``None``.
    """
    steps = _trace_steps(row)
    if steps is None:
        return [], "no_simulation_step_trace"
    if len(steps) < window_steps:
        return [], f"trace_too_short(<{window_steps}_steps)"

    scenario_id = str(row.get("scenario_id") or "unknown")
    planner_key = _resolve_planner_key(row)
    seed = row.get("seed")
    episode_id = str(row["episode_id"]) if row.get("episode_id") is not None else None
    row_status = row.get("status") or row.get("row_status")
    outcome = row.get("outcome") or row.get("termination_reason") or row_status
    source_stem = Path(source_path).stem

    records: list[WindowRecord] = []
    stride = max(1, stride_steps)
    for window_index, start in enumerate(range(0, len(steps) - window_steps + 1, stride)):
        window = steps[start : start + window_steps]
        features, missing = compute_window_features(window)
        window_id = (
            f"{source_stem}#r{row_index}#w{window_index}#s{int(window[0].get('step', start))}"
        )
        records.append(
            WindowRecord(
                window_id=window_id,
                source_episode_path=source_path,
                episode_id=episode_id,
                scenario_id=scenario_id,
                planner_key=planner_key,
                seed=seed,
                t_start_s=_finite(window[0].get("time_s")),
                t_end_s=_finite(window[-1].get("time_s")),
                step_start=int(window[0].get("step", start)),
                step_end=int(window[-1].get("step", start + window_steps - 1)),
                n_steps=len(window),
                row_status=str(row_status) if row_status is not None else None,
                outcome=str(outcome) if outcome is not None else None,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                features=features,
                missing_features=missing,
            )
        )
    if not records:
        return [], "no_windows_after_striding"
    return records, None


def _expand_patterns(patterns: list[str]) -> list[str]:
    """Expand glob patterns (recursive ``**`` supported) into a sorted file list."""
    resolved: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        for match in sorted(matches):
            if match not in seen and Path(match).is_file():
                seen.add(match)
                resolved.append(match)
    return resolved


def _write_jsonl(path: Path, records: list[WindowRecord]) -> None:
    """Write window records as one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json_dict(), sort_keys=True))
            handle.write("\n")


def _write_csv(path: Path, records: list[WindowRecord]) -> None:
    """Write a flattened CSV with one row per window (features as columns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_cols = [
        "window_id",
        "source_episode_path",
        "episode_id",
        "scenario_id",
        "planner_key",
        "seed",
        "t_start_s",
        "t_end_s",
        "step_start",
        "step_end",
        "n_steps",
        "row_status",
        "outcome",
        "feature_schema_version",
    ]
    columns = meta_cols + list(FEATURE_NAMES) + ["missing_features"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for record in records:
            payload = record.to_json_dict()
            meta = [payload[col] for col in meta_cols]
            feats = [payload["features"][name] for name in FEATURE_NAMES]
            writer.writerow(meta + feats + [";".join(payload["missing_features"])])


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser for window extraction."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract interaction-trace windows into diagnostic feature vectors from saved "
            "episode JSONL rows. Offline and read-only; never runs a new campaign."
        ),
        epilog=CLAIM_BOUNDARY,
    )
    parser.add_argument(
        "--episode-jsonl",
        action="append",
        default=None,
        metavar="GLOB",
        required=True,
        help="Episode JSONL path or glob (recursive ** supported). Repeat to add sources.",
    )
    parser.add_argument(
        "--window-steps",
        type=int,
        default=10,
        help="Number of trace steps per window (default: 10).",
    )
    parser.add_argument(
        "--stride-steps",
        type=int,
        default=5,
        help="Step stride between consecutive windows (default: 5).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on the number of episode rows processed.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("output/experiments/behavior_tokens/windows.jsonl"),
        help="Output JSONL path for window records.",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=None, help="Optional flattened CSV output path."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_arg_parser().parse_args(argv)
    if args.window_steps <= 0:
        print("error: --window-steps must be positive", file=sys.stderr)
        return 2

    paths = _expand_patterns(args.episode_jsonl)
    if not paths:
        print(
            "error: no episode JSONL files matched the given --episode-jsonl patterns "
            "(offline mode never generates traces)",
            file=sys.stderr,
        )
        return 1

    records: list[WindowRecord] = []
    skip_reasons: Counter[str] = Counter()
    processed = 0
    for source_path, row_index, row in iter_episode_rows(paths):
        if args.max_episodes is not None and processed >= args.max_episodes:
            break
        processed += 1
        if row is None:
            skip_reasons["malformed_json_row"] += 1
            continue
        row_records, reason = extract_windows_from_row(
            source_path,
            row_index,
            row,
            window_steps=args.window_steps,
            stride_steps=args.stride_steps,
        )
        if reason is not None:
            skip_reasons[reason] += 1
            continue
        records.extend(row_records)

    _write_jsonl(args.output_jsonl, records)
    if args.output_csv is not None:
        _write_csv(args.output_csv, records)

    summary = {
        "files_scanned": len(paths),
        "rows_processed": processed,
        "windows_written": len(records),
        "skipped_rows": dict(sorted(skip_reasons.items())),
        "output_jsonl": str(args.output_jsonl),
        "output_csv": str(args.output_csv) if args.output_csv else None,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    print(json.dumps(summary, indent=2))
    if not records:
        print(
            "warning: no windows extracted; check that traces include "
            "algorithm_metadata.simulation_step_trace.steps",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
