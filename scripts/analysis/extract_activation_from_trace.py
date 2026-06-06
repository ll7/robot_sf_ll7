#!/usr/bin/env python3
"""Simple trace activation extractor for issue #2306.
Reads a JSONL trace file and emits required activation fields as YAML-ish JSON.
This is intentionally small and defensive for local analysis runs.

Usage: python scripts/analysis/extract_activation_from_trace.py <trace.jsonl> [--row-id ID]
"""

import json
import sys
from pathlib import Path


def _command_source(record: dict) -> str | None:
    if "command_source" in record:
        return record.get("command_source")
    selected_command = record.get("selected_command")
    if isinstance(selected_command, dict):
        return selected_command.get("source")
    return None


def _command_source_changed(record: dict) -> bool | None:
    sources = [
        event.get("source")
        for event in record.get("command_history", [])
        if isinstance(event, dict)
    ]
    return (len(set(sources)) > 1) if sources else None


def _progress_delta_after_activation(record: dict, first_step: int | None) -> float | None:
    progress = record.get("progress")
    if not isinstance(progress, list) or first_step is None:
        return None
    after = progress[first_step : first_step + 5]
    return (after[-1] - after[0]) if len(after) >= 2 else None


def _mean_position(points: list) -> tuple[float, float] | None:
    valid_points = [
        point for point in points if isinstance(point, list | tuple) and len(point) >= 2
    ]
    if not valid_points:
        return None
    x_mean = sum(point[0] for point in valid_points) / len(valid_points)
    y_mean = sum(point[1] for point in valid_points) / len(valid_points)
    return (x_mean, y_mean)


def _trajectory_delta(record: dict, first_step: int | None) -> float | None:
    trajectory = record.get("trajectory")
    if not isinstance(trajectory, list) or first_step is None:
        return None
    pre_mean = _mean_position(trajectory[max(0, first_step - 3) : first_step])
    post_mean = _mean_position(trajectory[first_step : first_step + 3])
    if pre_mean is None or post_mean is None:
        return None
    dx = post_mean[0] - pre_mean[0]
    dy = post_mean[1] - pre_mean[1]
    return (dx * dx + dy * dy) ** 0.5


def extract_activation(record: dict) -> dict:
    """Extract the activation review fields from one compact trace row."""
    out = {
        "activation_count": None,
        "first_activation_step": None,
        "selected_command_source": None,
        "command_source_changed": None,
        "progress_delta_after_activation": None,
        "trajectory_delta": None,
        "terminal_outcome_changed": None,
    }
    # Example heuristics: look for 'activations' list or 'events' with type 'activation'
    acts = record.get("activations") or []
    if not isinstance(acts, list):
        acts = []
    out["activation_count"] = len(acts) if acts else 0
    if acts:
        out["first_activation_step"] = acts[0].get("step") if isinstance(acts[0], dict) else acts[0]
    out["selected_command_source"] = _command_source(record)
    out["command_source_changed"] = _command_source_changed(record)
    try:
        first_step = (
            int(out["first_activation_step"]) if out["first_activation_step"] is not None else None
        )
    except (TypeError, ValueError):
        first_step = None
    out["progress_delta_after_activation"] = _progress_delta_after_activation(record, first_step)
    out["trajectory_delta"] = _trajectory_delta(record, first_step)
    if "terminal_outcome" in record:
        out["terminal_outcome_changed"] = record.get("terminal_outcome")
    return out


def main() -> None:
    """Run the activation extractor CLI."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Trace file not found: {path}")
        sys.exit(2)
    row_id_arg = None
    if len(sys.argv) >= 3:
        row_id_arg = sys.argv[2]
    results = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row_id_arg and str(rec.get("row_id", i)) != row_id_arg:
                continue
            out = extract_activation(rec)
            out["_row_index"] = i
            results.append(out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
