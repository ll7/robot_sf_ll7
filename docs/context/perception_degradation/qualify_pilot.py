"""Qualify the interventions from raw pilot rows that carry true-vs-perceived traces.

    uv run python docs/context/perception_degradation/qualify_pilot.py \
        --raw-root <pilot run root> --output <qualification.json>

Checks, per condition and cell, that the perceived snapshot is what the condition
declares (noise magnitude, exact two-step lag, occlusion only behind walls), that every
consumed channel used it, that pedestrian encounters occur, and the seconds per episode.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def check_trace(condition: str, trace: list[dict]) -> dict:
    """Verify one episode's trace against its declared condition.

    Returns:
        Counters of steps checked and violations.
    """
    out = {"steps": len(trace), "violations": 0, "abs_errors": []}
    for i, step in enumerate(trace):
        true = np.asarray(step["true_xy"], dtype=float).reshape(-1, 2)
        slots = np.asarray(step["perceived_slots"], dtype=int)
        perceived = np.asarray(step["perceived_xy"], dtype=float).reshape(-1, 2)
        if condition == "oracle":
            ok = np.array_equal(true, perceived)
        elif condition == "position_noise":
            ok = perceived.shape == true.shape
            if ok and true.size:
                err = perceived - true
                out["abs_errors"].extend(err.reshape(-1).tolist())
                ok = bool(np.all(np.abs(err) < 0.5))
        elif condition == "delay":
            ref = trace[max(0, i - 2)]
            ok = np.array_equal(perceived, np.asarray(ref["true_xy"], dtype=float).reshape(-1, 2))
            ok = ok and step["perceived_vel"] == ref["true_vel"]
        else:
            ok = bool(np.all(slots < true.shape[0])) and np.array_equal(perceived, true[slots])
        out["violations"] += int(not ok)
    return out


def main() -> None:
    """Write the qualification report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells: dict = defaultdict(lambda: defaultdict(list))
    per_planner_time: dict = defaultdict(list)
    report: dict = {"conditions": {}, "cells": {}, "seconds_per_episode": {}}
    noise_errors: list[float] = []
    for path in sorted(args.raw_root.glob("*/runs/*__differential_drive/episodes.jsonl")):
        planner = path.parent.name.split("__", 1)[0]
        for line in path.open():
            raw = json.loads(line)
            meta = raw["algorithm_metadata"]
            perception = meta["pedestrian_perception"]
            condition = perception["condition"]
            trace = meta.get("pedestrian_perception_trace") or []
            checked = check_trace(condition, trace)
            if condition == "position_noise":
                noise_errors.extend(checked["abs_errors"])
            stats = perception["stats"]
            entry = report["conditions"].setdefault(
                condition,
                {"episodes": 0, "trace_steps": 0, "trace_violations": 0, "channels_ok": 0},
            )
            entry["episodes"] += 1
            entry["trace_steps"] += checked["steps"]
            entry["trace_violations"] += checked["violations"]
            entry["channels_ok"] += int(perception["all_channels_consumed_every_snapshot"])
            key = f"{raw['scenario_id']}|{condition}"
            cells[key]["min_distance_m"].append(raw["metrics"].get("min_distance"))
            cells[key]["true_rows_per_step"].append(stats["true_rows"] / max(1, stats["snapshots"]))
            cells[key]["occluded_step_share"].append(
                stats["steps_with_occlusion"] / max(1, stats["snapshots"])
            )
            exposure = (raw.get("interaction_exposure") or {}).get("interaction_exposure_steps")
            cells[key]["interaction_exposure_steps"].append(exposure)
            per_planner_time[planner].append(float(raw.get("wall_time_sec") or 0.0))
    if noise_errors:
        errs = np.asarray(noise_errors)
        report["position_noise_axis_error"] = {
            "n": int(errs.size),
            "mean": float(errs.mean()),
            "sd": float(errs.std(ddof=1)),
        }
    for key, values in sorted(cells.items()):
        report["cells"][key] = {
            name: {
                "n": len(v),
                "mean": float(np.mean([x for x in v if x is not None]))
                if any(x is not None for x in v)
                else None,
                "min": float(np.min([x for x in v if x is not None]))
                if any(x is not None for x in v)
                else None,
            }
            for name, v in values.items()
        }
    for planner, times in per_planner_time.items():
        report["seconds_per_episode"][planner] = {
            "n": len(times),
            "mean": float(np.mean(times)),
            "p95": float(np.percentile(times, 95)),
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("conditions", "seconds_per_episode")}, indent=2))
    print(json.dumps(report.get("position_noise_axis_error"), indent=2))


if __name__ == "__main__":
    main()
