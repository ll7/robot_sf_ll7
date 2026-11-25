"""Exploratory (not preregistered) descriptive summaries of the stage-1 main rows.

    python explore.py --evidence evidence/ --raw-dir <run_dir> --output <dir>

``--raw-dir`` is optional; without it the collision-type split is skipped.
Outputs ``exploratory.json``.  Nothing here enters an outcome rule.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _f(value: str) -> float | None:
    return None if value in ("", "None", None) else float(value)


def main() -> int:  # noqa: C901
    """Write exploratory descriptive summaries.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default="steps.csv.gz", help="step table inside --evidence")
    args = parser.parse_args()
    eps = 0.05
    episodes = {
        (r["config"], r["cell"], int(r["seed"])): r
        for r in csv.DictReader(open(args.evidence / "episodes.csv", encoding="utf-8"))
    }
    steps = defaultdict(list)
    with gzip.open(args.evidence / args.steps, "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["config"] in ("orca_dd", "orca_holo"):
                steps[(r["config"], r["cell"], int(r["seed"]))].append(r)

    gap_terminal = defaultdict(list)
    gap_min_success = defaultdict(list)
    onset = defaultdict(list)
    solver_speed_before = defaultdict(list)
    for key, rows in steps.items():
        cfg, cell, _seed = key
        ep = episodes[key]
        gaps = [_f(r["door_gap_m"]) for r in rows]
        if ep["success"] == "0":
            tail = [g for g in gaps[-100:] if g is not None]
            if tail:
                gap_terminal[(cfg, cell)].append(float(np.median(tail)))
            # Onset of the final uninterrupted run of zero solver requests.
            speeds = [np.hypot(_f(r["solver_vx"]) or 0.0, _f(r["solver_vy"]) or 0.0) for r in rows]
            i = len(speeds)
            while i > 0 and speeds[i - 1] < eps:
                i -= 1
            if i < len(speeds):
                onset[(cfg, cell)].append({"step": int(rows[i]["step"]), "x": float(rows[i]["x"])})
                window = speeds[max(0, i - 50) : i]
                if window:
                    solver_speed_before[(cfg, cell)].append(float(np.median(window)))
        else:
            valid = [g for g in gaps if g is not None]
            if valid:
                gap_min_success[(cfg, cell)].append(min(valid))

    def stats(values: list[float]) -> dict | None:
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        return {
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    out: dict = {"schema": "xorca-exploratory.v1", "note": "exploratory; not preregistered"}
    out["terminal_window_door_gap_m_failed"] = {
        f"{k[0]}|{k[1]}": stats(v) for k, v in sorted(gap_terminal.items())
    }
    out["min_door_gap_m_successful"] = {
        f"{k[0]}|{k[1]}": stats(v) for k, v in sorted(gap_min_success.items())
    }
    out["final_zero_run_onset"] = {
        f"{k[0]}|{k[1]}": {
            "step": stats([o["step"] for o in v]),
            "robot_x_m": stats([o["x"] for o in v]),
        }
        for k, v in sorted(onset.items())
    }
    out["median_solver_speed_5s_before_onset_mps"] = {
        f"{k[0]}|{k[1]}": stats(v) for k, v in sorted(solver_speed_before.items())
    }
    if args.raw_dir:
        split = defaultdict(lambda: defaultdict(int))
        for path in sorted(args.raw_dir.glob("main__*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if not (rec.get("outcome") or {}).get("collision_event"):
                    continue
                m = rec.get("metrics") or {}
                key = f"{path.name.split('__')[1]}|{rec['scenario_id']}"
                split[key]["episodes"] += 1
                for field in ("ped_collision_count", "obstacle_collision_count"):
                    split[key][f"with_{field}"] += int((m.get(field) or 0) > 0)
        out["collision_type_split"] = {k: dict(v) for k, v in sorted(split.items())}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "exploratory.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
