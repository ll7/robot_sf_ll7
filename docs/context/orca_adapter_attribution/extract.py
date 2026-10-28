"""Project raw episode JSONL into compact per-episode and per-step tables.

    python extract.py --raw-dir <run_dir> --stage main --out-dir <evidence_dir>

Writes ``episodes.csv`` (one row per episode, with a ``mode`` column) and
``steps.csv.gz`` (one row per step: preferred velocity, raw solver velocity,
occupancy penalty, adapter command, feasibility-projected command, environment
action, pre/post pose and measured post-step speed).

Stall-step classes (predeclared in contract.yaml, thresholds there):
  pref_zero        ORCA preferred velocity below eps
  solver_zero      raw solver/world request below eps
  projection_zero  raw request >= eps but the executed command's translation < eps
  plant_zero       executed translation >= eps but measured speed < eps
  moving           measured speed >= eps (not a stall step)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path

import yaml

EPS = 0.05  # m/s; overwritten from contract thresholds when available


def _norm(v):
    return math.hypot(v[0], v[1]) if v else None


def _cmd_translation(sel: dict | None) -> float | None:
    if not isinstance(sel, dict):
        return None
    if "vx" in sel:
        return math.hypot(float(sel["vx"]), float(sel["vy"]))
    if "linear_velocity" in sel:
        return abs(float(sel["linear_velocity"]))
    return None


def _cmd_pair(sel: dict | None) -> tuple[float | None, float | None]:
    if not isinstance(sel, dict):
        return None, None
    if "vx" in sel:
        return float(sel["vx"]), float(sel["vy"])
    if "linear_velocity" in sel:
        return float(sel["linear_velocity"]), float(sel["angular_velocity"])
    return None, None


def classify(step: dict, *, uses_solver: bool, eps: float) -> str:
    """Return the stall class of one step (see module docstring).

    Returns:
        Stall-class label.
    """
    x = step.get("xorca") or {}
    pre, post = x.get("pre"), x.get("post")
    dt = step.get("_dt", 0.1)
    meas = math.hypot(post["x"] - pre["x"], post["y"] - pre["y"]) / dt if pre and post else None
    if meas is not None and meas >= eps:
        return "moving"
    if uses_solver:
        pref = _norm(x.get("pref_final_world") or x.get("pref_world"))
        if pref is not None and pref < eps:
            return "pref_zero"
        solver = _norm(x.get("solver_world"))
        if solver is not None and solver < eps:
            return "solver_zero"
    cmd = _cmd_translation(step["planner"].get("selected_action"))
    if uses_solver and cmd is not None and cmd < eps:
        return "projection_zero"
    return "plant_zero" if cmd is not None and cmd >= eps else "command_zero"


def main() -> int:  # noqa: PLR0915
    """Write the compact episode and step tables.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    contract = yaml.safe_load(args.contract.read_text(encoding="utf-8"))
    eps = float(contract.get("thresholds", {}).get("eps_mps", EPS))
    window = int(contract.get("thresholds", {}).get("terminal_window_steps", 100))
    wall_x_far = float(contract.get("thresholds", {}).get("wall_far_x_m", 16.0))
    configs = {c["key"]: c for c in contract["configurations"]}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ep_rows = []
    step_path = args.out_dir / "steps.csv.gz"
    step_fields = [
        "config",
        "cell",
        "seed",
        "step",
        "x",
        "y",
        "heading",
        "goal_x",
        "goal_y",
        "pref_vx",
        "pref_vy",
        "pref_final_vx",
        "pref_final_vy",
        "door_gap_m",
        "n_obstacles",
        "solver_vx",
        "solver_vy",
        "occ_penalty",
        "adapter_v",
        "adapter_w",
        "sel_a",
        "sel_b",
        "env_a",
        "env_b",
        "post_x",
        "post_y",
        "post_heading",
        "measured_speed",
        "min_ped_dist",
        "stall_class",
    ]
    with gzip.open(step_path, "wt", encoding="utf-8", newline="") as sh:
        sw = csv.DictWriter(sh, fieldnames=step_fields)
        sw.writeheader()
        for path in sorted(args.raw_dir.glob(f"{args.stage}__*.jsonl")):
            config = path.name.split("__")[1]
            uses_solver = configs[config]["algo"] in {"orca", "social_force"}
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                trace = rec["algorithm_metadata"].get("simulation_step_trace") or {}
                steps = trace.get("steps") or []
                dt = float(trace.get("dt", 0.1))
                cell = rec["scenario_id"]
                seed = int(rec["seed"])
                classes = []
                missing = 0
                fallback = 0
                max_x = -1e9
                for st in steps:
                    st["_dt"] = dt
                    x = st.get("xorca")
                    if not x or not x.get("pre") or not x.get("post"):
                        missing += 1
                        classes.append("missing")
                        continue
                    fallback += bool(x.get("heuristic_fallback"))
                    cls = classify(st, uses_solver=uses_solver, eps=eps)
                    classes.append(cls)
                    pre, post = x["pre"], x["post"]
                    max_x = max(max_x, post["x"])
                    sa, sb = _cmd_pair(st["planner"].get("selected_action"))
                    ea, eb = _cmd_pair(st["planner"].get("applied_environment_action"))
                    goal = (st.get("goal") or {}).get("current") or [None, None]
                    peds = st.get("pedestrians") or []
                    dists = [
                        math.hypot(p["position"][0] - post["x"], p["position"][1] - post["y"])
                        for p in peds
                        if isinstance(p, dict) and p.get("position")
                    ]
                    pref = x.get("pref_world") or [None, None]
                    solv = x.get("solver_world") or [None, None]
                    pfin = x.get("pref_final_world") or [None, None]
                    obst = x.get("obstacles") or {}
                    adc = x.get("adapter_cmd") or [None, None]
                    sh_ = x.get("safe_heading") or {}
                    sw.writerow(
                        {
                            "config": config,
                            "cell": cell,
                            "seed": seed,
                            "step": st["step"],
                            "x": pre["x"],
                            "y": pre["y"],
                            "heading": pre["heading"],
                            "goal_x": goal[0],
                            "goal_y": goal[1],
                            "pref_vx": pref[0],
                            "pref_vy": pref[1],
                            "pref_final_vx": pfin[0],
                            "pref_final_vy": pfin[1],
                            "door_gap_m": obst.get("door_gap_m"),
                            "n_obstacles": obst.get("n_obstacles"),
                            "solver_vx": solv[0],
                            "solver_vy": solv[1],
                            "occ_penalty": sh_.get("penalty"),
                            "adapter_v": adc[0],
                            "adapter_w": adc[1],
                            "sel_a": sa,
                            "sel_b": sb,
                            "env_a": ea,
                            "env_b": eb,
                            "post_x": post["x"],
                            "post_y": post["y"],
                            "post_heading": post["heading"],
                            "measured_speed": math.hypot(post["x"] - pre["x"], post["y"] - pre["y"])
                            / dt,
                            "min_ped_dist": min(dists) if dists else None,
                            "stall_class": cls,
                        }
                    )
                tail = list(classes[-window:])
                counts = {
                    k: tail.count(k)
                    for k in (
                        "moving",
                        "pref_zero",
                        "solver_zero",
                        "projection_zero",
                        "plant_zero",
                        "command_zero",
                        "missing",
                    )
                }
                stall_tail = [c for c in tail if c not in ("moving", "missing")]
                dominant = max(set(stall_tail), key=stall_tail.count) if stall_tail else "none"
                all_counts = {
                    k: classes.count(k) for k in ("solver_zero", "projection_zero", "plant_zero")
                }
                out = rec.get("outcome") or {}
                status = str(rec.get("status"))
                mode = "native"
                if fallback:
                    mode = "fallback"
                if missing or status not in {"success", "failure", "collision"}:
                    mode = "degraded" if mode == "native" else mode
                ep_rows.append(
                    {
                        "config": config,
                        "algo": configs[config]["algo"],
                        "kinematics": configs[config]["kinematics"],
                        "cell": cell,
                        "seed": seed,
                        "mode": mode,
                        "fallback_steps": fallback,
                        "missing_trace_steps": missing,
                        "status": status,
                        "termination_reason": rec.get("termination_reason"),
                        "success": int(bool(out.get("route_complete"))),
                        "collision": int(bool(out.get("collision_event"))),
                        "timeout": int(bool(out.get("timeout_event"))),
                        "deadlock_metric": int(bool((rec.get("metrics") or {}).get("deadlock"))),
                        "steps": len(steps),
                        "max_x": max_x,
                        "crossed_wall": int(max_x > wall_x_far),
                        "terminal_x": steps[-1]["xorca"]["post"]["x"]
                        if steps and steps[-1].get("xorca")
                        else None,
                        "min_ped_dist": (rec.get("metrics") or {}).get("min_distance"),
                        "init_x": steps[0]["xorca"]["pre"]["x"]
                        if steps and steps[0].get("xorca")
                        else None,
                        "init_y": steps[0]["xorca"]["pre"]["y"]
                        if steps and steps[0].get("xorca")
                        else None,
                        "init_heading": steps[0]["xorca"]["pre"]["heading"]
                        if steps and steps[0].get("xorca")
                        else None,
                        "tail_steps": len(tail),
                        **{f"tail_{k}": v for k, v in counts.items()},
                        "tail_dominant_stall": dominant,
                        **{f"all_{k}": v for k, v in all_counts.items()},
                        "git_hash": rec.get("git_hash"),
                        "source_file": path.name,
                    }
                )
    with open(args.out_dir / "episodes.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ep_rows[0]))
        w.writeheader()
        w.writerows(sorted(ep_rows, key=lambda r: (r["config"], r["cell"], r["seed"])))
    print(f"{len(ep_rows)} episodes -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
