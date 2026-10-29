"""Export the accepted seed's step traces to compact per-arm CSVs plus a metadata JSON.

Recorded columns come straight from algorithm_metadata.simulation_step_trace.steps[].
obstacle_envelope_distance_m_derived is NOT recorded by the runner: it is computed here
as min distance from the traced robot centre to the map's obstacle and boundary line
segments (MapDefinition.obstacles_pysf) minus the robot collision radius (1.0 m), i.e.
the quantity whose sign the occupancy check circle_collides_any_lines tests.
usage: export_trace.py <trace-run-root> <seed> <out-dir> <map-svg>
"""
import csv, gzip, hashlib, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import exsel_screen as ex
from robot_sf.nav.svg_map_parser import convert_map

root, seed, out, svg = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
ROBOT_RADIUS_M = 1.0
out.mkdir(parents=True, exist_ok=True)
md = convert_map(str(svg))
assert md is not None, svg
def seg(line):
    a = list(line)
    if len(a) == 2:  # ((x1,y1),(x2,y2))
        return a[0][0], a[0][1], a[1][0], a[1][1]
    x1, x2, y1, y2 = a  # pysf flat (x1, x2, y1, y2)
    return x1, y1, x2, y2
segs = [seg(l) for l in md.obstacles_pysf]
def dist(px, py):
    best = math.inf
    for x1, y1, x2, y2 in segs:
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
        best = min(best, math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))
    return best
COLS = ["step", "time_s", "robot_x", "robot_y", "robot_heading", "robot_vx", "robot_vy", "robot_speed",
        "selected_linear", "selected_angular", "applied_linear", "applied_angular",
        "collision_obstacle", "collision_pedestrian", "terminated", "truncated",
        "min_ped_surface_clearance_m", "obstacle_envelope_distance_m_derived"]
fmt = lambda v: "" if v is None else (repr(round(v, 6)) if isinstance(v, float) else str(v))
meta = {"seed": seed, "cell": ex.CELL, "map_file": svg.as_posix().split("/wt/")[-1],
        "map_sha256": hashlib.sha256(svg.read_bytes()).hexdigest(), "robot_radius_m": ROBOT_RADIUS_M,
        "n_obstacle_and_bound_segments": len(segs), "arms": {}}
for arm in ex.ARMS:
    line = ex.find_episodes(root, arm).read_bytes().splitlines()[0]
    raw = json.loads(line)
    tr = raw["algorithm_metadata"]["simulation_step_trace"]
    rows = []
    env_d: list[float] = []
    for s in tr["steps"]:
        r, p = s["robot"], s.get("planner") or {}
        sel, app = p.get("selected_action") or {}, p.get("applied_environment_action") or {}
        peds = [q["surface_clearance_m"] for q in s.get("pedestrians", []) if q.get("surface_clearance_m") is not None]
        x, y = r["position"]
        env_d.append(dist(x, y) - ROBOT_RADIUS_M)
        rows.append([s["step"], s["time_s"], x, y, r["heading"], r["velocity"][0], r["velocity"][1],
                     math.hypot(*r["velocity"]), sel.get("linear_velocity"), sel.get("angular_velocity"),
                     app.get("linear_velocity"), app.get("angular_velocity"),
                     s["collision"]["obstacle"], s["collision"]["pedestrian"],
                     s.get("rl", {}).get("terminated"), s.get("rl", {}).get("truncated"),
                     min(peds) if peds else None, env_d[-1]])
    p = out / f"trace_{arm}.csv"
    with p.open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n"); w.writerow(COLS)
        for row in rows:
            w.writerow([fmt(v) for v in row])
    last = tr["steps"][-1]
    meta["arms"][arm] = {
        "trace_csv": p.name, "n_steps": len(rows), "dt": tr["dt"], "trace_schema": tr["schema_version"],
        "row_steps": raw["steps"], "termination_reason": raw["termination_reason"],
        "terminal_step": {"step": last["step"], "time_s": last["time_s"], "collision": last["collision"],
                          "rl": last.get("rl")},
        "first_obstacle_collision_step": next((r[0] for r in rows if r[12]), None),
        "min_obstacle_envelope_distance_m_derived": round(min(env_d), 6),
        "min_ped_surface_clearance_m_trace": round(min(r[16] for r in rows if r[16] is not None), 6),
        "row_min_surface_clearance_m": round(raw["metrics"]["min_distance"] - 1.4, 6),
        "raw_row_sha256": hashlib.sha256(line).hexdigest(),
        "steps_compared": len(rows),
        "derived_sign_vs_recorded_obstacle_flag_disagreements": sum((d < 0) != bool(r[12]) for d, r in zip(env_d, rows)),
        "guard_rejected_commands": "not recorded (record_planner_decision_trace stayed false)",
    }
(out / "trace_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
print(json.dumps(meta, indent=2, sort_keys=True))
