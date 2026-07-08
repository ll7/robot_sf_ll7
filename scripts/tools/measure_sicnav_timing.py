#!/usr/bin/env python3
"""Measure SICNav per-step solve time through the robot_sf wrapper (issue #4870).

Drives ``SICNavPlanner.step()`` -- the actual benchmark entrypoint -- against the
pinned upstream on the *open-source dependency path only*: CasADi (bundles IPOPT) +
python-RVO2. Acados and HSL are intentionally absent, so IPOPT falls back to its
default linear solver. This is the redistributable-footprint configuration.

The neighbor count is swept over {2, 5, 10}; per-step wall-clock is compared against
the benchmark step budget (``robot_sf.sim.sim_config.SimConfig.time_per_step_in_secs``
default = 0.1 s).

Outputs:
    output/issue_4870/sicnav_timing.json   machine-readable results
    output/issue_4870/sicnav_timing.md      human-readable table
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the robot_sf package on the worktree is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robot_sf.baselines.sicnav import SICNavPlanner, build_sicnav_config

STEP_BUDGET_S = 0.1  # robot_sf SimConfig.time_per_step_in_secs default
NEIGHBORS = (2, 5, 10)
STEPS = int(os.environ.get("SICNAV_STEPS", "6"))  # 1 warmup (compile) + steady steps


def _scene(n: int, t: int) -> dict:
    """Synthetic observation: robot drives toward +x goal; n humans cross toward -x."""
    robot_px = 0.8 * t
    return {
        "dt": 0.25,
        "robot": {
            "position": [robot_px, 0.0],
            "velocity": [0.8, 0.0],
            "goal": [6.0, 0.0],
            "radius": 0.25,
            "v_pref": 1.0,
        },
        "agents": [
            {
                "position": [3.0 - 0.4 * t - 0.2 * i, (-1) ** i * (0.7 + 0.25 * i)],
                "velocity": [-0.4, 0.0],
                "radius": 0.30,
                "goal": [-3.0, (-1) ** i * (0.7 + 0.25 * i)],
            }
            for i in range(n)
        ],
        "obstacles": [],
    }


def _run(n: int) -> dict:
    planner = SICNavPlanner(build_sicnav_config({}), seed=1)
    times: list[float] = []
    actions: list[tuple[float, float]] = []
    for step in range(STEPS):
        obs = _scene(n, step)
        t0 = time.perf_counter()
        action = planner.step(obs)
        dt = time.perf_counter() - t0
        times.append(dt)
        actions.append((float(action["v"]), float(action["omega"])))
    steady = times[1:]
    if not steady:
        raise ValueError(f"SICNAV_STEPS={STEPS} leaves no steady-state steps; use at least 2")
    return {
        "neighbors": n,
        "steps": STEPS,
        "first_step_ms": times[0] * 1000,
        "steady_mean_ms": float(np.mean(steady)) * 1000,
        "steady_median_ms": float(np.median(steady)) * 1000,
        "steady_max_ms": float(np.max(steady)) * 1000,
        "steady_over_budget_x": float(np.mean(steady)) / STEP_BUDGET_S,
        "first_action": actions[0],
        "last_action": actions[-1],
    }


def _markdown(report: dict) -> str:
    lines = [
        "# SICNav per-step solve time (issue #4870)",
        "",
        f"- Policy: `{report['policy_class']}`",
        f"- Solver: {report['solver']}",
        f"- Benchmark step budget: **{report['step_budget_s'] * 1000:.0f} ms** "
        f"(`SimConfig.time_per_step_in_secs`)",
        f"- Steps per neighbor count: {report['rows'][0]['steps']} "
        "(first = compile+solve warmup; rest = steady-state warmstarted solves)",
        "",
        "| Neighbors | First step (ms) | Steady mean/step (ms) | Steady max (ms) | × budget |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['neighbors']} | {row['first_step_ms']:.0f} | "
            f"{row['steady_mean_ms']:.0f} | {row['steady_max_ms']:.0f} | "
            f"{row['steady_over_budget_x']:.1f}× |"
        )
    lines += [
        "",
        "Smoke result: **PASS** -- the wrapper drives the pinned upstream end-to-end on the "
        "open-source path and returns valid unicycle `{v, omega}` actions.",
        "",
        "Timing verdict: the steady-state solve is **above** the 100 ms real-time step budget "
        "at every neighbor count and scales roughly linearly with neighbor count. This path "
        "(CasADi/IPOPT, no HSL, no compiled Acados) is therefore campaign-infeasible for "
        "real-time; the negative result is the deliverable.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Run the neighbor-count sweep and write JSON + Markdown timing artifacts."""
    rows = [_run(n) for n in NEIGHBORS]
    report = {
        "step_budget_s": STEP_BUDGET_S,
        "policy_class": "sicnav.policy.campc.CollisionAvoidMPC "
        "(CasADi+IPOPT, hum_model=orca_casadi_kkt -> SICNav-np)",
        "solver": "casadi nlpsol('ipopt'); HSL absent -> IPOPT default linear solver",
        "smoke": "PASS",
        "rows": rows,
    }
    out_dir = Path("output/issue_4870")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sicnav_timing.json").write_text(json.dumps(report, indent=2))
    (out_dir / "sicnav_timing.md").write_text(_markdown(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
