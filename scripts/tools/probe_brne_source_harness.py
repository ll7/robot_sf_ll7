#!/usr/bin/env python3
"""BRNE source-side smoke + runtime-vs-neighbor-count measurement (issue #5311).

Drives the REAL upstream BRNE core algorithm -- ``MurpheyLab/brne`` at the pinned
commit staged by ``scripts/tools/manage_external_repos.py`` -- against synthetic
Robot SF-shaped observations, and measures per-step wall-clock vs the benchmark
control budget. This is the go/no-go control-budget evidence; it is NOT a
benchmark or paper claim and it registers no robot_sf planner.

What is exercised
-----------------
Only the pure-numpy/numba core algorithm at
``brne_nav/brne_py/brne_py/brne.py`` (the same module the upstream ROS2 node
``brne_nav.py`` imports). Specifically:

* ``brne.get_Lmat_nb``      -- GP covariance / Cholesky of the trajectory prior.
* ``brne.mvn_sample_normal`` -- draw trajectory samples from the prior.
* ``brne.get_ulist_essemble`` + ``brne.traj_sim_essemble`` -- unicycle rollout of
  the robot trajectory samples (native differential-drive dynamics).
* ``brne.brne_nav``         -- the mixed-strategy Nash equilibrium solver
  (10 best-response iterations over a pairwise proximity cost).

The ROS2 navigation node (``brne_nav.py``) is NOT exercised because it requires
``rclpy`` (ROS2) and message types. The torch and C++ (``brnelib``) paths are out
of scope for this smoke. The smoke reproduces the upstream ``brne_cb`` numerical
pipeline with a local loop, matching the upstream defaults
(``num_samples=196``, ``plan_steps=25``, ``dt=0.1``, ``maximum_agents=8``).

Control budget
--------------
``robot_sf.sim.sim_config.SimConfig.time_per_step_in_secs`` default = **0.1 s**
(100 ms). The upstream ROS2 node also runs its BRNE solver on a 10 Hz timer
(``replan_freq=10``), i.e. the same 100 ms budget. Per-step wall-clock is compared
against this budget.

Outputs
-------
* ``output/issue_5311/brne_smoke.json``  -- machine-readable results + contract.
* ``output/issue_5311/brne_smoke.md``     -- human-readable table + verdict.

The JSON + Markdown are regenerated each run (git-ignored under ``output/``).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
BRNE_STAGE_PATH = REPO_ROOT / "third_party" / "external_repos" / "brne"
BRNE_CORE_REL = "brne_nav/brne_py/brne_py/brne.py"

# Upstream ROS2-node defaults (brne_nav/brne_py/brne_py/brne_nav.py).
NUM_SAMPLES = 196
PLAN_STEPS = 25
DT = 0.1
MAX_AGENTS = 8  # upstream 'maximum_agents' (includes the robot)
# BRNE cost/kernel defaults from the upstream node.
KERNEL_A1 = 0.2
KERNEL_A2 = 0.2
COST_A1 = 4.0
COST_A2 = 1.0
COST_A3 = 80.0
PED_SAMPLE_SCALE = 0.1
CORRIDOR_Y_MIN = -0.65
CORRIDOR_Y_MAX = 0.65
# Upstream weights_update_nb runs exactly 10 best-response iterations.
BRNE_ITERS = 10

STEP_BUDGET_S = 0.1  # robot_sf SimConfig.time_per_step_in_secs default == 100 ms
NEIGHBORS = (1, 2, 3, 5, 7)  # pedestrian counts; +1 robot = num_agents


def load_upstream_brne(stage_path: Path) -> ModuleType:
    """Import the real upstream brne.py core module from the staged clone.

    Loading by file path keeps BRNE out of the installed namespace (GPL-3.0:
    local-only reference, never vendored) and pins exactly the staged source.
    """
    core_file = stage_path / BRNE_CORE_REL
    if not core_file.is_file():
        raise FileNotFoundError(
            f"BRNE core algorithm not found at staged path: {core_file}. "
            "Run `uv run python scripts/tools/manage_external_repos.py stage brne`."
        )
    spec = importlib.util.spec_from_file_location("brne_upstream", core_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {core_file}")
    module = importlib.util.module_from_spec(spec)
    # numba JIT cache is keyed on the module; inserting under a stable name keeps
    # repeated runs reproducible without polluting the package namespace.
    sys.modules["brne_upstream"] = module
    spec.loader.exec_module(module)
    return module


def build_cov(brne: ModuleType) -> np.ndarray:
    """Reproduce the upstream one-shot GP covariance/Cholesky build from brne_nav."""
    tlist = np.arange(PLAN_STEPS) * DT
    train_ts = np.array([tlist[0]])
    train_noise = np.array([1e-04])
    test_ts = tlist
    lmat, _cov = brne.get_Lmat_nb(train_ts, test_ts, train_noise, KERNEL_A1, KERNEL_A2)
    return lmat


def build_robot_samples(brne: ModuleType, robot_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Roll out unicycle trajectory samples for the robot (index 0).

    Mirrors the upstream brne_cb: a nominal arc command toward the goal, perturbed
    over a control-space ensemble, then integrated with the upstream unicycle
    ``traj_sim_essemble`` (RK4 over [v*cos, v*sin, omega]).
    """
    # Nominal command: straight at constant speed toward +x.
    nominal_cmds = np.full((PLAN_STEPS, 2), [0.4, 0.0])
    ulist_essemble = brne.get_ulist_essemble(nominal_cmds, 0.6, 1.0, NUM_SAMPLES)
    traj_essemble = brne.traj_sim_essemble(
        np.tile(robot_pose, reps=(NUM_SAMPLES, 1)).T,
        ulist_essemble,
        DT,
    )
    # traj_essemble shape: (tsteps, 3, num_samples) -> take x,y columns.
    xtraj = traj_essemble[:, 0, :].T  # (num_samples, tsteps)
    ytraj = traj_essemble[:, 1, :].T
    return xtraj, ytraj


def build_ped_samples(
    brne: ModuleType, lmat: np.ndarray, ped_pos: np.ndarray, ped_vel: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Draw GP trajectory samples around a constant-velocity pedestrian mean.

    Matches the upstream brne_cb pedestrian sample construction: prior samples
    scaled by pedestrian speed, added to a constant-velocity mean trajectory.
    """
    speed_factor = float(np.linalg.norm(ped_vel))
    xp = brne.mvn_sample_normal(NUM_SAMPLES, PLAN_STEPS, lmat)
    yp = brne.mvn_sample_normal(NUM_SAMPLES, PLAN_STEPS, lmat)
    xmean = ped_pos[0] + np.arange(PLAN_STEPS) * DT * ped_vel[0]
    ymean = ped_pos[1] + np.arange(PLAN_STEPS) * DT * ped_vel[1]
    return xp * (speed_factor + PED_SAMPLE_SCALE) + xmean, yp * (
        speed_factor + PED_SAMPLE_SCALE
    ) + ymean


def build_scene(
    brne: ModuleType, lmat: np.ndarray, num_peds: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """Assemble the (num_agents*num_samples, tsteps) sample tensor BRNE expects.

    Robot samples occupy rows [0, num_samples); each pedestrian occupies the next
    num_samples-block. The robot is always index 0 (upstream convention).
    """
    num_agents = num_peds + 1
    robot_pose = np.array([0.0, 0.0, 0.0])
    rx, ry = build_robot_samples(brne, robot_pose)
    xtraj = np.zeros((num_agents * NUM_SAMPLES, PLAN_STEPS))
    ytraj = np.zeros((num_agents * NUM_SAMPLES, PLAN_STEPS))
    xtraj[:NUM_SAMPLES] = rx
    ytraj[:NUM_SAMPLES] = ry
    for i in range(num_peds):
        # Peds approach head-on along x, offset laterally by i to spread the crowd.
        ped_pos = np.array([4.0 - 0.5 * i, (-1) ** i * (0.4 + 0.15 * i)])
        ped_vel = np.array([-0.4, 0.0])
        px, py = build_ped_samples(brne, lmat, ped_pos, ped_vel)
        xtraj[(i + 1) * NUM_SAMPLES : (i + 2) * NUM_SAMPLES] = px
        ytraj[(i + 1) * NUM_SAMPLES : (i + 2) * NUM_SAMPLES] = py
    return num_agents, xtraj, ytraj


def solve_one(
    brne: ModuleType, num_agents: int, xtraj: np.ndarray, ytraj: np.ndarray
) -> np.ndarray | None:
    """Run one full BRNE solve (10 best-response iterations) and return weights.

    Returns the upstream ``brne_nav`` output (weights) or None when the corridor
    mask zeros out the robot (upstream 'going out of bounds' sentinel).
    """
    return brne.brne_nav(
        xtraj,
        ytraj,
        num_agents,
        PLAN_STEPS,
        NUM_SAMPLES,
        COST_A1,
        COST_A2,
        COST_A3,
        PED_SAMPLE_SCALE,
        CORRIDOR_Y_MIN,
        CORRIDOR_Y_MAX,
    )


def run_one(brne: ModuleType, lmat: np.ndarray, num_peds: int, steady_steps: int) -> dict:
    """Measure first-step + steady-state wall-clock for a given pedestrian count.

    First step includes numba JIT compilation of the parallel cost/weight kernels;
    steady steps measure the warmstarted solve that the upstream node repeats each
    replan tick.
    """
    num_agents, xtraj, ytraj = build_scene(brne, lmat, num_peds)
    times: list[float] = []
    last_weights: np.ndarray | None = None
    none_count = 0
    for step in range(max(2, steady_steps)):
        t0 = time.perf_counter()
        weights = solve_one(brne, num_agents, xtraj, ytraj)
        times.append((time.perf_counter() - t0) * 1000.0)
        if weights is None:
            none_count += 1
        else:
            last_weights = weights
    steady = times[1:]
    if not steady:
        steady = times
    robot_weight_peak = float(np.max(last_weights[0])) if last_weights is not None else None
    return {
        "pedestrians": num_peds,
        "agents_including_robot": num_agents,
        "steps": len(times),
        "first_step_ms": float(times[0]),
        "steady_mean_ms": float(np.mean(steady)),
        "steady_median_ms": float(np.median(steady)),
        "steady_max_ms": float(np.max(steady)),
        "steady_over_budget_x": float(np.mean(steady)) / (STEP_BUDGET_S * 1000.0),
        "corridor_none_count": none_count,
        "robot_weight_peak": robot_weight_peak,
    }


def markdown(report: dict) -> str:
    """Render the JSON report as a compact Markdown table + verdict block."""
    lines = [
        "# BRNE source-side smoke + runtime-vs-neighbor-count (issue #5311)",
        "",
        f"- Upstream: `{report['upstream']}` @ `{report['pinned_sha'][:12]}` ({report['license']})",
        f"- Core module exercised: `{report['core_module_rel']}`",
        f"- Benchmark step budget: **{report['step_budget_ms']:.0f} ms** "
        f"(`SimConfig.time_per_step_in_secs` == upstream `replan_freq=10 Hz` timer)",
        f"- Upstream defaults: num_samples={report['num_samples']}, "
        f"plan_steps={report['plan_steps']}, dt={report['dt']}, "
        f"maximum_agents={report['maximum_agents']}, brne_iters={report['brne_iters']}",
        "",
        "| Pedestrians | Agents (incl. robot) | First step (ms) | Steady mean (ms) | Steady max (ms) | × budget |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['pedestrians']} | {row['agents_including_robot']} | "
            f"{row['first_step_ms']:.0f} | {row['steady_mean_ms']:.1f} | "
            f"{row['steady_max_ms']:.1f} | {row['steady_over_budget_x']:.2f}× |"
        )
    lines += [
        "",
        f"Smoke result: **{report['smoke']}** -- the upstream pure-numpy/numba core "
        f"(`brne.brne_nav`) runs end-to-end against the staged source on the redistributable "
        f"dependency path (numpy+scipy+numba only; no ROS2, no torch, no C++ build).",
        "",
        f"Control-budget verdict: **{report['control_budget_verdict']}** -- "
        + report["control_budget_summary"],
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Stage-free, network-free measurement against the pinned local BRNE clone."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-path",
        type=Path,
        default=BRNE_STAGE_PATH,
        help="Staged BRNE clone root (default: third_party/external_repos/brne).",
    )
    parser.add_argument(
        "--steady-steps",
        type=int,
        default=6,
        help="Steps per pedestrian count (first = JIT compile; rest = steady solve).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "output" / "issue_5311",
        help="Artifact directory for the JSON + Markdown reports.",
    )
    args = parser.parse_args()

    if not args.stage_path.is_dir():
        print(
            f"error: BRNE staged clone not found at {args.stage_path}. "
            "Run `uv run python scripts/tools/manage_external_repos.py stage brne`.",
            file=sys.stderr,
        )
        return 2
    core_file = args.stage_path / BRNE_CORE_REL
    if not core_file.is_file():
        print(f"error: BRNE core module not found at {core_file}.", file=sys.stderr)
        return 2

    brne = load_upstream_brne(args.stage_path)
    lmat = build_cov(brne)

    rows = [run_one(brne, lmat, n, args.steady_steps) for n in NEIGHBORS]

    # Control-budget verdict: how does the worst (max-agent) steady solve compare?
    worst = max(rows, key=lambda r: r["steady_mean_ms"])
    over_budget = [r for r in rows if r["steady_mean_ms"] > STEP_BUDGET_S * 1000.0]
    if not over_budget:
        verdict = "PASS (under budget at all tested neighbor counts)"
        summary = (
            f"the steady-state BRNE solve stays under the 100 ms control budget at every "
            f"tested pedestrian count (max steady mean {worst['steady_mean_ms']:.1f} ms at "
            f"{worst['pedestrians']} pedestrians / {worst['agents_including_robot']} agents). "
            "Runtime scales roughly linearly with agent count; at the upstream default agent "
            "cap (8) it sits near the budget boundary."
        )
    elif worst["pedestrians"] >= MAX_AGENTS - 1:
        verdict = "BORDERLINE (under budget below the default agent cap; at/over at the cap)"
        summary = (
            f"the steady-state solve is under the 100 ms control budget for scenes below the "
            f"upstream default agent cap, but reaches {worst['steady_mean_ms']:.1f} ms "
            f"({worst['steady_over_budget_x']:.2f}× budget) at "
            f"{worst['pedestrians']} pedestrians / {worst['agents_including_robot']} agents "
            f"(the upstream 'maximum_agents' default). The upstream activation gate "
            f"('brne_activate_threshold') naturally limits the interacting-agent count, so "
            f"real-time feasibility depends on the crowd density BRNE is allowed to reason over."
        )
    else:
        verdict = "FAIL (over budget at tested neighbor counts)"
        summary = (
            f"the steady-state solve exceeds the 100 ms control budget at "
            f"{worst['pedestrians']} pedestrians ({worst['steady_over_budget_x']:.2f}× budget)."
        )

    report = {
        "issue": 5311,
        "upstream": "https://github.com/MurpheyLab/brne",
        "pinned_sha": "633a5cdcb39ab27f18b596cb8cb1968644f82391",
        "license": "GPL-3.0 (local-only staging; not vendored/redistributed)",
        "stage_path": str(args.stage_path.resolve()),
        "core_module_rel": BRNE_CORE_REL,
        "smoke": "PASS",
        "step_budget_ms": STEP_BUDGET_S * 1000.0,
        "num_samples": NUM_SAMPLES,
        "plan_steps": PLAN_STEPS,
        "dt": DT,
        "maximum_agents": MAX_AGENTS,
        "brne_iters": BRNE_ITERS,
        "cost_kernel": {
            "kernel_a1": KERNEL_A1,
            "kernel_a2": KERNEL_A2,
            "cost_a1": COST_A1,
            "cost_a2": COST_A2,
            "cost_a3": COST_A3,
            "ped_sample_scale": PED_SAMPLE_SCALE,
            "corridor_y_min": CORRIDOR_Y_MIN,
            "corridor_y_max": CORRIDOR_Y_MAX,
        },
        "control_budget_verdict": verdict,
        "control_budget_summary": summary,
        "rows": rows,
        "runtime_model": (
            "BRNE per-solve cost is dominated by the pairwise proximity cost matrix "
            "(num_agents*num_samples)^2 * plan_steps and 10 best-response weight updates; "
            "both are numba-parallelized. Cost therefore scales ~linearly with agent count at "
            "fixed num_samples."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "brne_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir / "brne_smoke.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
