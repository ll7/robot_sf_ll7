#!/usr/bin/env python3
"""Generate the three CPU-only beginner quickstart notebooks for Issue #5798.

Writes:
    notebooks/01_run_first_episode.ipynb
    notebooks/02_compare_two_planners.ipynb
    notebooks/03_visualize_trace.ipynb

Each notebook is headless (``SDL_VIDEODRIVER=dummy`` / ``MPLBACKEND=Agg`` set in
its first code cell), deterministic (fixed seeds), and writes its small artifact
under ``output/notebooks/``. The notebooks reuse existing env/planner/trace APIs
and add no new simulation logic.

Regenerate with:
    uv run python scripts/dev/generate_quickstart_notebooks.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "notebooks"


def _dedent(src: str) -> str:
    """Strip surrounding blank lines and common leading indentation.

    Notebook source is written as indented triple-quoted strings inside this
    generator; cells must be emitted as clean top-level code/markdown.
    """
    return textwrap.dedent(src).strip("\n")


def _code(src: str) -> nbf.notebooknode:
    return nbf.v4.new_code_cell(_dedent(src))


def _md(src: str) -> nbf.notebooknode:
    return nbf.v4.new_markdown_cell(_dedent(src))


# --------------------------------------------------------------------------------------
# Notebook 01: run a first episode
# --------------------------------------------------------------------------------------
def build_notebook_01() -> nbf.notebooknode:
    """Build the "run a first episode" notebook."""
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    cells = [
        _md(
            """
            # 01 — Run your first Robot SF episode

            This notebook runs a short, **fully deterministic** episode in a Robot SF
            navigation environment using a random policy, then plots the per-step reward.

            **You will learn how to:**
            1. Build a robot navigation environment with the factory function.
            2. Reset it and step it with random actions.
            3. Read the reward signal and plot it with matplotlib.

            > CPU-only, headless, no rendering window, no training. Runs in a few seconds.
            >
            > It mirrors the canonical example at
            > `examples/quickstart/01_basic_robot.py`.
            """
        ),
        _code(
            """
            # Headless setup: no GUI window. We deliberately do NOT force a matplotlib
            # backend here, because several robot_sf modules call matplotlib.use("Agg",
            # force=True) at import time and would clobber an inline backend set now.
            # Instead each plotting cell re-enables the inline backend right before it
            # draws, so figures still render into the notebook output under nbconvert.
            import os
            import sys
            from IPython import get_ipython

            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


            def _inline_matplotlib() -> None:
                # (Re-)arm the notebook inline backend for the next figure.
                ip = get_ipython()
                if ip is not None:
                    ip.run_line_magic("matplotlib", "inline")


            # Keep log output quiet so the executed notebook stays readable.
            from loguru import logger
            logger.remove()
            logger.add(sys.stderr, level="ERROR")

            from pathlib import Path

            import matplotlib.pyplot as plt

            from robot_sf.gym_env.environment_factory import make_robot_env

            # Resolve the repo root robustly regardless of the working directory this
            # notebook is launched from (repo root, or its own notebooks/ folder).
            def _repo_root() -> Path:
                here = Path.cwd()
                for candidate in [here, *here.parents]:
                    if (candidate / "pyproject.toml").exists():
                        return candidate
                return here

            REPO_ROOT = _repo_root()

            # Where this notebook writes its small artifact (git-ignored).
            OUTPUT_DIR = REPO_ROOT / "output/notebooks/01_run_first_episode"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            print("Repo root:", REPO_ROOT)
            print("Artifacts will be written to:", OUTPUT_DIR.relative_to(REPO_ROOT))
            """
        ),
        _md(
            "## 1. Build and reset the environment\n\n`make_robot_env()` is the ergonomic entry point. We seed the factory, the reset, and the action space explicitly so the action/reward trace is reproducible."
        ),
        _code(
            """
            SEED = 87234

            # Seed the factory (Python random + NumPy + env RNGs), the reset, and the
            # action space explicitly. We deliberately avoid set_global_seed here: it
            # also seeds the private Torch/TensorFlow RNGs, which can crash the kernel
            # on the installed stack, and it does NOT seed Gymnasium's action_space RNG.
            env = make_robot_env(debug=False, seed=SEED)
            observation, info = env.reset(seed=SEED)
            env.action_space.seed(SEED)
            print("Environment created and reset.")
            print("Observation type:", type(observation).__name__)
            if hasattr(observation, "keys"):
                print("Observation keys:", list(observation.keys()))
            print("Action space:", env.action_space)
            """
        ),
        _md(
            """
            ## 2. Step the environment with random actions

            Each call to `env.step(action)` returns the Gymnasium 5-tuple
            `(observation, reward, terminated, truncated, info)`. We collect the
            reward at every step. If the episode ends early (collision or success),
            we reset and keep going so the budget is filled.
            """
        ),
        _code(
            """
            N_STEPS = 60
            rewards = []
            collisions = 0

            for step in range(1, N_STEPS + 1):
                action = env.action_space.sample()  # random policy
                observation, reward, terminated, truncated, info = env.step(action)
                rewards.append(float(reward))
                if bool(info.get("collision")):
                    collisions += 1
                if terminated or truncated:
                    observation, info = env.reset()

            print(f"Stepped {N_STEPS} times.")
            print(f"Total reward: {sum(rewards):.3f}")
            print(f"Collisions: {collisions}")
            """
        ),
        _md(
            "## 3. Plot the reward over time\n\nWe save the figure to `output/` and also display it inline."
        ),
        _code(
            """
            _inline_matplotlib()
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.plot(range(1, N_STEPS + 1), rewards, marker=".", linewidth=1)
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
            ax.set_xlabel("Simulation step")
            ax.set_ylabel("Reward")
            ax.set_title(f"Per-step reward (random policy, seed={SEED})")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            plot_path = OUTPUT_DIR / "reward_curve.png"
            fig.savefig(plot_path, dpi=120)
            print("Saved:", plot_path.relative_to(REPO_ROOT))
            plt.show()
            """
        ),
        _md(
            """
            ## Done

            You built an environment, ran a deterministic episode, and inspected the
            reward signal. Next, try **02 — Compare two planners** to see how two
            different navigation strategies behave, or **03 — Visualize a trace** to
            watch an episode as an interactive viewer + trajectory plot.
            """
        ),
        _code(
            """
            # Always release the environment's resources when finished.
            env.exit()
            print("Environment closed. Notebook 01 complete.")
            """
        ),
    ]
    nb.cells = cells
    return nb


# --------------------------------------------------------------------------------------
# Notebook 02: compare two planners
# --------------------------------------------------------------------------------------
def build_notebook_02() -> nbf.notebooknode:
    """Build the "compare two planners" notebook."""
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    cells = [
        _md(
            """
            # 02 — Compare two planners

            This notebook runs the **same scenario** with **two different planners**
            and compares their outcomes side by side.

            The two planners are:

            - **`simple_policy`** — a built-in goal-seeking policy that steers the
              robot toward its goal.
            - **`random`** — a random policy that ignores the goal (a control/baseline).

            We reuse the benchmark episode runner
            (`robot_sf.benchmark.runner.run_episode`), which loads each planner from
            the baseline registry, builds observations, translates actions, and
            computes a rich metrics dictionary. **No new logic is added** — we only
            call the existing public API and chart the results.

            > CPU-only, deterministic (fixed seed), no model weights required.
            >
            > The metrics here are a small illustrative sample for a single seed —
            > they are **not** a benchmark result.
            """
        ),
        _code(
            """
            import os
            import sys
            from IPython import get_ipython

            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


            def _inline_matplotlib() -> None:
                # (Re-)arm the notebook inline backend for the next figure.
                ip = get_ipython()
                if ip is not None:
                    ip.run_line_magic("matplotlib", "inline")


            # Keep log output quiet so the executed notebook stays readable.
            from loguru import logger
            logger.remove()
            logger.add(sys.stderr, level="ERROR")

            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np

            from robot_sf.training.scenario_loader import load_scenarios
            from robot_sf.benchmark.runner import run_episode
            from robot_sf.baselines import list_baselines

            # Resolve the repo root robustly regardless of launch directory.
            def _repo_root() -> Path:
                here = Path.cwd()
                for candidate in [here, *here.parents]:
                    if (candidate / "pyproject.toml").exists():
                        return candidate
                return here

            REPO_ROOT = _repo_root()

            OUTPUT_DIR = REPO_ROOT / "output/notebooks/02_compare_two_planners"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            print("Repo root:", REPO_ROOT)
            print("Artifacts will be written to:", OUTPUT_DIR.relative_to(REPO_ROOT))
            """
        ),
        _md(
            "## 1. Load the shared scenario\n\nWe use the bundled `quickstart_demo_crossing_basic` scenario so both planners face the identical situation."
        ),
        _code(
            """
            SCENARIO_PATH = REPO_ROOT / "configs/scenarios/single/quickstart_demo.yaml"
            SCENARIO_NAME = "quickstart_demo_crossing_basic"

            scenarios = load_scenarios(SCENARIO_PATH)
            scenario = next(s for s in scenarios if s["name"] == SCENARIO_NAME)
            print("Loaded scenario:", scenario["name"])
            print("Map file:", scenario.get("map_file"))
            print("Registered baseline planners:", sorted(list_baselines()))
            print("(plus the built-in goal-seeker known as 'simple_policy' / 'goal')")
            """
        ),
        _md(
            """
            ## 2. Run each planner for one episode

            `run_episode` returns a record whose `metrics` dictionary contains the
            quantities we compare. We keep the horizon short so the notebook stays
            fast.
            """
        ),
        _code(
            """
            SEED = 270
            HORIZON = 60
            DT = 0.1
            PLANNERS = ["simple_policy", "random"]

            records = {}
            for algo in PLANNERS:
                rec = run_episode(scenario, SEED, horizon=HORIZON, dt=DT, algo=algo)
                records[algo] = rec
                m = rec["metrics"]
                print(
                    f"{algo:>14s}: steps={rec['steps']:>3d}  "
                    f"collisions={int(m['collisions']):>2d}  "
                    f"avg_speed={float(m['avg_speed']):.3f}  "
                    f"path_length={float(m['socnavbench_path_length']):.2f}  "
                    f"success={bool(m['success'])}"
                )
            """
        ),
        _md(
            """
            ## 3. Compare key metrics

            We chart four beginner-friendly metrics for both planners:

            | Metric | Meaning |
            | --- | --- |
            | `collisions` | Number of pedestrian/obstacle collisions in the episode |
            | `avg_speed` | Mean robot speed (m/s) |
            | `socnavbench_path_length` | Total path length travelled (m) |
            | `path_efficiency` | Ratio of straight-line to actual path (1.0 = perfectly direct) |
            """
        ),
        _code(
            """
            metric_keys = ["collisions", "avg_speed", "socnavbench_path_length", "path_efficiency"]
            metric_labels = ["Collisions", "Avg speed (m/s)", "Path length (m)", "Path efficiency"]

            values = np.array(
                [[float(records[a]["metrics"][k]) for k in metric_keys] for a in PLANNERS],
                dtype=float,
            )

            _inline_matplotlib()
            fig, axes = plt.subplots(1, len(metric_keys), figsize=(13, 3.4))
            x = np.arange(len(PLANNERS))
            for j, (ax, label) in enumerate(zip(axes, metric_labels)):
                ax.bar(x, values[:, j], color=["#4C72B0", "#DD8452"])
                ax.set_xticks(x)
                ax.set_xticklabels(PLANNERS, rotation=20, ha="right")
                ax.set_title(label)
                ax.grid(True, axis="y", alpha=0.3)
                for xi, vi in zip(x, values[:, j]):
                    ax.text(xi, vi, f"{vi:.2f}", ha="center", va="bottom", fontsize=9)
            fig.suptitle(f"Two planners on the same scenario (seed={SEED}, horizon={HORIZON})", y=1.04)
            fig.tight_layout()
            plot_path = OUTPUT_DIR / "planner_comparison.png"
            fig.savefig(plot_path, dpi=120, bbox_inches="tight")
            print("Saved:", plot_path.relative_to(REPO_ROOT))
            plt.show()
            """
        ),
        _md(
            """
            ## 4. Summary table

            A compact, machine-readable comparison written to `output/` alongside the
            figure.
            """
        ),
        _code(
            """
            import json

            summary = {
                "schema_version": "robot_sf_notebook_02_planner_compare.v1",
                "scenario": SCENARIO_NAME,
                "seed": SEED,
                "horizon": HORIZON,
                "dt": DT,
                "claim_boundary": (
                    "Single-seed illustrative comparison for a beginner notebook; "
                    "not a benchmark result."
                ),
                "planners": {
                    algo: {
                        k: records[algo]["metrics"][k] for k in metric_keys
                    }
                    for algo in PLANNERS
                },
            }
            summary_path = OUTPUT_DIR / "comparison_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\\n", encoding="utf-8")
            print("Saved:", summary_path.relative_to(REPO_ROOT))
            print(json.dumps(summary["planners"], indent=2))
            """
        ),
        _md(
            """
            ## Interpretation

            - The **goal-seeker** (`simple_policy`) usually travels further and faster
              because it actively aims for the goal — but on a crowded map it may
              collide with pedestrians along the way.
            - The **random** planner wanders without purpose, so it tends to stall,
              cover less ground, and time out — but without committing to a direction
              it may also happen to avoid collisions.

            The exact numbers depend on the seed and the single short horizon; this is
            a teaching comparison, not a rigorous evaluation. To draw real conclusions
            you would repeat over many seeds and scenarios (see the benchmark tooling
            under `scripts/benchmark*`).

            Next: **03 — Visualize a trace**.
            """
        ),
    ]
    nb.cells = cells
    return nb


# --------------------------------------------------------------------------------------
# Notebook 03: visualize a trace
# --------------------------------------------------------------------------------------
def build_notebook_03() -> nbf.notebooknode:
    """Build the "visualize a trace" notebook."""
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    cells = [
        _md(
            """
            # 03 — Visualize an episode trace

            This notebook records one deterministic episode to disk (JSONL), then
            produces **three artifacts** that let you *see* what happened:

            1. A **2D trajectory plot** (robot path + pedestrian positions) via matplotlib.
            2. A **top-down map thumbnail** (PNG).
            3. An **interactive Three.js browser viewer** (open `index.html`).

            It reuses the exact recording + playback + viewer pipeline that powers
            `uv run robot-sf demo` (see `scripts/demo/quickstart_demo.py`) — no new
            logic, only the existing public APIs.

            > CPU-only, headless, deterministic (fixed seed).
            """
        ),
        _code(
            """
            import os
            import sys
            from IPython import get_ipython

            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


            def _inline_matplotlib() -> None:
                # (Re-)arm the notebook inline backend for the next figure.
                ip = get_ipython()
                if ip is not None:
                    ip.run_line_magic("matplotlib", "inline")


            # Keep log output quiet so the executed notebook stays readable.
            from loguru import logger
            logger.remove()
            logger.add(sys.stderr, level="ERROR")

            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np

            # Resolve the repo root robustly regardless of launch directory.
            def _repo_root() -> Path:
                here = Path.cwd()
                for candidate in [here, *here.parents]:
                    if (candidate / "pyproject.toml").exists():
                        return candidate
                return here

            REPO_ROOT = _repo_root()

            OUTPUT_DIR = REPO_ROOT / "output/notebooks/03_visualize_trace"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            print("Repo root:", REPO_ROOT)
            print("Artifacts will be written to:", OUTPUT_DIR.relative_to(REPO_ROOT))
            """
        ),
        _md(
            """
            ## 1. Run one deterministic, recorded episode

            We build the environment with JSONL recording enabled, drive it with the
            bundled random planner at a fixed seed, and stop the recording when the
            episode finishes. The per-step simulation states are written to a `.jsonl`
            file we can replay.
            """
        ),
        _code(
            """
            from robot_sf.gym_env.environment_factory import make_robot_env
            from robot_sf.gym_env.robot_env import RobotEnv
            from robot_sf.common.artifact_paths import resolve_artifact_path
            from robot_sf.training.scenario_loader import load_scenarios, build_robot_config_from_scenario
            from robot_sf.baselines.random_policy import RandomPlanner

            SEED = 270
            SCENARIO_NAME = "quickstart_demo_crossing_basic"
            SCENARIO_PATH = REPO_ROOT / "configs/scenarios/single/quickstart_demo.yaml"

            scenario = next(
                s for s in load_scenarios(SCENARIO_PATH) if s["name"] == SCENARIO_NAME
            )
            config = build_robot_config_from_scenario(scenario, scenario_path=SCENARIO_PATH)
            recording_dir = resolve_artifact_path(OUTPUT_DIR / "recordings")

            env: RobotEnv = make_robot_env(
                config=config,
                seed=SEED,
                debug=False,
                recording_enabled=True,
                use_jsonl_recording=True,
                recording_dir=str(recording_dir),
                suite_name="notebook",
                scenario_name=SCENARIO_NAME,
                algorithm_name="random",
                recording_seed=SEED,
            )

            planner = RandomPlanner({"mode": "velocity", "v_max": 1.5}, seed=SEED)
            env.reset(seed=SEED)
            planner.reset(seed=SEED)

            steps = 0
            terminated = truncated = False
            while not truncated:
                decision = planner.step({"dt": 0.1, "robot": {}, "agents": []})
                action = np.array([float(decision["vx"]), float(decision["vy"])], dtype=np.float32)
                _obs, _reward, terminated, truncated, _info = env.step(action)
                steps += 1
                if terminated:
                    break

            env.end_episode_recording()
            env.close_recorder()
            env.exit()
            print(f"Episode finished after {steps} steps.")
            """
        ),
        _code(
            """
            # Consume the exact path reported by the recorder. Do not glob the directory:
            # a previous notebook run may have left a newer-looking, stale recording there.
            env_path = getattr(env, "last_recorded_jsonl", None)
            if env_path is None:
                raise FileNotFoundError(
                    "No JSONL recording was produced; the episode recorder did not report "
                    "an output path."
                )
            episode_jsonl = Path(env_path)
            if not episode_jsonl.is_file():
                raise FileNotFoundError(
                    f"The episode recorder reported a missing JSONL file: {episode_jsonl}"
                )
            # Promote a stable copy next to the other artifacts.
            stable_jsonl = OUTPUT_DIR / "episode.jsonl"
            stable_jsonl.write_bytes(episode_jsonl.read_bytes())
            print("Recorded episode:", episode_jsonl.name)
            print("Stable copy:", stable_jsonl.relative_to(REPO_ROOT))
            """
        ),
        _md(
            """
            ## 2. Load the recording and plot the trajectory

            `JSONLPlaybackLoader` reads the JSONL back into a list of visualizable
            states. Each state exposes the robot pose and the pedestrian positions, so
            we can draw the robot's path and the pedestrians' final footprint.
            """
        ),
        _code(
            """
            from robot_sf.render.jsonl_playback import JSONLPlaybackLoader

            episode, map_def = JSONLPlaybackLoader().load_single_episode(stable_jsonl)
            states = episode.states
            print(f"Loaded {len(states)} recorded states.")

            robot_path = np.array([s.robot_pose[0] for s in states])
            # Final pedestrian positions (last recorded frame).
            final_peds = states[-1].pedestrian_positions

            _inline_matplotlib()
            fig, ax = plt.subplots(figsize=(6.4, 6.4))
            ax.plot(robot_path[:, 0], robot_path[:, 1], "-", color="#4C72B0", linewidth=2, label="Robot path")
            ax.plot(robot_path[0, 0], robot_path[0, 1], "o", color="green", markersize=9, label="Start")
            ax.plot(robot_path[-1, 0], robot_path[-1, 1], "s", color="red", markersize=9, label="End")
            if final_peds.size:
                ax.scatter(final_peds[:, 0], final_peds[:, 1], c="#DD8452", s=70, label="Pedestrians (last frame)", zorder=5)
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"Episode trace: {SCENARIO_NAME} (random, seed={SEED})")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            plot_path = OUTPUT_DIR / "trace_trajectory.png"
            fig.savefig(plot_path, dpi=120)
            print("Saved:", plot_path.relative_to(REPO_ROOT))
            plt.show()
            """
        ),
        _md(
            "## 3. Render a top-down map thumbnail\n\nWe reuse the map visualizer that the one-command demo uses."
        ),
        _code(
            """
            from robot_sf.maps.map_visualizer import visualize_map_definition

            thumbnail_path = OUTPUT_DIR / "map_thumbnail.png"
            visualize_map_definition(map_def, output_path=thumbnail_path, title=f"{SCENARIO_NAME} (random)")
            print("Saved:", thumbnail_path.relative_to(REPO_ROOT))
            """
        ),
        _md(
            """
            ## 4. Export the interactive browser viewer

            `export_threejs_viewer` turns the JSONL recording into a static
            `index.html` Three.js scene you can open in any browser (no server
            needed). Run the cell, then open the printed path.
            """
        ),
        _code(
            """
            from robot_sf.render.threejs_viewer import export_threejs_viewer

            viewer_dir = OUTPUT_DIR / "viewer"
            result = export_threejs_viewer(stable_jsonl, viewer_dir)
            print("Open this file in a browser to watch the recording:")
            try:
                viewer_rel = Path(result.html_path).relative_to(REPO_ROOT)
            except ValueError:
                viewer_rel = Path(result.html_path)
            print("  ", viewer_rel)
            print("Viewer files:", sorted(p.name for p in viewer_dir.iterdir()))
            """
        ),
        _md(
            """
            ## Summary

            Artifacts written under `output/notebooks/03_visualize_trace/`:

            | File | What it is |
            | --- | --- |
            | `episode.jsonl` | The recorded per-step simulation states |
            | `trace_trajectory.png` | Robot path + pedestrians (matplotlib) |
            | `map_thumbnail.png` | Top-down map thumbnail |
            | `viewer/index.html` | Interactive Three.js browser viewer |

            You have now run an episode, **and** seen it three ways. These three
            notebooks together form the beginner quickstart: **run → compare →
            visualize**.
            """
        ),
    ]
    nb.cells = cells
    return nb


def main() -> int:
    """Generate the three notebooks into ``notebooks/``."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builders = {
        "01_run_first_episode.ipynb": build_notebook_01,
        "02_compare_two_planners.ipynb": build_notebook_02,
        "03_visualize_trace.ipynb": build_notebook_03,
    }
    for name, builder in builders.items():
        nb = builder()
        path = OUT_DIR / name
        nbf.write(nb, path)
        try:
            display = path.relative_to(ROOT)
        except ValueError:
            display = path
        print(f"Wrote {display}")
    print("Done. Validate with:")
    print(
        "  SDL_VIDEODRIVER=dummy uv run jupyter nbconvert --execute "
        "notebooks/*.ipynb --to notebook --inplace"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
