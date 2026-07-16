# Robot SF — Beginner Notebook Quickstarts

Three short, **CPU-only, deterministic** notebooks that take you from *install →
run → see something* in a few minutes. No GPU, no training, no model weights.

> Part of the adoption/UX epic (#5791); see [Issue #5798](https://github.com/ll7/robot_sf_ll7/issues/5798).

## Run a notebook

From the repository root:

```bash
# Option A — execute a notebook in place and render its plots
uv run jupyter notebook notebooks/01_run_first_episode.ipynb

# Option B — run all three headless (the CI smoke path)
uv run python scripts/validation/run_notebooks_smoke.py
```

Each notebook writes its small artifacts under `output/notebooks/` (which is
git-ignored), so the repository stays clean.

## The three notebooks

| # | Notebook | What you learn | Artifact(s) under `output/notebooks/` |
| --- | --- | --- | --- |
| 01 | [`01_run_first_episode.ipynb`](./01_run_first_episode.ipynb) | Build an environment, step it with a random policy, read & plot the reward | `01_run_first_episode/reward_curve.png` |
| 02 | [`02_compare_two_planners.ipynb`](./02_compare_two_planners.ipynb) | Run the **same scenario** with two different planners (`simple_policy` vs `random`) and compare metrics | `02_compare_two_planners/planner_comparison.png`, `comparison_summary.json` |
| 03 | [`03_visualize_trace.ipynb`](./03_visualize_trace.ipynb) | Record an episode to JSONL and **see** it three ways: a trajectory plot, a map thumbnail, and an interactive browser viewer | `03_visualize_trace/trace_trajectory.png`, `map_thumbnail.png`, `viewer/index.html`, `episode.jsonl` |

The notebooks only call **existing** env/planner/trace APIs — they add no new
simulation logic:

- [`robot_sf.gym_env.environment_factory.make_robot_env`](../robot_sf/gym_env/environment_factory.py)
- [`robot_sf.benchmark.runner.run_episode`](../robot_sf/benchmark/runner.py)
- [`robot_sf.render.threejs_viewer.export_threejs_viewer`](../robot_sf/render/threejs_viewer.py)

## Reproducibility & scope

- **Deterministic.** Every notebook fixes its seed, so re-running reproduces the
  same outputs on a clean CPU checkout.
- **Teaching, not benchmarking.** Notebook 02 compares two planners over a single
  short episode for one seed. It is an illustration, **not** a benchmark result —
  rigorous evaluation needs many seeds/scenarios via the benchmark tooling under
  `scripts/benchmark*`.

## Regenerating the notebooks

The notebooks are generated from a single readable script so their structure
stays in sync:

```bash
uv run python scripts/dev/generate_quickstart_notebooks.py
```

## CI

The notebooks are exercised headless on every PR via
`scripts/dev/ci_driver.sh notebooks-smoke` (see
[`scripts/validation/run_notebooks_smoke.py`](../scripts/validation/run_notebooks_smoke.py)
and the `examples-smoke` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).
