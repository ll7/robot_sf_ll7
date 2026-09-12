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

The notebooks only call **existing** APIs — they add no new simulation logic.
Where the stable public facade owns an operation, the notebooks use the documented
top-level import:

- [`robot_sf.make_env`](../robot_sf/api.py) and [`robot_sf.load_scenario`](../robot_sf/api.py)

The remaining internal imports are documented exceptions because the facade does not expose an
equivalent yet; each carries an `internal-import-exception:` marker in the generated notebook:

- [`robot_sf.benchmark.runner.run_episode`](../robot_sf/benchmark/runner.py) — named-algorithm
  episode execution.
- [`robot_sf.baselines`](../robot_sf/baselines/__init__.py) — baseline planner registry and the
  bundled random planner.
- [`robot_sf.common.artifact_paths`](../robot_sf/common/artifact_paths.py) — artifact path policy.
- [`robot_sf.render.jsonl_playback`](../robot_sf/render/jsonl_playback.py) — JSONL playback loader.
- [`robot_sf.maps.map_visualizer`](../robot_sf/maps/map_visualizer.py) — map thumbnail renderer.
- [`robot_sf.render.threejs_viewer`](../robot_sf/render/threejs_viewer.py) — Three.js viewer export.

All three notebooks share one generated setup cell (headless environment, quiet logging,
repository/output discovery, plotting helper, fixed seed) and close their environment on normal and
exceptional paths.

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
# Verify committed notebooks match the generator without writing:
uv run python scripts/dev/generate_quickstart_notebooks.py --check
# Same check with the full canonical parity report:
uv run python scripts/dev/generate_quickstart_notebooks.py --check --json
```

`--check` rebuilds each notebook in memory, strips execution counts, outputs, transient cell ids,
widget state, and environment-specific metadata, and compares canonical JSON against the committed
file. It fails closed when a committed notebook is missing, drifts in source or stable metadata, or
contains executed output or an execution count. The report names the exact mismatch paths and stable
reason codes (`cell_source_changed`, `metadata_changed`, `cell_structure_changed`,
`content_changed`, `transient_state_present`, `missing_committed_notebook`).

The notebook smoke (`scripts/validation/run_notebooks_smoke.py`) runs the same parity check before
executing the notebooks, so a hand-edited or executed notebook fails CI before any kernel starts.
Use `--skip-parity` only for a deliberately local execution-only probe.

## CI

The notebooks are exercised headless on every PR via
`scripts/dev/ci_driver.sh notebooks-smoke` (see
[`scripts/validation/run_notebooks_smoke.py`](../scripts/validation/run_notebooks_smoke.py)
and the `examples-smoke` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).
