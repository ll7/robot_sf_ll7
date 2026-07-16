# Robot SF Recipe Catalog

A **recipe** is a thin, blessed workflow that points at existing configs and
scripts and hides path complexity for new users. Recipes never add new
simulation or training logic; they only orchestrate commands that already live
in the repository.

Issue #5795 introduced the recipe layer as part of the adoption/UX epic.

## Commands

```bash
uv run robot-sf recipe list            # browse every recipe grouped by category
uv run robot-sf recipe explain <id>    # show purpose, mapped config, runtime, outputs
uv run robot-sf recipe run <id>        # execute the recipe's declared command
```

Add `--dry-run` to `recipe run` to print the command without executing it.

## Recipes

| Category | Recipe | Runtime | One-line purpose |
| --- | --- | --- | --- |
| getting-started | [`first-demo`](#first-demo) | < 30s CPU | First headless random-policy rollout |
| maps | [`custom-svg-map`](#custom-svg-map) | < 30s CPU | Simulate on a bundled custom SVG map |
| maps | [`map-validation`](#map-validation) | < 1 min CPU | Validate repository SVG maps |
| planners | [`orca-smoke`](#orca-smoke) | < 1 min CPU | ORCA / social-force planner smoke benchmark |
| planners | [`planner-comparison`](#planner-comparison) | < 2 min CPU | Compare planners on a sanity scenario |
| training | [`ppo-smoke`](#ppo-smoke) | < 2 min CPU | PPO training entry point, dry-run |
| benchmark | [`benchmark-mini-run`](#benchmark-mini-run) | < 2 min CPU | Mini benchmark batch run via `robot_sf_bench` |
| telemetry | [`telemetry-headless-demo`](#telemetry-headless-demo) | < 30s CPU | Headless occupancy-grid / telemetry demo |
| visualization | [`trace-viewer-demo`](#trace-viewer-demo) | < 1 min CPU | JSONL episode recording + playback demo |
| visualization | [`scenario-thumbnail-generation`](#scenario-thumbnail-generation) | < 1 min CPU | Generate scenario thumbnail figures |

For the full, always-current detail of any recipe, run:

```bash
uv run robot-sf recipe explain <id>
```

### first-demo

The shortest path to seeing Robot SF run: a headless random-policy rollout on
the default map. Run this first to confirm the install works.

### custom-svg-map

Load an SVG map into Robot SF and run a short random-policy rollout on it — the
starting point for bringing your own map into the simulator.

### map-validation

Run the SVG map verifier over the CI-enabled map set to catch structural,
metadata, and runtime-compatibility problems.

### orca-smoke

A tiny headless benchmark sweep of the social-force (ORCA-style) planner on the
planner-sanity scenario. Confirms the benchmark runner and aggregation pipeline.

### planner-comparison

Run the one-command smoke benchmark that exercises both simple-policy and
social-force planners and writes a Markdown comparison table.

### ppo-smoke

Exercise the PPO training entry point on CPU with a tiny smoke config in
`--dry-run` mode. Confirms the training stack and config loading **without**
real PPO optimisation. Remove `--dry-run` for actual (longer) training.

> **Dependency:** this recipe requires the `training` extra
> (`uv sync --extra training`) because `scripts/training/train_ppo.py` imports
> Stable-Baselines3, which is an optional dependency.

### benchmark-mini-run

Run a small headless batch of episodes through the benchmark CLI
(`robot_sf_bench run`) on the planner-sanity scenario matrix. No training, no
checkpoint.

### telemetry-headless-demo

Run the occupancy-grid quickstart, which exercises the observation, sensor, and
telemetry surfaces headless on CPU.

### trace-viewer-demo

Demonstrate the per-episode JSONL recording and playback system end to end —
the entry point for inspecting recorded trajectories.

### scenario-thumbnail-generation

Produce thumbnail figures from a bundled golden episodes fixture — the cheapest
way to exercise the thumbnail / figure pipeline without a fresh campaign.

## Claim boundary

Recipe outputs are local, worktree-local convenience artifacts. They are **not**
durable benchmark evidence unless promoted separately through the normal
benchmark provenance path.
