# Benchmark Spec: Social Navigation (Classic Interactions)

Purpose: Define the canonical scenario split, seed policy, baseline categories, and reproducible
commands for the Robot SF social navigation benchmark.

## Scope

This spec documents the map-based benchmark workflow driven by scenario manifests under
`configs/scenarios/` . The **classic interactions** suite is the core benchmark set, while the
**francis2023** suite is a holdout/generalization set. This spec focuses on reproducible episode
generation, baseline labeling, and metric definitions/caveats.

## Scenario Split + Seeds

**Core suite: Classic interactions**
* Manifest: [`configs/scenarios/classic_interactions.yaml`](../configs/scenarios/classic_interactions.yaml)
* Archetypes: bottleneck, crossing, doorway, group_crossing, head_on_corridor, merging, 
  overtaking, t_intersection (see the included archetype YAML files referenced by the manifest).
* Seeds: Each scenario file defines `seeds` (e.g.,  `[101, 102, 103]` in the archetype entries).

**Holdout suite: Francis 2023**
* Manifest: [`configs/scenarios/francis2023.yaml`](../configs/scenarios/francis2023.yaml)
* Scenario set: single-pedestrian + crowd interaction cases aligned with Francis 2023 Fig. 7.
* Seeds: Each scenario file defines `seeds` (e.g.,  `[201, 202, 203]` in the single-scenario files).

**Fallback seed policy**
* If a scenario omits `seeds`, the map runner uses `configs/benchmarks/seed_list_v1.yaml`
  ( `classic_interactions: 101, 102, 103, 111, 112, 113, 121, 122, 123, 131` , 
`francis2023: 201–210` , `default: [0]` ).

## Baseline Categories (Oracle / Heuristic / Learned)

Baselines are labeled by category to keep comparisons explicit. Not every runner exposes every
baseline; use the entrypoint noted below.

| Baseline | Category | Entrypoint |
| --- | --- | --- |
| `fast_pysf_planner` | Oracle/GT | `scripts/tools/policy_analysis_run.py --policy fast_pysf_planner` |
| `social_force` | Heuristic | `scripts/run_classic_interactions.py --algo social_force` |
| `orca` | Heuristic | `scripts/run_classic_interactions.py --algo orca` |
| `goal` / `simple` | Heuristic | `scripts/run_classic_interactions.py --algo goal` |
| `random` | Heuristic | Baseline registry only (non-map matrices; not wired into policy_analysis_run.py) |
| `ppo` | Learned | `robot_sf.baselines.ppo` or `policy_analysis_run.py --policy ppo` |

Notes:
* Map-based benchmark runs use `scripts/run_classic_interactions.py` (see below) and accept the
`--algo` names shown above.
* Learned baselines require model checkpoints; for policy analysis, provide `--model-path`.
* `random`/`ppo` are exposed via the baseline registry for non-map scenario matrices; map-based
  suites should use policy analysis for learned/GT comparisons.

## Reproducible Command (One-Liner)

Run the core benchmark suite with a single command:

```bash
uv run python scripts/run_classic_interactions.py \
  --algo social_force \
  --workers 4 \
  --output output/benchmarks/classic_interactions/social_force.jsonl \
  --no-resume
```

Expected outputs:
* `output/benchmarks/classic_interactions/social_force.jsonl` (episode records)
* `output/benchmarks/classic_interactions/social_force.summary.json` (run summary)

To re-run a different baseline, change `--algo` (e.g., `orca` , `goal` ). For deterministic runs, 
keep seeds fixed in the scenario YAML and avoid stochastic planner noise.

## Metrics: Definitions + Caveats (Summary)

Full details live in
[`docs/dev/issues/social-navigation-benchmark/metrics_spec.md`](./dev/issues/social-navigation-benchmark/metrics_spec.md).

**Core metrics**
* `success`: goal reached before horizon without collision.
* `time_to_goal_norm`: steps-to-goal normalized by horizon (1.0 on failure).
* `collisions`,  `near_misses`: counts based on distance thresholds.
* `min_distance`,  `path_efficiency`: closest approach and shortest/actual path ratio.

**Force/comfort**
* `force_quantiles` (q50/q90/q95),  `per_ped_force_quantiles`
* `force_exceed_events`,  `comfort_exposure`

**Smoothness**
* `jerk_mean`,  `curvature_mean`,  `energy`

**SNQI (composite)**
* Weighted combination of normalized metrics using baseline statistics (median/p95).
* Weights and normalization stats can be supplied via `--snqi-weights` / `--snqi-baseline`.

**Caveats**
* If forces are not recorded, force-based metrics are `NaN`.
* If there are no pedestrians, distance/force metrics may be `NaN` while collisions are `0`.
* Curvature excludes near-zero velocities; invalid samples are filtered.
* Thresholds (e.g., collision/near-miss distances, force thresholds) are defined in the metrics
  spec and implemented in `robot_sf/benchmark/metrics.py` .

## Expected Schema & Provenance

Each episode record is schema-validated against
`robot_sf/benchmark/schemas/episode.schema.v1.json` and includes:
* `scenario_id`,  `seed`,  `scenario_params`,  `metrics`, timing fields
* Git/config hashes for reproducibility

For aggregation, use the utilities in `robot_sf/benchmark/aggregate.py` or the CLI
( `robot_sf_bench aggregate` ) to compute mean/median/p95 and optional bootstrap CIs.
