# Benchmark Spec: Social Navigation (Classic Interactions)

Purpose: Define the canonical scenario split, seed policy, baseline categories, and reproducible
commands for the Robot SF social navigation benchmark.

## Scope

This spec documents the map-based benchmark workflow driven by scenario manifests under
[`configs/scenarios/`](../configs/scenarios/). The benchmark uses **all scenarios** from the
classic interactions and Francis 2023 suites; holdout is defined by **seed sets**, not by
scenario exclusion. This spec focuses on reproducible episode generation, baseline labeling, and
metric definitions/caveats.

## Scenario Suite + Seeds

**Scenario manifests**
* All scenarios: [`configs/scenarios/classic_interactions_francis2023.yaml`](../configs/scenarios/classic_interactions_francis2023.yaml)
* Classic interactions subset: [`configs/scenarios/classic_interactions.yaml`](../configs/scenarios/classic_interactions.yaml)
* Francis 2023 subset: [`configs/scenarios/francis2023.yaml`](../configs/scenarios/francis2023.yaml)
* Map references can use `map_id` via [`maps/registry.yaml`](../maps/registry.yaml); see
  [`configs/scenarios/README.md`](../configs/scenarios/README.md) for details.

**Seed holdout policy**
* Training: use random seeds (no fixed list).
* Evaluation: use the fixed seed sets in
  [`configs/benchmarks/seed_sets_v1.yaml`](../configs/benchmarks/seed_sets_v1.yaml).
  * `dev`: `[101, 102, 103]`
  * `eval`: `[111, 112, 113]`
* Per-scenario `seeds` in YAML are still honored when no named seed set is supplied.
* If a scenario omits `seeds` and no seed set is requested, the map runner falls back to
  [`configs/benchmarks/seed_list_v1.yaml`](../configs/benchmarks/seed_list_v1.yaml).

## Baseline Categories (Oracle / Heuristic / Learned)

Baselines are labeled by category to keep comparisons explicit. Not every runner exposes every
baseline; use the entrypoint noted below.

| Baseline | Category | Entrypoint |
| --- | --- | --- |
| `fast_pysf_planner` | oracle | `scripts/tools/policy_analysis_run.py --policy fast_pysf_planner` |
| `social_force` | heuristic | `scripts/run_classic_interactions.py --algo social_force` |
| `orca` | heuristic | `scripts/run_classic_interactions.py --algo orca` |
| `goal` / `simple` | heuristic | `scripts/run_classic_interactions.py --algo goal` |
| `random` | heuristic | Registry-only baseline (not wired into policy_analysis_run.py) |
| `ppo` | learned | `scripts/tools/policy_analysis_run.py --policy ppo --model-path ...` |

Notes:
* Map-based benchmark runs use `scripts/run_classic_interactions.py` (see below) and accept the
  `--algo` names shown above.
* Learned baselines require model checkpoints; for policy analysis, provide `--model-path`.
* `random`/`ppo` are exposed via the baseline registry for non-map scenario matrices; map-based
  suites should use policy analysis for learned/GT comparisons. `random` is not currently wired
  into `policy_analysis_run.py`.
* ORCA requires the rvo2 binding; install with `uv sync --extra orca` or set `allow_fallback: true`
  in the algo config to use the heuristic fallback.

## Reproducible Command (One-Liner)

Run the full suite with a single command (seed-holdout eval set):

```bash
uv run python scripts/tools/policy_analysis_run.py \
  --scenario configs/scenarios/classic_interactions_francis2023.yaml \
  --policy-sweep \
  --seed-set eval \
  --output output/benchmarks/seed_holdout_eval
```

Expected outputs:
* `output/benchmarks/seed_holdout_eval/combined_report.md` (multi-policy summary)
* `output/benchmarks/seed_holdout_eval/<policy>/episodes.jsonl` (episode records)
* `output/benchmarks/seed_holdout_eval/<policy>/summary.json` (run summary)
* `output/benchmarks/seed_holdout_eval/<policy>/report.md` (policy report)

To re-run a different baseline, change `--policy` or supply `--policies`. For deterministic runs,
use `--seed-set dev|eval` and a fresh output folder (or delete existing JSONL files).

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
