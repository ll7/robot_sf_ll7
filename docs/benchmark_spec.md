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

## Baseline Categories (Diagnostic / Classical / Learning)

Baselines are labeled by category to keep comparisons explicit. Not every runner exposes every
baseline; use the entrypoint noted below.

| Baseline | Category | Entrypoint |
| --- | --- | --- |
| `fast_pysf_planner` | diagnostic | `scripts/tools/policy_analysis_run.py --policy fast_pysf_planner` |
| `random` | diagnostic | Registry-only baseline (not wired into policy_analysis_run.py) |
| `social_force` | classical | `scripts/run_classic_interactions.py --algo social_force` |
| `orca` | classical | `scripts/run_classic_interactions.py --algo orca` |
| `goal` / `simple` | classical | `scripts/run_classic_interactions.py --algo goal` |
| `ppo` | learning | `scripts/run_classic_interactions.py --algo ppo` or `scripts/tools/policy_analysis_run.py --policy ppo --model-path ...` |

Notes:
* Map-based benchmark runs use `scripts/run_classic_interactions.py` (see below) and accept the
  `--algo` names shown above.
* Learned baselines require model checkpoints; for policy analysis, provide `--model-path`.
* `random` is exposed via the baseline registry for non-map scenario matrices and is not currently
  wired into `policy_analysis_run.py`.
* `random` is the **stochastic reference baseline**: it samples actions from a configured uniform
  distribution (seeded RNG), and is intentionally distinct from deterministic `goal`.
* `ppo` is available in both map-based benchmark runs and policy-analysis runs; policy analysis
  remains the preferred path when you need richer learned-policy diagnostics (videos, per-policy
  reports, and policy sweep metadata).
* ORCA requires the rvo2 binding; install with `uv sync --extra orca` or set `allow_fallback: true`
  in the algo config to use the heuristic fallback.

## Algorithm Readiness Profiles

Canonical readiness profiles are versioned in
[`configs/benchmarks/paper_baseline_algorithms_v1.yaml`](../configs/benchmarks/paper_baseline_algorithms_v1.yaml).

CLI gating:
* `--benchmark-profile baseline-safe` (default): allows only baseline-ready algorithms.
* `--benchmark-profile paper-baseline`: publication profile; allows PPO only when paper-grade
  provenance and quality-gate fields are present in the algo config.
* `--benchmark-profile experimental`: allows baseline-ready + experimental algorithms.
* `--adapter-impact-eval` (optional): records adapter-impact metadata (native vs adapted step
  counts where measurable, currently most informative for PPO command conversion).

Placeholder planners (`rvo`, `dwa`, `teb`) are hard-blocked for benchmark runs.

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

**Recommended CI invocation (machine-readable):**

```bash
robot_sf_bench run \
  --matrix configs/scenarios/classic_interactions.yaml \
  --out output/benchmarks/ci/episodes.jsonl \
  --algo goal \
  --benchmark-profile baseline-safe \
  --structured-output jsonl \
  --external-log-noise auto
```

## Resume Semantics (Map Runs)

Map-runner resume identity is scoped to the full run unit, not just `(scenario, seed)`.
Each episode identity includes:

* scenario payload + seed
* algorithm (`algo`)
* algorithm config hash (`algo_config_hash`)
* run-shaping overrides when provided (`run_horizon`, `run_dt`, `record_forces`)

This guarantees that resuming across mixed algorithm/config batches does not accidentally skip
jobs that belong to a different algorithm or planner configuration.

## Metrics: Definitions + Caveats (Summary)

Full details live in
[`docs/dev/issues/social-navigation-benchmark/metrics_spec.md`](./dev/issues/social-navigation-benchmark/metrics_spec.md).

**Core metrics**
* `success`: goal reached before horizon without collision.
* `time_to_goal_norm`: backward-compatible horizon normalization (clamped to `1.0` on failure).
* `time_to_goal_norm_success_only`: same normalization, but only valid for successful episodes.
* `time_to_goal_ideal_ratio`: success-only ratio of achieved time to ideal time
  (`shortest_path_len / robot_max_speed`).
* `collisions`,  `near_misses`: counts based on distance thresholds.
* `min_distance`,  `path_efficiency`: closest approach and shortest/actual path ratio.

**Force/comfort**
* `force_quantiles` (q50/q90/q95),  `per_ped_force_quantiles`
* `force_exceed_events`,  `comfort_exposure`

**Smoothness**
* `jerk_mean`,  `curvature_mean`,  `energy`

**SocNavBench subset (vendored)**
* `socnavbench_path_length`,  `socnavbench_path_length_ratio`,  `socnavbench_path_irregularity`
  (subset of upstream SocNavBench metrics).

**SNQI (composite)**
* Weighted combination of normalized metrics using baseline statistics (median/p95).
* Weights and normalization stats can be supplied via `--snqi-weights` / `--snqi-baseline`.

**Caveats**
* If forces are not recorded, force-based metrics are `NaN`.
* If there are no pedestrians, distance/force metrics may be `NaN` while collisions are `0`.
* Curvature excludes near-zero velocities; invalid samples are filtered.
* `time_to_goal_norm` includes failures via clamp-to-`1.0`; for success-only reporting, filter with
  `time_to_goal_success_only_valid=true` and use `time_to_goal_norm_success_only`.
* Thresholds (e.g., collision/near-miss distances, force thresholds) are defined in the metrics
  spec and implemented in `robot_sf/benchmark/metrics.py` .

## Expected Schema & Provenance

Each episode record is schema-validated against
`robot_sf/benchmark/schemas/episode.schema.v1.json` and includes:
* `scenario_id`,  `seed`,  `scenario_params`,  `metrics`, timing fields
* `algorithm_metadata.baseline_category` (`diagnostic|classical|learning`) and
  `algorithm_metadata.policy_semantics`
* `algorithm_metadata.planner_kinematics` including `execution_mode` (`native|adapter|mixed`) and
  adapter markers for compatibility interpretation
* `metric_parameters.threshold_profile` + `metric_parameters.threshold_signature`
  for threshold provenance and reproducibility
* Git/config hashes for reproducibility

For aggregation, use the utilities in `robot_sf/benchmark/aggregate.py` or the CLI
( `robot_sf_bench aggregate` ) to compute mean/median/p95 and optional bootstrap CIs.
Aggregation validates threshold-profile consistency and rejects mixed profiles.

For threshold studies, run `scripts/benchmark_threshold_sensitivity.py` to quantify
distance/comfort threshold impacts across scenario families and to compare speed-aware
near-miss alternatives (relative-speed weighting and TTC-gating).
