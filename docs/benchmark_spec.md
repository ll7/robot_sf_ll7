# Benchmark Spec: Social Navigation (Classic Interactions)

Purpose: Define the canonical scenario split, seed policy, baseline categories, and reproducible
commands for the Robot SF social navigation benchmark.

Canonical benchmark fallback policy:

- `docs/context/issue_691_benchmark_fallback_policy.md`

Scenario certification contract:

- [`docs/scenario_certification.md`](./scenario_certification.md)

Francis-guideline crosswalk for the current benchmark contract:

- [Francis Guideline Mapping For Robot SF](./context/issue_759_francis_guideline_mapping.md)

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
* ORCA requires the rvo2 binding; install with `uv sync --extra orca`.
* Fallback execution may still be used for explicit diagnostics, but it is not benchmark-success
  evidence and must fail closed in benchmark mode.
* `scenario_cert.v1` certificates classify malformed, infeasible, stress-only, and
  hard-but-solvable scenarios before benchmark interpretation. Treat `excluded` certificates as
  non-benchmark evidence and `stress_only` certificates as caveated stress coverage unless a
  separate benchmark issue promotes them.
* `socnav_bench` requires SocNavBench prerequisites (including `skfmm`); install with
  `uv sync --extra socnav` for native upstream execution.
* `socnav_sampling` uses the in-repo sampling adapter baseline, while `socnav_bench` is the
  upstream SocNavBench sampling wrapper.

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
* `--observation-mode <mode>` (optional): requests a declared planner observation contract for
  controlled input-modality checks. Unsupported planner/mode combinations fail before episodes
  are written.

Observation-mode declarations are metadata contracts, not automatic environment rewrites. The
current standard modes are:

* `goal_state`: robot state plus route goal only.
* `socnav_state`: structured robot, goal, and pedestrian state.
* `headed_socnav_state`: structured social-navigation state with headed robot fields.
* `sensor_fusion_state`: configured sensor-fusion stack used by learned checkpoint policies.
* `lidar_human_state` / `gst_human_state`: upstream learned-wrapper input contracts.

The built-in `goal` planner declares both `goal_state` and `socnav_state`, making it the initial
two-mode demonstration path. Extra channels in `socnav_state` are ignored by that planner, so this
is useful for input-contract parity checks but is not a claim of pure planner-logic attribution.

For parallel grid/SocNav-state, LiDAR, privileged, and adapter-derived evidence lanes, use the
observation-track architecture in
[`docs/context/issue_1612_observation_track_architecture.md`](./context/issue_1612_observation_track_architecture.md).
That note defines the proposed `benchmark_track` metadata, aggregation boundary, and config/result
naming convention. Track-aware map benchmark runs now preserve `benchmark_track` and
`track_schema_version` in episode rows, `scenario_params`, algorithm metadata, and resume identity.
Keep incompatible observation contracts out of the same aggregate unless a report explicitly opts
into cross-track diagnostics. Aggregate and report CLIs (`aggregate`, `table`, `rank`,
`plot-pareto`, `plot-distributions`, `snqi-ablate`, and `seed-variance`) fail closed by default
when an input JSONL mixes tracks. Use
`--observation-track-mode diagnostic-cross-track` only for explicitly caveated diagnostic
comparisons; those reports namespace rows by `benchmark_track` and record that fallback, degraded,
failed, or diagnostic-stub rows are caveats rather than benchmark-success evidence.

Placeholder planners (`rvo`, `dwa`, `teb`) are hard-blocked for benchmark runs.

## Success And Collision Semantics

Episode-level `success` is true only when the route reaches the goal before the horizon and total
collisions are zero. Total collisions are the sum of:

* pedestrian collisions: robot-pedestrian footprint overlap, using the episode `robot_radius` and
  `ped_radius`;
* wall/obstacle collisions: robot center within `collision_distance_m` of a sampled obstacle point;
* other-agent collisions: robot center within `collision_distance_m` of another robot/agent.

Near misses use pedestrian surface clearance, not center distance:
`0 <= min_clearance_m < near_miss_distance_m`. Synthetic benchmark-runner episodes resolve
robot/pedestrian radii from scenario metadata when present and otherwise use the same defaults for
planner observations and metric `EpisodeData`.

## Planner Inclusion Check

Use the mechanical inclusion check before proposing that an experimental planner move into a
promoted or baseline-safe benchmark set:

```bash
uv run robot_sf_bench planner-inclusion-check \
  --algo orca \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --output-dir output/planner_inclusion/orca
```

The command runs one planner on a small reference slice, validates episode writes through the
standard benchmark runner/schema path, aggregates the output, and writes a versioned JSON report
under the selected output directory. The default gates are intentionally minimal:

* every scheduled episode must produce a schema-valid record,
* aggregate values must be finite, with no NaN or infinite values,
* runtime must stay within `--max-runtime-sec` (default `60`),
* at least `--min-episodes` records must be written,
* `success` mean must be at least `--min-success-rate` (default `0.5`),
* `collisions` mean must be at most `--max-collision-rate` (default `0.0`).

The report decision is either `pass` or `revise`, with per-check failure reasons. Passing the gate
does not automatically change planner status, create a leaderboard claim, or replace paper-facing
benchmark evidence; it is a reproducible review input for a promotion PR.

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
* observation-noise profile/hash when enabled (`observation_noise_profile`,
  `observation_noise_hash`)

This guarantees that resuming across mixed algorithm/config batches does not accidentally skip
jobs that belong to a different algorithm or planner configuration.

## Observation Noise Profiles

Benchmark map runs can inject controlled planner-input observation noise without changing the
simulator state or ground-truth metric computation. This is intended for robustness checks only; the
profile is not calibrated to any real sensor model and should be reported with that caveat.

Direct run usage:

```bash
robot_sf_bench run \
  --matrix configs/scenarios/planner_sanity_matrix_v1.yaml \
  --out output/benchmarks/noise_smoke/episodes.jsonl \
  --algo goal \
  --benchmark-profile baseline-safe \
  --observation-noise configs/benchmarks/observation_noise/robustness_smoke_v1.yaml \
  --horizon 5 \
  --workers 1 \
  --no-video \
  --no-resume
```

Campaign configs can set the same profile with a top-level block or path:

```yaml
observation_noise: configs/benchmarks/observation_noise/robustness_smoke_v1.yaml
```

Supported fields include:
* `profile`, `enabled`, and optional deterministic `seed`
* `pose_noise_std_m` and `heading_noise_std_rad`
* `lidar_dropout_prob` and `lidar_dropout_value`
* `pedestrian_false_negative_prob`
* `pedestrian_false_positive_prob`, `pedestrian_false_positive_radius_m`, and
  `pedestrian_false_positive_radius`

Episode records include `observation_noise`, `observation_noise_hash`, and
`observation_noise_stats`. Campaign preflight, matrix summaries, manifests, and campaign summaries
also record the profile/hash. Absent or all-zero settings normalize to `profile: none` and leave
planner observations unchanged.

## Metrics: Definitions + Caveats (Summary)

Full details live in
[`docs/dev/issues/social-navigation-benchmark/metrics_spec.md`](./dev/issues/social-navigation-benchmark/metrics_spec.md).

**Core metrics**
* `success`: goal reached before horizon without collision.
* `time_to_goal_norm`: backward-compatible horizon normalization (clamped to `1.0` on failure).
* `time_to_goal_norm_success_only`: same normalization, but only valid for successful episodes.
* `time_to_goal_ideal_ratio`: success-only ratio of achieved time to ideal time
  (`shortest_path_len / robot_max_speed`).
* `outcome.collision_event`: canonical per-episode collision event flag for new episode outputs.
* `metrics.collisions`: collision count metric based on distance thresholds. For schema v1 episode
  outputs it must agree with `outcome.collision_event`: positive when the canonical event is true
  and zero when the canonical event is false.
* `near_misses`: count based on distance thresholds.
* `min_distance`,  `path_efficiency`: closest approach and shortest/actual path ratio.

**Force/comfort**
* `force_quantiles` (q50/q90/q95),  `per_ped_force_quantiles`
* `force_exceed_events`,  `comfort_exposure`

**Smoothness**
* `jerk_mean`,  `curvature_mean`,  `energy`

**Experimental pedestrian-impact (opt-in)**
* `ped_impact_*` metrics (enabled only via
  `compute_all_metrics(..., experimental_ped_impact=True)`).
* Post-processed episode records also include a schema-backed
  `metrics.pedestrian_impact` block with `schema_version: pedestrian-impact.v1`.
* Current prototype exposes near-vs-far deltas for pedestrian acceleration and
  heading turn-rate, plus sample/validity counters.
* Near/far semantics: distance to robot `<= ped_impact_radius_m` is "near",
  `> ped_impact_radius_m` is "far".
* Time-window semantics: signal smoothing uses trailing rolling means over
  `ped_impact_window_steps`.
* Episode reduction strategy: compute per-pedestrian `(near_mean - far_mean)` first,
  then reduce across pedestrians (mean/median) to reduce density bias.
* Aggregate reduction path: `robot_sf.benchmark.aggregate` flattens
  `metrics.pedestrian_impact.canonical_reductions` into aggregate-ready
  `ped_impact_accel_delta_{mean,median,valid}` and
  `ped_impact_turn_rate_delta_{mean,median,valid}` columns, then reports the
  normal mean/median/p95 aggregate statistics.

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
* `time_to_goal_norm` includes failures via clamp-to-`1.0`; for success-only reporting, use
  `time_to_goal_norm_success_only` together with the numeric validity flag
  `time_to_goal_success_only_valid == 1.0`.
* Experimental `ped_impact_*` metrics are exploratory and should be reported
  with the associated validity/sample counters from `metrics.pedestrian_impact.sample_counts`.
  Deltas can be undefined when a pedestrian has only near or only far samples.
* Thresholds (e.g., collision/near-miss distances, force thresholds) are defined in the metrics
  spec and implemented in `robot_sf/benchmark/metrics.py` .

## Expected Schema & Provenance

Each episode record is schema-validated against
`robot_sf/benchmark/schemas/episode.schema.v1.json` and includes:
* `scenario_id`,  `seed`,  `scenario_params`,  `metrics`, timing fields
* `algorithm_metadata.baseline_category` (`diagnostic|classical|learning`) and
  `algorithm_metadata.policy_semantics`
* `algorithm_metadata.planner_kinematics` including `execution_mode` (`native|adapter|mixed`) and
  adapter markers for compatibility interpretation. Contract now also includes
  `planner_command_space` (`unicycle_vw|holonomic_vxy`) for kinematics-aware interpretation.
* `observation_mode` and `algorithm_metadata.observation_spec`, including the planner's default
  mode, supported modes, active mode, and whether an override was applied.
* `algorithm_metadata.kinematics_feasibility` with command-level intervention diagnostics:
  `commands_evaluated`, `infeasible_native_count`, `projected_count`,
  `projection_rate`, `infeasible_rate`, `mean/max abs delta` for linear and angular commands.
* `metric_parameters.threshold_profile` + `metric_parameters.threshold_signature`
  for threshold provenance and reproducibility
* Git/config hashes for reproducibility

Collision compatibility: schema v1 consumers should read `outcome.collision_event` for the
canonical per-episode collision status and treat `metrics.collisions` as the agreeing count metric.
Older bundles that only carry legacy `status`, `collision_rate`, or inconsistent collision counts
should be migrated with `scripts/tools/migrate_episode_schema_v1.py`; new bundles fail validation
when `outcome.collision_event` and `metrics.collisions` disagree.

Batch/campaign-level metadata returned by `run_map_batch` (not individual
records from `_run_map_episode`) includes:
* `preflight.learned_policy_contract` for learned planners (currently PPO), including:
  * contract schema (allowed observation/action modes)
  * captured runtime config values
  * `status` (`pass|warn|fail|not_applicable`)
  * explicit mismatch/warning lists for auditability

For aggregation, use the utilities in `robot_sf/benchmark/aggregate.py` or the CLI
( `robot_sf_bench aggregate` ) to compute mean/median/p95 and optional bootstrap CIs.
Aggregation validates threshold-profile consistency and rejects mixed profiles.
When bootstrap sampling is enabled, aggregate output also includes an additive
`pairwise_contrasts` block when at least two planner groups share paired episode identities. The
contrast pairing key is `(scenario_id, seed)` with `seed_index` as a fallback, the reported delta is
`right_minus_left`, and each
metric contrast includes the paired mean delta, percentile bootstrap interval, two-sided bootstrap
sign p-value, Holm-adjusted p-value, and paired Cohen's dz effect size. Holm correction is applied
within the current aggregate family (`family="all"`) separately for each metric; run aggregation on
scenario-family-filtered inputs when family-specific correction is required. These contrasts are
descriptive benchmark summaries and do not by themselves establish high-power inference for small
or weakly paired evidence bases.

For threshold studies, run `scripts/benchmark_threshold_sensitivity.py` to quantify
distance/comfort threshold impacts across scenario families and to compare speed-aware

## AMMV Benchmark Protocol Manifest

The AMMV benchmark protocol is versioned as a checked-in manifest:

```text
benchmarks/ammv_benchmark_v0.yaml
```

A benchmark run that claims compliance with this protocol should record the
protocol reference in its run metadata, report, or claim artifact as:

```yaml
benchmark_protocol:
  id: ammv_benchmark_v0
  path: benchmarks/ammv_benchmark_v0.yaml
```

The manifest declares the scenario classes, planner panel, metric layers, and
claim rules required for an AMMV benchmark comparison.  It is descriptive in
this slice: loading the manifest validates the protocol shape, but it does not
execute scenarios, instantiate planners, or enforce a CI/release gate.
near-miss alternatives (relative-speed weighting and TTC-gating).
