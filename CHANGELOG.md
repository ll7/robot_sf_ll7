# Changelog

All notable changes to the Robot SF project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), 
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* Added a **read-only readiness preflight for the compact CARLA native↔aligned parity bundle**
  (#1510). New module `robot_sf_carla_bridge/parity_bundle_preflight.py` exposes
  `check_parity_bundle_readiness` (and the pure `evaluate_payload_metadata`), which checks — per
  candidate scenario, fail-closed — that each T0 export manifest reads/validates, every referenced
  export payload exists and is schema-valid, the scenario fixture certificate is present with an
  eligible status, provenance carries a `robot_sf_commit`, and the recorded `trajectory_fields` are
  non-empty, plus that the intended output directory is path-safe (never created). It composes the
  canonical export readers (`resolve_export_manifest_payload_paths`, `read_export_payload`) and the
  canonical parity metric list (`parity.DEFAULT_PARITY_METRICS`). CLI
  `robot-sf-carla-parity-bundle-preflight` prints a JSON report and exits nonzero when not ready.
  This is the **agent-executable local slice only**: it does **not** import CARLA, run a simulator,
  execute a replay, or assert metric parity — a `ready` verdict means the static prerequisites are
  staged, not that native↔aligned parity holds (runtime eligibility must be confirmed on a capable
  CARLA host; see `docs/context/issue_1511_carla_native_aligned_parity_claim_boundary.md`).
* Added a **metadata-only staging-manifest preflight** for real AMV command-response actuation
  traces routed through the `amv-calibration` external-data path (#2415). New schema
  `robot_sf/research/schemas/amv_command_response_trace_manifest.v1.json` and checker
  `robot_sf/research/amv_command_response_trace_manifest.py` validate, per candidate trace bundle,
  the provenance/license, the command/response/timing channels the bundle would expose, and the
  declared calibration targets — fail-closed against the canonical synthetic-actuation envelope
  vocabulary (`robot_sf.benchmark.synthetic_actuation.actuation_variability_fields()`) so the
  manifest cannot drift from what calibration can consume. CLI
  `scripts/validation/check_amv_command_response_trace_manifest_issue_2415.py` prints a JSON report
  and (with `--probe-live-staging`) reconciles each trace's declared staging status against a live
  `manage_external_data.check_asset` presence probe. The shipped example manifest
  (`configs/research/amv_command_response_trace_manifest_issue_2415.yaml`) is
  `blocked-external-input` today, matching the maintainer decision on #2415 (2026-06-22) that no
  realistic real-data source is currently available. This is **manifest-contract only**: it does
  not ingest traces, run a calibration, or make a hardware-calibrated realism claim
  (`evidence_boundary = manifest_contract_only_no_trace_ingest_no_calibration_run_no_calibrated_claim`).
* Added an **opt-in, diagnostic-only closing-speed / time-to-collision (TTC) aware near-miss**
  surface (#3700). New module `robot_sf/benchmark/near_miss_ttc.py` exposes
  `near_miss_ttc_input_readiness` (a fail-closed validator of the timing/velocity inputs a TTC-aware
  near-miss requires: a finite positive `dt`, a `(T,2)` `robot_pos`/`robot_vel` with at least two
  frames, and a `(T,K,2)` `peds_pos`) and `compute_ttc_near_miss_diagnostic`, which counts steps
  whose minimum projected TTC falls below a threshold and reports the worst closing speed under
  distinct `near_miss_ttc__*` keys. It reuses the existing TTC convention from
  `time_to_collision_min` and the canonical pedestrian-velocity primitive. This is **additive and
  diagnostic-only**: it does **not** modify the canonical distance-based `near_misses` metric, wire
  anything into SNQI or any scoring path, calibrate the threshold (the default
  `DIAGNOSTIC_TTC_THRESHOLD_S` is an explicit uncalibrated placeholder, `decision-required` per
  #3700), or assert any safety result. The diagnostic fails closed — raising `NearMissTtcInputError`
  rather than returning zeros — when the timing/velocity inputs are missing or invalid. Tests cover
  fast-closing vs. opening synthetic trajectories, the fail-closed contract for missing timing
  fields, and that the canonical `near_misses` output is unchanged.
* Added a read-only diagnostic inventory / fail-closed preflight for **conflicting "canonical" SNQI
  weight sets** (#3723). New module `robot_sf/benchmark/snqi/weights_inventory.py` discovers every
  known SNQI weight source — the code default `recompute_snqi_weights("canonical")` plus the shipped
  JSON files under `model/` and `configs/benchmarks/` — records each set's dominant term and numeric
  scale (raw vs normalized), and reports provenance conflicts: two sources both claiming the
  "canonical" designation but yielding different weight *directions* (e.g. the collision-dominant
  code default vs the jerk-dominant `model/snqi_canonical_weights_v1.json`), raw-vs-normalized scale
  splits, and duplicate weights shipped under distinct labels. A new `inventory` subcommand on the
  SNQI CLI (`python -m robot_sf.benchmark.snqi.cli inventory [--json] [--no-fail-on-conflict]`) and
  the `preflight_snqi_weight_sets(strict=True)` API expose the report and **fail closed** (non-zero
  exit / `SNQIWeightProvenanceError`) when a blocking conflict is detected. This is provenance
  disambiguation only: it does **not** choose a canonical set, re-tune weights, change normalization
  (#3699), or alter SNQI scoring — picking the source of truth remains a maintainer decision. See
  `docs/snqi-weight-tools/weights_provenance.md`.
* Added a **diagnostic inventory** for the two incompatible collision/near-miss definitions
  (#3724). The benchmark metric (`robot_sf/benchmark/metrics.py`) classifies collision/near-miss
  with a radius-aware *clearance* rule, while the SNQI proxy (`robot_sf/gym_env/snqi_proxy.py`)
  and policy-search validation (`scripts/validation/policy_search_common.py`) use the *raw center
  distance* against `COLLISION_DIST=0.25` — so the same geometry is labeled differently (the
  clearance collision boundary sits at a center distance of ~1.4 m with default radii vs 0.25 m, a
  ~5× gap). New module `robot_sf/benchmark/collision_definition_inventory.py` classifies center
  distances under both regimes and reports where they diverge, and CLI
  `scripts/benchmark/collision_definition_inventory_report.py` prints/saves a preflight report with
  an optional `--fail-on-divergence` (fail-closed) exit. This is **diagnostic only**: it does not
  change any threshold, metric, proxy, or validation behavior, and does not choose a canonical
  definition (that remains `decision-required` on #3724).
* Added a **read-only heavy forecast-model family inventory / preflight** for the offline
  prediction study (#2845). New pure module `robot_sf/research/forecast_heavy_model_inventory.py`
  documents the candidate heavy predictor families (transformer, AgentFormer-like, CVAE,
  diffusion) with qualitative literature-derived planning estimates of compute cost, inference
  latency, uncertainty quality, and repository integration burden; probes that the
  offline-evaluation surfaces an experiment would touch are importable (the forecast
  metrics/calibration/conformal/dataset/batch/baseline surfaces; fail-closed on the required
  ones); and reports the minimum-offline-experiment prerequisites (a staged held-out dataset, a
  heavy-model→`ForecastBatch` adapter, a CPU runtime budget, the study report, plus external
  dependency/checkpoint decisions) as explicit blockers, rolled up into a `ready`/`blocked`
  minimum-experiment status. Thin CLI
  `scripts/research/check_forecast_heavy_model_inventory.py` (`--json`/`--list`) and study report
  `docs/context/forecast_heavy_model_study_2026-06-20.md`. Inventory slice only: trains no model,
  runs no inference, adds no dependency, runs no benchmark, and makes no model-quality claim
  (`evidence_tier` stays blocked → analysis_only).
* Added a metadata-only staging/preflight checker for external pedestrian-prior extraction (#2918).
  New module `robot_sf/benchmark/pedestrian_prior_extraction_manifest.py` exposes
  `check_pedestrian_prior_extraction_manifest`, which validates a
  `pedestrian_prior_extraction_manifest.v1` manifest (allowed external source type, the bounded
  prior parameters the run will emit — walking speed, crossing angle, density, interaction
  distance, stop/yield timing — provenance fields, and the authored-vs-dataset-backed separation)
  and reports missing parameters, provenance/separation blockers, and whether a dataset-backed
  prior claim is yet allowed. The checker **ingests no external data, stores no raw trajectories,
  and makes no calibrated- or representative-prior claim** (`evidence_boundary:
  prior_extraction_plan_only_no_calibrated_prior_claim`); a dataset-backed claim is gated behind
  accepted provenance and a source family with a registered external-data staging contract, while
  `blocked-external-input` (the default, matching issue #2918's external-data block) and
  `proxy-only` manifests cannot assert a prior, and a `proxy-only` manifest declaring a
  dataset-backed source is rejected as boundary conflation. The allowed source-type → external-data
  asset-id map is cross-checked against the canonical `scripts/tools/manage_external_data.py`
  registry. CLI: `scripts/tools/check_pedestrian_prior_extraction_manifest.py`; example manifest:
  `configs/research/pedestrian_prior_extraction_manifest_issue_2918_example.yaml`; context note:
  `docs/context/issue_2918_pedestrian_prior_extraction_preflight.md`.
* Added a metadata-only staging-contract checker for dataset-backed scenario priors (#3161). New
  module `robot_sf/research/scenario_prior_staging_contract.py` exposes
  `check_scenario_prior_staging_contract`, which validates a `scenario_prior_staging_contract.v1`
  contract (per-dataset provenance/license, the canonical scenario-prior distribution fields a
  dataset-backed prior would expose, and the explicit external-data blocker) for the Stanford Drone
  Dataset, SocNavBench ETH, and AMV candidates. Declared distribution fields are checked against the
  live `PARAMETER_GROUPS` vocabulary of the #2919 comparison harness so the contract cannot drift
  from what the comparison can compute, and a dataset declared `staged` is reconciled against a live
  `manage_external_data.check_asset` presence probe (fail-closed). The checker **ingests no dataset,
  stores no raw trajectories, runs no comparison, and makes no real-world realism claim**; with no
  dataset staged it reports `blocked-external-input`. Example contract
  `configs/research/scenario_prior_staging_contract_issue_3161.yaml`, CLI
  `scripts/analysis/check_scenario_prior_staging_contract_issue_3161.py`, context note
  `docs/context/issue_3161_scenario_prior_staging_contract.md`.
* Added a metadata-only measurement/intake-manifest checker for autonomous micromobility vehicle
  (AMV) actuation latency and rider-coupling response (#3283). New module
  `robot_sf/benchmark/actuation_latency_measurement_manifest.py` exposes
  `check_amv_actuation_latency_measurement_manifest`, which validates an
  `amv_actuation_latency_measurement_manifest.v1` manifest (declared sensor channels, per-channel
  sampling rate, time synchronization, provenance, and the synthetic-vs-measured separation)
  against the canonical command-response latency and rider-coupling quantity contract and reports
  missing channels, synchronization/provenance/separation blockers, and whether a measured-value
  claim is yet allowed. The checker **collects no data, fabricates nothing, and makes no
  measured-value or calibrated-actuation claim** (`evidence_boundary:
  measurement_intake_plan_only_no_measured_value_claim`); a `measured` claim is gated behind
  accepted provenance, while `blocked-external-input` (the default, matching issue #3283's
  external-data block) and `synthetic-only` manifests cannot assert measured values, and a
  `synthetic-only` manifest declaring a measured source is rejected as boundary conflation. It also
  proposes the latency and rider-coupling fields a future measured AMV actuation profile would
  expose, extending the synthetic actuation-envelope schema in
  `robot_sf/benchmark/synthetic_actuation.py` without promoting placeholders into calibration
  evidence. CLI: `scripts/tools/check_amv_actuation_latency_measurement_manifest.py`; example
  manifest: `configs/benchmarks/issue_3283_amv_actuation_latency_measurement_manifest_example.yaml`;
  protocol note: `docs/context/issue_3283_amv_actuation_latency_measurement_protocol.md`.
* Added a **diagnostic-only** inventory of SNQI per-term normalization status (#3699). New module
  `robot_sf/benchmark/snqi/normalization_inventory.py` and CLI
  `scripts/benchmark/snqi_normalization_inventory_report.py` enumerate each SNQI term's scaling
  regime and surface that `compute_snqi` mixes *raw, unbounded* penalty terms (`time`, `comfort`)
  with *baseline-normalized* `[0, 1]` terms (collisions, near-misses, force-exceed, jerk), which
  makes the weight coefficients non-comparable as relative priorities. The report flags the
  mixed-scale condition and any baseline-normalized term lacking median/p95 coverage, and can fail
  closed (`--fail-on-mixed-scale`, `--fail-on-missing-baseline`). It does **not** change the SNQI
  formula, weights, `normalize_metric`, or any emitted score, and does **not** choose between the
  normalize vs. clip-and-document remedies (that remains `decision-required` on #3699). An anti-drift
  test reconstructs the SNQI score from the inventory's term table and asserts it equals
  `compute_snqi` exactly.
* Added a fail-closed Package A readiness checker so a rank-stability / held-out-family transfer
  campaign can verify its input prerequisites before execution (#3078). New manifest
  `configs/benchmarks/issue_3078_package_a_readiness.yaml` declares the held-out-family scenario
  inputs, seed-plan metadata, and frozen-protocol entry points; new checker
  `scripts/validation/check_package_a_readiness.py` verifies every declared input exists, that the
  seed-plan metadata is explicit, and that the output location is disposable under `output/`,
  exiting non-zero with `status: not_ready` when any prerequisite is missing. This is a
  provenance/readiness gate only — it does not execute the benchmark, submit Slurm, or interpret
  ranks.
* Added a **static launch preflight** for crossing-conflict predictive retraining configs (#3214).
  New module `robot_sf/training/predictive_retrain_preflight.py` and CLI
  `scripts/validation/validate_predictive_retrain_preflight.py` validate a predictive training
  pipeline config (the kind consumed by `scripts/training/run_predictive_training_pipeline.py`)
  *before* any SLURM/GPU launch: config structure, data prerequisites (scenario matrix, hard-seed
  manifest, planner grid, and weighting spec) exist, base/hard-case feature-width compatibility (the
  checkpoint-lineage contract), the navigation gate kept separate from the trajectory gate
  (`max_val_ade`/`max_val_fde`), a present evaluation block, and a declared output root. The check is
  cheap and CPU-only — it fills the gap left by the pipeline's own guards, which only run mid-pipeline
  after expensive dataset collection. It does **not** collect data, train, submit Slurm, change
  augmentation semantics, or make any model-improvement claim; the actual weighted retraining run is
  owned by #3254. Text and `--json` reports; exit code 0 when valid, 2 when invalid.
* Added a metadata-only validation-contract checker for candidate real-world micromobility traces
  (#3278). New module `robot_sf/analysis_workbench/real_trace_validation_contract.py` exposes
  `check_real_trace_validation_contract`, which maps a candidate dataset descriptor
  (`real_trace_validation_contract.v1` schema) onto the existing trace-failure predicate input
  contract and reports, per predicate, whether the declared channels make it *validatable* or
  *blocked*, which required channels are missing, whether a directly observed ground-truth label
  exists, and the resulting limitation (e.g. `late_evasive_reaction` / `oscillatory_local_control`
  are computable from kinematics but usually have no directly observed label to cross-check). It
  also surfaces metadata, provenance/access, and missing-data blockers. The checker **does not
  ingest, copy, or read any external/private data and makes no real-world validation claim**
  (`evidence_boundary: contract_check_only_no_real_world_validation`); the committed example
  descriptor (`configs/benchmarks/issue_3278_real_trace_validation_contract_example.yaml`) is a
  placeholder with `access_status: blocked` because external data access is not yet accepted. CLI:
  `scripts/tools/check_real_trace_validation_contract.py`. Decision note:
  `docs/context/issue_3278_real_trace_validation_contract.md`. Tests cover complete, incompatible,
  and missing/placeholder-metadata descriptors.
* Added a **read-only pedestrian-model assumption inventory / preflight** for the proposed
  headed social-force (HSFM) + time-to-collision (TTC) predictive-force experiments (#3481).
  New pure module `robot_sf/research/ped_model_assumption_inventory.py` documents the current
  force-model assumptions the upgrade would change (no field-of-view attenuation on ped-ped
  repulsion, heading coupled to instantaneous velocity, Euclidean-distance repulsion with no
  TTC term), probes that the entry-point surfaces an HSFM/TTC experiment would touch are
  importable (the vendored Social Force core plus the `robot_sf/ped_npc` behavior/population
  surfaces; fail-closed if any required surface is missing), and reports the still-missing
  prerequisites (HSFM heading state, FoV weight, TTC term, narrow-passage and bottleneck
  fixtures, versioned parameters, external calibration data) as explicit blockers. A documented
  CLI (`scripts/research/check_ped_model_assumption_inventory.py`, `--json`/`--list`) renders the
  inventory. This is the **assumption-and-artifact inventory** slice only: it implements **no**
  force law, changes **no** scenario behavior, runs **no** benchmark, and makes **no** realism
  claim (the force-model upgrade itself stays `evidence_tier: idea`).
* Added a **dry-run Robot SF -> external-benchmark scenario converter** that emits a deterministic,
  schema-validated **intermediate representation (IR)** for Robot SF scenario-matrix entries (#3285).
  New pure module `robot_sf/benchmark/scenario_interop.py` exposes `convert_scenario_to_ir(scenario)`,
  which maps geometry, agent paths, start/goal states, timing, and environment semantics into a
  target-neutral IR (`robot_sf/benchmark/schemas/scenario_interop_ir.v1.json`), preserves source
  provenance (scenario id, source file, source fields, metadata), and reports every unmapped source
  field explicitly in `unsupported_fields` instead of silently dropping it. A documented dry-run CLI
  (`scripts/tools/convert_scenario_interop.py --matrix <file>`) prints the IR per scenario and a
  validity summary. This is the local, asset-free slice of the cross-benchmark interop work: it emits
  **no** SocNavBench/HuNavSim asset and makes **no** cross-benchmark validity or score-parity claim;
  producing real external assets remains blocked on #1456/#1498/#2414/#1134. See
  `docs/context/issue_3285_scenario_interop_converter.md`.
* Added a **readiness/preflight helper for closed-loop prediction Package C** (#3080).
  `scripts/tools/prediction_package_c_readiness.py` inventories the local prerequisites for the four
  Package C forecast arms — `no_forecast`, `cv`, `semantic_cv`, and `interaction_aware` — across the
  three coordination stages (open-loop forecast analysis #2915, observation-perturbation replay #2777,
  closed-loop forecast-risk coupling #2916). For each arm it reports the required configs and code
  entry points, the declared same-seed plan (`[111, 2868]` read from the #2915/#2916 configs), the
  declared output roots, and the named blockers, then classifies the arm as fail-closed
  `ready` / `blocked` / `missing`. Per the issue-audit on 2026-06-22, Package C is gated solely on
  #2916 producing a durable campaign result store, so the default status is `blocked` until a
  `--coupling-result-store` with a canonical `summary.json` is supplied; `blocked`/`missing` are never
  treated as success evidence. The helper inspects the repository only — it does **not** execute any
  benchmark campaign, alter predictor semantics, or claim forecast performance. Text and `--markdown`
  reports plus optional `--output-json`; exit code 0 only when every arm is `ready`, 1 otherwise. New
  fixture tests `tests/tools/test_prediction_package_c_readiness.py` cover the ready/blocked/missing
  states (synthetic trees) and assert the real repo is wired (fails closed to `blocked`, not
  `missing`).
* Added a **presence-only tournament-readiness helper** for the predictive hard-case breakthrough
  portfolio (#3215). `scripts/tools/predictive_tournament_readiness.py` inventories the local
  prerequisites (expected configs, harness scripts, and output path) for the three concurrent
  tournament arms — Selection (#3204), Authority (#3213), and Model (#3214) — plus the shared
  `predictive_hard_seeds_v1` benchmark protocol, and classifies each as `ready` or `blocked` with the
  missing paths named as blockers. The check is deliberately fail-closed and **never** asserts run
  authorization: `run_authorized` is always `False` and the standing `run_gates` (open child bets,
  maintainer-set overnight SLURM/GPU budget, Autonomous Usage Stop Guard) are reported so a
  "prerequisites ready" result is not mistaken for "authorized to launch". It does not submit Slurm
  jobs, run the tournament, rank arms, or edit any claim. Text and `--json` reports; exit code 0 when
  local prerequisites are ready, 1 when blocked.
* Added a read-only **re-export readiness preflight** for stale dissertation table bundles
  (`scripts/tools/reexport_readiness_preflight.py`, #3203). It composes the existing
  `scripts/tools/stale_artifact_detector.py` freshness classifier with a required-input availability
  check and reduces both signals to a single `fresh` / `stale` / `blocked` state: `fresh` when the
  payload is present and checksums match (no re-export needed), `stale` when a re-export is needed and
  all required inputs (campaign config, generation script, source-commit provenance) are present
  (re-export unblocked), and `blocked` when a re-export is needed but a required input is missing or
  the provenance cannot be reconstructed. This prevents a stale bundle from being silently cited as
  current evidence without first checking that the bounded campaign can be reproduced here. The tool
  never runs the campaign, regenerates a table, or edits dissertation claims.
* Added a machine-readable classic archetype density / tier index
  (`configs/scenarios/archetypes/classic_density_tier_index.yaml`) that documents the *current*
  per-archetype density-tier coverage, density bands, and pedestrian spawn semantics of the classic
  interaction archetypes (#3725). It clarifies the benchmark scenario denominator (the in-matrix
  graded total is **23 rows across 11 configs**, not "12 archetypes × 3 densities = 36") and the
  overloaded meaning of `simulation_config.ped_density`: for `spawn_mode: markers` configs
  (`classic_bottleneck`, `classic_realworld_bottleneck`) `ped_density: 0.0` is a placement-mode
  placeholder (pedestrians come from fixed markers, with `density_advisory:
  zero_baseline_route_spawn`), **not** an empty scene. The index is a *derived, documentation-only*
  artifact — it does not change scenario generation behavior, rename any field, or make tier coverage
  uniform. New test `tests/test_classic_archetype_density_index.py` re-derives the same facts from the
  `classic_*.yaml` configs and `classic_interactions.yaml` and fails closed on drift or missing
  config/tier coverage. The scenario zoo index links the new file for discoverability.
* Added a **presence-only publication-prerequisites preflight** for the v0.1 validation/falsification
  benchmark package (epic #2910). A declarative checklist
  (`configs/benchmarks/releases/benchmark_v0_1_publication_prerequisites.yaml`) enumerates the
  canonical prerequisite owners a v0.1 package must be built on — ODD/scenario contracts, scenario
  certification, benchmark/row claim metadata, ODD/hazard coverage matrix, the release protocol and
  pinned v0.1 release manifest, the seed schedule, the release checklist doc, and citation metadata.
  The companion checker `scripts/validation/check_benchmark_v0_1_publication_prerequisites.py`
  verifies every referenced path exists and fails closed (exit `1`) when a required prerequisite is
  missing. It is deliberately presence-only: it does **not** release, tag, upload artifacts, run a
  benchmark/falsification campaign, judge scenario certification, or declare the benchmark "ready" —
  every report carries an explicit `claim_boundary` to that effect. Run with
  `uv run python scripts/validation/check_benchmark_v0_1_publication_prerequisites.py`.
* Recorded metric-affecting run configuration in benchmark result provenance so result artifacts are
  self-describing (#3701). New pure module `robot_sf/benchmark/run_config_provenance.py` exposes
  `metric_affecting_run_config(config)`, which serializes the two run-config toggles that change *what*
  the reported safety metrics mean — LiDAR `scan_noise` (noisy default `[0.005, 0.002]` vs deterministic
  `[0.0, 0.0]`; see `robot_sf/sensor/range_sensor.py`) and the collision-handling regime
  (`terminate_on_contact` vs `bounce_back`; the robot benchmark env terminates on collision per
  `RobotState.is_terminal`). The map-runner now embeds this block under
  `provenance.config_identity.metric_affecting_config` in every batch summary
  (`robot_sf/benchmark/map_runner.py`), derived once per batch via the fail-soft
  `representative_metric_affecting_config` helper, so two result sets can be checked for comparability
  without out-of-band knowledge of each run's config. The helper is descriptive provenance only — it
  does not redefine metrics, rerun campaigns, or promote benchmark claims, and degrades to a
  `status: not_available` block rather than ever breaking a run.
* Added an **opt-in factorial-ablation manifest + checker** for the merged planner-agnostic safety
  wrapper (#3501). `robot_sf/benchmark/safety_wrapper_ablation_manifest.py` enumerates the
  `planner × {wrapper off, wrapper on}` ablation cells from a tracked config
  (`configs/research/safety_wrapper_ablation_v1.yaml`) and **checks** the design so later runs can
  compare the wrapper on/off consistently: exactly the two `{wrapper_off, wrapper_on}` arms with one
  baseline (off), the off arm disabled and the on arm enabled, the on-arm thresholds validated as a
  real `SafetyWrapperConfig` (predeclared, no per-planner tuning), every planner present in both
  arms, and paired seeds shared identically across arms. The manifest records provenance fields
  (`schema_version`, `safety_wrapper_schema`, `config_path`, `git_head`, `claim_boundary`,
  `evidence_status`) and a `factorial_check` summary; a thin runner
  (`scripts/benchmark/build_safety_wrapper_ablation_manifest.py --dry-run`) writes it. This is a
  **dry-run design contract only** — it binds no runtime objects, runs no benchmark episodes, tunes
  no thresholds, and makes no mitigation-effectiveness claim (the factorial campaign run is the
  deferred downstream follow-up tracked by #3501).
* Added a fail-closed readiness check for scenario-horizon Results evidence (#3266). New module
  `robot_sf/benchmark/scenario_horizon_readiness.py` and CLI
  `scripts/validation/check_scenario_horizon_results_readiness.py` read a re-exported scenario-horizon
  campaign artifact (the dissertation `campaign_table.md` from PR #3263 / issue #3203, or an equivalent
  campaign-summary JSON) and classify it as `valid`, `diagnostic_only`, or `blocked`. The check is
  diagnostic-only — it never reruns a campaign or promotes evidence — and fails closed: a missing or
  unparseable artifact is `blocked`, any non-benchmark-success planner row (e.g. the PPO
  `partial-failure` recorded for #3266) caps the verdict at `diagnostic_only`, and an unasserted SNQI
  contract status keeps the evidence diagnostic-only rather than assuming it passed. Per-row status
  normalization reuses the canonical `fallback_policy.classify_planner_row_status` owner. Run against
  the real #3203 bundle, the check reports `diagnostic_only` (exit 2), confirming those tables remain
  diagnostic/provenance evidence only.

### Fixed

* Fixed **`scripts/dev/pr_ready_check.sh` mishandling a missing `BASE_REF`** on fresh checkouts
  (#3702). When the default `origin/main` (or any configured `BASE_REF`) was not present locally, the
  `git diff "$BASE_REF...HEAD"` comparison emitted a raw `fatal: ambiguous argument` error; because the
  command runs inside a process substitution, `set -e` silently swallowed the failure and the script
  proceeded with an empty changed-file set while still passing the unresolved ref to the downstream
  coverage, perf-evidence, and freshness checks. The script now resolves `BASE_REF` once up front via a
  new `resolve_base_ref` helper: a resolvable ref is used unchanged, a remote-tracking ref (e.g.
  `origin/main`) triggers a best-effort `git fetch`, and an otherwise-unresolvable ref falls back to
  `HEAD` with a clear message instead of crashing. Regression coverage lives in
  `tests/dev/test_pr_ready_preflight.py`. This is a developer-tooling fix only — it changes no test
  lane coverage semantics, coverage thresholds, or CI policy.
* Fixed **OSM SVG `viewBox` parsing failing on comma-separated values** in
  `fast-pysf/pysocialforce/map_osm_converter.py` (#3708). The converter parsed the `viewBox`
  attribute with bare `str.split()`, which only handles whitespace separators, so SVG exports using
  the equally valid comma-delimited form (e.g. `"0,0,488.48,458.33"`) raised
  `ValueError: could not convert string to float`. A new `parse_viewbox` helper splits on any run of
  commas and/or whitespace per the SVG spec, and both call sites (`extract_buildings_as_obstacle` and
  `add_scale_bar_to_root`) now use it. Whitespace-separated viewBoxes are unaffected.
* Fixed the PPO baseline adapter **crashing on partial dict observations** instead of backfilling
  missing keys (#3704). In `robot_sf/baselines/ppo.py`, `_align_model_obs_dict` previously raised
  `ValueError: Missing required dict observation keys` whenever a runner emitted a subset of the keys
  a checkpoint's `Dict` observation space declares (after flattening, prefix-expansion, and alias
  resolution); this failed all 144/144 serial-worker jobs in the #3203 campaign. Missing keys are now
  backfilled with an in-bounds default derived from the target subspace via the new
  `_default_for_space` helper: nested `Dict` subspaces recurse so each declared leaf is materialized,
  and leaf `Box` subspaces yield zeros **clipped to the declared low/high bounds** so a non-zero
  lower bound (e.g. counts/radii) stays in range. The backfill is logged at debug level so the
  substitution stays visible. Genuine shape mismatches and non-Dict payloads for nested keys still
  raise. This keeps PPO evaluation running on partial observations; heavily backfilled runs should be
  treated as degraded rather than faithful evidence. Gym wrapper observation spaces are unchanged.
* Fixed the **SAC baseline velocity action space ignoring `action_semantics`** when converting model
  output to benchmark commands (#3705). In `robot_sf/baselines/sac.py`, `_action_vec_to_dict` always
  scaled the `velocity` (`vx`, `vy`) output vector to `v_max`, even when `action_semantics="delta"`.
  Deltas are increments the env later accumulates (`new_v = old_v + delta`) and are already bounded by
  the SAC action space, so scaling them toward `v_max` distorted the accumulated trajectory. The
  velocity branch now mirrors the existing unicycle contract: the speed cap is applied only for the
  absolute-velocity contract, while delta increments pass through unchanged. Behavior for the
  `unicycle` action space (the current benchmark config default) is unchanged; the default
  `action_semantics="delta"` is intentionally preserved per the canonical training setting.
* Fixed `_spaces_compatible` **rejecting wrapper- or subclass-derived `Tuple` spaces** under strict
  top-level type checking (#3709). In `robot_sf/training/scenario_sampling.py`, the helper opened with
  `type(base) is not type(other)`, so a `gymnasium.spaces.Tuple` and a Tuple subclass (as produced by
  some vectorized-env wrappers/adapters) were reported incompatible even when their element spaces
  matched. Two `Tuple`-compatible spaces are now compared **structurally on their child spaces** while
  every other space family keeps the strict type check, so `Box`/`Dict`/`Discrete` comparison
  semantics are unchanged. The recursive child checks (length, per-element compatibility, Tuple
  vs. non-Tuple) still reject genuine mismatches.
* Fixed `SNQIWeights.load` / `from_dict` **raising unhandled `KeyError`s on malformed config**
  (#3710). In `robot_sf/benchmark/snqi/types.py`, reconstructing an `SNQIWeights` from JSON used bare
  `data["key"]` access (so a missing provenance field surfaced as an opaque `KeyError`) and never
  validated the weights mapping values. Loading now **fails closed with structured diagnostics**:
  non-mapping input, missing or non-string required metadata fields, a non-mapping `bootstrap_params`,
  a non-list `components`, and weight values that are non-numeric, non-finite, or negative each raise a
  descriptive `ValueError` naming the offending field and the source path. Invalid JSON is also
  wrapped in a `ValueError` that includes the file path. This is bounded to input validation — SNQI
  metric definitions, weights, normalization, and benchmark results are unchanged.
* Fixed differential-drive velocity updates **ignoring timestep scaling** (#3711). In
  `DifferentialDriveMotion._robot_velocity` the commanded acceleration (clipped to
  `max_linear_accel` / `max_angular_accel`, the follow-up to #3666/#3689) was applied directly as a
  per-step velocity delta without multiplying by the simulation timestep `d_t`. At the default
  `d_t = 0.1` this permitted 10× the labeled acceleration per step and made the dynamics
  timestep-dependent (smaller timesteps inflated the effective acceleration — the unstable velocity
  spikes reported in the issue). The action is now integrated over `d_t`
  (`max_delta = max_accel * d_t`), making the velocity update timestep-invariant and consistent with
  the `bicycle_drive` model. All pre-existing differential-drive tests use `d_t = 1.0`, where
  `accel * d_t == accel`, so their behavior is unchanged; new regression tests cover `d_t != 1.0`.
  **Behavioral note:** at the simulator's default `d_t = 0.1` this reduces the realized per-step
  velocity change for a given action by 10×; differential-drive controllers tuned against the
  previous (timestep-dependent) integration may need re-tuning. Controller re-tuning and benchmark
  re-validation are out of scope for this fix.
* Fixed the HEIGHT planner adapter's lidar raycasting **ignoring dynamic pedestrians** (#3629).
  `CrowdNavHeightAdapter._raycast_obstacles` intersected each ray only against cached static obstacle
  segments, so the HEIGHT policy's lidar channel was blind to moving pedestrians (they were used for the
  human spatial-edge tensor but never fed into the raycast). Rays are now also intersected against
  pedestrian discs (reusing `circle_line_intersection_distance` from `robot_sf/sensor/range_sensor.py`,
  the same primitive the live env's range sensor uses); the nearest of {static, pedestrian, sensor
  range} wins per ray, and the disc radius is read from the observation when present (default 0.3 m).
  Backward-compatible: an empty pedestrian set reproduces the previous static-only behavior exactly.
* Fixed collision **undercounting** in `summarize_collision_metrics` (#3627): the aggregator read the
  sampled `metrics.collisions` value even when it was a finite `0.0`, only falling back to the exact
  `outcome.collision_event` flag when the metric key was entirely absent — so an episode whose exact
  environment collision fired but whose sampled metric missed the contact was silently aggregated as
  **zero** collisions (a validation-integrity hole independent of the per-episode write-time and
  `EpisodeEventLedger.v1` guards, since this aggregator also reads resumed/loaded JSONL records).
  `_episode_collision_value` now fails closed: when the exact collision flag fired and the sampled
  value is `<= 0`, the count is floored to `1` (sourced as `episode.outcome.collision_event`). No
  inflation — no-collision episodes stay `0` and a larger sampled count is never reduced.
* Fixed a `scripts/dev/check_skills.py` false-positive path validation (#3623): backticked prose
  placeholders such as `SLURM/data-gated` (in a SKILL.md) were validated as repo paths and failed the
  preflight, because `SLURM/` is a real top-level dir. The broken-path check now only flags a
  non-resolving token when it *looks like* a path (has a file extension or nests ≥2 segments below its
  prefix), so single-segment extension-less placeholders are treated as prose — while genuinely broken
  path references (e.g. `docs/does_not_exist.md`, `SLURM/missing/template.sl`) are still caught. The
  offending SKILL.md was left untouched; the fix is in the validator.
* Added a **fail-closed route-clearance guard** to the camera-ready benchmark (#3628). The
  originally-reported `0.0 m` centerline clearances in `classic_merging_*` / `classic_station_platform_*`
  were already re-routed to positive margins on `main` (now +0.56 m and +2.5 m), but the preflight
  only emitted *informational* warnings, so a route with a **negative** margin (centerline-to-obstacle
  distance < robot radius = guaranteed collision) would still run silently. `prepare_campaign_preflight`
  / `run_campaign` now raise `RouteClearanceError` on any `min_clearance_margin_m < 0` (certification
  cannot excuse hard geometric infeasibility; tangent/positive-narrow geometry stays a warning), so
  both `--mode preflight` and `--mode run` fail closed. Also corrected the stale
  `route_clearance_certifications_v1.yaml` entries for the 3 repaired scenarios (`negative clearance` →
  `repaired_geometry`). User-facing: the benchmark can no longer silently run a geometrically-impossible
  route.
* **`stream_gap` was blind in the real benchmark runner** (found while promoting the #3556 belief-mode safety contrast to `map_runner`). `StreamGapPlannerAdapter._extract_state` read the nested SOCNAV observation (`obs["robot"]`, `obs["pedestrians"]`) but `map_runner` feeds a flat observation (`robot_position`, `pedestrians_positions`, `goal_current`), so the planner extracted `robot=[0,0]`, `n_peds=0` and drove blind every episode. `_extract_state` now accepts both the nested and flat observation formats (backward-compatible; the ScenarioBelief harness and 24 existing tests still pass), and `robot_sf/benchmark/scenario_belief_policy_hook.py` reads the flat observation and writes the uncertainty sidecar to `pedestrians_uncertainty` where the fixed extractor reads it. After the fix the planner engages pedestrians and the belief modes differentiate in the real runner.

### Added

* Added a **fail-closed archive-readiness checker** for the adversarial proposal-vs-random study
  (#3275). Before the proposal runner consumes a *real* certified failure archive, the archive must
  carry the fields the downstream disjoint split, overlap provenance, candidate certification, and
  null tests depend on — otherwise those steps cannot be computed. The new pure checker
  (`assess_archive_readiness` / `assess_archive_file_readiness` in
  [`robot_sf/adversarial/disjoint_evaluation.py`](robot_sf/adversarial/disjoint_evaluation.py),
  composing the existing `scenario_family_key` / `disjoint_family_split`) reports a precise
  `ready` / `not_ready` verdict over schema, per-entry `archive_id` / `candidate.scenario_seed` /
  `failure_attribution` presence, derivable scenario families, and whether a disjoint scenario-family
  split with non-empty fit/eval sides is even possible. Unlike the runner's loader, it **never falls
  back to a synthetic fixture**: a missing, empty, unreadable, malformed, or under-populated archive
  is reported `ready=False` with the blocking reasons. A thin CLI
  [`scripts/tools/check_adversarial_archive_readiness.py`](scripts/tools/check_adversarial_archive_readiness.py)
  prints the JSON report and exits non-zero when not ready, so it is safe to use as an input gate.
  This is a readiness/input-hygiene slice only — it runs no proposal-model evaluation, fabricates no
  archive inputs, and makes no held-out-yield or benchmark claim.
* Added a **plan-level preflight for the paper-grade reactivity-vs-replay rank study** (#3637,
  split from #3573). The new pure checker
  [`robot_sf/benchmark/reactivity_replay_preflight.py`](robot_sf/benchmark/reactivity_replay_preflight.py)
  (`reactivity_replay_rank_study_preflight.v1`) verifies a proposed run plan's **preconditions**
  before any compute is spent: ≥3 planners, exactly the reactive/replay arms with **paired** seeds
  (common random numbers), a seed budget at/above the S20 rank-stability floor and above the #3573
  diagnostic matrix, a contrast-registering horizon, and that the **replay limitation** is stated
  (`'replay' = robot→pedestrian force-off in a live sim, NOT trajectory playback`). The limitation
  constants now live in the canonical owner `robot_sf/benchmark/reactivity_ablation.py`
  (`REPLAY_LIMITATION`, `REPLAY_IS_TRAJECTORY_PLAYBACK`, `REACTIVITY_ARMS`) and are imported, not
  duplicated. Thin CPU-only CLI
  [`scripts/benchmark/preflight_reactivity_replay_rank_study_issue_3637.py`](scripts/benchmark/preflight_reactivity_replay_rank_study_issue_3637.py)
  emits a `ready`/`blocked` manifest from the launch packet
  [`configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml`](configs/benchmarks/reactivity_replay_rank_study_issue_3637_launch_packet.yaml).
  This is an evidence-control layer only: it does **not** run the benchmark, interpret rank
  stability, or make any paper-facing claim (sufficiency stays with
  `scripts/tools/seed_sufficiency_gate.py` post-run).
* Added a **context-note archival-sweep planner** (#3190):
  [`scripts/tools/plan_context_note_archival.py`](scripts/tools/plan_context_note_archival.py)
  consumes the existing freshness checker plus `docs/context/catalog.yaml` and proposes which notes
  should move to `docs/context/archive/`. It classifies **high-confidence** candidates (catalog rows
  marked `status: superseded` that already name a valid, existing replacement) and **review**
  candidates (notes flagged stale by `check_context_note_freshness`), computes per-note archive
  targets, and detects blocking conflicts (basename collisions or a pre-existing target). The planner
  is **plan-only** — it never moves, writes, or deletes a note (`--json-output` emits a
  maintainer-reviewable plan) — so the actual archival sweep stays a separate, review-gated PR per
  issue #3190. It reuses (does not duplicate) the checker's catalog parsing and stale-note rules.
* Added an **opt-in three-wheeled rollover proxy** runtime hook in `RobotEnv.step()` (#3479). When
  `rollover_proxy_enabled` is set, the env feeds the executed robot `current_speed`
  `(linear_velocity, yaw_rate)` into the existing `rollover_proxy.v1` closed-form stability margin,
  surfaces rollover telemetry in step info / telemetry-analyzer metrics, emits a `ROLLOVER_CRITICAL`
  termination reason, terminates the step, and applies a configured non-positive
  `rollover_proxy_penalty` when the internal proxy trips. **Disabled by default**, so existing
  benchmark/training runs keep identical episode semantics; config validation rejects a non-typed
  `rollover_proxy_params` and a non-finite or positive penalty. This remains an explicit
  **internal non-hardware proxy** (governance gate #2416/#2417), not a hardware-calibrated AMV
  stability claim. The `(linear_velocity, yaw_rate)` assumption holds for differential/holonomic
  drives but not bicycle drive (tracked follow-up #3683).
* Added a regression guard for **pygame-free headless execution** (#3631). A subprocess-probe test
  (`tests/test_pygame_headless.py::test_headless_step_does_not_import_pygame`) builds and **steps** the
  env under forced-headless drivers and asserts `pygame` is never imported and no `sim_ui` is created —
  extending the no-pygame contract from env *construction* (already covered) through a full step, which
  the issue's reproduction specifically targets. The headless path was verified already pygame-free, so
  this is a test-only guard (a negative probe with an injected `import pygame` flips it red, confirming
  it is not vacuous).
* Added a **performance-PR evidence contract** that fails PR readiness when a `perf`-typed
  conventional-commit change (the #3611 → #3613 failure mode: a `perf(planner): cache ...` whose
  claimed speed-up targeted the wrong layer and had to be reverted) lacks a `## Performance
  Evidence` PR-body section with concrete before/after runtime, a representative command, a
  rollback criterion, and — when caching is claimed — a cache-hit/reuse counter. The trigger is the
  local `perf(...)` commit subject (no GitHub label needed), so it runs identically in
  `pr_ready_check.sh` (fail-closed in final mode, advisory in interim) and CI. Check:
  [`scripts/dev/check_perf_evidence.py`](scripts/dev/check_perf_evidence.py); template section in
  [`.github/PULL_REQUEST_TEMPLATE/pr_default.md`](.github/PULL_REQUEST_TEMPLATE/pr_default.md);
  tests: [`tests/dev/test_check_perf_evidence.py`](tests/dev/test_check_perf_evidence.py).
* Added a **fail-closed planner observation-view integrity guard** in the benchmark runner (#3634,
  the runtime guard deferred from #3568). New `robot_sf/benchmark/map_runner_view_integrity.py`
  (`evaluate_effective_view_integrity` + `DegeneratePlannerViewError`, reason `degenerate_planner_view`)
  is checked on the first step in `map_runner_episode.py`: when the observation handed to the planner
  contains pedestrians (`observation_ped_count > 0`) but the planner's own extractor returns zero — and
  the planner is not pedestrian-blind-by-design (per its `observation_spec.inputs`) — the runner raises
  **before any metrics are recorded**, instead of silently emitting collision "results" from a blind
  planner (the `stream_gap` failure class fixed in #3567). Ground truth is the *observation's own*
  pedestrian count, so visibility-masked/occluded or genuinely-empty scenarios do not false-trip. The
  passing diagnostic is stored at `record["integrity"]["effective_view"]`. 20 guard tests + 145
  regression + 13 end-to-end episode tests pass.
* Began the #3463 corrective engineering for the topology near-parity selector lane (diagnostic-tier,
  no benchmark claim). `TopologyGuidedLocalPolicyConfig` gains two current-behavior-preserving knobs:
  `primary_route_progress_gate_use_monotone_accounting` (default `False`) hardens primary-route
  progress accounting against premature stalls — a single A\* re-plan that transiently raises
  `route_remaining` no longer masks real progress (uses `max_sample − newest` instead of
  `oldest − newest`); and `primary_route_progress_gate_min_samples` (default `2`, the historical
  hardcoded value) makes the progress-gate sample threshold explicit. Progress-gate config is now
  fail-closed (non-finite threshold or `< 2` / non-integer min-samples raises `ValueError`), and the
  accounting mode + sample threshold are surfaced in the `topology_reuse_penalty` diagnostics. Deferred
  to follow-ups: command-arbitration-strength tuning, a registered candidate config + paired benchmark
  diagnostic for the monotone variant, and any benchmark promotion (#3463).
* Added a pre-commit lint that fails when a file under `configs/**` contains an absolute
  home-dir path (`/home/`, `/Users/`, `/root/`), which is non-portable for other contributors
  and automated runners. Intentional absolute paths (e.g. private-ops SLURM routing targets
  that live outside this repo) are exempted by annotating the line with `# allow-abs-path: <reason>`.
  Hook: [`hooks/check_config_abs_paths.py`](hooks/check_config_abs_paths.py); tests:
  [`tests/integration/test_check_config_abs_paths.py`](tests/integration/test_check_config_abs_paths.py) (#3605).
* Ran the deferred #3558 **`stream_gap` gate-threshold calibration sweep**:
  [`scripts/validation/run_stream_gap_gate_threshold_sweep_issue_3558.py`](scripts/validation/run_stream_gap_gate_threshold_sweep_issue_3558.py)
  reuses the #3471 episode harness (now accepting optional per-run gate-threshold overrides;
  backward-compatible, existing #3471 tests still pass) to roll the `uncertain_dropped` mode across a
  grid of `uncertainty_min_existence_probability` thresholds and feeds the per-setting safety
  aggregates + conservative-retention baseline into the merged `calibrate_stream_gap_gate` decision
  layer. Diagnostic-tier result on the controlled crossing scenario: a **safe region exists** only at
  thresholds permissive enough (≤ the degraded existence 0.2) to *retain* the corridor agent — every
  setting that actually drops it (≥0.3, including the **0.5 production default**) is `less_safe`,
  confirming the #3471 finding that the default gate is unsafe. The sweep exercises only the existence
  axis (the scenario degrades existence alone) and does not change the production default (#3558).
* Added a pre-commit lint that catches issue-provenance mismatches in cloned training configs:
  when a config under `configs/training/**` has identity fields (`policy_id` / tracking `group` /
  `tags` / `campaign_id`) that agree on a single issue number, its leading header comment must not
  name a *different* issue (the #3570 pattern, where a header read "Issue-1024" while every identity
  field targeted issue-3068). Headers that also cite a source/leader issue alongside the canonical
  one pass; an intentional mismatch can be annotated with `# allow-provenance-mismatch: <reason>`.
  Hook: [`hooks/check_config_provenance.py`](hooks/check_config_provenance.py); tests:
  [`tests/integration/test_check_config_provenance.py`](tests/integration/test_check_config_provenance.py) (#3606).
* Ran the deferred #3573 **reactive-vs-replay pedestrian-reactivity ablation** through the real
  runner. Added the open-loop replay (non-reactive) pedestrian mode as a scenario-driven toggle:
  `build_env_config` reads `peds_have_robot_repulsion` and sets `sim_config.prf_config.is_active`
  (backward-compatible; default reactive). New campaign
  [`scripts/benchmark/run_reactivity_ablation_campaign_issue_3573.py`](scripts/benchmark/run_reactivity_ablation_campaign_issue_3573.py)
  runs each planner under reactive vs replay over identical scenarios + seeds (common random numbers)
  via `map_runner.run_map_batch` and feeds the per-planner contrast into the merged
  `assess_reactivity_ablation` quantifier. Diagnostic-tier result on `classic_crossing_subset`
  (goal + orca, 4 seeds, horizon 150): disabling reactivity raised collisions (orca 0.125 → 0.50,
  goal 0.50 → 0.625) and cut separation — reactive (yielding) pedestrians flatter planners. Evidence +
  caveats (sign convention, horizon-sensitivity, small N):
  `docs/context/evidence/issue_3573_reactivity_ablation_2026-06-25/` (#3573).
* Classified the ORCA-residual progress-probe lane decision (#2445):
  [`scripts/analysis/classify_orca_residual_progress_probe_issue_2445.py`](scripts/analysis/classify_orca_residual_progress_probe_issue_2445.py)
  reads the existing v1 bounded smoke result (SLURM job 12913) and applies the #2408/PR #2420 stop
  rule deterministically. Decision: **`stop`** the current residual-BC lane shape — the v1 smoke
  reproduces the v0 failure pattern (`success_rate=0.0` + `timeout_low_progress`) and is itself
  `failed_closed` with all four required smoke-evidence fields null, either of which forbids
  continuation/nominal escalation. Reopening requires a named objective/dataset redesign. The
  classifier fails closed (missing/invalid artifact or absent required fields → `stop` naming the
  gap). Analysis-only over failed-closed evidence — not readiness evidence; decision doc
  [`docs/context/issue_2445_orca_residual_progress_probe_decision.md`](docs/context/issue_2445_orca_residual_progress_probe_decision.md)
  with a #1358/#1475 handoff note (#2445).
* Added the #3556 belief-mode safety campaign harness: [`scripts/benchmark/run_belief_mode_safety_campaign_issue_3556.py`](scripts/benchmark/run_belief_mode_safety_campaign_issue_3556.py) runs the `oracle` / `uncertain_retained` / `uncertain_dropped` contrast through the **real** `map_runner.run_map_batch` on a crossing scenario family + seed matrix and classifies the episode-level safety result (`revise` / `retention_dominates` / `inconclusive` / `inconclusive_oracle_unsafe`). Current result on `classic_crossing_subset` is **`inconclusive`**: the gate now acts (the dropped mode differs on worst-case clearance), but the oracle baseline collides every episode so there is no near-safe headroom to attribute a collision effect — the remaining #3556 gap is a near-safe crossing family. Evidence + reproduction: `docs/context/evidence/issue_3556_belief_mode_real_runner_2026-06-24/`.

### Added

* Audited the headline 7-planner benchmark comparison for the silent-blind-planner failure mode found in #3556 (`stream_gap` read nested observations while `map_runner` feeds flat ones, so it saw nothing yet emitted collision "results"). **Result: the headline comparison is clean** — its classical planners share `socnav_occupancy._socnav_fields`, which handles both nested and flat observations; `stream_gap` was a standalone exception (not in the headline) and is already fixed (#3567). Added a regression guard `tests/benchmark/test_planner_observation_contract.py` pinning that the shared extractor returns a non-degenerate view of the flat benchmark observation (sees the robot pose and real pedestrian count, not the `[0,0]`/empty blind default) and stays backward-compatible. Audit write-up: `docs/context/evidence/issue_3556_planner_obs_audit_2026-06-24/`.
### Changed

* Made the `pr-body-contracts` CI check **advisory** and fixed two false-positive sources in `scripts/dev/check_pr_followups.py`. (1) The domain-approval trigger is now **negation-aware**: a body that honestly *describes* an evidence boundary ("makes no paper-facing claim") no longer trips the gate meant for bodies that *make* such a claim. (2) The **Follow-Up Issues section is optional** — it is only required when a PR closes an issue while declaring residual scope, so self-contained PRs no longer need a boilerplate "Follow-Up Issues: none" section. (3) New `--advisory` flag (used by `.github/workflows/pr-body-contracts.yml`) reports violations as GitHub `::warning::` annotations and exits 0 instead of blocking the PR; re-promote to blocking by dropping the flag once body conventions settle. Covered by 5 new tests (48 total).

### Added

* Wired the ScenarioBelief drop-vs-retain safety contrast into the **real benchmark runner** (#3556, follows #3471/PR #3553). [`robot_sf/benchmark/scenario_belief_policy_hook.py`](robot_sf/benchmark/scenario_belief_policy_hook.py) builds a ScenarioBelief from each benchmark observation, degrades the existence confidence of out-of-field-of-view agents (keeping uncertainty rows 1:1 with the observation so the gate can act), projects through the production `project_scenario_belief_for_planner`, and merges the uncertainty sidecar into the planner observation; `BeliefModeStreamGapAdapter` wraps the real `StreamGapPlannerAdapter` (fail-closed: a projection failure omits the sidecar, so the planner keeps every agent). `map_runner`'s `stream_gap` branch now reads a `belief_mode` knob (`oracle`/`uncertain_retained`/`uncertain_dropped`) and sets the uncertainty gate (ON only for dropping). Mechanism proof (unit): retaining the uncertain agent matches oracle (both wait), while dropping it makes the planner commit — the #3471 result, now through the real adapter+gate. Ships [`configs/benchmarks/scenario_belief_drop_vs_retain_issue_3556.yaml`](configs/benchmarks/scenario_belief_drop_vs_retain_issue_3556.yaml) as the campaign launch packet. Integration + mechanism + map_runner wiring tested; the nominal-benchmark campaign run is the next step. The FOV uncertainty source is not a calibrated perception model — no paper-grade claim. Refs #3556.
* Added the episode-level ScenarioBelief uncertainty → planner-safety experiment (#3471): [`scripts/validation/run_scenario_belief_episode_safety_issue_3471.py`](scripts/validation/run_scenario_belief_episode_safety_issue_3471.py) is the bounded follow-up PR #3450 named — it rolls a controlled crossing scenario over time with the real `stream_gap` planner under three belief modes (`oracle`, `uncertain_retained` = gate-off conservative, `uncertain_dropped` = gate-on) sharing one ground truth. Finding (`revise`, diagnostic-tier, 12 seeds): retaining uncertain agents is *identical* to oracle (so representational uncertainty alone is harmless), while **dropping** them raises collisions 0.42→0.92 and unsafe-commit steps 45→246 — the uncertainty-dropping default should be revised/blocked for safety-relevant use. Controlled-scenario evidence, **not** the full benchmark env and **not** paper-grade (the oracle baseline itself collides ~42%; the result is the relative contrast). Config `configs/benchmarks/scenario_belief_episode_safety_issue_3471.yaml`; evidence `docs/context/evidence/issue_3471_scenario_belief_episode_safety_2026-06-24/`. Refs #3471.
* Made research leverage a first-class part of autonomous issue selection. `docs/project_prioritization.md` now defines a **Research-Leverage Interpretation** (claim-boundary/hypothesis → `Improvement`; headline-companion/unblocks-downstream → `Unlock Factor`; local-implementable vs SLURM/data-gated → `Success Probability`; deadline → `Time Criticality`) plus a **verify-before-scoring gate** (already-merged/superseded issues route to closure, not a high score). `gh-issue-priority-assessor` gains an **auto mode** that assesses **only empty `Priority Score` items** and never overwrites an existing priority, and `goal-autopilot` runs it as a new `prioritize` phase before `implement` so unassessed issues become leverage-aware scores without churning human-set ones. Mechanically enforced by a new `--only-empty` flag on [`scripts/tools/project_priority_score.py`](scripts/tools/project_priority_score.py) (skips any item that already has a score), with test coverage. `goal-issue-implementation` documents that its Project #5 queue order is now leverage-aware.
* Added a CPU-only pre-submit evidence-contract preflight (#1475): [`scripts/validation/preflight_evidence_contract.py`](scripts/validation/preflight_evidence_contract.py) checks that the current public commit will emit a named evidence contract's required fields BEFORE a SLURM job is submitted, so a job is never run only to fail closed on missing-field bookkeeping (the #1475 / job 12913 GPU-hour waste). It composes the canonical contract definition (a public alias `REQUIRED_ORCA_RESIDUAL_SMOKE_FIELDS` exported from `robot_sf/training/orca_residual_lineage_packet.py`, identity-asserted in a test so there is no second source of truth) and the production `_attach_orca_residual_smoke_evidence` builder, evaluated against a representative `GuardedPPOAdapter` row that mirrors the real on-main shape. Exit 0 = conforms (safe to submit), non-zero = block (names the missing fields). Seeded with the `orca_residual_smoke` contract; adding another is a one-entry registry change. The private-ops sbatch wrapper should call it before `sbatch` (doc: `docs/context/preflight_evidence_contract.md`). Refs #1475.
* Added the headline planner-comparison CI + rank-stability report harness (#3216): [`scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py`](scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py) reports per-cell confidence intervals (bootstrap/per-seed) and a Kendall-τ / rank-flip stability statistic across seed resamples for the 7×7 planner×scenario headline grid, REUSING the canonical owners (`seed_variance`, `fidelity_rank_stability.rank_planners`/`kendall_tau`, `canonical_table_export`) rather than reinventing the statistics. Fail-closed (degraded/fallback/not_available cells never count as success) and it **never self-certifies paper-grade**: insufficient seed budget returns `diagnostic`/`blocked_until_run` pending the S20/S30 SLURM run (via #1554) and claim-card review. Ships the SLURM run as a launch packet (real referenced configs + sha256). Coordinates with #1554 (per-seed bundle) and #3078 (package A) without duplication. The paper-grade run remains SLURM (Refs #3216).
* Added S20/S30 seed-budget archival tooling + SLURM launch packet (#1554): [`scripts/benchmark/build_s20_s30_seed_budget_bundle_issue_1554.py`](scripts/benchmark/build_s20_s30_seed_budget_bundle_issue_1554.py) builds a durable per-planner-by-seed comparison bundle (success/collision/near-miss/timeout/clearance-uncertainty/`time_to_goal_norm`) with bootstrap/per-seed uncertainty and a seed-resampling rank-flip analysis, REUSING the canonical stats (`seed_variance`, `snqi.bootstrap.bootstrap_stability`, `rank_metrics.kendall_tau`/`rank_order`) rather than reinventing them. Fail-closed (fallback/degraded/unavailable/failed/partial/not_available/diagnostic_only rows never count as success); when no real S20/S30 rows exist it emits `blocked_until_run` instead of fabricating a bundle. Ships the SLURM run as a launch packet (`configs/benchmarks/s20_s30_seed_budget_issue_1554_launch_packet.yaml`, real referenced configs + sha256) and a context note recording the claim-map gate and #1545 seed-budget methodology. The heavy S20/S30 h500 run remains SLURM (Refs #1554).

* Clarified token-efficient goal-thread startup rules for workflow-improvement turns: agents now
  explicitly park stale prior work after compaction or user pivots, scope meta-workflow requests to
  fresh docs-or-workflow worktrees, and classify high-priority SLURM work through the appropriate
  single-lane or capacity-aware submit workflow before burning parent-thread context.
* Documented how to obtain, place, and **share external datasets across git worktrees** in
  [`docs/external_data_setup.md`](docs/external_data_setup.md) (#1498): a per-asset obtain/place/
  validate quick-reference for `sdd`, `socnavbench-s3dis-eth`, and `socnavbench-control` (all
  license/agreement-gated — no scriptable download), plus a "Sharing external data across git
  worktrees" section. Staging paths resolve per-worktree, so the guide shows staging one physical
  copy at a machine-stable location and symlinking it into each worktree's git-ignored expected
  path, the `--source` validation option, and the planned `ROBOT_SF_EXTERNAL_DATA_ROOT` shared-root
  follow-up. Docs-only; actual licensed acquisition remains a manual user step (#1498).
* Added legacy PPO snapshot parity checks (#3469):
  [`scripts/validation/check_legacy_ppo_snapshot_parity.py`](scripts/validation/check_legacy_ppo_snapshot_parity.py)
  inventories supported BR-06 PPO registry checkpoints, verifies durable GitHub-release pointers,
  explicitly classifies root-local debug `.zip` snapshots as unsupported for durable compatibility,
  and provides an opt-in hydrated-checkpoint one-step Gymnasium smoke path. Focused tests cover the
  inventory, fail-closed durable-pointer checks, unsupported local snapshots, JSON CLI output, and a
  mocked model/factory step smoke (#3469).
* Added a lightweight PR body-contract workflow (#3472):
  [`.github/workflows/pr-body-contracts.yml`](.github/workflows/pr-body-contracts.yml)
  now validates live pull-request bodies with
  [`scripts/dev/check_pr_followups.py`](scripts/dev/check_pr_followups.py), requiring body input,
  open linked follow-up issues for declared residual work, domain-aware approval for
  evidence-validity-sensitive claims, and a human-written description for substantive
  source/configuration changes. The checker now accepts a changed-file list and rejects empty or
  bot-only substantive PR descriptions, with fixtures covering the #3414/#3415 empty-body pattern,
  the #3416 CodeRabbit-only pattern, and the #3449/#3450 evidence-sensitive missing-contract
  pattern (#3472).
* Added a bounded robot-influence-on-pedestrian-flow v0 slice (#3066):
  [`scripts/benchmark/run_robot_influence_flow_slice_issue_3066.py`](scripts/benchmark/run_robot_influence_flow_slice_issue_3066.py)
  runs a small same-seed campaign (reusing `robot_sf.benchmark.runner.run_batch`) comparing two
  rule-based robot policies (`social_force` vs `orca`, CPU-only) on a corridor/crossing subset of the
  issue-3059 suite (seeds 111/112/113, 12 episodes, 0 degraded) and measures whether the robot policy
  changes pedestrian-flow dynamics (near-vs-far pedestrian accel/turn-rate), reporting per-row
  success/safety + flow/comfort + denominator/uncertainty with fail-closed status and a robot-influence
  interpretation kept separate from nav performance. Result: `diagnostic` — 3 of 4 powered flow deltas
  exceed pooled seed variance, but tiny n, low density, an outlier-dominated corridor cell, and
  confounding with nav outcome keep it diagnostic-only. Smoke/diagnostic, not paper-grade; no
  real-world or sim-to-real claim. Tracked summary under
  `docs/context/evidence/issue_3066_robot_influence_flow_2026-06-23/`. Unblocked once #3062 merged
  (#3066).
* Added the PPO curriculum-learning launch packet (#3068): a pre-launch spec
  ([`configs/training/ppo_curriculum_issue_3068_launch_packet.yaml`](configs/training/ppo_curriculum_issue_3068_launch_packet.yaml),
  schema `ppo-curriculum-launch-packet.v1`) defining a 4-stage density/complexity curriculum over
  real env knobs, a matched fixed-difficulty PPO baseline comparator, seeds, run budget, stop rule,
  metrics, expected artifacts, validation command, and durable-artifact policy — with the three
  competing explanations (extra-budget vs curriculum; train-curve vs final benchmark; insufficient
  provenance) encoded as discriminating checks. All referenced configs exist with verified sha256
  checksums; the context doc
  [`docs/context/issue_3068_ppo_curriculum_launch_packet.md`](docs/context/issue_3068_ppo_curriculum_launch_packet.md)
  records the no-claim boundary and flags the per-stage scheduler wiring as a downstream
  prerequisite. Authoring-only and `evidence:proposal` — no training-result or benchmark claim; the
  long training run remains a separate SLURM issue (#3068).
* Automated stale state-label cleanup at issue closure (#3456): a new GitHub Action
  [`.github/workflows/strip-closed-state-labels.yml`](.github/workflows/strip-closed-state-labels.yml)
  triggers on `issues: closed` and strips any live `state:*` label (`state:ready`/`state:running`/
  `state:blocked`) from the just-closed issue — covering all close paths (manual, duplicate/wontfix,
  PR-merge) at one choke point, with least-privilege `issues: write`, a read-then-write CLOSED
  re-confirm, and no-op when no stale label is present. `scripts/dev/closed_state_label_hygiene.py`
  gains a `--fix` mode that reuses the same `LIVE_STATE_LABELS` (single source of truth), so the
  read-only detector remains the periodic CI/backstop while the event-driven Action removes the root
  cause of recurring manual scrubs (134 stale labels on 2026-06-18, ~66 more in 5 days). Documented
  in [`docs/context/issue_3098_closed_state_label_hygiene.md`](docs/context/issue_3098_closed_state_label_hygiene.md) (#3456).
* Added the Qwen-RobotNav feasibility assessment (#2952):
  [`docs/context/issue_2952_qwen_robotnav_assessment.md`](docs/context/issue_2952_qwen_robotnav_assessment.md)
  audits availability (weights/code/license/size/hardware) and compares Qwen-RobotNav's
  observation/action/tool interface against the real Robot SF local-navigation surfaces
  (`robot_sf/nav/`, `robot_sf/gym_env/`, differential-drive `step()` contract). Decision:
  **`blocked_asset_tracking`** — the report is real (arXiv 2606.18112) and on-target, but public
  weights, a concrete code repo, and a usable license are all **unverified** (with one source
  conflict explicitly flagged), and the interface gaps are large (camera-history obs vs LiDAR+ego;
  8-waypoint output vs single `(v, ω)`; agentic two-tier vs stateless numeric `step`). Every
  availability claim is tabulated as verified/unverified with its source; `evidence_tier: idea`,
  exploratory only — no integration, no weights downloaded. Indexed in
  [`docs/context/INDEX.md`](docs/context/INDEX.md) (#2952).
* Added a bounded sensor-noise / partial-observability robustness slice (#3067):
  [`scripts/benchmark/run_sensor_noise_robustness_slice_issue_3067.py`](scripts/benchmark/run_sensor_noise_robustness_slice_issue_3067.py)
  runs a same-seed clean / noisy / partial-observation comparison on a pedestrian-dominated fixture,
  reusing the observation-noise/perturbation wrappers, and reports clean-vs-perturbed deltas
  (min observed distance, observation continuity, near-field exposure, observation count) per row
  with fail-closed status — and the overall classification kept separate from nominal performance.
  Result: `diagnostic` (non-null — perturbations measurably change the observed state the policy
  would consume; partial observation drops actor observations/continuity/near-field), trace-derived,
  single fixture/seed. Diagnostic-only/smoke, not paper-grade; explicitly NOT a real-sensor
  certification or sim-to-real claim. Tracked summary under
  `docs/context/evidence/issue_3067_sensor_noise_robustness_2026-06-23/`. Unblocked once #3062 merged
  (#3067).

* Added a bounded, same-seed forecast-risk closed-loop coupling gate (#2916): a deterministic risk
  adapter [`robot_sf/benchmark/forecast_risk_adapter.py`](robot_sf/benchmark/forecast_risk_adapter.py)
  mapping a `ForecastBatch.v1` to a bounded `[0,1]` per-step risk signal (fail-closed on
  degraded/fallback/oracle batches), a config
  [`configs/research/forecast_risk_coupling_issue_2916.yaml`](configs/research/forecast_risk_coupling_issue_2916.yaml)
  with `no_forecast`/`cv_risk`/`semantic_risk`/`interaction_risk` rows pinned to identical
  seed/scenario, and a fixture-driven runner
  [`scripts/benchmark/run_forecast_risk_coupling_gate.py`](scripts/benchmark/run_forecast_risk_coupling_gate.py)
  that scores collision/near-miss, route progress, stop/yield timing, false-positive stopping,
  runtime, and SNQI per row and emits a `continue|revise|stop` verdict with the
  false-positive-stopping vs safety-benefit trade-off explicit. On the bundled single-pedestrian
  occluded-emergence fixture the gate returns `continue` (forecast risk removed the near-miss with 0
  false-positive stops at a throughput cost). Evidence is `diagnostic_only`/`stress`, not
  paper-grade, and promotes no learned predictor; the three forecast sources are indistinguishable on
  this single fixture (caveat recorded). Tracked summary under
  `docs/context/evidence/issue_2916_forecast_risk_coupling_2026-06-23/` (#2916).
* Added a bounded ScenarioBelief uncertainty diagnostic (#2546):
  [`scripts/analysis/run_scenario_belief_uncertainty_diagnostic_issue_2546.py`](scripts/analysis/run_scenario_belief_uncertainty_diagnostic_issue_2546.py)
  compares a fixed scenario across five belief conditions (oracle/deterministic, visibility-limited,
  covariance-inflated, class-probability uncertainty, existence-confidence degradation), projecting
  each through `project_scenario_belief_for_planner` for both a consuming planner (`stream_gap`) and
  an unsupported one (`predictive_planner_v2`). The run finds a difference (not null): every
  uncertain condition flips the consuming planner's single-step decision from WAIT to COMMIT versus
  oracle by dropping the uncertain corridor agent (at the visibility-projection step or the opt-in
  `stream_gap` uncertainty gate), while the unsupported planner fails closed
  (`unsupported_uncertainty_planner`). The WAIT→COMMIT flip is explicitly flagged as the expected
  consequence of dropping uncertain agents, **not** a safety improvement. Follow-up decision:
  `continue` (a runtime uncertainty producer + end-to-end stress run is the next bounded step).
  Evidence is `diagnostic_only`/`stress`, not paper-grade; tracked summary under
  `docs/context/evidence/issue_2546_scenario_belief_uncertainty_2026-06-23/` (#2546).
* Added an AMMV mechanism-divergence classification diagnostic (#2444):
  [`scripts/analysis/run_ammv_divergence_classification_issue_2444.py`](scripts/analysis/run_ammv_divergence_classification_issue_2444.py)
  runs direct `SocialForcePlanner` mechanism probes (bypassing the differential-drive benchmark
  adapter that produced #2434's zero-delta result) with an **AMMV-isolated control** -- the same
  AMMV-aware config with `ammv_aware_enabled` toggled off, so the paired delta reflects only the
  AMMV term. Result: `nonzero_divergence_found` -- the AMMV term activates (~2.64 N) and changes the
  same-seed trajectory (robot-state delta 0.21-0.78 m) under isolation, so #2434's zero result is an
  adapter-mode artifact, not a globally inactive mechanism. Includes a zero-divergence guard so
  identical (#2434-style) pairs cannot be reused as behavioral evidence in #2159/#2227 panels.
  `scripts/tools/run_ammv_social_force_pair_diagnostic.py::_run_mechanism_probe` gained a
  backward-compatible `default_config` parameter to support the isolated control. Evidence is
  `diagnostic_only`/`stress`, not paper-grade; tracked under
  `docs/context/evidence/issue_2444_ammv_divergence_2026-06-23/` (#2444).
* Made SDD scenario-prior staging reproducible (#2657): added
  [`scripts/data/stage_sdd_dataset_issue_2657.py`](scripts/data/stage_sdd_dataset_issue_2657.py)
  (subcommands `plan`/`validate`/`status`/`mode`/`download`) and a provenance manifest
  [`configs/data/sdd_staging_manifest.yaml`](configs/data/sdd_staging_manifest.yaml) (source URL,
  CC BY-NC-SA 3.0 license + readme pointer, version tag, expected files/size, checksum placeholders,
  `local_availability`). The tool **never auto-downloads**: a bare run only plans/reports (disk check,
  availability) and exits without network access; `download` requires explicit `--confirm-download` +
  y/N (or `--yes`), checks free disk vs expected size and fails closed if insufficient, stages into a
  git-ignored subfolder (`output/external_data/sdd`), and validates a SHA-256 tree checksum before
  marking `staged`. Wired a `proxy_schema_smoke` vs `dataset_backed_prior` gate into the scenario-prior
  generator (#2726) so a missing/unvalidated SDD can never be presented as dataset-backed evidence.
  Audit doc: [`docs/context/issue_2657_sdd_staging.md`](docs/context/issue_2657_sdd_staging.md) (#2657).
* Added the AMMV contrastive mechanism panel (partial #2227, AMV sub-target):
  [`scripts/analysis/build_ammv_mechanism_panel_issue_2227.py`](scripts/analysis/build_ammv_mechanism_panel_issue_2227.py)
  runs `SocialForcePlanner` twice on one fixed scenario (seed 42) toggling only `ammv_aware_enabled`
  in `configs/baselines/social_force_ammv_aware.yaml`, exports both arms as schema-validated
  `simulation_trace_export.v1` traces, and renders contrastive trajectory panels via
  `generate_trajectory_panel_bundle`. Applies #2444's finding (PR #3451) that the AMMV term is a
  genuine same-seed divergent pair: AMMV-off max force 0.0 vs AMMV-on 2.64, final-position
  divergence 0.58 m. Evidence is `diagnostic_only`/`stress`, not paper-grade — a planner-level
  mechanism difference, not a navigation-success or benchmark claim. Tracked panels + captions +
  provenance under `docs/context/evidence/issue_2227_ammv_mechanism_panel_2026-06-23/`. The
  static-recentering and topology-guided-recovery panel sub-targets of #2227 remain follow-up
  (Refs #2227).
* Completed #2227 mechanism panels with the static-recentering and topology-guided-recovery
  contrastive sub-targets:
  [`scripts/analysis/build_recenter_topology_panels_issue_2227.py`](scripts/analysis/build_recenter_topology_panels_issue_2227.py)
  runs each mechanism's planner twice on one fixed activation-capable scenario, toggling ONLY the
  mechanism flag (`static_recenter_enabled`; `topology_command_enabled`), exports schema-valid
  `simulation_trace_export.v1` traces, and renders contrastive trajectory panels with per-step
  activation diagnostics and command-source. Findings (real, diagnostic-only/stress): static
  recentering activates (recenter term positive from step 7) and the on-arm reaches the goal where
  the off-arm fails; the topology command activates but the on-arm *degrades* the outcome versus the
  off-arm — reported honestly as a single-row degradation, not a benefit (consistent with the prior
  #2752 "no useful topology alternative" diagnosis). Together with the AMV/AMMV panel sub-target this
  completes #2227. Tracked panels + captions + provenance under
  `docs/context/evidence/issue_2227_recenter_topology_panels_2026-06-23/` (#2227).
* Added a canonical [`docs/glossary.md`](docs/glossary.md) defining the project's acronyms and
  domain terms (VRU, AMV, AMMV, SNQI, occluder, the evidence ladder, and run modes) in plain
  language, and made "understandable" a first-class maintainer value. [`maintainer_values.md`](docs/maintainer_values.md#clarity)
  now carries a `## Clarity` rule, `AGENTS.md` scopes token-efficiency to agent-internal surfaces
  (clarity wins on human-facing surfaces), and the README, docs index, and CONTRIBUTING checklist
  link the glossary so jargon is defined on first use.
* Added a diagnostic-only multi-robot research smoke foothold (#3069): a small
  runnable scenario config (`configs/multi_robot/issue_3069_smoke.yaml`), a smoke
  runner (`scripts/validation/run_multi_robot_smoke_issue_3069.py`) that rolls out
  the multi-robot environment via `make_multi_robot_env` and reports per-agent
  inter-robot collision and goal-progress telemetry from the environment's native
  `info["agents"]` metadata, and a guard test
  (`tests/gym_env/test_multi_robot_smoke_issue_3069.py`). The report carries an
  explicit `claim_boundary: diagnostic_only` / `evidence_tier: smoke` and
  `multi_robot_benchmark_claim: false`; it proves only that the multi-robot path
  runs end-to-end and is not a fleet-scale, MAPPO, or paper-facing claim (#3069).
* Added durable issue-2904 forecast-risk eligibility fixtures
  (`tests/fixtures/benchmark/forecast_risk_eligibility/`) and the regenerated
  `ForecastCalibrationReport.v1` evidence bundle under
  `docs/context/evidence/issue_2904_forecast_risk_eligibility_2026-06-20/`. The pedestrian
  (`deployable_tracked`) and bicycle (`deployable_observation`) fixtures yield calibrated,
  risk-scoring-eligible reliability rows so the forecast-risk calibration filter's
  `calibration_filtered`, `actor_class_aware`, and `observation_tier_aware` modes are no longer
  blocked for lack of eligible rows: the calibration report recommendation flips from
  `wait`/blocked to `continue`/analysis-only and the filter's overall recommendation flips from
  `wait` to `diagnostic_only`. Fixtures and evidence are diagnostic/analysis-only and do not change
  `_risk_scoring_eligibility`, calibration thresholds, or planner risk coupling (#2904).

### Fixed

* Fixed the ORCA-residual BC smoke evidence contract (#1475): the per-decision `action_adaptation`
  payload from `GuardedPPOAdapter` now always carries a truthful `residual_clipped` signal — on the
  direct pass-through, guard-fallback, and default branches no bounded residual is applied, so
  `residual_clipped: False` is emitted (a real signal, not fabricated). This lets the smoke
  finalizer populate `residual_clipping_rate` (and, with the existing `shield_stats`,
  `guard_veto_rate`, `fallback_degraded_status`, `artifact_pointer_status`) instead of failing
  closed on `missing_required_smoke_evidence` — which is what blocked the #1475 SLURM smoke rerun
  (job 12913). Contract-completeness only: it does not make the BC lane succeed (the run may still
  be `timeout_low_progress`/`success_rate=0.0` per #2445), and no fail-closed gate is weakened
  (a genuinely degraded run still classifies degraded). (#1475)

* Fixed `scripts/tools/issue_template_audit.py` so it recognizes the repo's own YAML issue
  forms (`epic`/`execution-run`/`test-debt`/`blocked-external-artifact`) and agent-authored
  bodies instead of only the markdown templates. The audit previously matched a single strict
  12-section markdown contract and reported 8–12 false "missing section" gaps on nearly every
  issue filed from the leaner YAML forms. The auditor now (a) maps heading variants such as
  `Scope and non-goals`/`Out of scope`/`Scope / decision needed`, `Acceptance criteria`/
  `Acceptance / stop rule`/`Accept / revise / reject`, `Estimate`/`Estimate metadata`,
  `Validation commands`, and `Question / Goal`/`Background`/`Summary`/`What to build` to their
  canonical sections; (b) strips trailing parenthetical qualifiers (e.g.
  `Acceptance criteria (mirrors body)`); (c) credits `###` contract subheadings carried inside
  the appended `agent-exec-spec` block; and (d) audits leaner-template issues against a shared
  agent-ready core (`Goal / Problem`, `Scope`, `Definition of Done`, `Validation / Testing`)
  while bare/markdown-style bodies keep the strict full 12-section contract. A standalone effort
  estimate is intentionally not in the core: leaner/agent issues state effort under
  `Project Metadata` or defer it on blocked work, and the agent-exec-spec agents execute from
  carries no estimate. Across the 91 open issues this cut spurious "missing section" findings
  from 762 (≈87 issues, many with 8–12 false gaps) to ~30 (26 issues, genuinely-absent
  sections), with 65 issues now fully clean. The archetype-metadata audit and
  `audit_archetype_metadata` API are unchanged.
* Expanded the canonical issue-archetype taxonomy in
  `docs/context/issue_1512_issue_archetypes.md` and the validator
  (`scripts/tools/issue_template_audit.py`) to bless the work-type archetypes the issue
  automation already emits — `implementation`, `test`, `refactor`, `data` — and to accept the
  deprecated spellings `agent_task` (→ `implementation`) and the evidence tier `proposal`
  (→ `idea`) on read without rewriting issue bodies. This clears the validator's
  invalid-metadata findings on the four affected open issues (#2000, #3069, #3071, #3206)
  without per-issue edits, and wires `implementation`/`data` into
  `issue_archetype_sync.py`'s `SAFE_ARCHETYPE_LABEL_MAP` (`type:implementation`/`type:data`,
  both existing labels).
* Fixed `scripts/validation/validate_forecast_risk_calibration_filter.py` so its
  `_any_eligible_for_risk_scoring` gate recognizes the canonical `eligible_analysis_only` token
  emitted by `build_forecast_calibration_report`. The gate previously matched only the legacy
  `{"eligible", "calibrated"}` strings, which the report never emits, leaving the
  `calibration_filtered` mode permanently blocked for every real calibration report regardless of
  eligible rows. Legacy tokens are retained for backward compatibility (#2904).

### Changed

* Consolidated SDD staging into the canonical external-data subsystem (#3473): the SDD asset
  checksum policy, proxy-vs-dataset-backed availability gate, pinned-`expected_tree_sha256`
  validation, disk-space fail-closed check, and no-auto-download safety contract (originally added
  for #2657) now live in
  [`scripts/tools/manage_external_data.py`](scripts/tools/manage_external_data.py) (new
  `load_sdd_staging_spec`/`validate_sdd_staging`/`resolve_sdd_scenario_prior_mode`/`run_sdd_download`
  functions and `sdd-plan|sdd-status|sdd-validate|sdd-mode|sdd-download` CLI commands). The `sdd`
  `AssetSpec` now references the single staging manifest
  ([`configs/data/sdd_staging_manifest.yaml`](configs/data/sdd_staging_manifest.yaml)) and carries
  the availability/mode states, removing the previous two-sources-of-truth split.
  [`scripts/data/stage_sdd_dataset_issue_2657.py`](scripts/data/stage_sdd_dataset_issue_2657.py) is
  reduced to a thin compatibility wrapper that delegates to the canonical subsystem while preserving
  its CLI/exit surface, and scenario-prior gating
  ([`scripts/analysis/calibrate_scenario_priors_from_traces_issue_2726.py`](scripts/analysis/calibrate_scenario_priors_from_traces_issue_2726.py))
  now consumes the canonical gate. Behavior preserved: the proxy-vs-dataset-backed gate stays
  fail-closed and downloads still require explicit `--confirm-download` + y/N (or `--yes`). Migration
  tests in
  [`tests/tools/test_sdd_staging_consolidation_issue_3473.py`](tests/tools/test_sdd_staging_consolidation_issue_3473.py)
  prove the proxy and dataset-backed decisions are identical through the canonical and wrapper paths
  (#3473).
* Refined the `goal-pr-review` skill (`.agents/skills/goal-pr-review/SKILL.md`) for clarity and
  determinism: added a mapping table that bridges `scripts/dev/pr_loop_policy.py` classifications
  (`pending_ci`, `failed_ci`, `missing_artifacts`, `stale_worktree`, `ready_to_merge`, `no_action`)
  to the skill's own state machine, noting that `ready_to_merge` is a candidate signal still gated by
  the intended-design and proof bars; split `## Read First` into always-read versus
  read-when-applicable (`gh-pr-comment-fixer` only for unresolved threads, `review-benchmark-change`
  only for benchmark-facing PRs); and tidied the `## When to use` section formatting. Docs-only; the
  skill validator (`scripts/dev/check_skills.py`) passes.
* Continued the `_build_policy` decomposition (#3384) by migrating the `risk_surface_dwa` and
  `lidar_social_force` adapter families from the inline dispatcher into
  `robot_sf/benchmark/map_runner_policies/adapters.py`, registered in `_POLICY_BUILDERS`. The builders
  reuse the neutral `map_runner_policy_common.build_adapter_policy` (#3403) and import their adapter
  classes from `robot_sf.planner.*`, so no import cycle is introduced. Behavior-preserving (the
  `map_runner` regression suite passes; one test monkeypatch repoints to the new builder namespace).
  Adapter families that still depend on `map_runner`-local helpers (e.g. `trivial_reference` via
  `_build_socnav_config`) are deferred to a later slice.
* Consolidated the RFC6901 JSON-pointer renderer that was copy-pasted in eight modules into a single
  shared helper `robot_sf.common.json_pointer.json_pointer` (#3386). The eight schema-validation call
  sites (`benchmark/{scenario_schema,odd_contract,hazard_traceability,scenario_contract,odd_hazard_coverage_matrix,artifact_catalog}.py`
  and `analysis_workbench/{simulation_trace_export,trace_annotation}.py`) now import it, removing the
  latent hazard of an escaping fix diverging between copies. As part of unifying the two prior copies,
  the three `analysis_workbench`/`artifact_catalog` sites now render the *root* path (empty error path)
  as the RFC6901-correct empty string `""` instead of `"/"`; this only affects the rare root-level
  schema-error message string and is covered by a new unit test
  (`tests/common/test_json_pointer.py`). Also fixed `_copy_figures` in `research/imitation_report.py`
  to actually `return` its `dict[str, Path]` mapping (it previously fell through to `None` despite the
  annotation), with a regression test (#3386).
* Began decomposing the ~4967-line `robot_sf/benchmark/camera_ready_campaign.py` (the largest module in
  `robot_sf/`) by extracting its 6 config dataclasses (`AmvProfileConfig`, `SeedPolicy`,
  `ScenarioCandidateSelection`, `PlannerSpec`, `SnqiContractConfig`, `CampaignConfig`) plus the
  `_AMV_DIMENSIONS` / `DEFAULT_SEED_SETS_PATH` constants into a new
  `robot_sf/benchmark/camera_ready_campaign_config.py` (#3405, first slice of #3385). The names are
  re-exported from `camera_ready_campaign`, so existing imports (e.g. in `release_protocol.py`,
  `orca_preflight.py`, and tests) are unchanged. Behavior-preserving verbatim move; the
  camera-ready/release regression suites pass.
* Consolidated the duplicated durable atomic-JSON-write helper into a single shared
  `robot_sf.common.atomic_io.atomic_write_json` (#3386). The `mkstemp -> fsync -> os.replace` helper
  previously copy-pasted in `benchmark/imitation_manifest.py` and `benchmark/full_classic/io_utils.py`
  now imports the shared version; behavior is preserved (the helper resolves the path and creates the
  parent directory, a safe superset of the prior two). `benchmark/forecast_dataset_recorder.py` keeps
  its own writer for now because it has a different output contract (trailing newline, no `fsync`).
  Covered by `tests/common/test_atomic_io.py`.
* Began decomposing the ~1270-line `_build_policy` dispatcher in `robot_sf/benchmark/map_runner.py`
  into a `robot_sf/benchmark/map_runner_policies/` builder package backed by a registry (#3400, first
  slice of Issue #3384). The built-in goal/simple policy family now lives in `map_runner_policies/goal.py`
  (`build(...) -> (policy_fn, meta)`); `_build_policy` consults `_POLICY_BUILDERS` before its remaining
  inline branches, which are unchanged. Behavior-preserving (the regression net in
  `tests/benchmark/test_map_runner_utils.py` still passes); the only test change repoints one
  monkeypatch to the new builder namespace.
* Relocated the shared `_build_adapter_policy` helper out of `robot_sf/benchmark/map_runner.py` into a
  neutral `robot_sf/benchmark/map_runner_policy_common.py` (`build_adapter_policy`), re-exported under
  the old private name so all ~17 in-module call sites are unchanged (#3403, prerequisite for the
  Issue #3384 adapter-family decomposition). Behavior-preserving — the helper is a verbatim move and the
  `map_runner` regression suite passes. This lets future `map_runner_policies/` builder modules reuse
  the helper without an import cycle back into `map_runner`.
* Cut pull-request CI wall-clock by parallelizing the `fast-feedback` test phase. Pull requests now
  split the suite into 4 `pytest-split` shards (a matrix on the existing job, so the CI topology is
  unchanged) while `main`/`workflow_dispatch` keep a single full pass so the advisory coverage
  baseline stays accurate without cross-shard combining. The shared `run_tests_parallel.sh` wrapper
  gained `PYTEST_SHARD_COUNT`/`PYTEST_SHARD_INDEX` env hooks (`--splits`/`--group`); coverage is
  skipped while sharding. Also enabled the CPython 3.12+ `COVERAGE_CORE=sysmon` backend to lower
  coverage instrumentation overhead, and added a docs-only `paths-ignore` filter so Markdown/`docs/`
  changes no longer trigger the full CI workflow. Adds `pytest-split` to the `dev` dependency group.
* Removed redundant Black/autopep8 formatter configuration and the unused Black optional
  development dependency; Ruff format remains the canonical formatter.
* Extended the issue-1515 hybrid-evidence matrix validator with an opt-in git-history proof path:
  default validation still only checks `commit_artifact` shape plus provenance pointers, while
  `scripts/validation/validate_hybrid_evidence_matrix.py --check-git-history` now additionally fails
  closed on unknown local commit SHAs before issue-1489 synthesis consumes stress/full-matrix rows.
* Updated the issue-1519 predictive ego-feature contract so ego-conditioned schema metadata now
  records a machine-readable motion-channel producer key for same-seed/runtime vs standalone
  collection, and runtime inference rejects explicit standalone-producer checkpoints instead of
  silently treating mixed producer rows as equivalent.
* Tightened the issue-1519 mixed predictive dataset flow so ego-conditioned base/hardcase inputs now
  fail closed on missing or mismatched producer metadata before mixing/training, and successful
  mixed NPZ outputs preserve `feature_schema_json` producer provenance instead of silently dropping
  it.
* Updated the issue-template audit flow for issue-1513 so `scripts/tools/issue_template_audit.py`
  now parses the `## Archetype Metadata` YAML block, flags invalid canonical `archetype` and
  `evidence_tier` values plus malformed metadata, reports those findings alongside the existing
  section audit, and aligns the issue-audit skill docs with the bounded metadata triage behavior.
* Implemented the issue-1187 render-helper slice: `capture_frames()` now samples real RGB frame
  buffers or one direct render result, `generate_video_contact_sheet.py` writes deterministic PNG
  contact sheets from episode frame metadata, and the helper catalog documents the supported
  surfaces and failure modes.
* Updated the issue-1186 post-`#243` output-root cleanup slice so remaining active benchmark and
  baseline commands now point at `output/benchmarks/...`, the factory-performance baseline tooling
  resolves its default path through canonical artifact helpers, and the social-navigation benchmark
  runner no longer advertises a legacy `results/` output root.
* Updated the issue-1180 `goal-pr-review` workflow so autonomous PR review defaults to a
  fix-first repair loop on writable branches: proof failures are now classified as auto-fixable
  now, deferred follow-up, or handoff-only blockers; safe actionable gaps should be repaired and
  revalidated before withholding `merge-ready`; and the skill index plus shared goal-loop note now
  reflect the new validation-and-reassess contract.
* Removed deprecated `robot_sf_snqi recompute --method` aliases (`pareto_optimization`,
  `equal_weights`, and `safety_focused`). Use canonical method names
  `canonical`, `balanced`, or `optimized`; deprecated names now fail argument parsing.

### Fixed

* Filled the issue-1572 AMV actuation metadata gaps so the issue-1556 synthetic slice can carry
  slice-local scenario AMV taxonomy without editing global scenario files, local `social_force`
  rows now expose explicit command-space/projection-policy metadata, and compact actuation summary
  artifacts distinguish AMV coverage status from adapter projection metadata without weakening
  fail-closed unavailable/fallback semantics.
* Fixed the issue-1527 ORCA benchmark preflight gap by moving the `rvo2` fail-fast check into the
  shared camera-ready campaign layer, so direct `prepare_campaign_preflight(...)`,
  `run_campaign(...)`, and wrapper entrypoints such as `scripts/tools/run_benchmark_release.py`
  cannot bypass the actionable `uv sync --extra orca` / `uv sync --all-extras` guidance for
  enabled ORCA-dependent rows.
* Fixed the README Zenodo DOI header by replacing the fragile badge image with a plain-text DOI
  link while preserving the canonical `https://doi.org/10.5281/zenodo.19563812` target.
* Fixed the crowd-only Gymnasium environment contract so `CrowdSimEnv` keeps a stable
  observation-space shape across resets, `make_crowd_sim_env()` preserves preconfigured
  config values unless callers override them explicitly, reset-time map selection no longer
  mutates NumPy's global RNG state, and compact JSONL recordings keep static scene metadata
  on reset events instead of repeating it on every step.

### Added

* Added the `robot_sf/py.typed` marker and `Typing :: Typed` classifier so downstream type checkers
  can consume the package's inline annotations under PEP 561.
* Added the issue-1556 synthetic AMV actuation stress slice: the new
  `configs/benchmarks/issue_1556_amv_actuation_stress_slice_v0.yaml` config stays
  non-paper-facing and differential-drive-only, camera-ready campaign manifests/reports now carry
  synthetic actuation provenance plus a diagnostic `actuation_envelope_summary.{json,md}` artifact,
  and the map runner reports auditable saturation metrics or fails closed when the synthetic
  profile cannot be applied.
* Added the issue-1550 predictive same-seed row-summary validation surface:
  `robot_sf/benchmark/predictive_same_seed_row_summary.py` plus
  `scripts/validation/validate_predictive_same_seed_row_summary.py` now validate small durable
  predictive comparison rows for variant/scenario/seed/planner-grid outcomes, provenance pointers,
  and explicit `unavailable`/`unknown` handling, with a checked-in template and focused regression
  coverage under `tests/benchmark/test_predictive_same_seed_row_summary.py`.
* Added the issue-1515 hybrid-evidence matrix validation surface: `robot_sf/benchmark/hybrid_evidence_matrix.py`
  and `scripts/validation/validate_hybrid_evidence_matrix.py` now validate the #1499 row contract
  for enums, nullability, repository-relative provenance, fail-closed fallback/degraded semantics,
  and synthesis-candidate guard-veto consistency, with checked-in fixtures and targeted regression
  coverage under `tests/benchmark/test_hybrid_evidence_matrix.py`.
* Added the issue-1214 docs/proof consistency checker surface: `scripts/validation/check_docs_proof_consistency.py`
  now scans changed docs/evidence files for high-confidence PR handoff drift, the
  `scripts/dev/check_docs_proof_consistency_diff.sh` wrapper exposes the branch-diff command, and
  the contributor workflow docs now point at the new check before proof-heavy PR handoff.
* Added the issue-1185 dummy-backend smoke surface: `robot_sf/sim/backends/dummy_backend.py`
  now exposes the map metadata and minimal simulator contract that `RobotEnv` expects, so
  `examples/advanced/01_backend_selection.py` can run headlessly in CI and stays covered by the
  manifest-driven examples smoke suite.
* Added the issue-1181 `ml-intern` bounded-assistant assessment note, including the local-only
  proof ladder, trace/privacy boundary, verified Robot SF prompt/context stack, and the explicit
  recommendation to keep `ml-intern` as a bounded experiment assistant rather than a replacement
  for the repository's local/HPC/SLURM proof-first workflow.
* Added the issue-1168 multi-AMV planner support classification surface: multi-AMV episode
  metadata now carries explicit planner-family support records, planner support preflight checks
  fail closed for unsupported or smoke-only planner families, and the docs index records the
  current boundary between smoke execution and real multi-robot planner support.
* Added the issue-1153 manual-control replay/export helper surface: BC samples now carry source
  provenance, replay/profile helpers reject directory inputs fail-closed, and replay/export JSON
  writers preserve NumPy-backed payloads by normalizing them into JSON-safe builtins before
  serialization.
* Added the issue-1151 manual-control MVP foundation surface: append-only JSONL recording,
  fail-closed mode/session helpers, baseline comparison primitives, and BC export utilities now
  reject invalid mode values, non-finite speed multipliers, negative tolerances, and malformed
  record payloads while keeping NumPy-backed observations serializable for recorder/export flows.
* Added the issue-1152 manual-control mode experiment surface: typed control/view mode specs,
  cruise and mouse-target differential-drive mapper variants, runtime mode selection config, and
  manifest/record metadata for `control_mode`, `view_mode`, and `input_mapping_version`, while
  keeping unsupported ego-up renderer hooks fail-closed until the camera transform exists.
* Added the issue-1162 manual-control rewind boundary: manual-control records now carry explicit
  rewind metadata, replay JSON preserves rewind events, and BC export skips samples invalidated by
  append-only rewind records while repeated-rewind planning fails closed until active-timeline
  derivation exists.
* Added the issue-1110 CARLA oracle replay parity adapter surface: `compare_oracle_replay_metrics(...)`
  and its CLI now emit conservative parity reports, reject degraded CARLA `mode` or `status`
  fail-closed, treat non-finite numeric values as unavailable instead of serializing invalid JSON,
  and document the adapter as a comparison boundary rather than live CARLA evidence.
* Added the issue-1128 multi-AMV episode extension surface: `multi_amv_episode_extension(...)`
  now emits an additive namespaced `multi_amv` block for smoke outputs, requires explicit
  planner status, omits optional planner notes when absent, and keeps invalid single-robot or
  empty-metric inputs fail-closed through targeted regression coverage and the documented
  multi-AMV smoke validation path.

* Added the issue-857 horizon-alignment experiment surfaces: manifest-level `scenario_overrides`
  support in `robot_sf/training/scenario_loader.py`, the new
  `configs/scenarios/sets/ppo_full_maintained_eval_v1_horizon100.yaml` eval/training surface,
  the seed-123 retrain clone
  `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_horizon100.yaml`,
  the diagnostic benchmark probe
  `configs/benchmarks/paper_experiment_matrix_v1_issue_791_horizon400_probe.yaml`, and focused
  regression coverage for the new horizon-matched workflow.
* Added the issue-856 broad-training evidence surfaces for the completed seed-123 control: the 12223 baseline adapter (`configs/baselines/ppo_issue_856_all_scenarios_12223.yaml`), the dedicated camera-ready comparison matrix (`configs/benchmarks/paper_experiment_matrix_v1_issue_856_all_scenarios_compare.yaml`), the campaign and pre-PR verification context notes, and the linked experiment-memory write-back. Recorded the verdict that, at fixed 10M budget, broad-training underperforms the eval-aligned leader on the camera-ready matrix (success −0.035, collisions +0.007, SNQI −0.040), strengthening the alignment-vs-diversity claim for the manuscript.
* Added a file-based policy-search workflow under `docs/context/policy_search/` with a canonical candidate registry, stage-gated runner (`scripts/validation/run_policy_search_candidate.py`), comparison/failure/promotion tools, and SLURM handoff notes for training-heavy follow-up work. The first non-training candidate, `hybrid_orca_sampler_v1`, now runs end to end in the benchmark stack and emits markdown/json validation artifacts locally.
* Added the deterministic `hybrid_rule_local_planner` policy-search record and ablation surfaces. The current best non-learning candidate is `hybrid_rule_v3_static_margin0_waypoint2`, which preserved zero collision terminations on nominal sanity and stress slices, while rejected static-escape, route-commitment, route-lookahead, and speed variants remain documented as reproducible ablations rather than promoted planner defaults.
* Added DreamerV3 BR-08 close-out documentation (2026-04-30), including the program-level stop decision, updated full-run outcome note for Slurm 12159, and linked follow-up surfaces from the 2026-04-28 handoff: checked-in Auxme launchers (`SLURM/Auxme/dreamer_br08_gate.sl`, `SLURM/Auxme/dreamer_br08_full.sl`), eval-parity regression coverage for the BR-08 full config, a `#782` pretraining design note, and a fail-closed `#789` note documenting that Ray 2.53.0 DreamerV3 still lacks mixed-observation support without a larger catalog/module fork.
* Documented durable artifact, linked-worktree bootstrap, branch-sync, and
  worktree-output handling rules in `AGENTS.md` and `docs/dev_guide.md`,
  clarifying that `output/` is temporary/local by default; fresh linked
  worktrees should detect the shared main checkout before symlinking
  `local.machine.md`; a worktree counts as fresh only when both
  `local.machine.md` and `.venv` are absent; active feature branches should
  merge latest `origin/main` early and again before PR creation; and PR
  preparation now includes reviewing ignored `output/*` files before handoff.

* Promoted the issue-791 Wave-5 leader (job 11724, WandB `ll7/robot_sf/ibo3aqus`,
  best success 0.929 / collision 0.071 / SNQI 0.353 on
  `ppo_full_maintained_eval_v1`) into the canonical PPO baseline at
  [configs/baselines/ppo_15m_grid_socnav.yaml](configs/baselines/ppo_15m_grid_socnav.yaml).
  All ~17 benchmark configs that reference this file (camera-ready, paper
  matrix, planner sanity, seed variability) now resolve PPO to the eval-aligned
  large-capacity grid_socnav policy with predictive foresight enabled and
  `fallback_to_goal: false` per the fail-closed benchmark policy. The previous
  BR-06 v3 15M artifact remains on disk for reproducibility but is no longer the
  default. Reported performance is benchmark-set (in-distribution), not OOD.

* Restored the asymmetric-critic robot training contract so `RobotEnv(..., asymmetric_critic=True)` and `make_robot_env(..., asymmetric_critic=True)` once again add the critic-only `critic_privileged_state` observation for SocNav structured runs. This keeps the public factory API stable for issue-791 training jobs and closes the regression that broke fresh SLURM submissions after the main-branch merge.

* Corrected the issue-791 reward-curriculum `resume_best` overlay so it preserves the original `grid_socnav` / `MultiInputPolicy` checkpoint contract when warm-starting from 11566 best.zip. This avoids the observation-space mismatch that appeared when the retry was pointed at the wrong model family.

* SVG obstacle-repair warnings now include the source `*.svg` filename, so invalid map repairs point directly to the offending asset; the regression test also checks the captured warning text.
* Added issue-832 paper-matrix extended seed schedules (`paper_eval_s5`, `paper_eval_s10`,
  `paper_eval_s20`), S5/S10 paper-matrix sibling configs that preserve the frozen
  `paper-matrix-v1` contract, `run_camera_ready_benchmark.py --campaign-id` for resumable fixed
  campaign roots, and `scripts/tools/compare_seed_schedule_campaigns.py` for reproducible
  S3-vs-higher-seed CI-width, mean-drift, ranking, and scenario-winner comparisons.

* Added fail-fast SoNIC / GenSafeNav benchmark wrapper aliases for the model-only checkpoint path: base aliases `sonic_crowdnav`, `gensafenav_ours_gst`, and `gensafenav_gst_predictor_rand`, plus guarded variants `ours_gst_guarded` and `gst_predictor_rand_guarded` with short-horizon guard telemetry (`guard_stats`) and goal fallback. This exposes the adapter surface explicitly for users and benchmark configs while keeping remaining risks visible: the base wrappers are still prototype-level model-only integrations, guarded behavior exists only on the guarded aliases, and guarded holonomic `vx_vy` runs now fail closed until the guard can preserve upstream ActionXY without a lossy `(v, w)` round-trip.

* Recorded-step playback analyzer for issue #585: telemetry replay now records per-step reward
  terms plus selected step metrics with episode IDs, JSONL episode metadata can link back to the
  corresponding telemetry stream, and interactive playback can show a synchronized analyzer overlay
  with reward tables, visible-metric filtering, and a scrub-aligned metric timeline while remaining
  backward compatible with older recordings.

* TEB corridor-commitment planner improvements for issue #805 (second iteration): fixed corner-cutting by tuning embedded route-guide `waypoint_lookahead_cells` to 5 (1.0 m target at 0.2 m/cell, full 0.9 m/s speed without diagonal clipping), increased `obstacle_inflation_cells` to 3 for 0.6 m corner clearance, added `stop_distance=0.5` for earlier front-clearance stops, and added `_rescue_or_stop` so all-blocked commitment falls back to the route guide rather than driving into obstacles;  `plan()` refactored into `_try_route_command`,  `_commitment_step`, and `_rescue_or_stop` helpers to satisfy the complexity limit.
* TEB corridor-commitment planner improvements for issue #805 (first iteration): multi-step corridor occupancy scoring, escalated lateral commitment gains, blocked-side flip fallback, route-waypoint guidance integration, and reproducible topology-slice artifacts (`configs/scenarios/sets/issue_805_teb_topology_slice.yaml`,   `docs/context/issue_805_teb_corridor_commitment_iteration.md`).

* Added a repo-local agent memory layer with `CLAUDE.md` startup imports, a canonical
`memory/MEMORY.md` index, typed memory subdirectories ( `architecture` , `decisions` , 
`experiments` , `failures` , `benchmarks` ), example notes for each type, and linked guidance in
`AGENTS.md` , `docs/dev_guide.md` , `docs/README.md` , and the AI-facing overview/deferral docs.
* Added canonical `classic_cross_trap_*` scenario IDs and manifests for the former
`classic_crossing_*` cross-shaped local-minimum trap scenarios, with legacy
  compatibility aliases and migration notes for existing configs.
* Added a canonical Markdown context-note workflow under `docs/context/README.md`, explicit
`AGENTS.md` / Copilot / dev-guide guidance for linked handoff notes, a new
`context-note-maintainer` repo-local skill, and an issue-796 policy note so reusable agent
  knowledge is easier to preserve, discover, and update without adding external infrastructure.

* Added an SB3 SAC training workflow for issue 790 with a config-first entrypoint (`scripts/training/train_sac_sb3.py`), gate/full configs under `configs/training/sac/`, strict config validation, and targeted training tests covering config load plus dry-run checkpoint creation.

* Added an all-scenarios SAC training config with random scenario sampling, non-fixed per-episode seeds, and W&B telemetry enabled (`configs/training/sac/gate_socnav_struct_ego_all_random_wandb_v3.yaml`), plus an autoresearch-list entry for repeatable experiment orchestration.

* Added scenario-weighted SAC curriculum configs plus a small SAC experiment harness (`scripts/tools/sac_autoresearch.py`) so issue-790 training loops can compare benchmark-style evaluation results reproducibly.

* Added config-driven SAC multi-env training support (`num_envs`) in `scripts/training/train_sac_sb3.py`, including subprocess vectorization for `num_envs > 1` and an updated all-scenarios W&B config using `num_envs: 4`.

* Paper results handoff export (`robot_sf.benchmark.paper_results_handoff`) now emits interval-inclusive planner-summary rows from frozen publication bundles, with CLI wrapper (`scripts/tools/paper_results_handoff.py`), confidence-interval metadata, seed/repeat provenance, deterministic JSON/CSV outputs, and contract documentation in `docs/context/issue_750_paper_results_handoff.md`.

* Policy analysis issue-768 coverage now includes ORCA variant CLI entries (`socnav_orca_nonholonomic`,        `socnav_orca_dd`,        `socnav_orca_relaxed`,        `socnav_hrvo`), targeted regression tests for the new builder paths and sweep policy resolver, and an issue note comparing the branch results against `main` plus full-eval artifacts under `output/experiments/768_*`.

* GitHub workflow guidance now prefers MCP / app tools for interactive issue, PR, and project work, keeps `gh` as the documented scripted fallback, and hardens `project_priority_score.py` with clearer auth diagnostics plus a retry path for the `--owner ll7` / `unknown owner type` gh quirk.
* Camera-ready campaign analysis now emits `scenario_difficulty_analysis.{json,md}` plus richer `campaign_analysis.{json,md}` content that ranks scenario difficulty from existing artifacts, summarizes family-level hardness, flags planner residual mismatch on easier scenarios, and records whether the verified-simple subset needs a bounded pilot before it can be treated as a calibration aid.
* Added adversarial pedestrian policy demo (`examples/advanced/32_demo_adversarial_pedestrian.py`) with factory-based environment setup, CLI overrides for map/model paths, and per-episode collision/kinematics summary logging.
* Added pedestrian policy collision benchmark script (`scripts/benchmark_ped_policy_collisions.py`) to evaluate pedestrian checkpoints with episode metrics plus collision kinematics (robot speed, pedestrian speed, and impact angle at robot-pedestrian collision), front/back/side collision-zone percentages, explicit timeout/robot-goal counts, and a console outcome-breakdown log line.
* Added differential-drive pedestrian policy debug runner (`scripts/debug_ped_policy_differential_drive.py`) combining the pedestrian debug loop with run_023-compatible differential-drive robot profile and observation adapter behavior.
* Added differential-drive pedestrian PPO training entrypoint (`scripts/training_ped_ppo_differential_drive.py`) mirroring the existing pedestrian PPO workflow with `DifferentialDriveSettings`.
* Pedestrian collision telemetry now records `collision_impact_angle_rad/deg` in per-step metadata, with aggregated TensorBoard scalars and histograms for overall and per-collision-type impact kinematics.
* APF model comparison benchmark script (`scripts/benchmark_ped_apf_models.py`) to run `run_023` and `run_043` with APF off/on (100 episodes each by default) and report aggregated episode metrics (steps, collisions, success/timeout, rewards) with JSON export under `output/benchmarks/`.
* Added benchmark release protocol v0.1 surfaces: canonical release manifests for the paper-facing matrix, a `run_benchmark_release.py` entrypoint layered on the camera-ready workflow, release/reproducibility docs,          `CITATION.cff`, and a benchmark-focused `RELEASE.md` checklist.
* Shipped adversarial-pedestrian assets for the new demos and training workflow: SVG maps `maps/svg_maps/masterthesis/corner.svg`,         `maps/svg_maps/masterthesis/headon.svg`,         `maps/svg_maps/masterthesis/intersection.svg`, plus pedestrian PPO checkpoints `model/pedestrian/ppo_corner.zip`,         `model/pedestrian/ppo_headon.zip`, and `model/pedestrian/ppo_intersection.zip` used by `examples/advanced/32_demo_adversarial_pedestrian.py`, `scripts/benchmark_ped_policy_collisions.py`, `scripts/debug_ped_policy_differential_drive.py`, `scripts/training_ped_ppo_differential_drive.py`, and `scripts/benchmark_ped_apf_models.py`.
* Added benchmark release protocol v0.1 surfaces: canonical release manifests for the paper-facing matrix, a `run_benchmark_release.py` entrypoint layered on the camera-ready workflow, release/reproducibility docs,         `CITATION.cff`, and a benchmark-focused `docs/RELEASE.md` checklist.
* Added Project #5 task prioritization support: a documented benchmark-oriented scoring model, a `project_priority_score.py` sync helper, and a GitHub Actions workflow for manual/scheduled/issue-event score synchronization into a numeric `Priority Score` field.
* Default PR template now prompts for summary, validation/proof, risks/rollout, docs/provenance, and follow-up issues so reviews stay proof-first and easier to act on.
* Canonical benchmark fallback policy note (`docs/context/issue_691_benchmark_fallback_policy.md`) plus shared availability helpers so benchmark CLI and camera-ready campaigns fail closed on fallback, degraded, skipped, and partial-failure outcomes.
* Repository agent guidance now requires proof-first validation for new planners, metrics, skills, and tests, with task-appropriate executable or reproducible evidence before a change is considered complete.
* Added a Codex-native repository context stack: strengthened `AGENTS.md`, new `docs/code_review.md` benchmark/provenance checklist, `.agents/PLANS.md` planning convention, repo-local context skills under `.agents/skills/`, and AI-facing overview/decision docs under `docs/ai/`.
* Promoted predictive planner checkpoint `predictive_proxy_selected_v2_xl_ego` into the model registry with a W&B-backed download entry (`ll7/robot_sf/i17pmely`).
* Added explicit BR-06 v6, v7, and v9 auto-env PPO resubmit configs for 24-CPU Auxme jobs so queued reruns can resolve `num_envs` from the host allocation instead of staying pinned to 8 env workers.
* Added `scripts/dev/sbatch_use_max_time.sh` plus SLURM submission docs so new batch jobs can query partition/QoS wall-time limits and default to the effective maximum at submit time.
* Added a SLURM resource-audit runbook documenting how to retrieve W&B `system` metrics, compare them against Slurm allocations, and reason about PPO `num_envs` vs reserved CPU count.
* `ClassicGlobalPlanner.plan_random_path()` now supports `allow_inflation_fallback=False` to keep the configured inflated area instead of shrinking inflation during random path sampling.
* Unified configs now support `map_id` for deterministic map selection when building environments.
* Config-first RLlib DreamerV3 training workflow for `drive_state` + `rays`, including:
`scripts/training/train_dreamerv3_rllib.py` , `configs/training/rllib_dreamerv3/drive_state_rays.yaml` , 
  deterministic observation flattening/action normalization wrappers, and a dedicated runbook.
* Optional dependency split for training stacks: `imitation` moved to the `imitation` dependency group so BC pre-training can be installed separately from RLlib-oriented environments.
* Added an explicit uv dependency conflict declaration between `--extra rllib` and `--group imitation` to provide actionable install errors when mixed.
* Scenario split helper (`robot_sf.training.scenario_split`) and CLI (`scripts/tools/split_scenarios.py`) for train/holdout validation.
* Optuna sqlite inspection helper (`scripts/tools/inspect_optuna_db.py`) for quick study summaries.
* Optuna expert PPO sweep report with recommended hyperparameter ranges (`docs/training/optuna_expert_ppo_sweep_2026-02-11.md`).
* Shared occupancy collision helpers to de-duplicate dynamic and obstacle collision checks.
* New force flags `peds_have_static_obstacle_forces` and `peds_have_robot_repulsion` with legacy alias support.
* Benchmark CLI warn-only scenario preview (`robot_sf_bench preview-scenarios`) plus warnings/coverage summary in validate-config output.
* Expert PPO training now supports `ppo_hyperparams`/`best_checkpoint_metric` overrides and saves a best-checkpoint snapshot per run.
* Optuna sweep helper for expert PPO training configs (`scripts/training/optuna_expert_ppo.py`).
* Safety-gated Optuna objectives for expert PPO sweeps (`constraint_collision_rate_max`, 
  optional `constraint_comfort_exposure_max` , and `constraint_handling=penalize|prune` )
  with per-trial feasibility metadata and feasible/infeasible study summaries.
* Policy analysis episodes now store `shortest_path_len` in metrics to enable diagnostics of path-efficiency saturation.
* Policy analysis sweep mode can run multiple policies in one invocation (`--policy-sweep`,          `--policies`).
* Policy analysis now records `jerk_mean_eps0p1` and `curvature_mean_eps0p1` with a low-speed filter.
* Policy analysis supports named seed sets (`--seed-set` via `configs/benchmarks/seed_sets_v1.yaml`) and writes combined sweep reports.
* Policy analysis can optionally extract failure frames from report.json (`--extract-frames`), and the frame extractor accepts report inputs.
* Example SocNav social-force algorithm config for map-based benchmarks (`configs/algos/social_force_example.yaml`).
* Fast-pysf ground-truth planner option for scenario video rendering (`--policy fast_pysf` in `scripts/tools/render_scenario_videos.py`).
* Policy analysis sweep script with metrics + optional videos (`scripts/tools/policy_analysis_run.py`).
* Benchmark outputs now include wall collision and clearing distance metrics by default.
* Vendored GA3C-CADRL (SA-CADRL) checkpoint under `model/ga3c_cadrl/` with provenance + license metadata.
* Added `sacadrl` optional dependency extra (TensorFlow) for GA3C-CADRL baseline support.
* ORCA planner uses the rvo2 binding when available; added `orca` optional dependency extra.
* Added SocNavBench subset metrics (`socnavbench_path_length`,             `socnavbench_path_length_ratio`,             `socnavbench_path_irregularity`).
* Map registry (`maps/registry.yaml`) with generator script and `map_id` support for scenario files.
* Occupancy grid polish: ego-frame transforms applied consistently, query aggregation returns per-channel means without scaling errors, new quickstart/advanced/reward-shaping examples (`examples/quickstart/04_occupancy_grid.py`,             `examples/advanced/20_occupancy_grid_workflow.py`,             `examples/occupancy_reward_shaping.py`), and an expanded guide (API/config/troubleshooting + docs index link).
* Telemetry visualization (feature 343): docked Pygame telemetry pane with live charts, JSONL telemetry stream under `output/telemetry/`, replay/export helpers, headless smoke script/test, and a demo (`examples/advanced/22_telemetry_pane.py`).
* Automated research reporting pipeline (feature 270-imitation-report): multi-seed aggregation, statistical hypothesis evaluation (paired t-tests, effect sizes, threshold comparisons), publication-quality figure suite (learning curves, sample efficiency, distributions, effect sizes, sensitivity), ablation matrix orchestration, telemetry section, and programmatic + CLI workflows (`scripts/research/generate_report.py`,             `scripts/research/compare_ablations.py`). Includes success criteria tests and demo (`examples/advanced/17_research_report_demo.py`).
* Research reporting polish: metadata manifest aligned with `report_metadata` schema, schema validation tests for metrics/hypotheses, and smoke/performance harnesses (`scripts/validation/test_research_report_smoke.sh`,             `scripts/validation/performance_research_report.py`,             `tests/research/test_performance_smoke.py`,             `tests/research/test_schemas.py`).
* Multi-extractor training now auto-collects convergence/sample-efficiency metrics, baseline comparisons, and learning-curve/reward-distribution figures, emitting schema-compliant summaries (`summary.json`/`summary.md`) plus legacy `complete_results.json`.
* New extractor report generator: `scripts/research/generate_extractor_report.py` converts multi-extractor `summary.json` into research-ready `report.md`/`report.tex` with figures, reproducibility metadata, and baseline comparisons.
* Inkscape SVG map template (`maps/templates/map_template.svg`) and quickstart updates in `docs/SVG_MAP_EDITOR.md`.
* Classic planner backend option in `RobotSimulationConfig` (`planner_backend="classic"` with `planner_classic_config`) plus ORCA demo consuming it (`examples/advanced/31_classic_planner_orca_demo.py`); planner attachment now supports grid-based planning without manual wiring.
* Scenario-level single-pedestrian overrides now support POI-based goals/trajectories, per-ped speeds, wait notes, and a preview helper (`scripts/tools/preview_scenario_trajectories.py`) documented in `docs/single_pedestrians.md`, including clean rendering and `--all` batch export.
* Single pedestrian runtime behavior controller with waypoint advancement, wait handling, and role tags (follow/lead/accompany/join/leave) for robot-relative interactions.
* Added Francis 2023 crowd/traffic scenario maps (crowd navigation, parallel/perpendicular traffic, circular crossing, robot crowding) and scenario entries in `configs/scenarios/francis2023.yaml`.
* SVG map inspection helper (`scripts/validation/svg_inspect.py`) plus reusable API (`robot_sf.maps.verification.svg_inspection`) for route-only mode detection, route/zone index checks, risky path-command warnings, and obstacle-interior route checks.
* Full classic benchmark now emits `run_meta.json` traceability metadata (repo/branch/commit, CLI invocation, matrix path, seed plan, environment), with a mirror at `artifacts/<run_id>/run_meta.json` for paper artifact workflows.
* Full classic benchmark supports freeze-manifest validation (`--freeze-manifest`) with runtime contract checks (matrix hash/path, baselines/planner config, seed plan, metric subset, bootstrap settings, software identifiers) and structured status/mismatch reporting in `run_meta.json`.
* Added a machine-readable freeze manifest template at `configs/benchmarks/classic_benchmark_freeze.example.yaml` and documented the pilot -> freeze -> final workflow.
* Benchmark episodes now include canonical threshold profile metadata (`metric_parameters.threshold_profile` + signature).
* Aggregation now validates threshold-profile consistency and rejects mixed-threshold reports.
* Added threshold sensitivity tooling (`scripts/benchmark_threshold_sensitivity.py`) to report near-miss and comfort-threshold impacts by scenario family, including speed-aware alternatives.
* Root `README.md` now includes an explicit attribution section documenting that this project builds on Caruso et al. (Machines 2023) and listing upstream reference repositories.
* Root `README.md` attribution section now also documents fast-pysf acknowledgements (svenkreiss/socialforce and pedsim_ros) and links the core Social Force references (Helbing & Molnar 1995; Moussaid et al. 2010).

### Fixed

* Fixed scenario-driven timeout semantics so `max_episode_steps` now expires on the configured
  discrete step count instead of one step late from a floating elapsed-time comparison.

* Legacy `scripts/training_ppo.py` invocations now fail closed and point to the canonical PPO
  training workflow docs.
* Restored the existing `socnav_sacadrl` CLI choice in `scripts/tools/policy_analysis_run.py` after the issue-768 ORCA variant additions, preventing a regression where the builder still supported SA-CADRL but the parser rejected it.
* Paper results handoff export now fails closed on malformed metadata, validates JSONL rows as objects while streaming them, aggregates planner rows per run to reduce peak memory pressure, and writes standards-compliant JSON/CSV artifacts without permissive `NaN` serialization.

* Restored the backward-compatible `avg_collision_impact_angle_rad` pedestrian metric alias and corrected the adversarial-force plot labels in `scripts/debug_ped_apf.py`.
* Issue-708 PPO SLURM launcher now runs the training command directly in the batch shell instead of using a nested `srun`, avoiding an immediate allocation confirmation failure on Auxme.
* Differential-drive kinematics now match standard straight-line and in-place rotation formulas.
* Auxme GPU SLURM submissions now default to `48G` host memory instead of the previous `120G`, matching the documented PPO host-memory sizing guidance more closely while preserving headroom for `num_envs` auto-resolution on 24-CPU jobs.
* Auxme GPU SLURM submissions now default to the current `a30` partition maximum wall time instead of the previous 8-hour fallback.
* Auxme GPU batch runs now bootstrap the environment modules shell function when available and default staging/results paths to repo-local `output/slurm/` locations instead of failing on unavailable `/scratch/${USER}` paths.
* Auxme GPU batch runs now support forwarding extra `uv run` flags, allowing shared max-time submissions for jobs such as `uv run --extra rllib python ...`.
* Expert PPO warm-start runs now emit direct W&B `rollout/*`,             `train/*`, and `time/*` metrics during training so resumed runs show progress before the next scheduled evaluation checkpoint and do not depend solely on TensorBoard discovery.
* `robot_sf/benchmark/perf_trend.py` now treats `--history-limit 0` as "load no history reports" instead of unintentionally loading all matched files.
* `robot_sf_bench plot-scenarios` now avoids thumbnail filename collisions for name-only matrices by applying deterministic `id -> name -> scenario_id -> hash` fallback resolution, sanitization, and collision suffixing.
* DreamerV3 RLlib launcher now hardens Ray runtime environment setup by disabling `uv run` worker propagation, pinning worker interpreter execution, and applying packaging excludes to reduce startup fragility and upload size.
* Dreamer observation contract now emits float32-consistent reset/step payloads (`drive_state`,  `rays`) so Gymnasium space checks pass without dtype warnings.
* Simulation throughput perf guard now supports cluster-aware calibration through env overrides and enforce mode instead of hard-failing heterogeneous nodes by default.
* SVG map parsing now honors indexed spawn/goal labels to keep route ordering stable.
* Pedestrian env now validates robot action-space compatibility and falls back to null actions on mismatch.
* SocNavBenchSamplingAdapter now shares the upstream loader logic with SamplingPlannerAdapter to avoid drift and keep the vendored planner path working.
* SocNavBench path irregularity now uses heading vectors instead of origin-dependent position vectors.
* SocNavBench upstream planner loader rejects untrusted roots unless explicitly allowed via `ROBOT_SF_SOCNAV_ALLOW_UNTRUSTED_ROOT`.
* Vendored SocNavBench helpers now guard missing data files, fix numpy API mismatches, and correct plotting/trajectory utilities.
* Renamed `thrid_party` to `third_party` to fix the vendor directory typo.
* Vendored the MIT-ACL Python-RVO2 fork to keep ORCA builds compatible with newer CMake versions.
* SocialForcePlannerAdapter now uses fast-pysf social-force interactions (goal, pedestrian, obstacle forces) instead of the heuristic placeholder.

### Added

* Map Verification Workflow (Feature 001-map-verification)
  + Single-command map validation tool (`scripts/validation/verify_maps.py`) for SVG map quality checks
  + Rule-based validation engine checking file readability, SVG syntax, file size, and layer organization
  + Scope filtering supporting 'all', 'ci', 'changed', specific filenames, or glob patterns
  + Structured JSON/JSONL manifest output for tooling and dashboard integration
  + CI mode with strict exit codes for automated quality gates
  + Loguru-based diagnostics with human-readable console output
  + Map inventory system with tag-based classification and filtering
  + Verification results include timing, rule violations, and remediation hints
  + Documentation in `docs/SVG_MAP_EDITOR.md` and `specs/001-map-verification/quickstart.md`
  + Sample manifest artifacts under `output/validation/`
* Run tracking & telemetry for the imitation pipeline (Feature 001-performance-tracking)
  + Progress tracker with deterministic step ordinals, ETA smoothing, and manifest-backed step history
  + JSONL manifests enriched with telemetry snapshots, rule-based recommendations, and perf-test results stored under `output/run-tracker/`
  + CLI tooling (`scripts/tools/run_tracker_cli.py`) for status/watch/list/show/export/summary plus optional TensorBoard mirroring
  + Performance smoke CLI (`scripts/telemetry/run_perf_tests.py`) that wraps the existing validation harness and records pass/soft-breach/fail statuses with remediation hints
  + Documentation updates spanning quickstart, dev guide, and docs/README.md so teams can enable the tracker and interpret telemetry in CI or local runs
  + CI guard step invoking `scripts/validation/run_examples_smoke.py --perf-tests-only` so the tracker smoke + telemetry perf wrapper fail fast before pytest
* PPO Imitation Learning Pipeline (Feature 001)
  + Expert PPO training workflow with convergence criteria and evaluation schedules
  + Trajectory dataset collection and validation utilities
  + Behavioral cloning (BC) pre-training from expert demonstrations
  + PPO fine-tuning with warm-start from pre-trained policies
  + Comparative metrics CLI for sample-efficiency analysis
  + Playback and inspection tool for trajectory datasets
  + Bootstrap confidence intervals for metric aggregation
  + Complete artifact lineage tracking (expert → dataset → pre-trained → fine-tuned)
  + Configuration dataclasses for all imitation workflows
  + Integration tests for end-to-end pipeline validation
  + Sample-efficiency target: ≤70% of baseline timesteps to convergence
  + Documentation in `docs/dev_guide.md` and `specs/001-ppo-imitation-pretrain/quickstart.md`
  + Default pretraining configs for behavioral cloning and PPO fine-tuning (`configs/training/ppo_imitation/bc_pretrain.yaml`,        `configs/training/ppo_imitation/ppo_finetune.yaml`)
* Canonical artifact root enforcement and tooling (Feature 243)
  + Introduced `output/` hierarchy as single destination for coverage, benchmark, recording, wandb, and tmp artifacts
  + Added migration helper (`scripts/tools/migrate_artifacts.py`) and guard (`scripts/tools/check_artifact_root.py`) with console entry point and regression tests
  + Wired guard + migration into CI workflow, publishing artifacts from canonical paths only
  + Refreshed core docs (`docs/dev_guide.md`,               `docs/coverage_guide.md`,   `docs/README.md`, root `README.md`) with policy overview, quickstart links, and updated coverage instructions
  + Extended quickstart guidance to cover guard execution, artifact overrides, and validation expectations
* Visibility-graph global planner (Feature 342)
  + Planner API with POI routing, caching, smoothing; new `use_planner` flag and clearance config in unified configs
  + Map POI parsing, planner demo (`examples/advanced/20_global_planner_demo.py`), validation script (`scripts/validation/verify_planner.sh`), and benchmark script (`scripts/benchmark_planner.py`)
  + Tests for path planning, POI sampling, caching, smoothing, and navigation integration under `tests/test_planner/`
* Comprehensive configuration architecture documentation (#244)
  + Created `docs/architecture/configuration.md` with configuration precedence hierarchy
  + Documented three-tier precedence system: Code Defaults < YAML < Runtime
  + Added migration guide from legacy config classes to unified config
  + Documented all configuration modules (canonical vs legacy)
  + Linked from `docs/README.md` and `docs/dev_guide.md`
* Automated example smoke harness (`scripts/validation/run_examples_smoke.py`,               `tests/examples/test_examples_run.py`) wired into validation workflow (#245)

### Fixed

* Pedestrian route sampling now fails fast when obstacle constraints cannot be satisfied instead of silently ignoring obstacles.
* Model registry resolution now fails fast for `local_only` entries with explicit replacement guidance instead of attempting ambiguous off-machine fallback, and predictive promotion metadata now preserves portable W&B entries when provenance is supplied as either a full run path or split run identifiers.

### Changed

* Retired the PAT-dependent `project-priority-score-sync.yml` GitHub Actions workflow and updated Project #5 prioritization docs to treat `scripts/tools/project_priority_score.py sync` as the supported local/manual score-sync path.
* Policy analysis timestamps now use CET/CEST (Europe/Berlin) instead of UTC for output folders and episode metadata.
* Policy analysis runs now force `use_planner=False` in the episode config to avoid attaching global planners during metrics/video sweeps.
* Classic global planner defaults now use 0.5m grid cells (`cells_per_meter=2`) with zero inflation for shortest-path planning, reducing invalid start/goal cell failures.
* Default global planner selection now prefers the classic Theta* (v2) grid planner, and benchmark shortest-path calculations use the same planner.
* `SimulationView` now starts with lidar-ray visualization disabled by default (`show_lidar=False`); use `show_lidar=True` or press `O` to toggle observation-space overlays (auto-selecting lidar/grid/image based on the active observation mode).
* Occupancy grid rasterization now logs out-of-bounds obstacle segments at DEBUG instead of the custom SPAM level.
* Benchmark CLI list-algorithms now reports only implemented baseline planners to avoid registry KeyErrors.
* Expert PPO training and trajectory collection now honor scenario YAML entries, including map files, simulation overrides, and scenario identifiers, while publishing `scenario_coverage` metadata consistent with dataset validators.
* **[BREAKING for internal imports]** Consolidated utility modules into single `robot_sf/common/` directory (#241)
  + Moved `robot_sf/util/types.py` → `robot_sf/common/types.py`
  + Moved `robot_sf/utils/seed_utils.py` → `robot_sf/common/seed.py` (renamed)
  + Moved `robot_sf/util/compatibility.py` → `robot_sf/common/compat.py` (renamed)
  + Removed empty `robot_sf/util/` and `robot_sf/utils/` directories
* Example catalog reorganization and automation improvements (#245)
  + Moved benchmark and plotting scripts into dedicated `examples/benchmarks/` and `examples/plotting/` tiers
  + Regenerated manifest-backed `examples/README.md` and refreshed docs (`README.md`,   `docs/README.md`,               `docs/benchmark*.md`,               `docs/distribution_plots.md`) to reference new paths
  + Updated `examples/examples_manifest.yaml` metadata (tags, CI flags, summaries) and added quick links from docs
  + Imitation pipeline example now auto-selects simulator backends and generates run-specific BC/PPO configs under `output/tmp/imitation_pipeline/` to keep CLI invocations aligned with script requirements
* Visualization stack ownership clarified: the Full Classic pipeline (`robot_sf.benchmark.full_classic.visuals.generate_visual_artifacts`) is now the canonical path that emits manifest-backed plot/video artifacts; the legacy helper API (`robot_sf.benchmark.visualization.*`) is deprecated for benchmark runs and retained only for ad-hoc JSONL plotting.

### Fixed

* Policy analysis reports now include video links for problem episodes when videos are recorded, by attaching metadata after video write.
* Model registry W&B downloads now handle file-like download responses, avoiding path resolution errors.
* Policy analysis video sweeps now close environments via `exit()` to flush recordings, preventing empty output folders.
* OSM map conversion now decomposes obstacle polygons with holes before building `MapDefinition` obstacles, preventing walkable areas from being treated as obstacles during spawning and grid generation.
* Resolved OSM dependency pinning by aligning `networkx` with current `osmnx` constraints to avoid unsatisfiable installs.
* OSM driveable-area fallback now checks obstacle containment safely, preventing `AttributeError` when `allowed_areas` is absent.
* OSM map extraction now subtracts explicit obstacle features from walkable areas and respects configured buffer widths.
* OSM background rendering now validates projected CRS usage and records the UTM CRS in metadata for correct meter-scale overlays.
* Scenario switching during training now tolerates observation-space bound differences across maps while still enforcing action-space compatibility, preventing crashes when sampling mixed-size scenarios.
* Scenario-level single-pedestrian overrides now clone map definitions to prevent cross-scenario contamination during randomized training.
* Goal-distance observations now use a fixed 50m cap with runtime clipping, keeping observation bounds consistent across mixed map sizes.
* SocNav structured observations now cap map-dependent position bounds at 50m with clipping to avoid scenario-switching bound mismatches.
* Expert PPO evaluation now applies the same environment overrides as training, preventing observation-space mismatches during evaluation.
* fast-pysf group gaze force now guards against zero-distance divisions to prevent training crashes.
* Expert PPO CLI now supports `--log-file` to tee stdout/stderr into a log file.
* Scenario loader map resolution now anchors relative paths to `base_dir` directories instead of CWD, preventing off-by-one directory lookups and unintended map matches.
* Pedestrian route sampling now raises a descriptive error instead of dropping obstacle checks after repeated anchor failures.

### Documentation

* Reorganized documentation index with categorized sections (#242)
  + Added clear navigation sections: Getting Started, Benchmarking & Metrics, Tooling, Architecture & Refactoring, Simulation & UI, Figures & Visualization, Performance & CI, Hardware & Environment
  + Added cross-links between core guides for improved discoverability
  + Normalized H1 headings and purpose statements across key documentation files
  + Collapsed legacy detailed index into expandable section for backward compatibility
* Added a benchmark spec doc covering scenario splits, seeds, baseline categories, reproducible commands, and metric caveats.
* Updated benchmark spec for seed-holdout evaluation and policy sweep commands; documented map_id usage in scenario README.
  

### Migration Guide (Version 2.1.0)

**For robot_sf developers and contributors:**

All utility imports must be updated to reference `robot_sf.common` :

```python
# Before (old paths - no longer valid)
from robot_sf.util.types import Vec2D, RobotPose
from robot_sf.utils.seed_utils import set_global_seed
from robot_sf.util.compatibility import validate_compatibility

# After (new paths - required)
from robot_sf.common.types import Vec2D, RobotPose
from robot_sf.common.seed import set_global_seed
from robot_sf.common.compat import validate_compatibility

# Convenience imports also available:
from robot_sf.common import Vec2D, RobotPose, set_global_seed
```

**Why this change?**
* Eliminates navigation confusion from fragmented utility locations
* Improves IDE autocomplete and discoverability
* Reduces cognitive load for new contributors
* Establishes single canonical location for all shared utilities

**Impact:**
* ~50 import statements updated across codebase
* All 923 tests passing after migration
* No functional changes - pure reorganization

**For external consumers (if any):**
If your project imports from `robot_sf.util` or `robot_sf.utils` , update your imports using the patterns above. The behavior of all utilities remains unchanged.

### Added

* **Architecture Decoupling (Feature 149)**: Introduced simulator facade and backend/sensor registries scaffolding behind the existing factory pattern. Default backend is "fast-pysf" with future backend selection via unified config.
* Backend registry integrated into environment initialization: `BaseEnv` now resolves the simulator via a backend key (`env_config.backend`, default "fast-pysf") using `robot_sf.sim.registry`, with a safe fallback to legacy `init_simulators()` for full backward compatibility.
* **fast-pysf Integration Improvements (Feature 148)**: Enhanced fast-pysf integration with comprehensive testing and quality tooling
  + **Map Verification Enhancements (001-map-verification)**
    - Added informational rule `R005` (layer statistics) emitted when Inkscape-labeled groups (`<g inkscape:label="…">`) are present
    - Enhanced `R004` message and remediation guidance (descriptive labeling for obstacles/spawns/waypoints)
    - Inserted Map Verification section into `docs/benchmark.md` (CI invocation, rule table, manifest structure, extension guidance)
    - Added labeled example SVG (`maps/svg_maps/labeled_example.svg`) for demonstrating layer labeling and future semantic tagging
    - Improved internal type hints (`_load_inventory` return type, expanded docstrings) for verification modules
    - Test coverage extended (`test_layer_stats_info`) validating scope resolution and future R005 visibility

  + **Unified Test Suite**: Single `uv run pytest` command now executes both robot_sf (881 tests) and fast-pysf (12 tests) for total 893 tests
  + **Quality Tooling Extension**: Extended ruff linting, ty type checking, and coverage reporting to include fast-pysf subtree
  + **Type Annotations**: Added comprehensive type hints to fast-pysf public APIs (map_loader, forces, simulator, scene modules)
  + **Configuration**: Per-file ignores in pyproject.toml for gradual quality adoption, ty configuration includes fast-pysf/pysocialforce
  + **Code Quality**: Fixed circular imports, removed dead code, alphabetized imports, replaced wildcard imports
  + **Documentation**: Created annotation_plan.md with numba compatibility guidelines and implementation strategy
* **fast-pysf Subtree (Feature 146)**: Integrated `fast-pysf` as a git subtree for easier management and updates
* **pytest-cov Integration (Feature 145)**: Comprehensive code coverage monitoring and CI/CD integration
  + **Automatic Collection**: Coverage data collected automatically during test runs via `pytest-cov` without additional commands
  + **Multi-Format Reports**: Terminal summary, interactive HTML (`htmlcov/index.html`), and machine-readable JSON (`coverage.json`)
  + **Baseline Comparison**: CI/CD pipeline compares coverage against baseline with non-blocking warnings on decreases
  + **VS Code Integration**: Tasks for "Run Tests with Coverage" and "Open Coverage Report"
  + **CI/CD Workflow**: GitHub Actions integration with caching, baseline updates on main branch, and artifact uploads
  + **Library Infrastructure**: 
    - `robot_sf/coverage_tools/baseline_comparator.py`: CoverageSnapshot, CoverageBaseline, CoverageDelta entities with comparison logic
    - `robot_sf/coverage_tools/report_formatter.py`: Multi-format report generation (terminal/JSON/markdown)
    - `scripts/coverage/compare_coverage.py`: CLI tool for local and CI baseline comparison
  + **Comprehensive Testing**: 18 unit tests (5 smoke + 13 baseline comparator) with 91.51% coverage of comparison logic
  + **Documentation**: 
  + `docs/coverage_guide.md`: 500+ line comprehensive guide with quickstart, CI integration, troubleshooting
  + `examples/plotting/coverage_example.py`: Programmatic usage examples
    - Updated `docs/dev_guide.md` with coverage workflow section
  + **Configuration**: pyproject.toml with [tool.coverage.*] sections, automatic pytest integration, parallel execution support
  + Coverage excludes: tests, examples, scripts, fast-pysf submodule per omit configuration
  + Non-blocking CI design: warnings only, no build failures on coverage decreases
* Paper Metrics Implementation (Feature 144): Comprehensive implementation of 22 social navigation metrics from paper 2306.16740v4 (Table 1):
  + **NHT (Navigation/Hard Task) Metrics (11)**: `success_rate`,               `collision_count`,               `wall_collisions`,               `agent_collisions`,               `human_collisions`,               `timeout`,               `failure_to_progress`,               `stalled_time`,               `time_to_goal`,               `path_length`,  `success_path_length` (SPL)
  + **SHT (Social/Human-aware Task) Metrics (14)**: velocity statistics (`velocity_min/avg/max`), acceleration statistics (`acceleration_min/avg/max`), jerk statistics (`jerk_min/avg/max`), clearing distance (`clearing_distance_min/avg`),               `space_compliance`,               `distance_to_human_min`,               `time_to_collision_min`,               `aggregated_time`
  + Extended `EpisodeData` dataclass with optional `obstacles` and `other_agents_pos` fields for enhanced collision detection
  + Internal helper functions: `_compute_ped_velocities`,               `_compute_jerk`,               `_compute_distance_matrix`
  + Comprehensive unit test coverage (30+ tests) with edge case validation
  + All metrics documented with formulas, units, ranges, and paper references
  + Backward compatible integration with existing benchmark infrastructure
* Multi-extractor training flow refactor (Feature 141): `scripts/multi_extractor_training.py` now uses shared helpers in `robot_sf.training`, emits schema-backed `summary.json`/`summary.md` plus legacy `complete_results.json`, captures hardware profiles, honors macOS spawn semantics, and ships default/GPU configs alongside updated SLURM automation and analyzer support.
* Helper Catalog Consolidation (Feature 140): Extracted reusable helper logic from examples and scripts into organized library modules:
  + `robot_sf.benchmark.helper_catalog`: Policy loading (`load_trained_policy`), environment preparation (`prepare_classic_env`), and episode execution (`run_episodes_with_recording`) helpers.
  + `robot_sf.render.helper_catalog`: Directory management (`ensure_output_dir`) and frame capture utilities (`capture_frames`).
* Automated Research Reporting: Multi-seed aggregation & completeness (Feature 270-imitation-report)
  + Seed orchestration (`orchestrate_multi_seed`) combining baseline/pretrained manifests
  + Per-seed manifest parsing (`extract_seed_metrics`) tolerant of JSON/JSONL tracker outputs
  + Completeness scoring (`compute_completeness_score`) with PASS/PARTIAL classification
  + Standardized seed failure logging (`log_seed_failure`) and graceful missing-manifest handling
  + Seed summary table & completeness JSON artifact (`completeness.json`) rendered into `report.md`
  + Tracker manifest parsing (`parse_tracker_manifest`) for run-level status integration
  + Extended aggregation exports (JSON + CSV) and hypothesis evaluation incorporated in multi-seed flow
  + All US2 tasks (T033–T043) implemented with unit & integration test coverage
  + `robot_sf.docs.helper_catalog`: Documentation index management (`register_helper`) for automated helper catalog updates.
  + Complete refactoring of all maintained examples (`examples/demo_*.py`) and scripts (`scripts/*.py`) to use helper catalog functions instead of duplicate logic.
  + Helper registry data structures with typed interfaces for discoverable, testable helper capabilities.
  + All helper functions include comprehensive docstrings, error handling, and Loguru logging compliance.
* Episode Video Artifacts (MVP):
  + New CLI flags for benchmark runner: `--no-video` and `--video-renderer=synthetic|sim-view|none`.
  + Synthetic lightweight encoder that renders a red-dot path from robot positions and writes per‑episode MP4s under `results/videos/`. <!-- active-docs-check: allow historical changelog path -->
  + Episode JSON schema extension to include optional `video` manifest `{path, format, filesize_bytes, frames, renderer}`.
  + End‑to‑end wiring through CLI → batch runner → worker → episode; deterministic file naming `video_<episode_id>.mp4`.
  + Tests: CLI integration (`tests/test_cli_run_video.py`) and programmatic API (`tests/unit/test_runner_video.py`), both skipped when MoviePy/ffmpeg unavailable.
* Environment Factory Ergonomics (Feature 130): Structured `RenderOptions` / `RecordingOptions`, legacy kw mapping layer (`fps`,               `video_output_path`,               `record_video`), precedence normalization and logging diagnostics; performance guard (<10% creation mean regression) and new migration guide (`docs/dev/issues/130-improve-environment-factory/migration.md`). New example: `examples/demo_factory_options.py`.
* Governance: Constitution version 1.2.0 introducing Principle XII (Preferred Logging & Observability) establishing Loguru as the canonical logging facade for library code and prohibiting unapproved `print()` usage outside sanctioned CLI/test contexts.
* Documentation: Development guide updated with new Logging & Observability section summarizing usage guidelines (levels, performance constraints, acceptable exceptions).
* SVG Map Validation (Feature 131): Manual bulk SVG validation script (`examples/svg_map_example.py`) supporting strict/lenient modes, summary reporting, environment override (`SVG_VALIDATE_STRICT`), and compliance spec (FR-001–FR-014). Added missing spawn/goal zones and minimal `robot_route_0_0` / `ped_route_0_0` paths to classic interaction SVG maps and large map asset now includes explicit width/height and minimal routes.

### Fixed

* Normalized obstacle vertices to tuples to prevent ambiguous NumPy truth-value error during SVG path obstacle conversion in large map (`map3_1350_buildings_inkscape.svg`).
* Preserved per-algorithm benchmark aggregation (Feature 142): classic orchestrator now mirrors `algo` into `scenario_params`, aggregation raises on missing metadata, and summary outputs emit `_meta` diagnostics plus warnings when expected baselines are absent.
* Full Classic visuals: auto video renderer now falls back to synthetic when all SimulationView encodes fail, recording the downgrade in `performance_visuals`.

### Migration Notes

* No code changes required; existing Loguru usage already compliant. Any remaining incidental `print()` in library modules should be migrated opportunistically (PATCH) unless tied to user-facing CLI UX.

### Added (Performance Budget Feature 124)

* Per-test performance budget enforcement (soft 20s, hard 60s) with slow test report (top 10) and guidance suggestions.
* Environment variables: `ROBOT_SF_PERF_RELAX` (suppress soft breach enforcement) and `ROBOT_SF_PERF_ENFORCE` (escalate soft breaches to failures).
* Shared performance utilities (`tests/perf_utils/`): policy, reporting, guidance, minimal scenario matrix helper.
* Refactored benchmark integration tests (resume, reproducibility) to use minimal matrix for faster deterministic runs.
* Synthetic slow test and guidance validation tests.

### Added

* Classic Interactions PPO Visualization (Feature 128) – deterministic PPO-driven classic interaction scenario visualization script (`examples/classic_interactions_pygame.py`) with:
  + Constants-based configuration (no CLI) and dry-run validation path
  + Deterministic seed ordering & structured episode summaries (scenario, seed, steps, outcome, success/collision/timeout booleans, recorded)
  + Graceful recording guard (moviepy/ffmpeg optional) with informative skip notes
  + Logging verbosity toggle (`LOGGING_ENABLED`) and performance-friendly frame sampling
  + Improved model load error guidance (actionable download/help message)
  + Headless safety via SDL_VIDEODRIVER=dummy detection
  + Reward fallback integration log (env already falls back to simple_reward)
  + Summary table printer helper for human-readable output
* Benchmark visual artifact integration (plots + videos manifests) for Full Classic Interaction Benchmark:
  + Post-run single-pass generation of placeholder plots and representative episode videos
  + SimulationView-first architecture with graceful synthetic fallback (current release uses synthetic until replay support added)
  + Deterministic selection (first N episodes) and machine-readable manifests: `plot_artifacts.json`,               `video_artifacts.json`,               `performance_visuals.json`
  + Renderer attribution field (`renderer`) and budget timing flags
  + Renderer toggle flag (`--renderer=auto|synthetic|sim-view`) with forced mode diagnostics
  + Replay capture adapter enabling SimulationView reconstruction (episode + step validation)
  + Extended skip-note taxonomy: `simulation-view-missing`,               `moviepy-missing`,               `insufficient-replay`,               `render-error:<Type>`,               `disabled`,               `smoke-mode`
  + Performance split metrics (`first_video_render_time_s`,               `first_video_encode_time_s`) plus memory sampling & over‑budget flags
  + Dependency matrix + lifecycle documentation (`docs/benchmark_visuals.md`) covering fallback ladder and required optional deps (pygame, moviepy/ffmpeg, jsonschema, psutil)
* **Social Navigation Benchmark Platform** - Complete benchmark infrastructure for reproducible social navigation research
* **Full Classic Interaction Benchmark (Initial Implementation)**
  + Synthetic placeholder execution pipeline (planning → execution → aggregation → effect sizes → precision loop)
  + Adaptive sampling with CI half-width early stop targets (collision_rate, success_rate, placeholder snqi target)
  + Plot artifacts (distribution, trajectory, KDE/Pareto/force heatmap placeholders) and annotated video generation (graceful fallback)
  + CLI `scripts/classic_benchmark_full.py` exposing comprehensive flags (episodes, precision thresholds, videos)
  + Manifest instrumentation for runtime, episodes_per_second, scaling_efficiency placeholder metrics
  + Resume idempotency and performance smoke tests (T042, T044) plus failure injection test for videos (T043)
  + Dedicated documentation page `docs/benchmark_full_classic.md`

  + **Episode Runner**: Parallel execution with manifest-based resume functionality
  + **CLI Interface**: 15 comprehensive subcommands covering full experiment workflow
    - `run` - Execute episodes with parallel workers
    - `baseline` - Compute baseline statistics  
    - `aggregate` - Generate summaries with bootstrap confidence intervals
    - `validate-config` - Schema validation for scenarios
    - `list-scenarios` - Display scenario configurations
    - `figure-*` - Distribution plots, Pareto frontiers, force field heatmaps, thumbnails
    - `table` - Generate baseline tables (Markdown/LaTeX)
    - `snqi-*` - SNQI weight recomputation and ablation analysis
    - Additional utilities for trajectory extraction and episode validation
  + **SNQI Metrics Suite**: Composite Social Navigation Quality Index with component breakdown
  + **Statistical Analysis**: Bootstrap confidence intervals and robust aggregation
  + **Unified Baseline Interface**: PlannerProtocol for consistent algorithm comparison
  + **Figure Orchestrator**: Publication-quality visualization pipeline
  + **Comprehensive Testing**: 108 tests including 33 new tests for benchmark functionality

* Classic interactions refactor (Feature 139): Extracted small reusable visualization and formatting helpers into `robot_sf.benchmark` (`visualization.py`,               `utils.py`) and added contract tests and a dry-run smoke test. See docs/dev/issues/classic-interactions-refactor/design.md.
* **Complete Documentation**: Step-by-step quickstart guide with example workflows
* **Performance Validation**: 20-25 steps/second with linear parallel scaling

### Changed

* **fast-pysf Type Annotations (Feature 148)**: Enhanced type safety across fast-pysf public APIs
  + Added return type annotations to all Force classes (`__call__` methods return `np.ndarray`)
  + Added type hints to Simulator and Simulator_v2 classes (step methods, state management)
  + Improved PedState class with proper array type annotations (`np.ndarray | None` for optional attributes)
  + Added `str | Path` type support to map_loader.load_map() for flexible path handling
* Classic interactions pygame demo now respects per-scenario `map_file` entries in the scenario matrix: on each selected scenario it loads the referenced SVG (via converter) or JSON map definition and injects a single-map `MapDefinitionPool` into the environment config. Falls back gracefully (with a warning) to the default pool if loading fails.
* SVG map conversion now validates presence of at least one `robot_route_*_*` path; missing robot routes raises a clear `ValueError` (callers can fallback to default maps) preventing downstream division-by-zero in simulator initialization. Added richer conversion logging (route and zone counts).
* Enhanced baseline planner interface with unified PlannerProtocol
* Improved test coverage with comprehensive benchmark validation
* Updated main documentation with prominent benchmark platform section
* Simplified video encode invocation by replacing signature introspection with ordered fallback attempts (keyword → positional → minimal) for improved maintainability and clearer failure surface.

### Technical Details

* All baseline planners (SocialForce, PPO, Random) now implement PlannerProtocol interface
* Episode schema includes comprehensive metadata and provenance tracking
* Deterministic execution with seed-based reproducibility
* Manifest-based resume enables incremental experiment extension
* Bootstrap statistical analysis provides uncertainty quantification
* Publication-ready figure generation with LaTeX table support

### Migration Notes

* No breaking changes to existing functionality
* New benchmark CLI commands available via `python -m robot_sf.benchmark.cli`
* Legacy environment interfaces remain fully supported
* Comprehensive backward compatibility maintained

### Fixed

* **fast-pysf Code Quality Issues (Feature 148 / PR #236)**: Resolved 24 PR review comments
  + Fixed circular import in forces.py (changed `from pysocialforce import logger` to `from pysocialforce.logging import logger`)
  + Removed dead code from scene.py (commented desired_directions method)
  + Alphabetized imports in simulator.py for consistency
  + Removed duplicate simulator assignment in ex09_inkscape_svg_map.py
  + Replaced wildcard import with explicit import in TestObstacleForce.py
  + Fixed file path resolution in test_map_loader.py to work with dynamic paths
  + Added per-file ruff ignores for complexity and print statements in examples
* Video artifact manifest now emits a per-episode `skipped` entry with `moviepy-missing` note instead of silently omitting episodes when SimulationView encoding is unavailable for that episode only.
* Robot / multi-robot environments now gracefully fallback to `simple_reward` when `reward_func=None` is passed via factory functions, preventing a `TypeError: 'NoneType' object is not callable` during `env.step` (affects new classic interactions PPO visualization demo).

---

## Guidelines for Contributors

When adding entries to this changelog:

1. **Group changes** by `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`
2. **Write for users** - focus on user-visible changes and their benefits
3. **Include migration notes** for breaking changes
4. **Reference related issues/PRs** where applicable
5. **Keep descriptions concise** but informative

## Version Numbering

This project uses semantic versioning:
* **MAJOR** version for incompatible API changes
* **MINOR** version for backwards-compatible functionality additions  
* **PATCH** version for backwards-compatible bug fixes
