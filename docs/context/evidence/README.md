# Evidence Bundles

This directory contains small, reviewable copies of generated artifacts that are worth preserving in
git because they support durable context notes, benchmark decisions, or PR handoff.

Do not mirror `output/` here wholesale. Generated files under `output/` remain worktree-local and
may be deleted when they are reproducible from a tracked config, seed schedule, commit, and command.

## What Belongs Here

- compact `summary.json` files that anchor a promoted benchmark or policy-search decision,
- Markdown/JSON analysis reports used by tracked context notes,
- small CSV/JSON tables needed to review a campaign without rerunning it,
- manifests and checksums for the copied evidence.

## What Stays Out

- raw episode JSONL files unless a tiny curated subset is required for a regression fixture,
- large Slurm stderr/stdout logs,
- model checkpoints and model caches,
- coverage HTML,
- temporary repos, caches, and scratch outputs.

For larger artifacts that are not cheap to regenerate but are not reviewable in git, use an external
artifact store and track only a manifest or registry pointer here.

Git LFS is not the default storage type for generated benchmark artifacts. Use it only for a
deliberately versioned, non-regenerable binary fixture after an explicit maintainer decision. If an
artifact can be regenerated from tracked configs, seed schedules, commands, and commits, it is fine
to leave it ignored or delete it locally once the durable summary/report evidence is preserved.

## Review Fallback

When automated AI review is rate-limited, unavailable, or configured to skip a small evidence file
under this tree, reviewers should apply the fallback checklist in `docs/code_review.md`. CSV
evidence is not exempt from review merely because broad CSV filters exist: inspect the header,
representative rows, provenance fields, parent-issue conclusion alignment, and explicit claim
boundary before merging an evidence-producing PR.

## Evidence Bundle v1

Use `evidence_bundle.v1` for the smallest reproducible packet that lets a future reviewer recover
what compact evidence supported a claim boundary without depending on local `output/` contents.
The helper is intentionally explicit-file-only:

```bash
uv run python scripts/tools/benchmark_publication_bundle.py evidence-bundle \
  --source-root docs/context/evidence/<source-or-fixture> \
  --out-dir docs/context/evidence \
  --bundle-name <issue-or-claim-bundle> \
  --file summary.json \
  --file metric_table.csv \
  --file trace_manifest.yaml \
  --file claim_boundary.md \
  --command "<canonical reproduction or validation command>" \
  --commit <git-commit> \
  --claim-boundary diagnostic_only_not_benchmark_evidence
```

Each bundle writes:

- `payload/`: exactly the selected compact files;
- `evidence_bundle_manifest.json`: `schema_version: evidence_bundle.v1`, command, commit,
  claim boundary, source root, file index, sizes, SHA-256 checksums, and policy caveats;
- `checksums.sha256`: checksum lines for every payload file.

The schema contract lives at `robot_sf/benchmark/schemas/evidence_bundle.v1.json`.

For large artifacts that should remain outside git, the same command supports opt-in mirror
metadata:

```bash
uv run python scripts/tools/benchmark_publication_bundle.py evidence-bundle \
  --source-root docs/context/evidence/<source-or-fixture> \
  --out-dir docs/context/evidence \
  --bundle-name <issue-or-claim-bundle> \
  --file summary.json \
  --command "<canonical reproduction validation command>" \
  --commit <git-commit> \
  --claim-boundary diagnostic_only_not_benchmark_evidence \
  --mirror-dry-run-base-uri s3://example-bucket/evidence
```

`--mirror-dry-run-base-uri` writes `mirror_manifest.json` with deterministic remote URI targets,
sizes, SHA-256 checksums, MIME hints, and `upload_status: dry_run` without contacting a remote
service. `--mirror-local-dir <path>` is a credential-free local backend for tests and staging; it
copies the selected payload files and records `file://` URIs with `upload_status: uploaded`.
Mirroring is off by default, credentials must not appear in manifests or logs, and mirror metadata
does not upgrade a diagnostic bundle into benchmark-strength or paper-facing evidence.

## Claim Readiness Check

Before promoting compact evidence into a stronger claim, run the claim-readiness guardrail against
the claim note and evidence path:

```bash
uv run python scripts/validation/check_claim_readiness.py \
  --claim-file docs/context/issue_1542_manuscript_claim_evidence_map.md \
  --evidence docs/context/evidence/<bundle-or-report>
```

The checker reports missing fields for evidence tier, comparator or baseline, mechanism activation,
trace support, artifact provenance, seed or slice boundary, claim boundary, and fallback/degraded
limitation handling. It is a guardrail only: readiness output does not establish scientific truth,
benchmark success, safety, or paper-grade sufficiency.

Required provenance fields:

- `command`: the command that produced or validates the evidence;
- `commit`: the repository commit associated with the evidence;
- `claim_boundary`: the conservative interpretation boundary for the bundle;
- `files`: compact, reviewable payload entries with `path`, `size_bytes`, `sha256`, and `kind`.

Policy caveats:

- large raw logs, videos, checkpoints, raw episode streams, and mirrored `output/` trees stay out;
- a bundle makes evidence references reproducible, but does not by itself establish a
  benchmark-strength, paper-grade, or scientifically sufficient claim;
- fallback/degraded/diagnostic status must be preserved in `claim_boundary` or a payload note.

## Current Bundles

- `issue_1554_job_13198_constraints_first_analysis/`: diagnostic-only constraints-first artifact
  bundle for completed job 13198. Preserves public completed-job metadata, records missing retained
  private metrics/decision-packet inputs as blockers, fails closed SNQI adjacent-rank statements,
  confirms no Slurm submission and no paper/dissertation claim promotion.

- `issue_1554_slurm_evidence_2026-06-30/`: queue-decision packet over completed #1554
Slurm jobs 13192, 13198, and 13203. Classifies job 13198 as the completed S20/H500
result matrix to analyze before any duplicate rerun, preserves its soft SNQI contract
warning as a paper-claim blocker, and confirms no Slurm/GPU submission, artifact
deletion, or paper/dissertation claim edit.

- `issue_3798_post_13175_s20_s30_evidence_gap_packet.{md,json}`: diagnostic-only packet over
retrieved job 13175 S20/H500 artifacts for the post-#1554 evidence gap. Names the compact
reviewable metadata files, records that S30 remains an unexecuted escalation path, and preserves
the no-submit/no-claim boundary for paper or dissertation use.

- `issue_3653_snqi_decision_disagreement_job_13175/`: diagnostic-only SNQI scalarization-sensitivity
application on hydrated job 13175 S20/H500 episode evidence. Includes report JSON, planner rows,
decision-disagreement CSV, Markdown, Pareto SVG, and provenance; raw episode JSONL stays ignored.

- `issue_3207_simulator_dependence_validity_boundary_packet_2026-06-29/`: checker packet over
  the merged #3207 bounded actual fidelity-sensitivity slice. Classifies the current evidence as
  `no_claim` / `not_benchmark_evidence` because the slice is not full fixed scope and rank evidence
  is non-identifiable (`primary_metric_zero_variance`). Does not run a simulator study or promote a
  simulator-dependence, simulator-realism, sim-to-real, paper-facing, or dissertation claim.
- `issue_2557_replica_readiness_packet_2026-06-29/`: diagnostic-only
  fixed-seed queue-fill replica readiness packet for Issue #2557. It records the
  public tracked completed/running status, retrieved compact evidence, remaining
  artifact-promotion gap, no-new-Slurm recommendation, cost/risk, and exact
  local packet command without promoting benchmark or paper-facing claims.
- `issue_3484_feasibility_diagnostics/`: reserved location for small dry-run
  manifests and follow-up summaries for universally-failing scenario-family
  feasibility diagnostics. Outputs from
  `scripts/tools/build_feasibility_diagnostic_manifest.py` are diagnostic planning
  artifacts only: rows start as `not_run` / `needs_evidence`, do not run planners,
  do not certify route clearance, and are not benchmark, safety, paper, or
  dissertation evidence.
- `issue_3254_predictive_crossing_conflict_13042_2026-06-23/`: analysis-only
  negative result for the schema-fixed Issue #3254 predictive crossing-conflict
  rerun. Training completed on a non-degenerate `predictive_ego_v1` dataset, but
  final evaluation failed the success-rate gate (`0.08696 < 0.3`), so the run is
  not benchmark-strength or paper-facing evidence and should not be blindly
  resubmitted.
- `issue_3342_nearfield_turn_budget_2026-06-21/`: diagnostic-local S20 follow-up
  for the Issue #3215 near-field turn-budget signal. It compares `baseline`,
  `nearfield_turn`, and `nf_speedcap_only` across clean and observation-noise
  slices with explicit success, collision, min-distance, and uncertainty fields.
  The local S20 result does not support adopting the near-field signal; S30 is
  configured but not run. Not benchmark-strength or paper-facing evidence.
- `issue_2777_live_observation_noise_replay/issue_3330_seed_amplitude_grid/`: diagnostic-only
  native live replay over the #3323 near-field #2756 fixture with a predeclared 2x3 medium/high
  bounded-Gaussian perturbation grid. The grid is `medium_amplitude_sensitive`: seed 3328 is
  behavior-sensitive at medium amplitude, seeds 3328 and 3330 are behavior-sensitive at high
  amplitude, and seed 2755 remains policy-insensitive. Not robustness, calibrated sensor-realism,
  planner-superiority, paper-facing benchmark, or scenario-general evidence.
- `issue_3294_release_claim_matrix/`: reviewable v0.1 release claim matrix assembled from
  existing release artifacts, leaderboard row-claim sidecars, release config metadata, and ODD
  coverage rows. It classifies rows as benchmark evidence, diagnostic evidence, or non-claim
  without running a new benchmark campaign or promoting fallback/degraded/unavailable rows.
- `issue_2919_scenario_prior_gap_2026-06-21/`: analysis-only authored-vs-repository-trace-derived
  scenario-prior gap report from the Issue #2917 card registry. It compares pedestrian density,
  pedestrian speed, and timing-offset parameter families, emits scenario-family proposals, and
  explicitly defers dataset-backed SDD/ETH/AMV comparison to #3161. Not planner-ranking,
  benchmark-superiority, or real-world representativeness evidence.
- `issue_2924_counterfactual_pair_2026-06-21/`: analysis-only counterfactual pair
  runner evidence for a matched prediction-risk fixture. It records invariant enforcement,
  mechanism-trace activation delta, `min_clearance_m` outcome delta, and diagnostic trace panels.
  Not benchmark-strength, paper-grade, or planner-superiority evidence.
- `issue_1470_oracle_imitation_traces_12911_2026-06-17/`: tracked closeout evidence
  for Slurm job `12911` on #1470/#2441. The bundle includes the small six-row oracle
  imitation trace JSONL and manifest. It is trace-collection evidence only, not final
  imitation training or planner promotion evidence.
- `issue_1475_orca_residual_bc_smoke_12913_2026-06-17/`: tracked failed-closed
  smoke evidence for Slurm job `12913` on #1475. The smoke reached finalization but
  blocked nominal escalation because required residual/guard/artifact evidence fields
  were missing and the single episode timed out with low progress.
- `issue_2767_benchmark_table_candidates/`: draft-only benchmark-results table candidates generated
  from tracked claim/evidence inputs. The bundle is a synthesis review surface only; stale,
  diagnostic, unavailable, fallback, degraded, proxy-only, and missing-denominator evidence remains
  caveated or blocked and is not promoted to manuscript evidence.
- `issue_2788_negative_result_scenario_candidates/`: diagnostic-only generated scenario candidate
  manifests derived from negative-result register entries NR-001 and NR-002. The manifests encode
  clearance-targeted topology variants and a near-field observation-noise successor candidate;
  all remain `not_promoted` with null severity/diversity metrics and are not benchmark evidence.
- `issue_2751_topology_reselection_runtime/`: diagnostic-only runtime evidence for the
  clearance-targeted topology-reselection successor. The result is `revise`: all hard slices
  remained `horizon_exhausted`, while the negative-control rows succeeded with zero topology
  switching. Not benchmark or paper-facing evidence.
- `issue_2755_observation_noise_envelope_2026-06-13/`: diagnostic trace-derived
  near-field observation-noise robustness envelope. Evaluates seven perturbation conditions
  (noop, low/medium Gaussian noise, missed detection, occlusion, delay, combined) against
  the occluded-emergence trace fixture. Delay-only is trace-derived robustness evidence;
  other conditions are diagnostic_only or scenario_too_weak. Not paper-facing benchmark evidence.
- `issue_2782_observation_noise_mechanisms/`: diagnostic mechanism-layer classification for the
  Issue #2755 observation-noise envelope. Maps each perturbation condition to the pipeline layer
  where the perturbation did, did not, or could not be shown to propagate. Not paper-facing
  benchmark evidence.
- `issue_2765_dense_pedestrian_stress_2026-06-14/`: diagnostic trace-derived dense-pedestrian
  observation-noise envelope. Evaluates ten perturbation conditions against a deterministic
  three-pedestrian stress fixture; eight conditions expose forecast ambiguity while the full missed
  detection condition is classified as scenario-too-weak. Uses stored trace action proxies, not
  live planner replay or paper-facing benchmark evidence.
- `issue_3200_density_runtime_smoke_summary.json`: diagnostic-only same-seed runtime smoke over
  coverage-entropy-selected pedestrian-density candidates. The coverage report had no literal
  redundant rows, so the comparator uses lowest-novelty review rows; all four rows ended
  `horizon_exhausted` and count as no benchmark-success evidence.
- `issue_2756_occluded_emergence_2026-06-13/`: smoke/diagnostic note for the
  deterministic occluded-emergence trace fixture. The fixture separates ground-truth and observed
  pedestrian state, records first visibility and conflict timing, and feeds observation-noise plus
  constant-velocity forecast diagnostics.
- `issue_2780_occluded_emergence_variants/`: smoke/diagnostic/stress evidence for five simulated
  occluded-emergence variant fixtures varying emergence side, pedestrian speed, first-visible
  distance, conflict timing, and robot approach speed. Four variants are safety-relevant under
  live replay; none replace #2777 live replay evidence.
- `issue_2757_cv_forecast_eval_2026-06-13/`: diagnostic-only constant-velocity
  pedestrian forecast baseline evaluation on bounded durable trace fixtures. The corridor traces
  produce short-horizon metrics; crossing, bottleneck, signalized, occluded, and dense-interaction
  coverage remains limited or unavailable.
- `issue_2774_motion_rich_forecast_traces_2026-06-14/`: diagnostic-only refresh of the
  constant-velocity pedestrian forecast baseline that promotes the durable occluded-emergence
  trace fixture as non-corridor forecast evidence. Corridor and occluded-emergence traces are
  evaluated; crossing and bottleneck remain zero-motion limitations, and signalized/dense/bottleneck
  motion-rich coverage remains unavailable.
- `issue_2915_forecast_baselines_2026-06-20/`: analysis-only comparison of constant-velocity,
  semantic CV, and interaction-aware CV baselines on identical configured trace-origin forecast
  batches. The bundle includes one ForecastBatch.v1 JSONL per baseline, a comparison table, and a
  report naming strongest baselines per scenario family with fail-closed unavailable rows.
- `issue_2749_observation_noise_diagnostics/`: diagnostic-only paired no-op vs
  perception-limited observation-noise step diagnostics for `hybrid_rule_v0_minimal` on a
  pedestrian-present stress-slice scenario; perturbation plumbing activated, but the distant
  pedestrian produced no measurable planner degradation.
- `issue_2777_live_observation_noise_replay/`: diagnostic stress-slice live replay for all seven
  Issue #2755 perturbation families on the Issue #2756 occluded-emergence boundary added by
  Issue #3320 and moved into near-field geometry by Issue #3323. The wrapper ran `risk_surface_dwa_v0` on
  `issue_2756_occluded_emergence` seed 111, preserved no-op first visibility at step 5 and
  delay-only first observation at step 7, reached a 1.6552 m closest robot-pedestrian distance,
  and classified the seven-family run as `policy_insensitive` because perturbations changed
  observations but not commands or progress/risk summaries. The nested
  `issue_3328_behavior_probe/` evidence adds an opt-in high-amplitude noise probe that changes
  commands/progress on the same one-seed fixture and is classified
  `behavior_sensitive_diagnostic_only`. Diagnostic stress evidence only; not a robustness claim.
- `issue_3201_observation_noise_live_smoke/`: diagnostic-only same-seed clean vs perturbed
  step-diagnostics replay on the `dense_pedestrian_stress` live matrix candidate. Perturbation
  changed planner-input pedestrian observations, but commands and progress/risk summaries stayed
  identical because the live scenario remained outside the intended 2 m near-field target. Not
  benchmark or sensor-realism evidence.
- `issue_2927_observation_quality_live_smoke/`: smoke/diagnostic observation-quality wrapper over
  the committed near-field live step-diagnostics summary. It attaches validated
  `observation_quality.v1` metadata, reports false-negative safety effects, and explicitly excludes
  false-positive actor rows as not available; not planner-superiority or hardware-calibrated sensor
  evidence.
- `issue_3300_false_positive_actor_injection/`: diagnostic same-seed live step-diagnostics replay
  adding an observed-only false-positive actor to the near-field fixture. The false-positive slice
  changed the command sequence to repeated stop commands without collision flags over the five-step
  smoke, while the missed-detection slice hit one pedestrian collision flag by step 3. Not
  benchmark-strength, hardware-calibrated sensor, or paper-facing evidence.
- `issue_3206_pedestrian_archetype_reporting_2026-06-20/`: composition-report-only packet for
  the shipped pedestrian speed-archetype MVP. Records deterministic counts, speed factors, and
  no-result boundaries for later homogeneous-vs-heterogeneous smoke runs.
- `issue_3207_fidelity_sensitivity_launch_packet_2026-06-20/`: launch-packet-only simulation
  fidelity sensitivity contract for Issue #3207. Defines the tracked config, fidelity axes, required
  rank-stability and metric-drift outputs, and no-evidence boundary before any sweep is run.
- `issue_3207_fidelity_sensitivity_smoke_2026-06-20/`: diagnostic-only two-planner live smoke for
  Issue #3207 over the same compact scenario surface. It materializes rank-stability and metric-drift
  calculations across clean timestep variants and the existing observation-noise smoke profile; it
  is not benchmark-strength planner-ranking, sensor-realism, or sim-to-real evidence.
- `issue_3207_fidelity_sensitivity_actual_slice_2026-06-20/`: bounded actual two-planner local
  campaign slice for #3207 over the compact scenario surface. It runs 54 episodes across timestep,
  pedestrian-archetype, observation-noise, and clearance-radius variants; rank order stayed stable
  on the success-rate tie-breaker with no rank flips. This remains bounded sensitivity evidence,
  not full fixed-scope, simulator-realism, sim-to-real, or paper-facing planner-ranking evidence.
- `issue_3233_near_field_observation_noise/`: diagnostic-only same-seed clean vs perturbed
  step-diagnostics replay on a deterministic near-field live fixture. The clean baseline reached
  1.45 m closest robot-pedestrian distance and selected low-speed commands under dynamic-collision
  pressure; the perturbed run changed command sequence/progress and ended with a pedestrian
  collision. Not benchmark or sensor-realism evidence.
- `issue_3335_observation_noise_cross_fixture_2026-06-21/`: diagnostic-only cross-fixture
  synthesis of tracked native live observation-noise grid summaries. It records the committed #3330
  near-field seed/amplitude grid as behavior-sensitive, then preserves a #3335 second-matrix attempt
  that failed closed because `issue_3320_occluded_emergence_live_replay` was not near-field
  (`closest_robot_ped_distance=21.80 m`). This is useful negative external-validity evidence, not
  benchmark, robustness, planner-superiority, or hardware-calibrated sensor-realism evidence.
- `issue_3202_ammv_anticipatory_conflict/`: diagnostic-only default-vs-AMMV Social Force
  comparison on a waiting-then-crossing fixture plus a direct AMMV mechanism probe. Adapter rows
  stayed identical and lacked AMMV metadata; the direct probe activated AMMV force and produced
  speed/lateral-velocity/clearance deltas. Not benchmark or paper-facing behavior-model evidence.
- `policy_search_h500_2026-05-06/`: h500 policy-search leader summaries and failure reports that
  support the v1 raw-success leader and v2 strict-gate promotion decision.
- `issue_1023_scenario_horizons_preflight_2026-05-06/`: compact preflight artifacts for the
  paper-facing scenario-horizon benchmark config.
- `issue_1023_scenario_horizons_local_full_2026-05-06/`: compact local non-Slurm full-campaign
  artifacts, analyzer output, and fixed-vs-scenario comparison for issue #1023.
- `issue_2542_dissertation_export_bundle/`: compact dissertation table bundle refreshed by
  Issue #3203 with tracked scenario-horizon table payloads, manifest, checksums, and durable compact
  source snapshots. The source campaign exited invalid because PPO partial-failed and SNQI contract
  status was `fail`; use only for discussion/provenance, not benchmark-success or ranking claims.
- `issue_1045_h500_solvability_mechanisms_2026-05-07/`: aggregate mechanism classification for
  h500 fixed-timeout relief, including explicit trace-required boundaries for wait-then-go claims.
- `issue_1111_carla_setup_smoke_2026-05-18/`: compact setup-only CARLA T1 oracle smoke evidence
  proving the optional Python API and T0 payload-selection boundary without live replay claims.
- `issue_1239_human_model_transfer_2026-05-18/`: compact human-model transfer smoke evidence with
  explicit variant/source rows and fail-closed upstream adapter availability.
- `issue_1169_carla_live_replay_2026-05-18/`: compact Docker-backed CARLA live replay summaries
  proving client/server connectivity and the fail-closed static-geometry boundary.
- `issue_1467_carla_replay_metrics_2026-05-24/`: compact native CARLA replay metric smoke
  evidence showing the T1 live replay path can emit metrics and produce comparable parity rows.
- `issue_1442_carla_native_spawn_probe_2026-05-24/`: compact CARLA runtime evidence showing the
  certified #1111 payload still adapts by projection while a generated CARLA-aligned native-spawn
  probe reaches `oracle-replay` without adaptation.
- `issue_1344_paired_amv_primary_2026-05-20/`: compact paired nominal/stress AMV primary-row
  campaign summaries and tables.
- `issue_1569_amv_actuation_smoke_2026-05-27/`: compact local smoke summary for the synthetic AMV
  actuation-envelope stress slice, including row-status classification, actuation diagnostics, and
  the explicit non-paper-facing claim boundary.
- `issue_2224_amv_actuation_ranking_2026-06-04/`: compact matched `amv_actuation_smoke`
  comparison showing `actuation_aware_hybrid_rule_v0` reduced command clipping versus
  `hybrid_rule_v3_fast_progress` on one smoke row, while both candidates still timed out.
- `issue_3181_amv_feasibility_ranking_2026-06-20/`: compact 2-scenario x 2-seed paired
  diagnostic slice showing the actuation-aware variant reduced or tied command clipping, while
  success stayed zero and the result remains non-paper-facing.
- `issue_2259_amv_clipping_success_boundary_2026-06-05/`: compact parent-lane synthesis for
  Issue #2259, separating synthetic command-feasibility improvement from unchanged navigation
  success and keeping calibrated AMV actuation blocked.
- `issue_2308_amv_timeout_trace_2026-06-05/`: compact trace-level AMV timeout analysis showing
  reduced clipping did not change the route/task progress blocker on the matched smoke row.
- `issue_2404_amv_timeout_decomposition_2026-06-06/`: compact analysis-only synthesis mapping the
  Issue #2308 AMV timeout trace rerun onto the Issue #2404 requested decomposition fields and
  decision-output vocabulary.
- `issue_2405_amv_step_export_2026-06-06/`: compact diagnostic proof that the Issue #2168
  default/AMMV Social Force path can regenerate selected aggregate rows with step frames and export
  one selected row per side as loader-valid `simulation_trace_export.v1`.
- `issue_2428_mechanism_trace_panels_2026-06-06/`: compact diagnostic-only AMMV/default Social
  Force trace-panel bundle proving the #2405 single-row export path can feed the trajectory-panel
  renderer and preserve reviewable PNG/PDF/checksum artifacts.
- `issue_2981_orca_residual_emission_2026-06-17/`: compact diagnostic-only ORCA residual
  mechanism-trace emission proof from tracked planner-decision fixture input through the scripted
  `mechanism_trace.v1` row emitter.
- `issue_2430_ammv_trace_annotation_2026-06-06/`: compact frame-level parity decision showing the
  Issue #2428 selected default/AMMV trace pair is telemetry-rich but frame-identical, so it is not
  behavioral-difference evidence.
- `issue_2432_ammv_trace_selection_2026-06-06/`: compact three-seed AMMV/default trace-selection
  check showing the local Issue #2168 head-on-corridor slice still has no non-identical
  behavioral-difference pair.
- `issue_2434_ammv_scenario_sweep_2026-06-06/`: compact five-family AMMV/default episode-metric
  screen showing the selected classic close-interaction adapter slice still has no non-identical
  pair.
- `issue_3170_amv_feasibility_ranking_stress_2026-06-20/`: compact diagnostic synthesis showing
  that the broadest tracked multi-scenario AMMV/default evidence is frame-identical while the
  nonzero actuation-aware feasibility signal remains a one-scenario/one-seed slice.
- `issue_2704_progress_gated_topology_successor/`: compact paired topology diagnostic summary
  showing the progress-gated primary-route reselection successor runs but remains a `revise`
  result, with unchanged route progress and `horizon_exhausted` outcome on the canonical h160
  double-bottleneck slice.
- `issue_2313_local_baseline_quarantine_2026-06-05/`: compact metadata-only planner summary
  showing the seven absent local-only baseline rows are explicitly unavailable.
- `issue_2409_local_baseline_quarantine_2026-06-06/`: compact follow-on synthesis showing the
  Issue #2313 quarantine already covered the seven local-only baseline rows and no durable artifact
  source was newly found or claimed.
- `issue_1454_s10_preflight_2026-05-22/`: compact preflight evidence for the staged S10 fixed-h100
  and scenario-horizon h500 robustness configs.
- `issue_1454_stage_a_fixed_h100_2026-05-22/`: compact Stage A full-campaign, analyzer, and
  May 4 comparison evidence for the issue #1454 S10 fixed-h100 broader-baseline gate.
- `issue_1454_s10_h500_candidates_2026-05-23/`: compact exploratory S10 scenario-horizon h500
  evidence for the seven functioning Stage A rows plus local policy-search candidate rows, with a
  pointer to the non-package GitHub artifact release for the raw campaign archive.
- `issue_1608_seed_sensitivity_2026-05-30/`: derived scenario seed-sensitivity classifications
  over the issue #1454 S10/h500 candidate evidence, with top-four planner selection and hard/easy
  seed summaries.
- `issue_1462_s10_h500_failure_modes_2026-05-24/`: compact scenario, candidate-vs-core, and seed
  failure-mode tables derived from the issue #1454 S10/h500 candidate evidence.
- `issue_1428_orca_residual_lineage_2026-05-24/`: compact diagnostic-contract evidence for the
  pre-SLURM ORCA-residual behavior-cloning lineage packet.
- `issue_1396_shielded_ppo_launch_packet_2026-05-24/`: comparison-freeze fixture for the
  shielded-PPO repair pre-SLURM launch-packet validator.
- `issue_1395_learned_risk_launch_packet_2026-05-24/`: trace-contract and baseline-freeze
  fixtures for the learned-risk-model pre-SLURM launch-packet validator.
- `issue_1397_oracle_imitation_launch_packet_2026-05-24/`: dry-run fixture and checksum evidence
  for the pre-Slurm oracle-imitation launch-packet validator.
- `issue_2441_oracle_imitation_traces_2026-06-06/`: compact SLURM finalization,
  split-check, metric summary, and source checksums for the Issue #2441 train/validation
  oracle-imitation trace collection; raw traces remain local-only until durable promotion.
- `issue_2557_reward_curriculum_partial_2026-06-08/`: compact partial 10M-step reward-curriculum
  seed-replica evidence for the first five completed Issue #2557 training jobs, with W&B pointers
  and an explicit incomplete-batch claim boundary.
- `issue_1318_teb_corridor_deadlock_2026-05-20/`: compact TEB/ORCA/hybrid-rule
  classic-merging corridor-deadlock comparison summary for issue #1318.
- `issue_1484_broader_cross_kinematics_2026-05-28/`: compact broader cross-kinematics campaign
  summaries, parity tables, analyzer output, and checksums for issue #1484.
- `issue_1501_adversarial_smoke_2026-05-28/`: compact row-status, sampler-comparison, failure
  archive, and checksum evidence for the first `crossing_ttc` adversarial smoke job.
- `issue_1502_adversarial_two_family_2026-05-31/`: compact row-status, sampler-comparison,
  failure-archive, guided-route-search, and checksum evidence for the bounded two-family adversarial
  comparison run.
- `camera_ready_all_planners_2026-05-04/`: compact camera-ready all-planners campaign summaries and
  reports from the May 4 run.
- `issue_1692_topology_hypothesis_probe_2026-05-30/`: compact diagnostic-only evidence for the
  topology-hypothesis trace probe on `classic_realworld_double_bottleneck_high` seed 111.
- `issue_2403_topology_selection_score_2026-06-06/`: compact analysis-only synthesis mapping the
  Issue #2307 instrumented topology rerun onto the Issue #2403 requested selection-score fields and
  decision-output vocabulary.
- `issue_1904_scenario_perturbation_pilot_2026-05-31/`: compact diagnostic-only paired
  no-op-versus-route-offset pilot summary for the first #1610 scenario perturbation criticality
  slice.
- `issue_1933_perturbation_seed_coverage_2026-05-31/`: compact diagnostic-only seed-limit-4
  paired no-op-versus-route-offset summary for the #1610 scenario perturbation criticality pilot.
- `issue_1935_stronger_perturbation_planner_2026-05-31/`: compact diagnostic-only seed-limit-4
  paired no-op-versus-route-offset summary with one stronger policy-search local planner added to
  the #1610 scenario perturbation criticality pilot.
- `issue_1937_ped_route_offset_2026-05-31/`: compact diagnostic-only seed-limit-4 paired
  no-op-versus-route-offset summary with the pedestrian-route-offset perturbation family added to
  the #1610 scenario perturbation criticality pilot.
- `issue_1939_corridor_trace_response_2026-05-31/`: compact diagnostic-only closest-approach trace
  slices for the `classic_head_on_corridor_low` pedestrian-route-offset response observed in the
  Issue #1610/Issue #1937 pilot.
- `issue_1941_ped_timing_phase_2026-05-31/`: compact diagnostic-only paired
  no-op-versus-start-delay summary for the #1610 single-pedestrian timing phase perturbation pilot.
- `issue_1943_ped_speed_perturbation_2026-05-31/`: compact diagnostic-only paired
  no-op-versus-speed summary for the #1610 single-pedestrian speed perturbation pilot.
- `issue_1945_orca_leave_group_speed_trace_2026-06-01/`: compact diagnostic-only ORCA
  closest-approach trace slices for the `francis2023_leave_group` speed-offset outcome flip.
- `issue_1947_intersection_wait_timing_speed_trace_2026-06-01/`: compact diagnostic-only
  closest-approach trace slices comparing `francis2023_intersection_wait` timing and speed
  perturbation responses.
- `issue_1949_ped_wait_duration_perturbation_2026-06-01/`: compact diagnostic-only paired
  no-op-versus-wait-duration summary for the #1610 single-pedestrian wait-at perturbation pilot.
- `issue_1951_intersection_wait_phase_grid_2026-06-01/`: compact diagnostic-only paired
  no-op-versus-perturbation summary and family/magnitude rollup for the `francis2023_intersection_wait`
  single-pedestrian phase grid.
- `issue_1953_intersection_wait_speed_grid_trace_2026-06-01/`: compact diagnostic-only
  closest-approach trace slices for the strongest #1951 intersection-wait speed-grid response.
- `issue_2176_remaining_one_factor_h80_2026-06-03/`: compact diagnostic-only h80 summary for the
  remaining selected Issue #2170 one-factor hybrid component comparisons, including the partial
  selector-only row caveat.
- `issue_2178_selector_orca_extra_h80_2026-06-03/`: compact diagnostic-only h80 summary for the
  corrected selector-only one-factor comparison after syncing the `orca` extra and proving `rvo2`
  availability.
- `issue_2180_one_factor_h500_2026-06-03/`: compact local h500 diagnostic summary for the
  Issue #2170 one-factor hybrid component manifest after all rows completed 18/18 with zero failed
  jobs.
- `issue_2182_component_synthesis_2026-06-03/`: compact component-effect classification table
  synthesized from the Issue #2180 h500 run for Issue #2104 closeout.
- `issue_2225_learned_policy_failure_synthesis_2026-06-04/`: compact learned-policy failure-mode
  evidence table synthesizing BC warm-start PPO, shielded PPO repair, ORCA-residual BC, learned risk
  model, and oracle imitation status against hybrid-rule mechanism evidence for Issue #2225.
- `issue_2261_static_recenter_slice_local_2026-06-05/`: compact analysis-only summary for
  Issue #2261, explaining why static recentering stays slice-local after the held-out smoke and
  naming the missing activation trace needed for definitive attribution.
- `issue_2306_static_recenter_activation_trace_2026-06-05/`: compact instrumented rerun evidence
  showing static recentering stayed inactive on the unsolved held-out row from Issue #2221.
- `issue_2402_static_recenter_activation_2026-06-06/`: compact analysis-only synthesis mapping the
  Issue #2306 instrumented static-recenter rerun onto the Issue #2402 requested activation fields
  and decision-output vocabulary.
- `issue_2438_static_recenter_activation_closure_2026-06-07/`: compact diagnostic-only closure for
  Issue #2438, reusing the Issue #2306/#2402 activation evidence and stopping the current held-out
  transfer route without another rerun.
- `issue_2588_static_deadlock_controlled_trace/`: compact controlled-trace summary for the
  Issue #2452 static-deadlock suite after Issue #2586 reportability, showing one active
  trace-change row and no terminal success rescue.
- `issue_2590_escape_recenter_static_deadlock_controlled_trace/`: compact controlled-trace summary
  for the Issue #2452 `escape_recenter_pair` follow-up, showing the same active trace-change row
  and no terminal success rescue against the escape-only baseline.
- `issue_2592_static_deadlock_active_row_h500/`: compact h500 horizon-sensitivity summary for the
  shared active static-recenter row, showing both recenter pairings reached success at step 122
  while matched baselines remained 500-step local-minimum failures.
- `issue_2594_static_deadlock_broader_h500/`: compact broader h500 static-deadlock summary for the
  predeclared 3-scenario x 3-seed slice, showing complete trace fields and terminal rescue on the
  only unsolved active row while the other pair-rows were already solved.
- `issue_2273_learned_risk_trace_preflight_2026-06-05/`: compact trace-status preflight for
  Issue #2273 / parent Issue #1472, recording that learned-risk training still lacks durable trace
  and baseline artifact URIs despite a valid launch packet fixture.
- `issue_2410_hybrid_component_readiness_refresh_2026-06-06/`: compact Issue 2410 refresh of
  hybrid-learning component readiness after the ORCA-residual progress-probe launch-packet update,
  preserving the Issue 1489 still-blocked synthesis boundary.
- `issue_2408_orca_residual_low_progress_2026-06-06/`: compact analysis-only classification of
  the ORCA-residual BC v0 low-progress smoke failure before a revised progress-probe rerun.
- `issue_2275_predictive_v2_fate_2026-06-05/`: compact decision matrix for Issue #2275 / parent
  Issue #1490, recording the stop-old-expansion decision after the #1543 negative audit and #1897
  failed coupling gate.
- `issue_2411_predictive_v2_child_classification_2026-06-06/`: compact classification of old
  predictive-v2 child issues after the stop-old-expansion decision.
- `issue_2274_hybrid_component_matrix_2026-06-05/`: validator-readable hybrid-learning component
  status matrix for Issue #2274 / parent Issue #1489, with compact CSV and YAML rows for learned
  risk, ORCA-residual BC, oracle imitation, shielded PPO repair, and BC warm-start PPO.
- `issue_2269_research_v1_trace_case_selection_2026-06-05/`: analysis-only case-selection
  manifest for Issue #2269 / parent Issue #2159, selecting five research-v1 trace-review cases and
  separating trace-slice-ready cases from AMV-specific trace-export blockers.
- `issue_2270_panel_candidate_manifest_2026-06-05/`: analysis-only panel candidate manifest for
  Issue #2270 / parent Issue #2227, recording static-recenter and topology panel candidates that
  remain blocked on durable `simulation_trace_export.v1` trace pairs.
- `issue_2753_signalized_crossing_metrics/`: fixture/report-table evidence for the four canonical
  signalized crossing metric row types (`red_required_stop`, `green_proceed`,
  `unavailable_no_claim`, `proxy_only_denominator_excluded`).  No simulator or runtime traces;
  fixture-only claim boundary.
- `issue_2799_signalized_runtime/`: simulator-backed runtime smoke for signalized-crossing
  denominator and exclusion semantics. Includes `red_required_stop`, `green_proceed`,
  `unavailable_no_claim`, and `proxy_only_denominator_excluded` rows; proves runtime denominator
  plumbing, not traffic-light realism, forced-waiting reasoning, or planner-ranking performance.
- `issue_3544_signalized_failure_pack_real/`: live-execution signalized failure-pack negative
  control. Uses real trace exports and runtime metrics from the signalized smoke matrix; detected no
  failure cases, so the pack is analysis-only and figure-ineligible.
- `issue_3546_signalized_failure_predicate_contract/`: contract-validation evidence that
  planner-observable `signal_red_phase_violations` and
  `signal_stop_line_crossings_under_red` intentionally count as signalized failure-pack
  predicates. Not a live positive benchmark result.
- `issue_2752_topology_reselection_mechanism/`: analysis-only mechanism diagnosis of three hard
  slices from Issue #2751 runtime evidence. All hard slices remained `horizon_exhausted`;
  `bottleneck_transfer` and `doorway_transfer` classified as `no_useful_topology_alternative`
  (likely scenario/geometry insufficiency), `t_intersection_transfer` as `candidate_route_blocked`
  (ambiguous between blocked geometry and excessive switching without per-step switch_timeline).
  Not benchmark or paper-facing evidence.
- `issue_2766_mixed_scenario_matrix/`: synthesis-only mixed-scenario coverage matrix comparing
  topology reselection, signal metrics, prediction baseline, observation perturbation,
  denominator health, stale/current status, and claim eligibility across canonical scenario
  slices. All rows remain non-claimable or diagnostic-only; unavailable, proxy-only, stale, and
  missing-denominator cells include explicit reasons.
- `issue_2801_topology_successor_recommendation/`: analysis-only successor recommendation after
  consuming #2751 runtime evidence and #2752 mechanism diagnosis. Recommends **stop** for
  topology-reselection-as-clearance on the current hard-slice set. Two of three hard slices are
  scenario/geometry insufficiency; the third is ambiguous but switching did not produce clearance.
  Not benchmark or paper-facing evidence.
- `issue_2804_non_topology_successor/`: analysis-only launch packet for local_policy_scoring
  investigation of the t_intersection_transfer hard slice. Follows #2801 stop decision and
  implements the recommended non-topology successor target. Hypothesis: per-step scoring will
  separate blocked_geometry from switch_too_often. Not benchmark or paper-facing evidence.
- `issue_2758_semantic_forecast_baselines_2026-06-14/`: diagnostic-only comparison of
  constant-velocity and semantic (signal-aware, goal-aware, combined) forecast baselines on
  bounded repository trace fixtures. Includes a compact summary plus JSON/Markdown comparison
  report. Existing durable fixtures lack signal/intent metadata, so the comparison proves
  baselines run correctly but does not show semantic conditioning improving calibration or
  collision relevance yet. Not benchmark or paper-facing evidence.
- `issue_2781_interaction_aware_forecast_2026-06-15/`: diagnostic-only comparison of all five
  forecast baselines (cv, signal_aware, goal_aware, semantic, interaction_aware) on bounded
  repository trace fixtures. Adds interaction-aware deterministic forecast baseline with
  proximity-based repulsion and crowding uncertainty. On matched evaluated rows, interaction-aware
  conditioning improves the Gaussian likelihood/calibration proxy while worsening 1s point
  accuracy, so the result is mixed, diagnostic-only, and a revise signal before closed-loop
  coupling. Not benchmark or paper-facing evidence.
- `issue_2868_semantic_metadata_fixtures_2026-06-15/`: diagnostic-only comparison after adding
  four metadata-bearing trace fixtures for signalized crossing, goal-directed crossing,
  waiting-with-intent-change, and route-conflict goal cases. The report distinguishes
  metadata-present and metadata-absent rows across cv, signal_aware, goal_aware, semantic, and
  interaction_aware baselines. Not human-realism, closed-loop, benchmark, or paper-facing evidence.
- `issue_2865_forecast_calibration_report_2026-06-15/`: analysis-only calibration/reliability
  report converted from the #2868 forecast comparison rows. Covers cv, signal_aware, goal_aware,
  semantic, and interaction_aware baselines with scenario-family, observation-tier, horizon,
  semantic-metadata, actor-class availability, miss-rate, failure-taxonomy, and risk-scoring
  eligibility columns. Decision: wait; claim status: diagnostic-only.
- `issue_2843_closed_loop_forecast_coupling_gate_2026-06-15/`: diagnostic-only closed-loop
  forecast coupling gate synthesizing #2781 forecast comparison with #1897 closed-loop gate
  metrics. Recommendation: revise. Forecast interaction_aware worsened 1s ADE by 0.0246 m vs CV
  while improving NLL by 0.6717; closed-loop gate failed (global_success_delta 0.0). No learned
  training involved. Not benchmark or paper-facing evidence.
- `issue_2759_forecast_risk_policy_stack_2026-06-15/`: diagnostic-only same-seed comparison
  of baseline (forecast_risk_weight=0) vs diagnostic (forecast_risk_weight=5.0) scoring in
  PolicyStackV1Adapter. Two deterministic fixture cases: high_risk_diagnostic_slows_goal
  (penalty shifts selection to risk_dwa, speed/progress proxy reduced) and
  false_positive_suppresses_penalty (penalty suppressed, goal retained with zero unnecessary
  slowdown count).
  claim_boundary=diagnostic_only_not_benchmark_evidence. Not a safety or live benchmark claim.
- `issue_2869_forecast_risk_calibration_filter_2026-06-15/`: diagnostic-only comparison of
  five forecast-risk scoring modes (`no_risk`, `raw_risk`, `calibration_filtered`,
  `actor_class_aware`, `observation_tier_aware`) before trusting planner coupling. Uses the
  Issue #2865 calibration report to mark `calibration_filtered`, `actor_class_aware`, and
  `observation_tier_aware` as blocked. Tradeoffs include route-progress reduction and
  false-positive stopping avoidance. Recommendation: `wait`.
  claim_boundary=diagnostic_only_not_benchmark_evidence. Not a safety or live benchmark claim.
- `issue_2866_transferability_matrix_rows/`: tracked bounded-input evidence for the Issue #2866
  transferability matrix follow-up, including generated JSON/Markdown matrix output with explicit
  observation-tier, actor-class, scenario-family, horizon, semantic-metadata, and blocked-cell rows.
- `issue_2837_horizon_timestep_ablation_2026-06-15/`: analysis-only horizon x output-dt_s ablation
  for forecast presets. Compares horizon ladder 0.5/1.0/1.6/2.0/3.0 s and dt_s ladder 0.1/0.2/0.4/0.5 s
  on durable repository trace fixtures. Reports collision relevance, miss rate, NLL, calibration
  error, runtime, artifact size, and memory proxies where available. Recommends short/medium/long
  presets with explicit unavailable rows for fixture-limited long horizons.
  claim_boundary=analysis_only_not_navigation_evidence. Not a safety, closed-loop, benchmark,
  or paper-facing claim.
- `issue_2903_horizon_denominator_health_2026-06-16/`: analysis-only denominator-health audit for
  the issue #2837 horizon x timestep ablation. Classifies each missing (horizon, dt_s, trace) cell
  by reason, verifies category totals sum to the expected 180-cell matrix, spot-checks
  representative missing cells, and proposes minimum fixture additions to reach at least 90%
  evaluated coverage. Forecast defaults remain unchanged.
  claim_boundary=analysis_only_not_navigation_evidence. Not a safety, closed-loop, benchmark,
  or paper-facing claim.
- `issue_2937_horizon_denominator_health_2026-06-16/`: analysis-only denominator-health fixture
  repair for the issue #2837 horizon x timestep ablation. Repairs the seven issue #2903 fixture
  gaps, regenerates the 180-cell audit at 164 evaluated cells (91.1%), and leaves only the
  intentionally short corridor-interaction fixtures as remaining blockers. Forecast defaults
  remain unchanged.
  claim_boundary=analysis_only_not_navigation_evidence. Not a safety, closed-loop, benchmark,
  or paper-facing claim.
- `issue_3206_heterogeneous_pedestrian_smoke_2026-06-20/`: smoke-only paired
  homogeneous-vs-mixed pedestrian composition run on classic crossing with three seeds per
  condition. It proves the archetype composition scenario knob reaches benchmark runtime and can
  produce reviewable metric deltas. The distributional-disruption blocks are present but not
  computable because the smoke has no control trace support. Not benchmark-strength evidence,
  planner ranking, real-world fairness evidence, or a real-world pedestrian-behavior realism claim.
