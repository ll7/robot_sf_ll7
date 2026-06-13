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
- `issue_2749_observation_noise_diagnostics/`: diagnostic-only paired no-op vs
  perception-limited observation-noise step diagnostics for `hybrid_rule_v0_minimal` on a
  pedestrian-present stress-slice scenario; perturbation plumbing activated, but the distant
  pedestrian produced no measurable planner degradation.
- `policy_search_h500_2026-05-06/`: h500 policy-search leader summaries and failure reports that
  support the v1 raw-success leader and v2 strict-gate promotion decision.
- `issue_1023_scenario_horizons_preflight_2026-05-06/`: compact preflight artifacts for the
  paper-facing scenario-horizon benchmark config.
- `issue_1023_scenario_horizons_local_full_2026-05-06/`: compact local non-Slurm full-campaign
  artifacts, analyzer output, and fixed-vs-scenario comparison for issue #1023.
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
- `issue_2430_ammv_trace_annotation_2026-06-06/`: compact frame-level parity decision showing the
  Issue #2428 selected default/AMMV trace pair is telemetry-rich but frame-identical, so it is not
  behavioral-difference evidence.
- `issue_2432_ammv_trace_selection_2026-06-06/`: compact three-seed AMMV/default trace-selection
  check showing the local Issue #2168 head-on-corridor slice still has no non-identical
  behavioral-difference pair.
- `issue_2434_ammv_scenario_sweep_2026-06-06/`: compact five-family AMMV/default episode-metric
  screen showing the selected classic close-interaction adapter slice still has no non-identical
  pair.
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
- `issue_2752_topology_reselection_mechanism/`: analysis-only mechanism diagnosis of three hard
  slices from Issue #2751 runtime evidence. All hard slices remained `horizon_exhausted`;
  `bottleneck_transfer` and `doorway_transfer` classified as `no_useful_topology_alternative`
  (likely scenario/geometry insufficiency), `t_intersection_transfer` as `candidate_route_blocked`
  (ambiguous between blocked geometry and excessive switching without per-step switch_timeline).
  Not benchmark or paper-facing evidence.
- `issue_2801_topology_successor_recommendation/`: analysis-only successor recommendation after
  consuming #2751 runtime evidence and #2752 mechanism diagnosis. Recommends **stop** for
  topology-reselection-as-clearance on the current hard-slice set. Two of three hard slices are
  scenario/geometry insufficiency; the third is ambiguous but switching did not produce clearance.
  Not benchmark or paper-facing evidence.
- `issue_2804_non_topology_successor/`: analysis-only launch packet for local_policy_scoring
  investigation of the t_intersection_transfer hard slice. Follows #2801 stop decision and
  implements the recommended non-topology successor target. Hypothesis: per-step scoring will
  separate blocked_geometry from switch_too_often. Not benchmark or paper-facing evidence.
