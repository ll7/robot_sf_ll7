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

## Current Bundles

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
- `issue_2273_learned_risk_trace_preflight_2026-06-05/`: compact trace-status preflight for
  Issue #2273 / parent Issue #1472, recording that learned-risk training still lacks durable trace
  and baseline artifact URIs despite a valid launch packet fixture.
- `issue_2275_predictive_v2_fate_2026-06-05/`: compact decision matrix for Issue #2275 / parent
  Issue #1490, recording the stop-old-expansion decision after the #1543 negative audit and #1897
  failed coupling gate.
- `issue_2274_hybrid_component_matrix_2026-06-05/`: validator-readable hybrid-learning component
  status matrix for Issue #2274 / parent Issue #1489, with compact CSV and YAML rows for learned
  risk, ORCA-residual BC, oracle imitation, shielded PPO repair, and BC warm-start PPO.
- `issue_2270_panel_candidate_manifest_2026-06-05/`: analysis-only panel candidate manifest for
  Issue #2270 / parent Issue #2227, recording static-recenter and topology panel candidates that
  remain blocked on durable `simulation_trace_export.v1` trace pairs.
