# Policy Search Context

This directory is the file-based execution surface for the local policy-search
project requested on 2026-04-29.

Use it for three things only:

- stable project contract and candidate registry,
- reasoning, validation, and result notes for each iteration,
- SLURM handoff instructions for work that is too expensive to run locally.

## Layout

- `INDEX.md`: compact retrieval index for current authority, lifecycle routing, claim eligibility,
  and first-stop policy-search context.
- `candidate_registry.yaml`: canonical candidate list and config pointers for implemented or
  concrete runnable Robot SF candidates.
- `candidate_registry_summary.md`: compact companion summary that separates active runnable,
  diagnostic-only, learned-policy intake, SLURM-handoff, monitor/source-first, and historical
  candidate lanes.
- `learned_policy_registry.md`: compact learned local-navigation registry for implemented,
  staged, monitor-only, and blocked learned-policy families. Use this before adding a learned
  policy to the runnable candidate registry.
- `reject_monitor_registry.md`: reusable negative/monitor trail for learned-policy families such as
  CrowdNav descendants, Dreamer/world-model approaches, diffusion/visual navigation, SAC/TD3/PPO
  variants, DRL-VO, and source-side-only candidates. Check this before proposing a new follow-up.
- `experiment_ledger.md`: compact execution log for implemented candidates.
- `contracts/`: project contract, gates, taxonomy, and runbook.
  - `external_policy_intake.md`: staged source-to-benchmark intake contract for external learned
    local-navigation policies; keeps source-side and adapter-only evidence out of benchmark claims
    until the registry and benchmark suite support them.
  - `oracle_imitation_dataset_split.md`: train/validation/evaluation seed-split policy, hard-slice assignment rules, relabeling boundaries, and manifest schema for oracle-imitation datasets (issue #1443).
- `reasoning/`: bounded planning and design notes.
- `reports/`: per-run markdown reports emitted by the policy-search runner.
- `validation/`: local smoke or narrow validation notes.
- `SLURM/`: deferred training TODO plus detailed handoff instructions.

## Current Diagnostic Notes

- `../issue_1365_social_graph_observation_adapter.md`: shared SocNav-to-graph observation adapter
  contract for graph/social-RL candidate screening; documents masks, caps, history, static-obstacle
  tokens, and the deployment-only leakage boundary.
- `../issue_1369_sage_mpc_transfer_assessment.md`: SAGE / `TIB-K330/drl_planner`
  reproducibility assessment; classifies the MPC-transfer graph-RL source as monitor-only because
  dependency pins, checkpoints, offline MPC source buffer, and an inference command are absent.
- `portfolio_overview_2026-05-05.md`: current candidate portfolio overview generated from the
  policy-search registry and tracked reports; includes current leaders, success/collision/near-miss
  evidence, why the best candidates look promising, coverage gaps, and reproduction commands.
- `reports/2026-05-05_full_matrix_all_candidates_analysis.md`: completed 29-candidate
  full-matrix sweep analysis, release-candidate decision, provenance caveat, and recommended next
  steps.
- `reports/2026-05-05_full_matrix_h500_analysis.md`: h500 long-horizon full-matrix analysis,
  current leader, promotion split, research directions, and scenario-specific horizon policy.
- `reports/2026-05-05_h500_horizon_recommendations.md`: generated per-scenario horizon
  recommendations from safe h500 incumbent JSONL evidence; machine-readable output lives at
  `configs/policy_search/scenario_horizons_h500.yaml`.
- `../evidence/policy_search_h500_2026-05-06/`: small tracked evidence bundle with selected h500
  summaries and failure reports promoted from `output/`.
- `reports/promotions/2026-05-05_full_matrix_h500_strict_gate/`: strict `nominal_sanity` gate
  reports for the h500 top candidates.
- `reasoning/2026-05-05_h500_research_plan.md`: next research workstreams for h500 blockers,
  comfort-preserving success, safety-accounted selectors, and MPC-as-proposer variants.
- `SLURM/004_h500_leader_clean_rerun.md`: clean pinned rerun handoff for
  `scenario_adaptive_hybrid_orca_v1` and `hybrid_rule_v3_fast_progress`.
- `validation/2026-05-02_hybrid_rule_failure_diagnostics.md`: issue #874 diagnosis of the
  remaining `hybrid_rule_v3_fast_progress_static_escape` static-route and leave-group failures.
- `2026-05-20_navdp_nomad_diffusion_assessment.md`: Issue #1356 Assessment of NavDP and NoMaD
  Diffusion-Navigation Source Contracts, with Monitor-Only Verdicts for Robot SF Local-Planner Use.
- `2026-05-30_diffusion_policy_feasibility_issue_1621.md`: Issue #1621 broader feasibility
  assessment for diffusion, consistency, and trajectory-diffusion local-navigation policy families;
  current verdict is no external adapter and monitor/source-first until a Robot SF-compatible
  observation/action contract and latency proof exist.
- `../issue_1318_teb_corridor_deadlock_eval.md`: issue #1318 classic-merging corridor-deadlock
  comparison showing current in-repo TEB collides on the selected #1022 seeds while the hybrid-rule
  incumbent solves four of five.
- `2026-05-20_learned_local_navigation_screen.md`: Issue #1355 Source-Backed Screening Matrix for
  learning-based local-navigation candidates, with implement/source-first/monitor/reject verdicts
  and links to existing Robot SF duplicate boundaries.
- `issue_1758_arena_rosnav_source_assessment.md`: Arena-Rosnav source-side assessment; current
  verdict is `source-side reproduction first` because the source requires a ROS Noetic/Gazebo
  Arena workspace and no durable trained-policy files were bundled in the checked source.
- `2026-05-30_external_learned_policy_ranking_issue_1620.md`: Issue #1620 ranked external
  learned-policy shortlist across graph/social-RL, ROSNav, diffusion/visual, transformer, VLA, and
  mapless-baseline families, with source/checkpoint/adapter-fit verdicts and follow-up routing.
- `2026-05-30_foundation_model_readiness_issue_1626.md`: Issue #1626 readiness boundary for
  foundation-model, VLA, and multimodal navigation policy families; current verdict is interface
  design or monitor-only, not model integration.
- `../issue_1368_neupan_point_obstacle_assessment.md`: source-side NeuPAN assessment; current
  verdict is monitor/source-side only because GPL-3.0, source-environment, runtime, and
  point-obstacle/social-claim boundaries block a Robot SF adapter for now.
- `issue_1367_crowdnav_family_verdict.md`: CrowdNav-family learned-policy consolidation for
  CrowdNav / SARL, RGL, DS-RNN, CrowdNav++ / IGAT, HEIGHT, and GenSafeNav / SoNIC; current verdict
  is no new first integration until the relevant graph/history, checklist, and source-reproduction
  gates for each candidate land.
- `issue_1394_crowdnav_height_source_harness.md`: CrowdNav HEIGHT source-harness proof for the
  current family representative; current verdict is blocked by missing legacy `gym` and local
  checkpoint assets.
- `issue_1366_gensafenav_sonic_conformal_contract.md`: GenSafeNav / SoNIC conformal uncertainty
  assessment; current verdict is source-side reproduction first before benchmark promotion.
- `issue_1393_gensafenav_source_harness.md`: fresh GenSafeNav `Ours_GST` source-harness
  reproduction record; current verdict remains blocked by the missing `gym` source dependency.
- `../issue_769_drl_vo_assessment.md`: DRL-VO metadata history plus issue #1364 privileged-state
  audit verdict; current status is prototype-only/tracked-agent diagnostic, not main-table ready.
- `2026-05-20_tentabot_motion_primitive_assessment.md`: issue #1357 assessment of Tentabot-style
  motion-primitive value policies, recommending a Robot SF-native scorer spike without source-code
  reuse.
- `../issue_1387_tentabot_value_scorer_spike.md`: clean-room Tentabot-style value-scorer spike
  history, including #1832 and #1877 negative progress-recovery/static-gate probe boundaries.
- `contracts/learned_local_policy_eligibility.md`: learned-policy eligibility checklist for
  observation/action leakage, registry entry, and required raw/adapted/guarded action logging.
- `../issue_1618_learned_policy_adapter_interface.md`: adapter-interface contract that turns the
  learned-policy eligibility checklist into runtime loading, inference, action-adaptation,
  diagnostics, and fail-closed status requirements.
- `../issue_1627_learned_policy_transfer_benchmark.md`: benchmark design for imported learned
  policies; defines source-intake/source-reproduction/adapter-metadata/smoke/benchmark stages,
  required transfer metadata, fail-closed statuses, and first implementation scope.
- `../issue_1761_learned_policy_transfer_metadata_validator.md`: metadata-only validator for
  `algorithm_metadata.transfer_benchmark`; includes native PPO and blocked CrowdNav HEIGHT fixtures
  and keeps fallback/degraded/not-available rows from claiming benchmark success.
- `reports/2026-05-20_orca_residual_guarded_ppo_v0_smoke.md`: issue #1358 ORCA-residual guarded
  PPO benchmark-surface smoke; validates the bounded residual action path for unsafe PPO proposals
  before the deferred training campaign.
- `SLURM/005_orca_residual_bc_lineage.md`: issue #1428 pre-SLURM handoff for behavior-cloning the
  first bounded ORCA-residual policy with runtime-only observations and explicit diagnostics.

## Reproducible Entry Points

- Retrieval index: `docs/context/policy_search/INDEX.md`
- Candidate lifecycle summary: `docs/context/policy_search/candidate_registry_summary.md`
- Candidate config home: `configs/policy_search/candidates/`
- Benchmark-facing algorithm configs: `configs/algos/`
- Candidate runner: `uv run python scripts/validation/run_policy_search_candidate.py`
- Candidate step diagnostics: `uv run python scripts/validation/run_policy_search_step_diagnostics.py`
- Candidate/learned-policy registry validator:
  `uv run python scripts/validation/validate_policy_search_registry.py`
- Candidate portfolio overview: `uv run python scripts/tools/summarize_policy_search_portfolio.py`
- Candidate comparison: `uv run python scripts/tools/compare_policy_search_candidates.py`
- Failure taxonomy: `uv run python scripts/tools/build_policy_search_failure_report.py`
- Horizon recommendation: `uv run python scripts/tools/suggest_policy_search_horizons.py`
- Pareto plot: `uv run python scripts/tools/plot_policy_search_pareto_front.py`
- Promotion decision: `uv run python scripts/tools/promote_policy_search_candidate.py`
- SLURM candidate sweep: `scripts/dev/sbatch_policy_search_sweep.sh --stage full_matrix --all-implemented`

Candidate names are resolved through `docs/context/policy_search/candidate_registry.yaml`; registry
entries point to files under `configs/policy_search/candidates/`. For example:

```bash
uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress \
  --stage smoke
```

```bash
uv run python scripts/validation/run_policy_search_step_diagnostics.py \
  --candidate hybrid_rule_v3_static_margin0 \
  --stage smoke
```

Use `configs/algos/` for benchmark-facing algorithm configs and wrappers. Do not assume a
policy-search candidate lives there just because a similarly named benchmark config exists.

## Learned-Policy Intake

Before adding a learned local-navigation method to `candidate_registry.yaml`, apply
`contracts/learned_local_policy_eligibility.md`. For external learned-policy families, first apply
`contracts/external_policy_intake.md` so source screen, license, checkpoint, observation/action,
source-side smoke, adapter, Robot SF smoke, and benchmark-suite evidence are not collapsed into one
ambiguous status. The registry is reserved for implemented or concrete runnable Robot SF candidates
with config pointers; source-only, monitor-only, adapter-only, or privileged-evaluation methods
should stay in context notes until they have a runnable adapter contract and smoke proof.

For repeatable checklist-input validation, record the candidate's learned-policy metadata as YAML or
JSON and run:

```bash
uv run python scripts/validation/check_learned_policy_eligibility.py <candidate-spec.yaml>
```

This helper checks completeness and consistency of the eligibility inputs only. It does not turn a
candidate into a benchmark-ready planner or replace adapter, smoke, or benchmark validation.

`learned_policy_registry.md` remains the durable current-state registry for learned-policy
families. The external intake contract supplies the stage checklist and status mapping; it must not
be maintained as a parallel source of truth.

## Scope Boundary

Local work may run smoke and narrow validation stages.

Full matrix evaluations, robustness extensions, and all learning-heavy campaigns
must be routed through `SLURM/todo.md` and the linked handoff notes.
