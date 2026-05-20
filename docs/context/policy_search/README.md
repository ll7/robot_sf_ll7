# Policy Search Context

This directory is the file-based execution surface for the local policy-search
project requested on 2026-04-29.

Use it for three things only:

- stable project contract and candidate registry,
- reasoning, validation, and result notes for each iteration,
- SLURM handoff instructions for work that is too expensive to run locally.

## Layout

- `candidate_registry.yaml`: canonical candidate list and config pointers for implemented or
  concrete runnable Robot SF candidates.
- `reject_monitor_registry.md`: reusable negative/monitor trail for learned-policy families such as
  CrowdNav descendants, Dreamer/world-model approaches, diffusion/visual navigation, SAC/TD3/PPO
  variants, DRL-VO, and source-side-only candidates. Check this before proposing a new follow-up.
- `experiment_ledger.md`: compact execution log for implemented candidates.
- `contracts/`: project contract, gates, taxonomy, and runbook.
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
- `../issue_769_drl_vo_assessment.md`: DRL-VO metadata history plus issue #1364 privileged-state
  audit verdict; current status is prototype-only/tracked-agent diagnostic, not main-table ready.
- `2026-05-20_tentabot_motion_primitive_assessment.md`: issue #1357 assessment of Tentabot-style
  motion-primitive value policies, recommending a Robot SF-native scorer spike without source-code
  reuse.
- `contracts/learned_local_policy_eligibility.md`: learned-policy eligibility checklist for
  observation/action leakage, registry entry, and required raw/adapted/guarded action logging.

## Reproducible Entry Points

- Candidate runner: `uv run python scripts/validation/run_policy_search_candidate.py`
- Candidate portfolio overview: `uv run python scripts/tools/summarize_policy_search_portfolio.py`
- Candidate comparison: `uv run python scripts/tools/compare_policy_search_candidates.py`
- Failure taxonomy: `uv run python scripts/tools/build_policy_search_failure_report.py`
- Horizon recommendation: `uv run python scripts/tools/suggest_policy_search_horizons.py`
- Pareto plot: `uv run python scripts/tools/plot_policy_search_pareto_front.py`
- Promotion decision: `uv run python scripts/tools/promote_policy_search_candidate.py`
- SLURM candidate sweep: `scripts/dev/sbatch_policy_search_sweep.sh --stage full_matrix --all-implemented`

## Learned-Policy Intake

Before adding a learned local-navigation method to `candidate_registry.yaml`, apply
`contracts/learned_local_policy_eligibility.md`. The registry is reserved for implemented or
concrete runnable Robot SF candidates with config pointers; source-only, monitor-only, or
privileged-evaluation methods should stay in context notes until they have a runnable adapter
contract.

## Scope Boundary

Local work may run smoke and narrow validation stages.

Full matrix evaluations, robustness extensions, and all learning-heavy campaigns
must be routed through `SLURM/todo.md` and the linked handoff notes.
