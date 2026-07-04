# Policy Search Candidate Registry Summary

Status: manually maintained retrieval companion to `candidate_registry.yaml`.

This summary helps agents route work without reading every candidate row or overinterpreting older
reports. It is not a benchmark ranking and does not change the canonical registry.

## Current Authority

- Canonical runnable registry: `candidate_registry.yaml`
- Learned-policy planning and intake: `learned_policy_registry.md`
- Rejected, monitor-only, and source-first learned families: `reject_monitor_registry.md`
- Local execution stages and gates: `contracts/agent_runbook.md`,
  `contracts/promotion_gates.md`, and `configs/policy_search/funnel.yaml`
- Long-run handoffs: `SLURM/todo.md`, the linked `SLURM/*.md` files, and
  `../slurm_issue_batch_status_2026-05-21.md` for execution status

## Lifecycle And Claim Vocabulary

| Field | Values | Meaning |
| --- | --- | --- |
| `lifecycle_status` | `active_runnable`, `diagnostic_only`, `learned_policy_intake`, `slurm_handoff_only`, `monitor_or_source_first`, `historical_report` | Retrieval/routing status for agents. |
| `claim_eligibility` | `benchmark_candidate`, `diagnostic_only`, `smoke_only`, `not_benchmark_evidence`, `blocked_or_monitor` | Strongest claim level supported before reading detailed evidence. |

These are companion-summary terms. If they are later added directly to
`candidate_registry.yaml`, keep the meaning aligned with this table.

## Current Active Runnable Anchors

These rows are the shortest current route into runnable policy-search work. Read their configs and
latest reports before expanding to older variants.

| Candidate | Lifecycle | Claim eligibility | Why start here |
| --- | --- | --- | --- |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | `active_runnable` | `benchmark_candidate` | Latest h500 collision-guard selector row; tied to leader-collision slice and h500 analysis. |
| `scenario_adaptive_hybrid_orca_v1` | `active_runnable` | `benchmark_candidate` | Main scenario-adaptive classical selector from the full-matrix policy-search lane. |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | `active_runnable` | `benchmark_candidate` | Current static-clearance/continuous-check variant for long-horizon failure analysis. |
| `hybrid_rule_v3_fast_progress` | `active_runnable` | `benchmark_candidate` | Important h500 leader comparator and clean-rerun handoff target. |
| `hybrid_rule_v3_static_margin0_waypoint2` | `active_runnable` | `benchmark_candidate` | Strong non-learning route/waypoint baseline with smoke, nominal, stress, and h500 evidence. |
| `risk_dwa_camera_ready` | `active_runnable` | `benchmark_candidate` | Canonical non-learning Risk-DWA comparison baseline for learned-style planner spikes. |
| `planner_selector_v1` | `active_runnable` | `smoke_only` | Adaptive ensemble route; useful as a selector reference, but check reports before claim use. |

## Diagnostic-Only Rows

Diagnostic rows probe comfort or mechanism sensitivity. They may be runnable, but their purpose is
not headline promotion without a separate issue and evidence decision.

| Candidate | Lifecycle | Claim eligibility | Notes |
| --- | --- | --- | --- |
| `proxemic_profile_conservative_issue_1676` | `diagnostic_only` | `diagnostic_only` | Conservative proxemic/comfort profile. |
| `proxemic_profile_neutral_issue_1676` | `diagnostic_only` | `diagnostic_only` | Neutral profile used as middle setting. |
| `proxemic_profile_open_issue_1676` | `diagnostic_only` | `diagnostic_only` | Open comfort profile with higher expected near-miss risk. |
| `issue_2170_static_recenter_only` | `diagnostic_only` | `diagnostic_only` | One-factor static-recenter diagnostic: local h500 component signal remains useful for activation-targeted slices, but #2438 classifies the current held-out transfer route as `mechanism_inactive`, so it must not be used as transfer or benchmark evidence. |
| `topology_guided_hybrid_rule_v0` | `diagnostic_only` | `diagnostic_only` | Masked-route topology-hypothesis selector adding a bounded selected-hypothesis command to the existing hybrid-rule safety scorer; current lane is `revise` after #2530, and the #2563 primary-route reuse-penalty proposal must pass a paired diagnostic before any benchmark claim. |
| `topology_guided_hybrid_rule_v0_reuse_penalty` | `diagnostic_only` | `diagnostic_only` | Issue #2540/#2563 variant that applies an explicit primary-route reuse penalty under eligible near-parity alternatives; intended only for a paired diagnostic against `topology_guided_hybrid_rule_v0`. |
| `topology_guided_hybrid_rule_v0_progress_gated_reselection` | `diagnostic_only` | `diagnostic_only` | Issue #2704 successor that suppresses the reuse penalty only when recent primary-route progress satisfies a predeclared threshold; intended as the #2660 progress-gated reselection diagnostic and not benchmark evidence. |
| `topology_guided_hybrid_rule_v0_progress_gated_reselection_monotone` | `diagnostic_only` | `diagnostic_only` | Issue #3463 successor variant of the progress-gated reselection lane that preserves the same threshold contract but uses monotone recent-progress accounting so transient re-plan bumps do not mask real primary-route progress; intended for bounded CPU-only sensitivity packets, not benchmark evidence. |
| `actuation_aware_hybrid_rule_v0` | `diagnostic_only` | `diagnostic_only` | Synthetic AMV actuation scorer; not calibrated hardware evidence. |
| `planner_selector_v2_diagnostic` | `diagnostic_only` | `diagnostic_only` | Deterministic selector over existing local candidates using predeclared topology/seed diagnostics and current local pedestrian context; not benchmark-strength evidence. |
| `adaptive_proxemic_selector_v0` | `diagnostic_only` | `diagnostic_only` | Deterministic local-context selector over the three fixed proxemic profiles; logs selected profile and trigger reason, but does not support benchmark or comfort claims. |
| `mpc_clearance_guarded_v1` | `diagnostic_only` | `diagnostic_only` | NMPC clearance sampler with an opt-in hard first-step static-clearance guard; intended to measure whether static-collision repair only creates low-progress failures. |
| `hybrid_rule_route_reacquire_recenter_probe` | `diagnostic_only` | `diagnostic_only` | Issue #1905 route-local-minimum probe combining corridor-subgoal recovery, static recenter, and narrow reorientation hooks; smoke passes, but nominal sanity remains `revise`, so keep it diagnostic-only. |

## Learned-Policy And Learned-Style Rows

Learned rows require the learned-policy registry, eligibility checklist, adapter metadata,
artifact provenance, and fail-closed behavior before benchmark claims.

| Candidate or family | Lifecycle | Claim eligibility | Authority |
| --- | --- | --- | --- |
| `ppo_issue791_best_v1` | `learned_policy_intake` | `benchmark_candidate` | Current learned-only PPO comparator; see `learned_policy_registry.md` and model registry. |
| `risk_guarded_ppo_v1` | `learned_policy_intake` | `smoke_only` | Guarded PPO tuning lane; check reports for caveats. |
| `orca_prior_guarded_ppo_v1` | `learned_policy_intake` | `smoke_only` | ORCA-prior guarded PPO variant. |
| `orca_prior_guarded_ppo_v2_static_global` | `learned_policy_intake` | `smoke_only` | Global ORCA-prior guarded variant. |
| `orca_primary_guarded_ppo_v1` | `learned_policy_intake` | `smoke_only` | ORCA-primary guarded variant. |
| `orca_residual_guarded_ppo_v0` | `learned_policy_intake` | `smoke_only` | Runtime residual surface exists; training/checkpoint lineage is deferred. |
| `orca_residual_guarded_ppo_progress_v1` | `learned_policy_intake` | `smoke_only` | Progress-probe residual candidate for the #1475 rerun; not learned-residual evidence until the bounded smoke produces durable artifacts. |
| `tentabot_value_scorer_v0` | `learned_policy_intake` | `diagnostic_only` | Clean-room value-scorer baseline; hand-authored Tentabot-style recovery is stopped after Issue #1910 unless a separate learned-value-estimator contract is opened. |
| `tentabot_value_scorer_v1_static_gated` | `learned_policy_intake` | `diagnostic_only` | Stopped diagnostic lane; the static-safety demotion tier reduced low-progress only with worse collisions/near misses. |
| `tentabot_value_scorer_v2_route_arc` | `learned_policy_intake` | `diagnostic_only` | Stopped diagnostic lane; scalar route-arc progress reduced low-progress only with worse static-collision/near-miss behavior. |
| `tentabot_value_scorer_v3_trace_recovery` | `learned_policy_intake` | `diagnostic_only` | Stopped diagnostic lane; trace-level recovery was executable but did not preserve the Issue #1832 static-collision baseline. |
| `learned_risk_model_v1` | `slurm_handoff_only` | `not_benchmark_evidence` | Launch-packet lane only until SLURM training and durable artifacts exist. |

## SLURM Handoff-Only Lanes

These are not local implementation tasks unless the issue is specifically about handoff docs,
launch-packet validation, or artifact provenance.

| Handoff | Lifecycle | Claim eligibility | Notes |
| --- | --- | --- | --- |
| `SLURM/001_learned_risk_model_v1.md` | `slurm_handoff_only` | `not_benchmark_evidence` | Learned risk model v1 launch lane. |
| `SLURM/002_shielded_ppo_repair_campaign.md` | `slurm_handoff_only` | `not_benchmark_evidence` | Shielded PPO repair campaign. |
| `SLURM/003_imitation_oracle_dataset_campaign.md` | `slurm_handoff_only` | `not_benchmark_evidence` | Oracle-imitation dataset campaign. |
| `SLURM/004_h500_leader_clean_rerun.md` | `slurm_handoff_only` | `not_benchmark_evidence` | Clean rerun for h500 leader candidates. |
| `SLURM/005_orca_residual_bc_lineage.md` | `slurm_handoff_only` | `not_benchmark_evidence` | ORCA-residual behavior-cloning lineage. |

## Monitor, Source-First, Blocked, And Rejected Learned Families

Keep these out of the runnable candidate registry until their reopen condition is met.

| Family | Lifecycle | Claim eligibility | Authority |
| --- | --- | --- | --- |
| CrowdNav / HEIGHT / IGAT graph policies | `monitor_or_source_first` | `blocked_or_monitor` | `learned_policy_registry.md`, `reject_monitor_registry.md`, and CrowdNav family verdict notes. |
| Arena-Rosnav stack | `monitor_or_source_first` | `blocked_or_monitor` | `issue_1758_arena_rosnav_source_assessment.md`. |
| DRL-VO family | `monitor_or_source_first` | `blocked_or_monitor` | `../issue_769_drl_vo_assessment.md` and reject monitor registry. |
| GenSafeNav / SoNIC | `monitor_or_source_first` | `blocked_or_monitor` | Source-harness and conformal-contract notes. |
| NeuPAN / SAGE / MPC-transfer | `monitor_or_source_first` | `blocked_or_monitor` | Source-side assessment notes. |
| NavDP / NoMaD / diffusion / visual navigation | `monitor_or_source_first` | `blocked_or_monitor` | Visual-policy and diffusion feasibility notes. |
| Foundation-model / VLA navigation | `monitor_or_source_first` | `blocked_or_monitor` | Foundation-model readiness note. |
| DreamerV3 / world-model navigation | `monitor_or_source_first` | `blocked_or_monitor` | DreamerV3 close-out and checkpoint import boundary. |

## Historical Candidate And Report Buckets

Older implemented rows remain useful for provenance, ablations, and regression analysis. They are
not automatically current just because they remain in the registry.

| Bucket | Examples | Routing |
| --- | --- | --- |
| Early hybrid-rule variants | `hybrid_rule_v0_minimal`, `hybrid_rule_v3_teb_like_rollout`, `hybrid_rule_v4_recovery_aware` | Read reports only when comparing mechanism history or reproducing a prior iteration. |
| Route/waypoint sweeps | `hybrid_rule_v3_waypoint2_*`, route-lookahead and static-margin variants | Use for targeted failure analysis; start from the current anchors above. |
| Model-based samplers | `hybrid_orca_sampler_v1`, `mpc_clearance_sampler_v1`, `mpc_clearance_guarded_v1` | Useful provenance for sampler ideas; check full-matrix reports before reuse. |
| Adaptive classical variants | `scenario_adaptive_orca_v1`, earlier hybrid-ORCA selectors | Use as selector baselines or comparison history. |
| Reports before 2026-05-05 | `reports/2026-04-*`, `validation/2026-04-*` | Historical development trail; prefer the 2026-05-05 overview/analysis for current interpretation. |

## Maintenance Rules

- Update this summary when a candidate becomes a new current anchor, a learned-policy row moves
  from intake to runnable evidence, or a SLURM handoff gains durable artifact proof.
- Do not use this summary to change planner rankings; link the report or issue that justifies any
  status change.
- Keep blocked/source-first learned families in `reject_monitor_registry.md` unless they gain a
  runnable Robot SF candidate config and proof path.
- If this summary and `candidate_registry.yaml` disagree, treat `candidate_registry.yaml` as the
  config authority and this summary as stale retrieval metadata.
