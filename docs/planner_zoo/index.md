# Planner Zoo

[Back to Documentation Index](../README.md)

This page is a user-facing navigation layer over the current Robot SF planner and policy-search
registries. It summarizes what is runnable, diagnostic-only, learned-policy intake, or blocked
without changing benchmark eligibility.

Canonical sources:

- Runnable policy-search candidates:
  [`docs/context/policy_search/candidate_registry.yaml`](../context/policy_search/candidate_registry.yaml)
- Registry companion summary:
  [`docs/context/policy_search/candidate_registry_summary.md`](../context/policy_search/candidate_registry_summary.md)
- Learned-policy planning and intake:
  [`docs/context/policy_search/learned_policy_registry.md`](../context/policy_search/learned_policy_registry.md)
- Policy-search funnel:
  [`configs/policy_search/funnel.yaml`](../../configs/policy_search/funnel.yaml)

Use this page as a map, not as evidence. Smoke, diagnostic, fallback, degraded, monitor-only, and
blocked rows are not benchmark-ready claims.

## Status Vocabulary

| Status | Meaning |
| --- | --- |
| Current runnable anchor | Runnable candidate with a registry config and local gates. Read the linked reports before making any benchmark claim. |
| Runnable historical candidate | Runnable candidate kept for ablations, regressions, and provenance. Prefer current candidates for new comparisons. |
| Diagnostic-only | Runnable or staged probe whose purpose is mechanism inspection. Do not treat outcomes as benchmark evidence without a separate issue and proof decision. |
| Learned-policy intake | Learned or learned-style policy row that also needs artifact, adapter, and eligibility checks. |
| SLURM handoff or blocked | Not a local benchmark row. Use only for launch-packet, artifact, or source-harness follow-up work. |

The local smoke command for ordinary runnable candidates is:

```bash
uv run python scripts/validation/run_policy_search_candidate.py --candidate <candidate_id> --stage smoke
```

Use `--stage amv_actuation_smoke` only for candidates whose required stage is
`amv_actuation_smoke`.

## Current Runnable Anchors

These are the shortest current entry points for locally runnable planner comparisons. Their smoke
commands exercise the repository runner, but not every row is benchmark-eligible. The strongest
supported claim still depends on the latest reports and promotion gates.

| Candidate | Family | Config | Local smoke command | Notes |
| --- | --- | --- | --- | --- |
| `scenario_adaptive_hybrid_orca_v2_collision_guard` | Scenario-adaptive classical selector | [`configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml`](../../configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml) | `uv run python scripts/validation/run_policy_search_candidate.py --candidate scenario_adaptive_hybrid_orca_v2_collision_guard --stage smoke` | Current h500 collision-guard selector row; tied to leader-collision slice evidence. |
| `scenario_adaptive_hybrid_orca_v1` | Scenario-adaptive classical selector | [`configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v1.yaml`](../../configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v1.yaml) | `uv run python scripts/validation/run_policy_search_candidate.py --candidate scenario_adaptive_hybrid_orca_v1 --stage smoke` | Main scenario-adaptive classical selector from the full-matrix lane. |
| `hybrid_rule_v3_fast_progress_static_escape_continuous` | Hybrid rule-based local planner | [`configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape_continuous.yaml`](../../configs/policy_search/candidates/hybrid_rule_v3_fast_progress_static_escape_continuous.yaml) | `uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_fast_progress_static_escape_continuous --stage smoke` | Current continuous static-clearance variant for long-horizon failure analysis. |
| `hybrid_rule_v3_fast_progress` | Hybrid rule-based local planner | [`configs/policy_search/candidates/hybrid_rule_v3_fast_progress.yaml`](../../configs/policy_search/candidates/hybrid_rule_v3_fast_progress.yaml) | `uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_fast_progress --stage smoke` | Important h500 leader comparator and clean-rerun handoff target. |
| `hybrid_rule_v3_static_margin0_waypoint2` | Hybrid rule-based local planner | [`configs/policy_search/candidates/hybrid_rule_v3_static_margin0_waypoint2.yaml`](../../configs/policy_search/candidates/hybrid_rule_v3_static_margin0_waypoint2.yaml) | `uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_static_margin0_waypoint2 --stage smoke` | Strong non-learning route/waypoint baseline with smoke and nominal evidence. |
| `risk_dwa_camera_ready` | Classical Risk-DWA baseline | [`configs/policy_search/candidates/risk_dwa_camera_ready.yaml`](../../configs/policy_search/candidates/risk_dwa_camera_ready.yaml) | `uv run python scripts/validation/run_policy_search_candidate.py --candidate risk_dwa_camera_ready --stage smoke` | Canonical non-learning Risk-DWA comparison baseline for learned-style spikes. |
| `planner_selector_v1` | Adaptive ensemble | [`configs/policy_search/candidates/planner_selector_v1.yaml`](../../configs/policy_search/candidates/planner_selector_v1.yaml) | `uv run python scripts/validation/run_policy_search_candidate.py --candidate planner_selector_v1 --stage smoke` | Selector reference with smoke-only claim boundary unless reports justify more. |

## Runnable Historical Candidates

These rows remain useful for reproducing prior policy-search iterations and comparing mechanism
history. Start from the current benchmark candidates above unless a task explicitly needs one of
these variants.

| Family | Candidates |
| --- | --- |
| Early hybrid-rule variants | `hybrid_rule_v0_minimal`, `hybrid_rule_v3_teb_like_rollout`, `hybrid_rule_v4_recovery_aware`, `hybrid_rule_v3_dynamic_relaxed`, `hybrid_rule_v3_static_margin0` |
| Route, waypoint, speed, and clearance sweeps | `hybrid_rule_v3_waypoint2_static_escape`, `hybrid_rule_v3_waypoint2_speed2p2`, `hybrid_rule_v3_waypoint2_route_commit`, `hybrid_rule_v3_static_margin0_waypoint3`, `hybrid_rule_v3_waypoint2_progress`, `hybrid_rule_v3_waypoint2_route_lookahead6`, `hybrid_rule_v3_waypoint2_route_lookahead8`, `hybrid_rule_v3_waypoint2_route_lookahead8_inflation4`, `hybrid_rule_v3_waypoint2_route_lookahead8_static05`, `hybrid_rule_v3_waypoint2_route_lookahead8_static02`, `hybrid_rule_v3_waypoint2_route_lookahead8_clearance1`, `hybrid_rule_v3_waypoint2_dynamic_clearance` |
| Comfort variants | `hybrid_rule_v3_static_margin0_comfort`, `hybrid_rule_v3_waypoint2_mild_comfort` |
| Model-based samplers | `hybrid_orca_sampler_v1`, `mpc_clearance_sampler_v1` |
| Guarded learned-policy comparators | `risk_guarded_ppo_v1`, `orca_prior_guarded_ppo_v1`, `orca_prior_guarded_ppo_v2_static_global`, `orca_primary_guarded_ppo_v1`, `ppo_issue791_best_v1` |
| Adaptive classical history | `scenario_adaptive_orca_v1` |

## Diagnostic-Only And Non-Benchmark Rows

These rows are useful probes, but the registry marks them as diagnostic-only, smoke-only, or
prototype-bound. Keep their reports out of benchmark-ready summaries unless a later issue promotes
the claim boundary with new evidence.

| Candidate | Family | Command or boundary |
| --- | --- | --- |
| `hybrid_rule_v3_progress_2p4_static_escape_probe` | Hybrid-rule probe | `uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_rule_v3_progress_2p4_static_escape_probe --stage smoke` |
| `risk_surface_dwa_v0` | Local risk-surface diagnostic | `uv run python scripts/validation/run_policy_search_candidate.py --candidate risk_surface_dwa_v0 --stage smoke` |
| `topology_guided_hybrid_rule_v0` | Topology-hypothesis diagnostic | `uv run python scripts/validation/run_policy_search_candidate.py --candidate topology_guided_hybrid_rule_v0 --stage smoke` |
| `proxemic_profile_conservative_issue_1676` | Proxemic/comfort diagnostic | `uv run python scripts/validation/run_policy_search_candidate.py --candidate proxemic_profile_conservative_issue_1676 --stage smoke`; comfort-profile evidence is diagnostic-only. |
| `proxemic_profile_neutral_issue_1676` | Proxemic/comfort diagnostic | `uv run python scripts/validation/run_policy_search_candidate.py --candidate proxemic_profile_neutral_issue_1676 --stage smoke`; comfort-profile evidence is diagnostic-only. |
| `proxemic_profile_open_issue_1676` | Proxemic/comfort diagnostic | `uv run python scripts/validation/run_policy_search_candidate.py --candidate proxemic_profile_open_issue_1676 --stage smoke`; comfort-profile evidence is diagnostic-only. |
| `tentabot_value_scorer_v0` | Learned-style value-scorer spike | `uv run python scripts/validation/run_policy_search_candidate.py --candidate tentabot_value_scorer_v0 --stage smoke`; upstream Tentabot remains source-side only. |
| `actuation_aware_hybrid_rule_v0` | Synthetic AMV actuation diagnostic | `uv run python scripts/validation/run_policy_search_candidate.py --candidate actuation_aware_hybrid_rule_v0 --stage amv_actuation_smoke`; not calibrated hardware evidence. |
| `adaptive_proxemic_selector_v0` | Proxemic selector diagnostic | `uv run python scripts/validation/run_policy_search_candidate.py --candidate adaptive_proxemic_selector_v0 --stage smoke` |
| `mpc_clearance_guarded_v1` | NMPC clearance diagnostic | `uv run python scripts/validation/run_policy_search_candidate.py --candidate mpc_clearance_guarded_v1 --stage smoke` |
| `planner_selector_v2_diagnostic` | Adaptive ensemble diagnostic | `uv run python scripts/validation/run_policy_search_candidate.py --candidate planner_selector_v2_diagnostic --stage smoke` |
| `orca_residual_guarded_ppo_v0` | Runtime residual prototype | Smoke-only runtime surface; learned residual training/checkpoint lineage is pending. |

## Learned-Policy Intake

Learned policies need additional checks before they can support benchmark claims: checkpoint
availability, artifact provenance, observation/action adapter metadata, fail-closed handling, and
the learned-policy eligibility checklist.

Use [`learned_policy_registry.md`](../context/policy_search/learned_policy_registry.md) and
[`learned_local_policy_eligibility.md`](../context/policy_search/contracts/learned_local_policy_eligibility.md)
before opening implementation work.

| Policy or family | Current status | Boundary |
| --- | --- | --- |
| `ppo_issue791_best_v1` | Implemented learned baseline | Comparison evidence exists for the configured PPO baseline; not a blanket safety promotion. |
| `guarded_ppo_orca_prior` and related guarded PPO rows | Implemented/smoke-proven | Inference-only guarded tuning lane; not benchmark evidence unless specific reports support the claim. |
| `orca_residual_guarded_ppo_v0` | Staged launch-packet lane | Runtime residual surface exists; training and checkpoint lineage are pending. |
| `learned_risk_model_v1` | SLURM handoff | Pre-SLURM launch packet only; not local benchmark evidence. |
| External learned-policy families | Monitor-only, source-first, or rejected | CrowdNav, Arena-Rosnav, DRL-VO, GenSafeNav/SoNIC, NeuPAN, SAGE/MPC-transfer, NavDP/NoMaD, diffusion-policy, foundation/VLA, DWA-RL, and DreamerV3 rows are not Robot SF benchmark rows until their registry reopen conditions are met. |

## Adding Or Updating A Planner Row

1. Add or update the candidate config under
   [`configs/policy_search/candidates/`](../../configs/policy_search/candidates/).
2. Add the registry row in
   [`candidate_registry.yaml`](../context/policy_search/candidate_registry.yaml) with family,
   status, config path, hypothesis, and required stages.
3. Keep diagnostic-only, fallback/degraded, monitor-only, and blocked boundaries explicit.
4. Run the narrowest applicable validation command, usually the local smoke stage.
5. Update the registry summary or this page only when the routing status changes.
