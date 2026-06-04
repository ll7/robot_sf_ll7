# Issue #2225 Learned-Policy Failure Synthesis (2026-06-04)

Issue: <https://github.com/ll7/robot_sf_ll7/issues/2225>
Date: 2026-06-04
Status: current first-pass synthesis over existing durable learned-policy and hybrid-rule evidence.

## Goal

Synthesize negative and weak learned-policy signals against mechanism-designed hybrid-rule evidence
without launching new training or treating fallback, degraded, or local-only outputs as success
evidence.

The compact evidence table lives at
[failure_modes.csv](evidence/issue_2225_learned_policy_failure_synthesis_2026-06-04/failure_modes.csv).

## Evidence Summary

| Method or lane | Evidence tier | Primary signal | Interpretation | Next step |
| --- | --- | --- | --- | --- |
| BC warm-start PPO | `cancelled_intermediate_negative_diagnostic` | `docs/context/issue_1961_bc_warm_start_recoverability.md` and `docs/context/evidence/issue_1977_bc_warm_start_cancelled_2026-06-02/artifact_manifest.md` record job `12689` cancelled after `5,361,664` timesteps with tail success rate `0`, no final checkpoint, and no policy-analysis comparison. | Preserved BC artifacts can be hydrated, but the same PPO continuation shape is not a useful next long allocation without a redesigned learned-progress objective. This is negative diagnostic evidence, not completed benchmark evidence. | Stop the same run shape; reopen only through intermediate-checkpoint analysis or a redesigned objective. |
| Camera-ready guarded PPO profile | `diagnostic_negative` | `docs/context/issue_602_guarded_ppo_profile.md` reports `guarded_ppo` success `0.0071`, collisions `0.0780`, and lower near misses than PPO on the guarded-PPO comparison matrix. | Generic hard guarding materially changes safety accounting but can suppress goal-reaching enough to make a planner unpromotable. | Revise around progress and guard/fallback accounting before treating safety gains as navigation improvement. |
| Shielded PPO collision-20 repair | `prototype_candidate` | `docs/context/issue_1474_shielded_ppo_repair_closeout.md` and `docs/context/evidence/issue_1474_shielded_ppo_repair_2026-06-01/artifact_manifest.md` preserve a durable W&B checkpoint with final training eval success `0.83` and collision `0.16`, while noting the launch-packet training target was missed. `docs/context/issue_2006_guarded_ppo_zero_motion_repair.md` and `docs/context/policy_search/reports/2026-06-02_shielded_ppo_issue1474_collision20_v1_smoke.md` repair the observation handoff and show a one-episode guarded smoke pass. | The lane is no longer purely negative: it has a viable checkpoint and repaired smoke handoff. It still does not prove nominal-sanity, stress, full-matrix, or benchmark-strength guarded navigation. | Continue only to the nominal-sanity gate with preserved guard diagnostics; treat smoke as prototype evidence. |
| ORCA-residual learned local policy | `staged_interface` | `docs/context/issue_1428_orca_residual_lineage.md` records the BC lineage packet and a one-episode `orca_residual_guarded_ppo_v0` runtime-surface smoke success, while explicitly stating this is not learned residual training evidence. | Residual-over-ORCA is the most mechanism-aligned learned interface in the current evidence stack, but the learned residual contribution is not yet trained or measured. | Require durable dataset, checkpoint, residual contribution, clipping, and guard-veto diagnostics before comparative claims. |
| Learned-risk and oracle-imitation components | `planned_component` | `docs/context/issue_1624_hybrid_learning_architecture.md` maps these lanes as launch-packet or dataset-prep components under an authoritative hard guard. | These are architecture candidates, not evidence that learned local navigation improves Robot SF outcomes today. | Defer comparative synthesis until component campaigns produce durable outputs accepted by the hybrid evidence matrix. |
| Mechanism-designed hybrid-rule planner | `mechanism_comparator` | `docs/context/policy_search/reports/2026-04-30_best_non_learning_local_policy_report.md` selects `hybrid_rule_v3_static_margin0_waypoint2` as the best current non-learning candidate on nominal and stress slices, while `docs/context/issue_2182_component_effect_synthesis.md` classifies static recentering as supported and several other knobs as neutral or weaker. | Mechanism-designed local planning currently gives clearer action-level explanations than generic learned warm starts or shields. Its evidence is still diagnostic and timeout-limited, not paper-grade causality. | Use as a mechanism comparator for recentering, guard authority, route commitment, and timeout recovery. |

## Research Decision

The current evidence supports a conservative mechanism-level direction with about `0.8`
confidence:

- stop repeating generic BC warm-start PPO continuations from the same preserved inputs;
- revise hard-guarded PPO objectives around progress and observation/action handoff, not only
  collision penalties;
- prioritize residual-over-ORCA and learned-risk-surface interfaces because they can expose bounded
  contribution diagnostics against a mechanism-designed command;
- use hybrid-rule component effects as comparators for learned components, while keeping those
  component effects diagnostic until broader scenario or seed evidence exists.

This does not support a global claim that learned local-navigation methods are ineffective. It says
the currently durable Robot SF evidence favors mechanism-aligned learned interfaces over generic
warm starts or shields.

## Limitations

- The #1977 BC warm-start run was cancelled and has no final checkpoint or policy-analysis
  comparison.
- Shielded PPO has a repaired smoke pass, but not nominal-sanity, stress, or full-matrix evidence.
- ORCA-residual has lineage and runtime-surface proof, not trained residual contribution evidence.
- Learned-risk and oracle-imitation lanes remain launch-packet or dataset-prep surfaces.
- Hybrid-rule component synthesis uses compact diagnostic slices; it guides design but should not be
  treated as paper-facing causality.
- Local `output/` paths referenced by source notes are not durable unless represented by tracked
  manifests or external artifact pointers.

## Validation

This synthesis changes docs and compact tracked evidence only. Validation should verify links and
diff hygiene:

```bash
for path in \
  docs/context/issue_1961_bc_warm_start_recoverability.md \
  docs/context/evidence/issue_1977_bc_warm_start_cancelled_2026-06-02/artifact_manifest.md \
  docs/context/issue_602_guarded_ppo_profile.md \
  docs/context/issue_1474_shielded_ppo_repair_closeout.md \
  docs/context/evidence/issue_1474_shielded_ppo_repair_2026-06-01/artifact_manifest.md \
  docs/context/issue_2006_guarded_ppo_zero_motion_repair.md \
  docs/context/policy_search/reports/2026-06-02_shielded_ppo_issue1474_collision20_v1_smoke.md \
  docs/context/issue_1428_orca_residual_lineage.md \
  docs/context/issue_1624_hybrid_learning_architecture.md \
  docs/context/policy_search/reports/2026-04-30_best_non_learning_local_policy_report.md \
  docs/context/issue_2182_component_effect_synthesis.md; do
  test -f "$path"
done
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
