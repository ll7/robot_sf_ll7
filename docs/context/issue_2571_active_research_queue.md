# Issue #2571 Active Research Queue 2026-06-07

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2571>

Status: current queue synthesis as of 2026-06-07. This note is a navigation layer, not a
separate backlog or benchmark result. GitHub issues remain the work system, and the linked context
notes remain authoritative for exact commands, artifacts, and evidence limits.

## Queue Rule

Prioritize work that can move a research claim boundary or unblock durable evidence. Do not treat
smoke, diagnostic, fallback, or local-output-only results as benchmark evidence. If a lane is gated
below, implement the named gate issue before scheduling the downstream training or benchmark task.

## Next-Cycle Threads

| Thread | Current evidence | Next concrete issue | Gate before expansion |
| --- | --- | --- | --- |
| Adversarial manifest quality | Issue #2524/PR #2559 added validator-backed adversarial manifest generation, Issue #2562/PR #2572 proved a route-materialized planner smoke path, Issue #2567/PR #2578 defines a compact quality-metric path, and Issue #2568 records the learned-expansion gate. These are diagnostic or pre-benchmark signals only. | Use [issue_2568_adversarial_expansion_gate.md](issue_2568_adversarial_expansion_gate.md) and [issue_2567_adversarial_manifest_quality.md](issue_2567_adversarial_manifest_quality.md) to compare manifest validity, degeneracy, novelty, perturbation distance, and smoke-yield signals before expansion. | Keep RL/diffusion adversarial expansion frozen unless the candidate batch has a #2567-style summary showing useful, non-degenerate behavior; even then, treat it as diagnostic until certified benchmark proof exists. |
| Oracle trace artifact access | PR #2556 finalized trace manifests/checksums for the #2441 oracle-imitation runs, but downstream consumers still need durable trace access rather than worktree-local `output/` assumptions. | [#2561](https://github.com/ll7/robot_sf_ll7/issues/2561) must promote durable pointers, checksums, and split routing for downstream training access. | [#2569](https://github.com/ll7/robot_sf_ll7/issues/2569) keeps oracle-imitation training gated until durable artifact access is available. |
| Scenario semantics and uncertainty | PR #2555 added the waiting-then-crossing fixture metadata, PR #2537 added the first ScenarioBelief uncertainty consumer smoke, Issue #2564/PR #2573 proved a trace-only signal-state proxy smoke, Issue #2565/PR #2574 proved an uncertainty-aware stream-gap planner-input smoke, and Issue #2538 adds the ScenarioBelief-to-planner projection smoke. These are diagnostic-only. | The next useful issue should connect a runtime ScenarioBelief producer or observation builder to this planner projection rather than adding another isolated unit smoke. | Treat signalized crossing and ScenarioBelief uncertainty as a coupled dependency pair; do not claim planner improvement until a runtime consumer or planner-observable contract exists. |

## Frozen Or Gated Lanes

| Lane | Current classification | Queue action |
| --- | --- | --- |
| Static recentering | Inactive on the current held-out transfer route after #2438/#2520; still useful as slice-local diagnostic evidence. | Keep out of primary transfer scheduling. Use [#2566](https://github.com/ll7/robot_sf_ll7/issues/2566) for any remaining mechanism-card/status propagation. |
| Topology mitigation | `revise` after #2530; #2563 selects primary-route reuse penalty as the next diagnostic hypothesis, and #2570 propagates the no-overclaim status. | Keep topology benchmark, leaderboard, and transfer claims frozen until #2540 or a narrower #2563-derived diagnostic shows corrective behavior. |
| Adversarial RL/diffusion | Proposal only until manifest quality is proven for the candidate batch. | Follow [issue_2568_adversarial_expansion_gate.md](issue_2568_adversarial_expansion_gate.md) before any learned adversarial expansion. |
| Oracle imitation training | Blocked on durable artifact access, not model code. | Do #2561 before #2569 or any new training run that depends on the traces. |

## Source Map

- Active dashboard: [issue_2228_research_dashboard.md](issue_2228_research_dashboard.md)
- Scientific lane states: [research_lane_states.md](research_lane_states.md)
- Adversarial manifest evidence: [issue_2524_adversarial_manifests.md](issue_2524_adversarial_manifests.md),
  [issue_2562_adversarial_manifest_smoke.md](issue_2562_adversarial_manifest_smoke.md),
  [issue_2567_adversarial_manifest_quality.md](issue_2567_adversarial_manifest_quality.md)
- Oracle trace evidence: [evidence/issue_2441_oracle_imitation_traces_2026-06-06/README.md](evidence/issue_2441_oracle_imitation_traces_2026-06-06/README.md)
- Scenario semantics evidence: [issue_2527_waiting_crossing_fixture.md](issue_2527_waiting_crossing_fixture.md),
  [issue_2528_scenario_belief_consumer_smoke.md](issue_2528_scenario_belief_consumer_smoke.md),
  [issue_2564_signal_state_proxy_smoke.md](issue_2564_signal_state_proxy_smoke.md),
  [issue_2565_uncertainty_gating_smoke.md](issue_2565_uncertainty_gating_smoke.md),
  [issue_2538_scenario_belief_planner_projection.md](issue_2538_scenario_belief_planner_projection.md);
  provenance PRs [#2555](https://github.com/ll7/robot_sf_ll7/pull/2555) and
  [#2537](https://github.com/ll7/robot_sf_ll7/pull/2537)
- Negative/revise guardrails: [issue_2438_static_recenter_activation_closure.md](issue_2438_static_recenter_activation_closure.md),
  [issue_2563_topology_corrective_revision.md](issue_2563_topology_corrective_revision.md),
  [issue_2570_topology_revise_status_propagation.md](issue_2570_topology_revise_status_propagation.md)

## Validation

This synthesis note should be validated as a docs-only context update:

```bash
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```
