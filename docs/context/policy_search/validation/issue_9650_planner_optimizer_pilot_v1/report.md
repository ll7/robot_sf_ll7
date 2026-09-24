# Planner optimizer bounded pilot: issue_9650_planner_optimizer_pilot_v1

- Source revision: `b461e7ad008596343eb789558fb85ed1f6a2fae4`
- Evidence class: diagnostic finite-budget simulator pilot; not nominal or release benchmark evidence
- Baseline candidate: `hybrid_rule_v3_fast_progress`
- Training episodes per optimizer trial: 2
- Trial budget: 3 per method, equal across Random and TPE
- Total search evaluations: 12 episodes
- Held-out identities: `44fc675dab630c932f04fd21cef155ee752e27be8d38de31649a2f37f40d359d`

## Objective order

1. Valid benchmark execution/admissibility
2. Collision-free fraction
3. Near-miss-free fraction using the policy-search surface-clearance semantics
4. Task completion fraction
5. Lower successful time-to-goal ideal ratio
6. Lower pedestrian 95th-percentile force (when available)

The recorded selection tuple preserves this order. Optuna receives the same component vector; no weighted aggregate is used.
Missing secondary metrics remain null in the evidence and receive a tied sentinel only in the sampler interface.

## Training results

- Baseline: `[1.0, 0.5, 0.5, 0.0, None, -1.4996569299856108]` (complete); valid 2/2; collision episodes 1/2; near-miss-free 0.500; successes 0/2; successful time ratio unknown; mean episode p95 force 1.500.
- RANDOM best trial `2`: `[1.0, 0.5, 1.0, 0.0, None, -1.4329363288945007]` (complete); valid 2/2; collision episodes 1/2; near-miss-free 1.000; successes 0/2; successful time ratio unknown; mean episode p95 force 1.433; parameters `{"goal_progress_weight": 5.975093803501739, "max_linear_speed": 2.166092040005835}`
  Invalid trials: 0 / 3; runtime `19.188s`.
- TPE best trial `0`: `[1.0, 0.5, 1.0, 0.0, None, -1.4357967887106322]` (complete); valid 2/2; collision episodes 1/2; near-miss-free 1.000; successes 0/2; successful time ratio unknown; mean episode p95 force 1.436; parameters `{"goal_progress_weight": 5.54963516349268, "max_linear_speed": 2.7672097183575333}`
  Invalid trials: 0 / 3; runtime `18.579s`.

## Frozen held-out comparison

- Selected method/config: `random` / `issue_9650_planner_optimizer_pilot_v1_best_random`.
- Baseline: `[1.0, 1.0, 0.5, 0.0, None, -0.6030521566868655]` (complete); valid 2/2; collision episodes 0/2; near-miss-free 0.500; successes 0/2; successful time ratio unknown; mean episode p95 force 0.603.
- Selected: `[1.0, 1.0, 1.0, 0.0, None, -0.4706898948348184]` (complete); valid 2/2; collision episodes 0/2; near-miss-free 1.000; successes 0/2; successful time ratio unknown; mean episode p95 force 0.471.
- Training-set objective improvement over baseline: `True`.
- Held-out objective improvement over baseline: `True`.

## Interpretation boundary

The lexicographic improvement is limited to near-miss rate and force on this tiny pilot: task completion did not improve, and the training collision remained. Efficiency is unknown because no episode completed successfully. Three trials per method are too few to infer that one search method is better. This demonstrates only observed planner/config behavior on these exact rows; it does not establish global optimality, feasibility for other inputs, or real-world safety. No mathematical feasibility oracle is claimed.

Canonical config export: `best_candidate.yaml`.
Existing policy-search runner registry: `candidate_registry.yaml`.

## Canonical export replay smoke

The selected configuration loaded from the exported registry and ran through the existing
`run_policy_search_candidate.py` CLI with decision `pass` for one `smoke` episode. The episode
ended at `max_steps` with zero task successes; fallback/degraded status was `clear`. This verifies
that the exported config is consumable by the canonical runner, not that it completes the task.
The compact per-episode receipt is `pilot_summary.json`.

The full original episode bundle is retained as a local companion artifact rather than committed:
some canonical episode records contain absolute local map paths. Its SHA-256 is
`1eec9bb8720105c00cd8d4b243a2bc024e19ba263423e8169758bfffc2bfaa59`.
