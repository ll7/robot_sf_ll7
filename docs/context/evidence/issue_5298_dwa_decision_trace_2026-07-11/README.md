<!-- AI-GENERATED (robot_sf#5298, 2026-07-11) - NEEDS-REVIEW -->
# Issue #5298 — DWA Decision Trace for the #5262 Timeout and T-Intersection Collision

Date: 2026-07-11

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/5298>
Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/5262>
Source diagnostic PR and packet: #5274 and 
`docs/context/evidence/issue_5262_dwa_config_sensitivity_2026-07-11/` (on the #5274 branch).
Archetype-matrix evidence: `docs/context/evidence/issue_5020_dwa_archetype_matrix_2026-07-10/`.

## Claim boundary and status

- **Evidence status:** analysis-only.
- **Claim boundary:** two CPU-only fixed-seed episodes traced with the canonical DWA config. This does not change DWA roster status, benchmark metric semantics, the frozen v0.1 suite, or any paper/dissertation claim. It diagnoses the observed failure mechanism; it is not a comparator run.
- **Major caveats:** the trace reproduces the two canonical-config episodes selected by the #5262 manifest. The non-canonical config points from #5262 are out of scope here. Two episodes cannot bound the full failure surface; they isolate the mechanism on the named rows.
- **Uncertainty:** about 85% confidence that the mechanism identified below is the dominant cause on these two rows. That conclusion would change if a deeper rollout-horizon or global-route probe isolates a distinct driver.

## Traced episodes

| Episode | Scenario | Seed | Config | Outcome | Steps |
| --- | --- | --- | --- | --- | --- |
| bottleneck_timeout | `classic_bottleneck_medium` | 131 | canonical `configs/algos/dwa_classic.yaml` | max_steps | 100 |
| t_intersection_collision | `classic_t_intersection_low` | 161 | canonical `configs/algos/dwa_classic.yaml` | collision | 96 |

The seeds come from the standard classic archetype matrix declaration 
(`configs/scenarios/classic_interactions.yaml`); the #5262 manifest's canonical config point applies 
no overrides, so these rows reproduce the #5262 canonical episode outcomes exactly.

## Per-step trace artifacts

- [`dwa_decision_trace_steps.csv`](dwa_decision_trace_steps.csv): one row per planner step with the 
  selected command, selected score, feasible/infeasible candidate counts, dynamic-window bounds, 
  constraint reason, distance-to-goal, and route-progress state.
- [`dwa_decision_trace_summary.json`](dwa_decision_trace_summary.json): per-episode mechanism 
  summary (constraint-reason counts, route-progress stats, first-unrecoverable step).

## Failure mechanism

### bottleneck_timeout — progress stall, not a clearance deadlock

- All 100 planner steps selected `best_feasible`; **no step ever 
  reached the all-candidates-infeasible safety fallback**. constraint_reason_counts={'best_feasible': 100}.
- Route progress: initial distance to goal 1.999 m, final 4.884 m, minimum 0.474 m (net progress -2.885 m, -144.3% of the initial gap closed).
- The robot keeps selecting a forward feasible command but never closes the final 4.884 m to within `goal_tolerance=0.25 m` within the 100-step horizon. The selected last command is v=1.000 m/s, omega=0.000 rad/s — full forward speed, straight. This is a local-minimum / route-progress stall against the bottleneck geometry, **not** a blocked dynamic window.
- **First observable unrecoverable point:** no single step is unrecoverable in the clearance sense; the episode becomes unrecoverable when the remaining-goal distance stops decreasing for the rest of the horizon. The bounded 15-step × 0.1 s rollout keeps scoring forward motion as feasible even though the global route never converges.

### t_intersection_collision — short rollout horizon misses the collision until the last steps

- 96 planner steps traced; constraint_reason_counts={'best_feasible': 96}.
- The first step at which **any** rollout candidate became infeasible was step 91; the controller still found a `best_feasible` forward command and continued.
- **No step reached the all-candidates-infeasible safety fallback**: the planner always found at least one feasible constant-velocity rollout under its bounded horizon, so it never switched to the zero-command brake. It collided at full forward speed before the horizon caught the contact.
- Route progress: initial distance to goal 4.388 m, minimum 1.575 m. Last selected command v=1.000 m/s, omega=0.000 rad/s — the robot was still driving forward into the junction when it collided.
- **First observable unrecoverable point:** the collision is observable in the trace as the shrinking feasible-candidate fraction over the final steps; the bounded 1.5 s prediction horizon cannot foresee the T-intersection contact early enough to trigger the all-infeasible brake, so the controller commits forward until contact.

## Verdict

**Bounded implementation repair is supported**, not a roster exclusion or a different diagnostic. The two traced mechanisms are both controller-horizon / route-progress properties rather than a 
config-sensitivity surface (consistent with the #5262 `needs-implementation-change` verdict):

1. The bottleneck timeout is a global route-progress stall that a one-period reactive window cannot 
   resolve — the controller never gets stuck on clearance, it just never converges to the goal.
2. The T-intersection collision is a bounded prediction-horizon miss — the 1.5 s constant-velocity 
   rollout keeps a forward command feasible until the contact is ~5 steps away.

The next bounded repair/experiment should target the DWA rollout horizon and its global-route / goal 
convergence behavior, not the velocity/acceleration/tolerance axes already swept in #5262. That 
follow-up should be tracked in its own scoped issue (see PR body).

## Reproduction

```bash
DISPLAY= SDL_VIDEODRIVER=dummy MPLBACKEND=Agg uv run python \
  scripts/benchmark/trace_dwa_decisions_issue_5298.py \
  --out-dir output/benchmarks/issue_5298 \
  --evidence-dir docs/context/evidence/issue_5298_dwa_decision_trace_2026-07-11
```

Executed at repo commit `ecc70db75f4fcb1089beb7da75b423ac348aa5cd`. Raw per-step trace is also written to the disposable 
`output/benchmarks/issue_5298/dwa_decision_trace.json`; this packet keeps the compact derived 
steps CSV and summary JSON needed to review the mechanism.

## Acceptance mapping (issue #5298 definition of done)

- [x] A committed trace artifact names the exact config, scenario, and seed for both selected 
      episodes.
- [x] The trace identifies the failure mechanism (bottleneck progress stall; T-intersection 
      bounded-horizon collision miss).
- [x] The conclusion names the next bounded repair/experiment direction (rollout-horizon and 
      global-route convergence), tracked as a follow-up issue.

