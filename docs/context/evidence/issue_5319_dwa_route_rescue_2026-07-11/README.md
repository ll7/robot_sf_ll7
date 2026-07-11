<!-- AI-GENERATED (robot_sf#5319, 2026-07-11) - NEEDS-REVIEW -->
# Issue #5319 — DWA Route-Rescue Diagnostic Probe

Date: 2026-07-11

Issue: <https://github.com/ll7/robot_sf_ll7/issues/5319>
Baseline trace: <https://github.com/ll7/robot_sf_ll7/issues/5298> and `docs/context/evidence/issue_5298_dwa_decision_trace_2026-07-11/`.

## Claim boundary and status

- **Evidence status:** diagnostic-only.
- **Claim boundary:** two CPU-only fixed-seed episodes traced with the DWA route-rescue config (`configs/algos/dwa_route_rescue.yaml`). This does not change DWA roster status, benchmark metric semantics, the frozen v0.1 suite, or any paper/dissertation claim.
- **Interventions:** (1) route-rescue extends the rollout horizon and boosts progress weight when the robot stalls for `route_rescue_patience` steps; (2) feasibility-slowdown reduces linear speed when infeasible-candidate fraction exceeds a threshold.
- **Caveat:** this is a diagnostic probe, not a comparator benchmark. Two episodes cannot bound the full failure surface. The intervention may not generalize beyond these rows.
- **Seeds:** 131, 161.
- **Config SHA-256 hash:** `80d70aba8abc2934492a51ac0cffb0eedb30afb65832f1478e48341bd23c3ea2` for `configs/algos/dwa_route_rescue.yaml`.

## Episodes traced

| Episode | Scenario | Seed | Config | Outcome | Steps |
| --- | --- | --- | --- | --- | --- |
| bottleneck_timeout | `classic_bottleneck_medium` | 131 | route-rescue | max_steps | 100 |
| t_intersection_collision | `classic_t_intersection_low` | 161 | route-rescue | max_steps | 100 |

## Comparison with baseline

| Episode | Metric | Baseline (#5298) | Route-rescue | Delta |
| --- | --- | --- | --- | --- |
| bottleneck | steps | 100 | 100 | 0.000 |
| bottleneck | termination | max_steps | max_steps | same |
| | min_distance | 0.474 m | 0.474 m | 0.000 m |
| | net_progress | -2.885 m | -2.885 m | 0.000 m |
| | rescue_steps | 0 | 0 | — |
| | slowdown_steps | 0 | 0 | — |
| t_intersection | steps | 96 | 100 | 4.000 |
| t_intersection | termination | collision | max_steps | changed |
| | min_distance | 1.575 m | 1.587 m | 0.012 m |
| | net_progress | 1.178 m | 1.306 m | 0.128 m |
| | rescue_steps | 0 | 0 | — |
| | slowdown_steps | 0 | 0 | — |

## Mechanism analysis

### bottleneck_timeout

- Termination: max_steps after 100 steps.
- Route progress: initial 1.999 m, final 4.884 m, minimum 0.474 m, net -2.885 m.
- Route-rescue was active for 0 steps (first activation step: None).
- Feasibility-slowdown was active for 0 steps.

### t_intersection_collision

- Termination: max_steps after 100 steps.
- Route progress: initial 4.388 m, final 3.082 m, minimum 1.587 m, net 1.306 m.
- Route-rescue was active for 0 steps (first activation step: None).
- Feasibility-slowdown was active for 0 steps.
- First infeasible candidate at step 14.

## Verdict

This is a diagnostic probe, not a success/fail benchmark claim. The table above identifies whether the route-rescue/feasibility-slowdown intervention improves, fails to improve, or changes the mechanism on both rows.

If either row remains unresolved or the intervention alters the contract unsafely, the diagnostic classification is retained and the next smallest probe is named.

## Reproduction

```bash
DISPLAY= SDL_VIDEODRIVER=dummy MPLBACKEND=Agg uv run python \
  scripts/benchmark/trace_dwa_route_rescue_issue_5319.py \
  --out-dir output/benchmarks/issue_5319 \
  --evidence-dir docs/context/evidence/issue_5319_dwa_route_rescue_2026-07-11
```

Executed at repo commit `4132ef70aef95ea937766f60e4bd7291a8e56146`.

