# Issue #3501 — Planner-agnostic safety wrapper (mitigation lever, first increment)

**Status:** diagnostic / proxy. **Evidence grade:** idea-level; wrapper thresholds are
predeclared modeling choices, diagnostic until durable evidence. No paper-facing
deployment-safety claim before the with/without ablation runs.

## What this is

`robot_sf/robot/safety_wrapper.py` is the **mitigation lever** both external dissertation
assessments flag as the largest improvement per implementation hour: a single, planner-agnostic
safety wrapper that post-processes any planner's commanded action through fixed, predeclared
stages, so a factorial `planner × {wrapper off, wrapper on}` ablation can quantify its causal
effect — "the framework identifies a mitigation lever and quantifies its effect."

## Wrapper (`safety_wrapper.v1`)

`apply_safety_wrapper(linear_velocity, angular_velocity, context, config)` applies, in precedence
order:

1. **hard stop / yield veto** — zero forward speed when `min_ttc_s ≤ ttc_veto_threshold_s` or
   `min_clearance_m ≤ clearance_veto_m` (angular velocity preserved so the robot can turn to yield);
2. **speed cap near pedestrians** — clamp forward speed to `capped_speed_m_s` within
   `pedestrian_caution_radius_m`;
3. **pass-through** otherwise.

It returns a versioned record with the corrected action, the `intervention` label
(`disabled` / `none` / `speed_cap` / `hard_stop`), an `intervened` flag (for intervention-rate
reporting), and the original action + context.

### Predeclared defaults (`SafetyWrapperConfig`, planner-agnostic, no per-planner tuning)

| field | default | meaning |
| --- | --- | --- |
| `enabled` | `False` | **off by default**, opt-in per run |
| `pedestrian_caution_radius_m` | 2.0 | speed-cap radius |
| `capped_speed_m_s` | 0.5 | defensive speed ceiling |
| `ttc_veto_threshold_s` | 1.0 | hard-stop TTC gate |
| `clearance_veto_m` | 0.3 | hard-stop clearance gate |

## Scope boundary

Pure per-step transform, **off by default** — changes no runtime/benchmark behavior. Deferred
follow-ups: the factorial `planner × {off, on}` **ablation campaign** (SLURM runs over a fixed
scenario set + paired seeds, emitting into the #3482 ledger), live wiring into
`robot_sf/robot/action_adapters.py`, and a stateful deadlock-recovery stage.

## Tests

`tests/test_safety_wrapper.py` (12 tests): disabled pass-through, safe pass-through, speed cap
near pedestrians (and no-cap when already slow), hard stop on low TTC and on low clearance, veto
precedence over the speed cap, schema/record contract, and config validation.

## Related

- Safety-event ledger (ablation emit target): #3482.
- Trace-level predicates (deadlock/oscillation/occlusion signals): #3483.
- MOTP speed-cap contract (a related wrapper component): #3480.
