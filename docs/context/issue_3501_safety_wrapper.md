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

## Deadlock recovery (`safety_wrapper_deadlock_recovery.v1`)

The stateful fourth stage lives in `DeadlockRecoveryMonitor` (same module), an **opt-in,
disabled-by-default** monitor composed *around* `apply_safety_wrapper`. Call `monitor.step(...)`
once per simulation step with the executed (post-transform) command and the same `SafetyContext`.
It breaks the *frozen robot* failure mode (planner keeps commanding near-zero forward speed while a
nearby pedestrian/obstacle blocks progress, so the episode stalls without a collision):

1. a step counts **frozen** when `|executed forward speed| ≤ frozen_speed_eps_m_s` **and** the step
   is hazard-blocked (nearest pedestrian within `hazard_proximity_m`, or finite predicted clearance
   ≤ `hazard_clearance_m`) — a legitimate goal-reached stop with clear surroundings is *not* a
   deadlock;
2. once the frozen run reaches `patience_steps`, a bounded in-place rotation
   (`recovery_turn_sign · recovery_angular_velocity_rad_s`) is applied for up to `recovery_steps`
   steps; a completed maneuver re-arms the patience window so a still-stuck robot pauses one cycle
   (letting the planner react) before rotating again.

**Safety property:** recovery only ever overrides *angular* velocity — forward speed is passed
through unchanged — so it never overrides the hard stop/yield veto and can never inject forward
motion into a hazard. Predeclared defaults: `patience_steps=20`, `recovery_steps=10`,
`recovery_angular_velocity_rad_s=0.5`, `recovery_turn_sign=+1`, `frozen_speed_eps_m_s=0.05`,
`hazard_proximity_m=2.0`, `hazard_clearance_m=0.5`.

## Scope boundary

`apply_safety_wrapper` is a pure per-step transform and `DeadlockRecoveryMonitor` is off by default,
so the wrapper **changes no runtime/benchmark behavior** unless explicitly opted in. Deferred
follow-ups: **wiring `DeadlockRecoveryMonitor` into the benchmark runtime step loop**
(`robot_sf/benchmark/safety_wrapper_runtime.py` + `map_runner_episode.py`, currently stateless), the
factorial `planner × {off, on}` **ablation campaign** (runs over a fixed scenario set + paired
seeds, emitting into the #3482 ledger), and live wiring into `robot_sf/robot/action_adapters.py`.

## Tests

`tests/test_safety_wrapper.py` (12 tests): disabled pass-through, safe pass-through, speed cap
near pedestrians (and no-cap when already slow), hard stop on low TTC and on low clearance, veto
precedence over the speed cap, schema/record contract, and config validation.
`tests/test_safety_wrapper_deadlock.py` (11 tests): disabled pass-through, no-recovery-before-patience,
recovery engages at patience and rotates in place, cyclic maneuver/patience re-arm, turn sign,
movement resets the frozen run, clear-surroundings-is-not-a-deadlock, forward speed never added under
a persistent hard stop, obstacle-only freeze, counter reset, and config validation.

## Related

- Safety-event ledger (ablation emit target): #3482.
- Trace-level predicates (deadlock/oscillation/occlusion signals): #3483.
- MOTP speed-cap contract (a related wrapper component): #3480.
