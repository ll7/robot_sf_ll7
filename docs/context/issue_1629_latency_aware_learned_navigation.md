# Issue #1629 Latency-Aware Learned Navigation Safety (2026-05-30)

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1629>

Date: 2026-05-30

## Goal

Assess whether learned local-navigation policies can be studied safely under realistic control and
perception latency, including 100 ms, 200 ms, and 500 ms delays.

This note is analysis-only. It does not run a latency campaign, train a policy, change simulator
timing semantics, or claim learned-policy safety under latency.

## Decision

Latency-aware learned navigation is a worthwhile AMV-specific research direction, but the first
implementation should be a **latency stress preflight contract**, not training or benchmark
execution.

The repository can already represent part of the problem:

- simulation `dt` is explicit through `SimulationSettings.time_per_step_in_secs`;
- the Issue #1556 synthetic AMV actuation slice already records action-delay/update-hold fields;
- `SyntheticActuationController` implements `zero-step-delay`, `one-step-delay`,
  `two-step-delay`, `10hz-matched`, and `5hz-hold`;
- LiDAR and graph observation paths already support bounded history ending at the current decision
  step;
- learned-policy adapter contracts require explicit observation, action, checkpoint, determinism,
  and fallback metadata.

The repository does **not** yet have a complete latency benchmark contract because observation
delay, planner update-rate holding, and latency-native metrics are not first-class in the primary
map-runner path.

## Latency Modes

Use these terms distinctly:

| Mode | Repository status | Interpretation boundary |
|---|---|---|
| Observation delay | Missing as a benchmark runner hook. `SensorFusion` and `SocialGraphObservationAdapter` keep history, but they do not expose delayed observation selection. | Needs an explicit queue/buffer contract so the policy consumes state from `t-k`, not just a stacked current-history tensor. |
| Action or actuation delay | Partly implemented by `SyntheticActuationController` and Issue #1556 profile metadata. | This is synthetic command-path delay, not measured hardware latency. It currently applies to differential-drive absolute commands. |
| Planner update frequency | Missing as a named map-runner control. Current policy calls are effectively once per environment step in `robot_sf/benchmark/map_runner.py`. | Needs `always`, periodic update, and hold-last semantics with stale-action provenance. |
| Policy inference deadline | Present in the older benchmark runner through `POLICY_STEP_TIMEOUT_SECS`, but not as a learned-policy latency metric in the main map-runner path. | Wall-clock timeout is not simulated delay; it is runtime reliability evidence and should be tracked separately. |

At the default benchmark `dt=0.1`, the requested delays map to:

| Delay | Step interpretation |
|---|---|
| 100 ms | one simulation step |
| 200 ms | two simulation steps |
| 500 ms | five simulation steps, currently beyond Issue #1556's built-in actuation delay labels |

The 500 ms case should therefore be represented as a planned latency mode in a new contract, not
overloaded onto the existing two-step synthetic actuation profile.

## Timing And Control Hook Inventory

| Surface | Existing coverage | Gap |
|---|---|---|
| `robot_sf/sim/sim_config.py` | Defines fixed simulation `time_per_step_in_secs`, default `0.1`. | No separate sensor, policy, or actuator latency fields. |
| `robot_sf/gym_env/robot_env.py` | Applies one parsed action per environment step and records `last_action` for reward and UI metadata. | No delayed-action queue, no stale-action ratio, no planner update cadence. |
| `robot_sf/benchmark/map_runner.py` | Calls `policy_fn(policy_obs)` once per episode step, applies observation noise, validates synthetic actuation, and records `dt`. | No observation-delay buffer, planner hold-last mode, or inference deadline counters in map-runner summaries. |
| `robot_sf/benchmark/synthetic_actuation.py` | Applies delayed, held, and clipped differential-drive commands and reports clip/yaw/braking summaries. | Delay labels stop at two steps and are scoped to synthetic actuation, not perception or policy update delay. |
| `robot_sf/sensor/sensor_fusion.py` | Emits oldest-to-newest `drive_state` and `rays` stacks. | Stack history is not equivalent to a delayed observation selector. |
| `robot_sf/sensor/social_graph_observation.py` | Maintains bounded pedestrian feature history and rejects future-like fields. | Does not encode observation age or latency mode. |
| `robot_sf/benchmark/runner.py` | Has `POLICY_STEP_TIMEOUT_SECS` and timeout fallback metadata for a process-isolated planner step path. | Not the canonical camera-ready/map-runner learned-policy path and not simulated latency. |
| `robot_sf/benchmark/metrics.py` | Provides safety, progress, smoothness, and timing-derived metrics. | No decision age, stale-action ratio, planner-miss rate, timeout fraction, or delay histogram metrics. |

## Candidate Learned-Policy Responses

| Candidate response | Fit for first latency work | Reason |
|---|---|---|
| Observation/history-aware PPO or LSTM policy | Later training candidate | Existing LiDAR plan includes a history-oriented PPO variant, but it needs a latency contract before training. |
| Delay randomization during training | Later training candidate | Useful only after the runtime delay modes are explicit and reproducible. |
| Recurrent memory | Later training candidate | Requires clear history semantics and reset/seed diagnostics in the learned-policy adapter. |
| Learned risk surface | Diagnostic only | Can consume current structured state, but it is not itself a latency-aware action policy. |
| Guarded policy or ORCA residual | Good first policy family after preflight | Guards make safety caveats observable, but fallback/guard output must not count as learned-policy success. |
| Planner arbitration | Not first | Switching under stale observations adds leakage and attribution risk before latency provenance exists. |

The first learned-policy candidate should not be selected until Issue #1744 or equivalent preflight
work can prove the latency modes are represented and reported correctly.

## Scenario And Metric Hypotheses

The latency stress slice should start as `paper_facing: false` and reuse compact scenarios already
identified as sensitive to delay, stop-go behavior, and yaw/curvature pressure:

| Scenario | Latency hypothesis |
|---|---|
| `classic_cross_trap_high` | Stale observations or held actions can amplify yaw deadlock and circulation. |
| `classic_bottleneck_high` | Delayed braking/restart decisions can increase stall, near-miss, or collision risk. |
| `francis2023_blind_corner` | Delayed turn-entry decisions can expose perception/action lag around occlusions. |
| `francis2023_intersection_wait` | Hold-last or delayed relaunch can separate safe waiting from failure to progress. |

Core metrics should include existing safety and progress metrics plus latency-native additions:

- existing: success, collisions, near misses, `min_clearance`, `time_to_collision_min`,
  `stalled_time`, `failure_to_progress`, `jerk_mean`, `jerk_max`, `curvature_mean`,
  `command_clip_fraction`, `yaw_rate_saturation_fraction`, and `signed_braking_peak_m_s2`;
- new: effective observation age, action age, held-action ratio, planner update interval,
  policy inference timeout count/fraction, fallback/degraded count, and delay-mode provenance.

## Follow-Up

Recommended first bounded implementation issue:
<https://github.com/ll7/robot_sf_ll7/issues/1744>

Issue #1744 was open when this note was validated on 2026-05-30.

That issue should add the smallest executable preflight/config contract for latency stress. It
should express observation delay, action or actuation delay, and planner update hold independently;
emit latency provenance and metrics; and fail closed for unsupported modes before benchmark evidence
is written.

Do not start latency-aware policy training, broad latency campaigns, or paper-facing AMV safety
claims before that preflight exists.

## Validation

This analysis was built from:

```bash
gh issue view 1629
gh issue list --search "repo:ll7/robot_sf_ll7 is:issue is:open latency learned navigation"
sed -n '1,240p' docs/context/issue_1546_amv_actuation_envelope_stress_slice.md
sed -n '1,180p' docs/context/issue_1556_amv_actuation_stress_slice.md
sed -n '1,140p' docs/context/issue_1618_learned_policy_adapter_interface.md
sed -n '1,220p' robot_sf/benchmark/synthetic_actuation.py
sed -n '1,340p' robot_sf/sensor/sensor_fusion.py
sed -n '1,180p' robot_sf/sensor/social_graph_observation.py
rg -n "POLICY_STEP_TIMEOUT_SECS|synthetic_actuation_profile|latency_mode|update_mode|observation_delay|history|dt" \
  robot_sf scripts configs tests docs/context -S
```

Validation for this docs-only change:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```

Both commands passed on 2026-05-30.
