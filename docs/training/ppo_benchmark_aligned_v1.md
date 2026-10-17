# PPO benchmark-aligned family v1

Implements issue #9628 sections 5 and 6 for the campaign defined there: 48
scenario cells, goal-zone entry, h600 and 0.1 s steps. This is development
configuration and plumbing evidence, with no training, navigation improvement,
release promotion or paper-admission claim.

| Setting | Old issue-791 recipe | New `benchmark_aligned_v1` | Benchmark contract in #9628 |
| --- | --- | --- | --- |
| Linear / angular cap | 3.0 m/s / 1.0 rad/s | 2.0 m/s / 1.0 rad/s | 2.0 m/s / 1.0 rad/s in the CPU PPO adapter |
| Reverse motion | Enabled | Disabled | Disabled by the differential-drive release plant |
| Policy action | Native acceleration | Signed velocity delta → bounded velocity target → acceleration | PPO adapter decodes delta → velocity command; runner converts target to acceleration |
| Acceleration / braking / turn ramp | Default ±1 limits | Explicit 1 m/s² linear/braking, 1 rad/s² angular | Same plant limits |
| Goal completion | Unset; resolved maps use `waypoint_radius_v1` | Explicit `goal_zone_entry_v1` | `goal_zone_entry_v1` release matrix |
| Episode budget / dt | Scenario-authored 120–700 steps on current main / 0.1 s | 600 steps / 0.1 s (60 s) in both roles | h600 / 0.1 s campaign specified in #9628 |
| Scenario coverage | Maintained superset (historical comment: 70; current main: 71) | 48 campaign cells | 48 campaign cells |
| Training episode seeds | Unbound randomisation | 100000–199999 under logged master 20261012; #10324 owns sampling | Disjoint from selection/reporting |
| In-training evaluation / checkpoint selection | 70 episodes, no selection manifest; deterministic evaluator falls back to training seeds | 1031–1060 only; every 48×30 pair, 1,440 episodes | Selection pool separated from reporting by the 2026-10-12 author ruling |
| Reporting seeds | No separately declared reporting pool | 1101–1130, metadata only; never passed to the trainer evaluator | 1101–1130 development reporting; forbidden release bands stay sealed |
| Best checkpoint / stopping | `success_rate`; convergence thresholds enabled | Selection `success_rate`; full inherited 10M budget, convergence threshold unreachable | Reporting results cannot choose checkpoints |

The old goal policy is confirmed by resolving the original leaf and its scenarios:
`sim_config.goal_completion_policy` is `None` and every resolved map has
`waypoint_radius_v1`. This confirms the config/runtime default, without claiming
historical job telemetry. Current-main horizons resolve to 120 (1 cell), 250 (10),
300 (4), 320 (3), 350 (1), 380 (1), 400 (27), 500 (13), 600 (9), 650 (1), 700 (1).
The old 70-episode evaluation therefore does not cover every current scenario.

The target here is #9628's uniform h600 campaign. Release 0.0.8 has different,
scenario-authored horizons and is not reproduced by this family. The derived
scenario file preserves geometry and explicitly binds the common 2 m/s plant
(including the current matrix's scenario-specific speed override). Use that same
derived matrix for future development reporting. Released files remain intact.

The runner reuses `unicycle_velocity_target_to_acceleration`, already called by
`RobotEnv.step`. Signed deltas use the existing `ppo_delta_to_velocity_target`
path and float32 observed speed; native benchmark environments keep acceleration
actions. The independent motion test checks braking, zero/reverse prevention,
speed caps and the turn-rate ramp, then compares actual speed and pose.

## Preflight and sampler integration hook

```bash
uv run python -m scripts.training.train_ppo_benchmark_aligned --preflight
```

Preflight resolves both scenario roles, plant, goal, horizon and the actual
selection manifest without constructing an environment. `sampler_available: false`
means conditions are valid but training is unavailable. Execution through this
entrypoint refuses until #10324's `training_seed_range`, `training_master_seed` and
`development_evaluation_seeds` interface exists in the trainer. The three fields
live in `configs/training/ppo/benchmark_aligned_v1_seed_policy.yaml`, a marked
sidecar, rather than the canonical leaf. The stock loader validates the leaf today;
its fixed seed/pool `[100000]` is a parsing template, not an approved campaign.
The leaf sets `scenario_sampling.require_training_seed_policy: true`. The stock
trainer checks it before CLI dispatch and again at direct API execution, before
backend/seed initialisation or artifacts. Loading still succeeds, while execution
(including dry runs) refuses an absent interface or an unbound single-seed template.
Always launch this family through the guarded entrypoint. Once #10324 is integrated,
it replaces the template with the sidecar's range/master/reporting metadata through
the supported dataclass interface, then delegates to the existing training API.
No sampler or reset logic is implemented here. The reviewed execution-boundary
fix adds only the dependency refusal to the stock trainer; seed sampling and worker
telemetry remain owned by #10324/#10343. The canonical inventory still requires
150 loadable leaves, retaining all key validation. Archive both
the leaf and its sidecar in future launch packets.

Use the guarded launcher for this family. The guard requires the exact approved
training interval and selection/reporting pools, strict integer/unique seeds and
pairwise disjointness. Positive allowlists exclude every retired/sealed identity
without exposing their values. Random selection, partial coverage, auxiliary
new-best analysis and range-seeded resume are refused. The real periodic and final
evaluator tests observe only selection identities, including best-checkpoint
tracking. Reporting is a separate future campaign, never an in-training phase.

Reward, network, optimiser, curricula and 10M training budget are inherited from
the original recipe. Reward defects, release-SNQI equivalence and broader selection
methodology remain tracked on #9628 in their owning lanes. Geometry/family overlap
means this is seed-held-out development tuning, not family-held-out generalisation.
No new checkpoint exists; do not reinterpret or replace a released checkpoint.
Future checkpoint registration must declare `action_semantics: velocity_delta`.
