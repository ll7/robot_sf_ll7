# Issue #10007 FXS: SA-CADRL inputs and sampling drive forecasts

Scope: B1-B4 on PR #9926 head `d41cceb7f9422337bd34b602ae49f3dddbddb2a5`.
Delivery base refreshed to `b3fa204dc4eda5004d67e4fcf518f3370fcd0966`.
The refreshed base has identical planner/drive/environment/map-runner/scenario/map/checkpoint
bytes on the inspected contract paths, so the baseline probes remain applicable.
Development diagnostics only; no held-out performance or publication claim. Frozen
manifests and historical 0.0.2/0.0.7 artifacts are unchanged.

## SA-CADRL release inputs (B1/B2)

The release arm reads `configs/algos/socnav_release_v0_0_8.yaml` through the
benchmark's `_build_socnav_config`. The release now explicitly pins
`sacadrl_pref_speed: 1.0` and retains `sacadrl_max_other_agents: 19`. Both 1.0 and
2.0 m/s are inside the checkpoint's training range, but the benchmark drive brakes
at 1 m/s² while upstream dynamics change speed instantly. At 2.0 m/s the policy's
stop action does not stop in time: stopping distance is 2 m, versus 0.5 m at 1.0.
The controlled split below identifies the preference change as the cause of the
observed outcome collapse. The delegated decision restores the working 1.0 m/s
configuration and keeps the 19-agent capacity. These SA-CADRL-specific keys do not
alter other SocNav arms sharing the bounds config. General defaults stay historical.
Reopen this speed decision only with new controlled evidence or changed drive dynamics.

Upstream source is pinned to gym-collision-avoidance commit
`903564097509e3fbdbbb850a3a89729a28377b81`:

- [Training testcase bounds, config.py lines 49-62](https://github.com/mit-acl/gym-collision-avoidance/blob/903564097509e3fbdbbb850a3a89729a28377b81/gym_collision_avoidance/envs/config.py#L49-L62): preferred speeds span 0.5–2.0 m/s.
- [Base observation sizing, lines 64-70](https://github.com/mit-acl/gym-collision-avoidance/blob/903564097509e3fbdbbb850a3a89729a28377b81/gym_collision_avoidance/envs/config.py#L64-L70): observed agents default to environment size minus one.
- [EvaluateConfig, lines 193-200](https://github.com/mit-acl/gym-collision-avoidance/blob/903564097509e3fbdbbb850a3a89729a28377b81/gym_collision_avoidance/envs/config.py#L193-L200): 19 total agents implies 18 observed others; dt is 0.1 s.
- [FullTestSuite, lines 254-265](https://github.com/mit-acl/gym-collision-avoidance/blob/903564097509e3fbdbbb850a3a89729a28377b81/gym_collision_avoidance/envs/config.py#L254-L265): explicitly sets 19 observed others. Its default test populations are 2, 3 and 4 total agents; capacity is not population.

The bundled IROS18 network has 138 input columns: one count, four host features,
and 19 seven-feature agent slots. The count excludes unused padded slots. The
release uses FullTestSuite's observation capacity, without claiming its scenario
population or dynamics match Robot SF. Physical robot radius 1 m remains above
the upstream training radius bound 0.8 m; this input change does not cure that limitation.

## Heading-action adaptation (B3: confirmed, documented)

Upstream [UnicycleDynamics.step](https://github.com/mit-acl/gym-collision-avoidance/blob/903564097509e3fbdbbb850a3a89729a28377b81/gym_collision_avoidance/envs/dynamics/UnicycleDynamics.py#L14-L42)
instantly adds the selected relative heading and then translates. Robot SF's
rate-limited differential drive cannot reproduce that instantaneous turn.

The adapter commands `clip(delta_heading / dt, -1, 1)` rad/s. This already turns
toward the selected heading at the rate limit over the current step. At dt=0.1,
nonzero choices ±π/12 and ±π/6 all exceed the 0.1 rad per-step rate bound.
The 11-action table becomes **nine distinct commands**, three turn rates for
each of three speed scales (1, 0.5, 0). The hunt's six was an empirical sample
of actions selected over 300 states, not the full table's cardinality. With the
restored 1.0 m/s preference, emitted speed scales remain {1, 0.5, 0} m/s and
the full cardinality remains nine.

The drive additionally limits angular acceleration to 1 rad/s²: from rest the
first endpoint rate is 0.1 rad/s and the first heading increment is 0.005 rad
(trapezoidal wheel odometry). Over that first step all positive heading choices
have the same realized turn. Smaller/larger choices remain indistinguishable
when the policy is queried anew every 0.1 s. This is an explicit kinematic
adaptation limitation, not action-side parity with upstream.

No heading-hold state is added. Holding an old target until completion would
suppress fresh checkpoint decisions while pedestrians move and would change the
policy's temporal semantics. Recomputing that target every step reproduces the
current saturation; scaling angles to manufacture extra rates would change their
meaning. A future persistent-heading method needs its own declared method and
validation. B3 makes no runtime behavior change in this PR.

## Sampling rollout (B4)

The old `_rollout` used endpoint linear velocity for displacement and instantly
set angular velocity from heading error. It neither initialized from observed
turn rate nor bound angular acceleration. That disagrees with both the drive's
0.1 rad/s per-step ramp and its trapezoidal wheel odometry.

Candidate forecasts now apply velocity targets as accelerations to the actual
`DifferentialDriveRobot`, seeded with measured linear/angular velocity and wheel
speeds. Bound drive settings include angular speed/acceleration and wheel geometry.
The same drive implementation underlies PR #9926's `native_drive_rollout`;
a heading-feedback controller requires recomputing each command after each forecast
step, rather than passing a fixed open-loop sequence to that helper. Candidate
horizons and release enhancement switches are unchanged. Future aligned escape
samples still describe translation after a stopped robot completes its in-place turn.

The diagnostic's `native_candidate` mirrors the new `_rollout`; its after-fix
agreement is by construction. The before numbers show the old rollout differed
from this native-drive model by up to 2.43 m and yielded 1,300 false-clear
candidates. Execution-match is not shown: executed commands add penalty scaling
and brake limits, and the planner replans every step.

## Reproduction and regression gate

Run `scripts/diagnostics/probe_fxs_sacadrl_sampling.py --output <durable-directory>`
through the project environment at the base and fixed checkout. It runs native map
episodes for seeds 1001–1010 on group crossing high, doorway medium and head-on
corridor medium, both arms: 60 episodes per revision. Horizon comes from each
scenario. No fallback, timeout overrides, held-out seeds or degraded backend.
The script compares every sampler candidate and swept blocking index against
the matching native-drive model, and records SA-CADRL input/command statistics.
The additional three-versus-nineteen-agent inference is observational only.

Regression file: `tests/planner/test_fxs_release_contract.py`.

| Test | Defect / credible regression | Existing coverage gap | Literal contract / reference |
|---|---|---|---|
| preferred speed | Release silently changes from the working 1.0 m/s preference | Release action-bound test injects its own preferred speed; it never tests release host input | Actual release YAML → benchmark filter → checkpoint host feature; literal 1.0 with the braking-lag reason |
| observed agents | Release silently truncates to three | Observation parity probe explicitly sets native row count | Actual release YAML and real 138-column tensor; shuffled input with independently sorted literal expected positions and valid count; padded rows excluded |
| first rollout step | Instant turn or endpoint Euler displacement returns | Existing rollout-limit test expects endpoint Euler distance and omits angular acceleration | Hand-calculated 5 mm displacement and 0.005 rad rotation from trapezoidal wheel integration |
| bound drive / turn state | Binding or flat angular velocity is lost | Existing binding fixture has only linear limits and radius | Real flat observation, nondefault drive settings, actual DifferentialDriveRobot applied action; production candidate compared across consumer boundary |

The model now rejects sequences exceeding the checkpoint's 19 slots before inference,
rather than cropping them with an inconsistent count. A separate regression builds
the real 20-slot input and asserts that inference is never called.

No production test-only seams. Historical base failures and initial episode outcomes
are in the [original evidence summary](evidence/issue_10007_fxs_summary.json); its
`after` phase used 2.0 m/s and is superseded for release configuration. FXS2 split
and replay evidence is in the [decision summary](evidence/issue_10007_fxs2_summary.json).
These probes support implementation integrity only.

## Measured outcomes and limitations

SA-CADRL split: the same three scenarios and seeds 1001–1010, 30 episodes per
arm, native TensorFlow checkpoint without fallback. Reviewer runs at
`1e1b9bcb8dbad2dda7d07db286678bb564e33c81` separate B1 from B2:

| Arm | Preferred speed (m/s) | Observed-agent capacity | Success / collision (of 30) |
|---|---:|---:|---:|
| A | 1.0 | 3 | 11 / 19 |
| B (B1 only) | 2.0 | 3 | 2 / 28 |
| C (B2 only; chosen release configuration) | 1.0 | 19 | 14 / 16 |
| D (initial combined change) | 2.0 | 19 | 3 / 27 |

B versus A and D versus C isolate the collapse to the 2.0 m/s preference under
the benchmark's braking lag. C versus A supports keeping the 19-agent capacity
(corridor successes 4 → 7). The FXS2 arm-C replay reproduces **14/30 successes**
and **16/30 collisions**, matching outcome, step count and termination reason on
all 30 reviewer episodes (crossing 7/3, doorway 0/10, corridor 7/3). These
are development diagnostics, not held-out or paper-grade performance evidence.
Doorway is 10/10 collisions in every arm; episodes contain only 4–6 pedestrians
and do not exercise the 19-slot ceiling. Physical radius and heading adaptation
limitations remain. No retuning is included.

Sampling's 213,120 baseline candidates differed from the native-drive model by
up to 2.427882 m, with 32,367 swept blocking-index differences and 1,300
false-clear candidates. The 213,252 fixed candidates agree within floating-point
precision (maximum error 1.4211e-14 m, zero blocking-index differences and zero
false-clear candidates) **by construction**, because the reference mirrors the
new rollout. Execution-match is not shown. All 30 sampling episodes succeeded
before and after; candidate classifications are not executed collision counts or
a safety guarantee.

The restored preference intentionally passes on the base, which already preferred
1.0 m/s; its new assertion fails on the reviewed 2.0 m/s head. The other four
original regression cases still fail on the base for their intended defects.
The additional oversized-input guard fails before implementation. The reviewer
no-sort mutation is killed by the shuffled fixture (17/19 slot mismatches).
All 44 original focused tests plus the guard pass: **45 passed**. Ruff check,
format check on all six PR Python paths and whitespace checks pass. The FXS2
summary records named test nodes and evidence hashes. No skip/xfail/timeout edits
are made; the broad readiness suite remains unrun under the seed/node restriction.

**Validation exception:** an additional run of `tests/test_socnav_env_integration.py`
executed its existing `env.reset(seed=123)` fixture before its seed values were
inspected. That violates the task's seed restriction. The run is excluded from
accepted evidence; all admitted before/after probe episodes use 1001–1010. The full
readiness suite is unrun because its broad test lanes include fixtures outside the
authorized seed set. This draft does not claim full readiness or held-out evaluation.
