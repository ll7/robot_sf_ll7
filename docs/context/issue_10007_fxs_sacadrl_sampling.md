# Issue #10007 FXS: SA-CADRL inputs and sampling drive forecasts

Scope: B1-B4 on PR #9926 head `d41cceb7f9422337bd34b602ae49f3dddbddb2a5`.
Development diagnostics only; no held-out performance or publication claim. Frozen
manifests and historical 0.0.2/0.0.7 artifacts are unchanged.

## SA-CADRL release inputs (B1/B2)

The release arm reads `configs/algos/socnav_release_v0_0_8.yaml` through the
benchmark's `_build_socnav_config`. Its 2 m/s speed cap did not override the
1 m/s preferred speed or three-agent default. The checkpoint therefore saw
`pref_speed=1` and at most three pedestrian records, independently of drive bounds.
The release config now supplies `sacadrl_pref_speed: 2.0` and
`sacadrl_max_other_agents: 19`. These names are specific to SA-CADRL and do not
alter other SocNav arms sharing the bounds config. General defaults stay historical.

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
of actions selected over 300 states, not the full table's cardinality. Raising
preferred speed changes these scales from {1, 0.5, 0} to {2, 1, 0} m/s and
leaves the full cardinality **9 before / 9 after**.

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

## Reproduction and regression gate

Run `scripts/diagnostics/probe_fxs_sacadrl_sampling.py --output <durable-directory>`
through the project environment at the base and fixed checkout. It runs native map
episodes for seeds 1001–1010 on group crossing high, doorway medium and head-on
corridor medium, both arms: 60 episodes per revision. Horizon comes from each
scenario. No fallback, timeout overrides, held-out seeds or degraded backend.
The script compares every sampler candidate and swept blocking index against
commands applied to the native robot, and records SA-CADRL input/command statistics.
The additional three-versus-nineteen-agent inference is observational only.

Regression file: `tests/planner/test_fxs_release_contract.py`.

| Test | Defect / credible regression | Existing coverage gap | Bytes / independent oracle |
|---|---|---|---|
| preferred speed | Release silently reverts to the 1 m/s default | Release action-bound test injects its own preferred speed; it never tests release host input | Actual release YAML → benchmark filter → checkpoint host feature; literal drive speed 2 |
| observed agents | Release silently truncates to three | Observation parity probe explicitly sets native row count | Actual release YAML and real 138-column tensor; ordered pedestrian positions and valid count; padded rows excluded |
| first rollout step | Instant turn or endpoint Euler displacement returns | Existing rollout-limit test expects endpoint Euler distance and omits angular acceleration | Hand-calculated 5 mm displacement and 0.005 rad rotation from trapezoidal wheel integration |
| bound drive / turn state | Binding or flat angular velocity is lost | Existing binding fixture has only linear limits and radius | Real flat observation, nondefault drive settings, actual DifferentialDriveRobot applied action; production candidate compared across consumer boundary |

No production test-only seams. Base failures and numeric episode outcomes are in the
adjacent evidence summary. These probes support implementation integrity only.
