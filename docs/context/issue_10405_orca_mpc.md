# Issue 10405: ORCA-initialised rollout MPC (B)

**Diagnostic-only empirical 0.1.0 research.** Opt in with algorithm `orca_mpc`
and [orca_mpc.yaml](../../configs/algos/orca_mpc.yaml). Native RVO2 and a bound
forward differential drive with exact map geometry are required. No safety,
full-suite, causal, publication or release claim; no default planner changes.

B reuses the ORCA/sampler composition from `hybrid_orca_sampler.py`, the existing
MPPI/CEM search, the CV predictor from `prediction_mpc.py`, native wheel odometry
in `drive_rollout.py`, and continuous geometry in `clearance_geometry.py`.
The two new MPPI hooks are identity operations for existing planners; default
MPPI commands/RNG and native ORCA commands/traces have byte-equivalence tests
against the frozen main source.

Every step seeds a 30-step, 0.1 s sequence search from ORCA, goal turning,
braking and the shifted previous solution. Two CEM iterations of 48 candidates
project sequences into live asymmetric acceleration/braking and speed limits.
The objective combines terminal progress, reciprocal minimum clearance and
command smoothness. Swept relative segments include time zero and their interior
minimum against CV pedestrians. Exact wall/polygon geometry is checked through
the horizon and a terminal static braking tail; curvature padding uses the live
acceleration/speed/turn bounds. The first feasible endpoint is applied and the
planner replans. No post-search escape override is permitted.

If no sampled sequence passes, reachable maximum braking is explicitly recorded
as `no_admissible_braking`; it is not a safety certificate or dependency fallback.
Finite sampling can miss feasible manoeuvres. CV extrapolation, constant body
radii, the finite horizon, chord interpolation assumptions and lack of pedestrian
prediction beyond the horizon prevent a general safety guarantee. Static braking
viability is checked beyond that horizon; dynamic safety is not claimed there.

The pre-registration and reconstructable private evidence packet are named
`orca_mpc-20261015` (custody locator in the task report). Population: the exact
11 scenarios and dev seeds 1001–1030 from A′ (#10404), 330 episodes per arm.
A = native ORCA, A′ = recorded #10414 with verified identical input/runtime
contracts, B = `orca_mpc`, H = frozen bottleneck-yield hybrid v4. These selected
confinement/collision cases are not representative of the full suite. Primary
contrasts use 20,000 paired bootstrap resamples of the 30 seed blocks, preserving
all scenarios/arms. Keep rule: B−A collision interval strictly below zero and
observed success loss no more than 5 pp. H is a separate leader comparison.
No parameter tuning after outcome inspection.

Validation and numerical results are recorded in the draft PR and preserved
packet. Independent domain review remains pending. No merge or self-approval.
