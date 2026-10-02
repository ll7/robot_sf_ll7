# Opt-in pedestrian contact and wall diagnostics

AI-GENERATED / NEEDS-REVIEW

Issue #10101; stacked on CALFIT #10094. This is diagnostic opt-in physics,
not release or empirical-model admission. The full-dev comparison is pending.

Select `pedestrian_contact_rule: projection_v1` and
`pedestrian_wall_rule: bounded_edge_v1` in simulation settings. Missing selectors
do not enter dataclass serialization or environment/configuration hashes.
The contact radius follows the effective physical pedestrian radius.

## Diagnosis at the CALFIT base

Dev1001, calibrated profile factor .003, offset .375m, radius .28m, cap2m/s,
positive desired N(1.29,.19). Three V2 walkers stop upstream of x=8m at
x=7.486782/7.561681/7.690361. The terminal forward drive2.934283m/s² is
balanced by wall force−2.934283; pedestrian force is zero and goal x=15m.
This confirms an excessive shifted-distance wall barrier, not a route or
pedestrian-pedestrian issue.

V4 width2.4m has402,824 overlap pair-steps:153,991 same-heading,
115,359 head-on and133,474 crossing/stationary. Median/95th/max overlap depth
.079144/.313356/.558394m; local neighbour density median2.864789,
95th4.774648 persons/m² in a1m disc. Heading classes use actual velocities;
stationary pairs are kept separate from evidence of crossing.

V5 measures lateral centre-of-mass to cylinder-edge clearance, not body-edge
clearance or a longitudinal ellipse. The legacy32-segment summed wall law gives
.858480m at dev1001 versus accepted[.4,.6]. V6 measures filtered own-X turning
onset relative to the closest-approach point and deterministic no-interferer
baselines. Dev1001 gives2.939276/2.929539/2.930805m for controlled speeds
1.15/1.42/1.78m/s. The first two fail their bands. V6 contains no walls;
contact acts much later than turning onset. No tolerance or estimator is changed.

## Law choice and numerical limits

Compare [HFV2000](https://arxiv.org/pdf/cond-mat/0009448), mass80kg,
k120000N/m and sliding coefficient240000kg/(m s), with hard projection.
For a two-body normal mode, dt√(2k/m)=5.477 at dt=.1s, exceeding the
semi-implicit stability bound2; dt must be below.036515s. At compression.16m,
the explicit tangential damping bound is.002083s. An uncapped real integration
probe at.1s reaches25.626m/s. Substepping resolves that numerical overshoot but
retains finite compression. The frictionless projection satisfies the strict
zero-overlap objective and leaves separated trajectories exactly unchanged.

Post-step equal-mass projection alternates swept pair and swept wall-capsule
constraints. A1µm numerical margin protects strict distance comparisons.
The4096-pass limit is explicit; nonconvergence restores the pre-step state and
raises an error. This is not a silently accepted approximate contact solve.
CALFIT's nonreactive V6 walker is prescribed; only the reactive subject yields.

The wall law uses the nearest finite surface, with tied normals averaged,
instead of multiplying force by tessellation count. Its body-edge exponential
has amplitude3m/s², decay.04m, range.20m, and reaches zero continuously at the
range boundary. Swept capsule constraints provide nonpenetration independently
of force stiffness and prevent tunnelling across thin walls. Both simulator
wrappers and the robot benchmark's separate pedestrian integration apply it.

## Test value

Protected behavior: actual body separation, separated-byte identity, dt stability,
swept wall safety, feasible-gap passage, prescribed-interferer preservation,
tessellation-independent force, live manifest consistency and benchmark wiring.
Credible regressions: removing post-step handling, calling only one wrapper,
allowing endpoint tunnelling, summing every wall segment, moving the prescribed
walker, or declaring parameters that the live force did not use.
Nearest existing wall-profile tests sample upstream force and configuration
identity; they do not exercise body exclusion or the benchmark's manual step.
Tests use actual Simulator/PedState arrays and existing force-factory/manual-step
interfaces. There is no production test-only seam; the wrapper fixture supplies
only its pedestrian-velocity view.

Six original witnesses fail on base: overlap/dt assertions, wall crossing and
`physically feasible aperture stalled upstream` at x7.578752. The separated
negative control passes on base, as expected. New manifest/configuration controls
are feature coverage, not claimed pre-existing bug witnesses.

## Acquisition

`scripts.validation.pedcontact_10101 run --arm off|on --seed 1001 --out DIR`
uses CALFIT's Slurm-only acquisition/priority guard, source/installed byte checks,
raw NPZ members and SHA256 manifest. Run each seed1001–1030 for both arms;
`collect --root ROOT --out comparison.json` verifies every seed bank before
constructing full-grid gates, Student95% intervals, post-step overlap and
wall-penetration counts, and real integration time per step. Censoring stays null.
Effective contact/wall/integration values come from live simulator/force objects,
are included in acquisition records and opt-in benchmark episode metadata, and
are checked by an independent declared-versus-live mismatch test.
