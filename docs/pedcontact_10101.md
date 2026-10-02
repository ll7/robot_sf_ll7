# Opt-in pedestrian contact and wall diagnostics

AI-GENERATED / NEEDS-REVIEW

Issue #10101; stacked on CALFIT #10094. This is diagnostic opt-in physics,
not release or empirical-model admission. The full 30-dev-seed comparison is complete;
[all values, intervals and gates](pedcontact_10101_step4.md) are versioned alongside
[the verified acquisition summary](pedcontact_10101_step4.json).

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

Nine original witnesses fail on base: overlap/dt assertions, wall crossing and
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

## Full-dev results and remaining target misses

At CALFIT defaults (radius .28m, cap 2m/s), dev1001–1030 off/on banks contain
1080 protocol cases. Post-step overlap pair-steps are 91,665,539/0; wall
penetration pedestrian-steps are 0/0. V2 traversal is 0/90 versus 90/90.
The maximum projection iteration count is 988, below the explicit 4096 limit.
Accepted numerical checks are 3/15 off and 4/15 on. MISSING is a failed gate,
not an imputed observation: V3 remains censored at most narrow widths.
The JSON and Markdown retain every value, observed n and Student 95% interval;
intervals are conditional on observed dev trials and are undefined for n=1.

V2 is physically fixed, but slowdown .0230/.000254/0 remains below .06–.24.
Finite-range wall strength/range can move slowdown without restoring the legacy
barrier; human anticipatory narrowing responses are also a model limitation.
The source-derived engineering band is not demonstrated to be erroneous.
V5 clearance changes .867806 to .386567m (95% interval .385275–.387858),
just below .4–.6. Wall strength/decay can move clearance. Gerin-Lajoie et al.
2008 describe a roughly .5m lateral and 2m longitudinal personal-space ellipse;
the suite cylinder-edge scalar is a proxy, not that complete human measurement.
Its ±20% tolerance is an engineering decision, not a paper confidence interval.
See https://doi.org/10.1016/j.gaitpost.2007.05.008 .

V6 slow/normal onset is 2.938506/2.928810m, unchanged by contact; fast changes
2.930136 to 2.904414m and passes. V6 is wall-free, so wall tuning cannot move
it. Huber et al. 2014 Table 1 gives 180° slow/normal/fast turn-onset means
2.1/2.4/2.7m. The onset rows give no standard deviations. The paper's threshold
uses five no-interferer trials per person and speed; deterministic straight-line
model baselines contain no sway and have zero maximum angular speed. This is
an estimator/reference comparability and model limitation, not evidence to
silently widen the bands or add noise. Radius and speed cap are screened as
possible fitting parameters. See https://doi.org/10.1371/journal.pone.0089589 .

Weighted timed integration costs 10.1419/218.4304 ms per step (21.54×).
Timers include cold JIT startup but exclude diagnostic counting/copying and V6's
five free-baseline integrations; host variation prevents treating this ratio as
a warm microbenchmark. All 1080 saved speed arrays are finite. Projection can
raise displacement-derived velocity above the pre-projection cap: on max 2.772594
m/s, 2075 samples >2+1e-9. This is an explicit limitation of hard correction.
Uncongested dev controls are byte-identical, including measured free speed and
flux-density slope; dense fundamental-diagram behavior can change materially.

## Compatibility and validation

Release/ecosystem identity checks: 94 passed. All 1513 tracked config/schema/golden
files match the CALFIT base byte for byte. New selectors are absent from default
serialization, and default simulator and benchmark metadata retain their identity.
The full slow suite has 40,821 passes, 67 skips, 7 existing xfails, 16 failures
and 3 setup errors. All 19 residual nodes reproduce on CALFIT base. Targeted
coverage also exercises actual public construction and effective law/radius
forwarding, rather than only a manually assembled wrapper.

The optimized-mode algorithm_metadata pin failure is inherited: source SHA256
bea0ba751ed1092d12b23d7cc23dcdeac5b1767c4e30cdafbc20378114f521d7 matches
base and current main; the old contract still pins 8b4bb11d… from before #9996's
GuardedPPO identity correction. Current main repairs that pin. This stack leaves
protected contract bytes unchanged. Strict Sphinx also fails on base's unchanged
131730-byte exclusions argument exceeding Linux's per-argument limit; current
main has a file-transport repair. Neither inherited failure is called green.

Acquisition summaries for diagnosis, HFV comparison, uncongested controls and
source-table verification are in [the diagnostic record](pedcontact_10101_diagnosis.json).
That record also retains seed-refusal attribution and fail-on-base assertions.

The robot gate roster is 48 main scenarios, three doorway widths, six nominal
probes, ORCA and all four release hybrid arms, dev1001–1010 off/on: 2850 pairs.
The release width campaign “90-cell” label means three widths × 30 seeds,
not ninety distinct geometries. Fit admission requires the full dev roster.
