<!-- AI-GENERATED (#10101) - NEEDS-REVIEW -->
# Opt-in pedestrian contact and wall response — Round 2

AI-GENERATED / NEEDS-REVIEW

Issue #10101; draft PR #10104 stacked on CALFIT #10094. Round 1's V6 pruning,
zero-survivor conclusion, displacement-derived speeds, radius-.40 robot gate,
startup timing claim and invalid-spawn explanation of the corner abort are withdrawn.
The corner abort was a solver defect. The final robot gate explicitly uses the
[milestone's .28m radius](https://github.com/ll7/robot_sf_ll7/milestone/11).

## Usable for 0.1.0?

The opt-in path fixes the measured overlap and wall traversal defects, with a bounded
runtime on the declared warmed congested probe. It remains development physics,
with numerical calibration misses and paired robot regressions to assess before adoption.
Neither these dev results nor an engineering range PASS establish empirical validation
or release admission. Default behavior and protected artifact bytes stay unchanged.

Select `pedestrian_contact_rule: projection_v1` and
`pedestrian_wall_rule: bounded_edge_v1`. An explicit pedestrian_radius_m is honoured;
otherwise selectors preserve the backend radius. [Corrected estimator, solver and
test-value protocol](pedcontact_10101_round2_protocol.md) documents all assumptions.

## Diagnosis retained from the CALFIT base

Dev1001 V2 terminal drive +2.934283m/s² balances wall force −2.934283; pair force
is zero and goal x=15m. This is the shifted-distance wall barrier, not routing.
V4 width2.4m has402,824 overlap pair-steps:153,991 same-heading,115,359 head-on,
133,474 crossing/stationary. Depth median/95th/max .079144/.313356/.558394m;
local neighbour density median/95th2.864789/4.774648 persons/m² in a1m disc.
[Original diagnostic evidence](pedcontact_10101_diagnosis.json) remains historical.

## Contact law and numerical checks

[HFV2000](https://arxiv.org/abs/cond-mat/0009448), mass80kg, k120000N/m and
sliding coefficient240000kg/(m s), has dt√(2k/m)=5.477 at dt=.1s, above the
semi-implicit limit2; dt<.036515s for the normal mode. At .16m compression the
explicit tangential bound is .002083s. The real .1s probe reaches25.626m/s;
substepping removes the overshoot but retains compression. Projection meets the
strict zero-overlap objective. Both candidates leave uncongested walking unchanged;
projection avoids the stiffness/substep cost and is selected on the exclusion evidence.

Grid broad phase, AABB wall culling and alternating Gauss-Seidel over-relaxation1.6
replace all-pair scans. All capsule constraints apply, including shared-vertex ties,
with initial push-out. The256-pass cap records deterministic fallback and continues;
an unresolved result remains a counted violation. Closing normal velocity is removed
and the integration speed cap reapplied, without injecting displacement/dt velocity.
The nearest surface's bounded shifted exponential uses A3m/s², decay.04m, range.20m;
normals blend over .30m. Actual-force gap witnesses damp over the bounded parameter
grid. The parallel-gap gradient bound345/s² is below360/s² at dt=.1s,tau=.5s.
This scoped force check is not a universal geometry/solve-time bound. Swept projection
provides independent body/wall exclusion. Live manifests record the effective laws,
parameters, radius, integration, velocity treatment and fallback policy (#10084).

## Validation and reference limitations

[All 30-seed off/on values, intervals and statuses](pedcontact_10101_step4.md)
include constituent widths and integration cost. [Complete verified summary](pedcontact_10101_step4.json)
keeps physical validity separate from the accepted numerical bands.
V2 traversal is fixed; its remaining speed-drop miss is movable by wall parameters
and limited by absent human anticipatory/shoulder response. No evidence establishes
that its accepted band is wrong. [Wilmut2015 crossing-phase results](https://doi.org/10.1371/journal.pone.0124695)
place the1.3–2.1 shoulder ratios in the same post-hoc reduction group, supporting
the suite's plateau anchor. The original .9 shoulder-ratio case is impossible
for rigid discs and remains explicitly excluded by the suite's declared policy.
V5 is abeam CM-to-cylinder-edge clearance. The final baseline .867806m is above [.4,.6].
[Gerin-Lajoie2008](https://doi.org/10.1016/j.gaitpost.2007.03.015) reports personal-space
geometry citing2005; this scalar proxy and its engineering band are not a published
confidence interval. Wall strength/decay/range can move it; protocol equivalence needs
author review, without silently relaxing the band.
V6 now uses .05rad/s sustained .3s, rejects numerical tails and observes25m with60m
initial separation. [Huber2014](https://doi.org/10.1371/journal.pone.0089589) supplies
filtering, angular velocity and own-X-to-PoMD definitions, but not that numerical
threshold. Human sway baselines differ from deterministic CM baselines. A remaining
range miss is chiefly a suite/reference-equivalence limitation for this criterion,
not evidence that all force-model classes are unfit. Threshold sensitivity remains
explicit; no target is changed and no single validation item prunes fit classes.
[Saved-path threshold sensitivity](pedcontact_10101_v6_sensitivity.json) is large:
.05rad/s gives onsets3.159/3.600/4.264m (all FAIL); .1rad/s gives2.515/2.803/2.931m
(all PASS) for slow/normal/fast. The fit retains the primary .05 criterion frozen
before results. Seed intervals do not cover this protocol uncertainty.

## Complete bounded fit and robot gate

[Frozen grid](pedcontact_10101_fit_grid.json) evaluates every108setting ×3devseeds,
all18cases per bank, including V2/V5 at every wall setting. Rank by passed checks,
then summed range residual, then candidate id; the residual mixes units and is only
a diagnostic tie-break. [Full ranking and each item's value/interval](pedcontact_10101_fit_result.json)
does not claim a global optimum or empirical calibration.
[Paired robot gate](pedcontact_10101_robot_gate.json) includes all2850pairs, .28m radius,
ORCA and four standard hybrids, dev1001–1010,48empty scenarios,3doorway widths,
and6probes. Improvements are reported alongside every classified new failure. The prior
radius-.40 gate had ORCA probe collision totals6 off and2 on, omitted from the
original narrative; that historical improvement is now recorded and distinguished
from the fresh shipped-radius totals. That old on bank has59 completed rows
and one solver abort, so6→2 is a raw count reduction, not a clean complete
paired success-rate comparison.
[Measured dev1005 reset receipts](pedcontact_10101_robot_starts.json) replace missing
spawn validity; the project's clearance check runs before each episode's step.
[Acquisition and comparison sources](pedcontact_10101_drivers.json) preserve replay code.
[Verified raw custody and member hashes](pedcontact_10101_custody.json) locate retained evidence.

Final producer ffdd8618b01c9bcba3272771aa97c4b8d8724a1d. Default numerical score off/on: 2/1 of15. Best bounded point c1547cf0fec2: 5/15; best shipped-radius point ac3ca720b117: 5/15. Full accepted settings: 0.

Best overall measured whole-bank integration cost 38.377ms/step; N350 width5.0m cost 55.592ms/step. [Every fitted point's measured cost and repair counters](pedcontact_10101_fit_runtime.json) prevents extrapolating the default snapshot cost across the fit. These three-seed timings are not a paired off/on benchmark of the fitted parameters.

Best shipped radius measured whole-bank integration cost 15.924ms/step; N350 width5.0m cost 44.784ms/step. [Every fitted point's measured cost and repair counters](pedcontact_10101_fit_runtime.json) prevents extrapolating the default snapshot cost across the fit. These three-seed timings are not a paired off/on benchmark of the fitted parameters.

Warmed integration mean ms/step, after one warm-up and20timedsteps:

| N | Off | Reviewed ead on | Final on | Final/off |
|---|---|---|---|---|
| 60 | 0.853 | 2.239 | 1.503 | 1.761 |
| 150 | 4.349 | 18.883 | 6.227 | 1.432 |
| 350 | 21.891 | 273.526 | 30.707 | 1.403 |

The complete 30-seed V4 width5.0m banks (N350) cost 23.910ms off and 44.175ms on per integration step, including startup and later congestion. Default on records 3461 fallback repairs and 0 unresolved results.

The shipped-radius gate records ORCA collisions3 off and4 on: three prior
collisions become successes, while crossing1001/1004/1008 add pedestrian collisions
and doorway1001 adds a wall collision. All four hybrid probe arms have60/60
successes in both modes. There are no runtime aborts or empty-world metric changes.

The final N350 cost meets3× on this fixed CALFIT snapshot; crowded/fallback cost
elsewhere is not bounded by that measurement. All nine uncongested dev controls
retain exact state bytes and free-speed/fundamental-diagram slope.
