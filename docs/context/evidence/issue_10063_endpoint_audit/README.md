<!-- AI-GENERATED (#10063, 2026-10-01) - NEEDS-REVIEW -->
# Endpoint audit for issue #10063

Diagnostic development evidence; author review and release admission remain separate.
`overlaps.json` lists all 48 release scenarios and three doorway widths: 102 full
robot spawn/goal rectangles, with 58 intersections before and 53 after. Five
unintended intersections are removed in three release-only successor maps.
Every retained intersection has an exact matrix/scenario/zone/actor fingerprint
and rationale in `configs/scenarios/release_0_0_8_endpoint_dispositions.yaml`.
Fingerprints bind density and any forced population, including dormant crowds.

The static audit uses the resolved 0.4 m pedestrian radius (substrate 0.35 m).
Its distance is rectangle-to-nominal-pedestrian-centres, including tangency and
round endcaps. A conservative 1e-9 m allowance includes decimal contact rounding.
Crowd routes include the production sampler's independent clipped
x/y spread of +/-1.5 m, a square Minkowski sum. Obstacle rejection, actual-pose
reset/respawn guards and dynamic contact are separate runtime constraints.
The 26 retained route-support intersections comprise seven dormant cases and
19 intended moving-flow cases; each rationale is reviewable individually.

Overtaking preserves h1 start x=1.5 and desired speed 0.8 m/s. Robot spawn y=4–4.5
and h1 lane y=6.6 give a 2.1 m gap; unchanged robot route y=5 gives a 1.6 m
passing gap. Robot cap 0.7 m/s and budget 600 steps restore an actual pass from
behind; merely shrinking the spawn allowed the faster robot to outrun h1.
Station reverse crowd spawn moves to y=16.5–19.5; its departing bend x=74 keeps
all nominal route-spawn support 1.5 m from the robot goal. The 4 m route detour
changes the unchanged-density population from 26 to 27, explicitly a distribution
change. Crowding uses x=6.5–14.5 and density 0.21, preserving 24 pedestrians.

Test-value: real YAML/SVG regression witnesses reject exact-base overtaking and
crowd destination geometry. A separate production-sampler witness fails on the
intermediate station bend x=75.5 and passes at x=74. Existing sampler tests
checked unfolding but did not compare the full support to pedestrian geometry.
The 30 focused static/importing-file tests cover radius-only intersections,
trajectory overrides, stationary actors, forced populations, stale fingerprints,
missing maps and fail-closed CLI. The CI runtime PPO repeat tests use a distinct
dev-1001 fixture, retaining the archived seed-111 identity without stepping it.
Hosted wheel smoke also uses 1001, and the reproducibility job resolves 1001–1002;
static seed-policy witnesses fail on base without stepping, and the repaired
local CLI passes. The two seed-123 example rollouts and the unresolved classic
campaign example are explicitly deselected in dedicated example CI; collection
proof verifies those exclusions. Saved command/collection receipts confirm that
retired-seed slow episode files were excluded from prior default-suite runs.

`experiments.json` contains both completed 270-cell Slurm cohorts, per-cell exact
event-ledger outcomes, raw hashes, source/config/input identities and physics byte
checks. It also inventories the 30 Q3 archive files read without re-stepping.
Q3's 11/30 early risk_dwa collisions are historical; fresh pinned base yields
3/30 because #10024 adds a stronger reset reaction buffer. The final cohort
covers all three changed scenarios, three planners and dev seeds 1001–1030 only.
`reproduction.json` includes exact producer/launcher bodies, resource limits,
source identities and checksummed raw archives. Recovery copies are on the local
host and imech192; this is not a durable publication or sealed evaluation.
Raw later collisions and timeouts in the dense crowd/platform interactions remain
visible in the outcome table, not masked by the endpoint geometry check.

Current release-template matrix pin is refreshed. The retained historical PEDFIX
pin in `docs/validation/pedfix_0_0_8/manifest.json` describes its measured checkout;
`git show bc85705d4f89c5f91a499c77362e30a8f68bdd8f:configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml`
reproduces `8b16cb1ba6155567f31855ec453d596e17e4ed859caf60b6e1092f49b4766ac8`.
Replacing that past measurement hash would falsify provenance. Regenerate live
release identities after integrating the approved combination base. Historical
SVGs, frozen candidates and 0.0.2/0.0.7 artifacts remain intact.

Final full-suite, exact-SHA routing/coverage and hosted-CI receipts are reported
in PR #10067 and `/home/luttkule/ovtfix_report.md`; this bundle does not admit
scientific, safety, performance or release claims.
