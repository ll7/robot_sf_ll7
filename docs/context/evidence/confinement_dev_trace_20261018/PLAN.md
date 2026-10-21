# Predeclared confinement development trace campaign

Evidence tier: diagnostic-only, descriptive development-seed analysis. No release rerun,
claim admission, manuscript, ledger, frozen artifact, planner, simulator or metric change.

## Goal and frozen inputs

Answer whether the four hybrids' small unconditional lead over ORCA is also observed among
episodes with at least one pedestrian completing a geometrically defined passage.
Use exactly the 15 cells and six configurations in `cells.json` and `policies.json` beside
this plan (the labels are Hybrid v4, Hybrid v4 continuous, Hybrid ORCA v2 yield/guard,
ORCA and SocNav sampling). Inputs come from release producer
`373dbfde4f39667cf9e8732dabe7df5118bdeab1` and the archived resolved 0.0.8 manifest,
verified against its SHA-256 config identities. Preserve differential-drive kinematics,
wrapped_v2 social force, all authored map/config bytes and per-cell horizons (400, 500,
600 or 700 steps), dt=0.1 s, record_forces=true and the release observation contract.

Exactly development seeds **1091–1120**, one complete 90-episode block per seed:
6 configurations × 15 cells × 30 seeds = 2,700 episodes. No release/evaluation seeds,
no legacy 111–140 schedule, and no held-out seeds. Set route_spawn_seed to the current
development seed. Enable `record_simulation_step_trace=true`. No outcome-dependent
extra episodes, seed replacement, horizon extension, parameter tuning or count-definition changes.
The source worktree stays clean at the producer commit; tooling executes outside it.

## Passage and stationarity definitions (frozen before any episode)

`cells.json` gives exact metric coordinates, physical faces, lateral apertures and map digests.
SVG coordinates equal simulator coordinates; validate geometry against reset map obstacles.
A passage is the axis-aligned rectangle between the two finite entry/exit plane segments.
Both directions count. An actor must be observed outside one face, enter through that face,
remain within the lateral aperture until leaving through the opposite face, with no reset,
respawn, missing identity or discontinuity in that segment. Starting inside is left-censored
and does not establish traversal. A new observation outside a face can start a later attempt.
Episode traversal is any actor completing any passage. Also report each double-bottleneck
neck and both-necks-by-the-same-actor separately. Cross-trap uses the four corridors around
the cross ends; hallway uses its two end faces. These are operational definitions, not
claims that every cell has a literal door. Report initial-inside/left-censored counts.

An actor is stationary near an opening when position-difference speed is strictly below
0.05 m/s for at least 3.0 consecutive simulated seconds, and its centre remains within
Euclidean distance 2 m of either finite face segment throughout. Reset the duration when
any condition fails. Episode stationary share means at least one actor; additionally report
stationary actor share among observed actors. Do not substitute robot stalled_time.

Retain stable simulator-slot IDs and reset positions. An observational wrapper records every
route respawn and excludes its entire affected transition; single-actor discontinuities are
also excluded at displacement >1 m per 0.1 s. Neither crossings nor stationary durations
may bridge these boundaries. Compare every recorded pedestrian velocity with successive
position differences (including reset to first frame), tolerance 1e-6 m/s; fail analysis
on unexplained mismatch. Count excluded transitions and maximum velocity error. Position
trace velocities are finite differences, not independent physical velocity observations.

## Estimands and uncertainty

Per cell: traversal episode count/share and stationary episode count/share pooled equally
across six arms (180 episodes), plus arm-specific values, actor denominators, duration and
censoring summaries. Per group and configuration: success numerator/denominator in traversal
and no-traversal strata, and unconditional success with percentage-point difference from
the release counts in the audit. Compare each hybrid's traversal success rate minus ORCA's
within each group. Report a positive observed difference as an observed lead; an interval
including zero is inconclusive about persistence. Zero/empty strata are unavailable, never 0%.

Bootstrap **10,000** draws, `random.Random(42)`, sampling 30 complete development seed blocks
with replacement. Use the SAME draws jointly across all cells/configurations/groups/strata.
Recompute stratum numerators and denominators inside each draw, and contrasts from that
replicate's two rates. This follows [the pinned ll7/diss seed-block implementation](https://github.com/ll7/diss/blob/98358209c41ae9fda313ccf63816662ae4da3ed1/scripts/dev/diss_s20_s30_rank_stability.py): sorted indices
`round(0.025*B)-1` and `round(0.975*B)` (zero-based). Report empty-replicate counts;
if any denominator vanishes in a replicate, flag the interval and use only defined draws
with the same rounded percentile convention on their count. These are percentile 95%
seed-block intervals, not independent Bernoulli intervals and not multiplicity adjusted.

Traversal is post-treatment and affected by robot behaviour and early termination. Preserve
observed durations, timeout/termination outcomes and horizon-hit counts; no traversal means
not observed during that episode, not impossible. These development seeds cannot establish
release equivalence, causal effects of pedestrian motion/wall forces, real-world validity,
or held-out performance. Yield/guard are parameter variants of one family, not independent
mechanistic replications. Unconditional differences from the release are descriptive.

## Execution, validation and custody

Before execution commit this plan and record its full commit/time in the lane receipt.
First add failing tests for analysis/tooling behavior and demonstrate failure on main;
then implement and pass synthetic passage, teleport, stationarity, velocity and paired-block
checks. Run the unchanged producer episode executor on CPUs, four processes initially,
one numerical thread each, no GPU. Cap at six workers after measured disk/memory/load checks;
leave host headroom. Check free disk before launch; stop new dispatch below 50 GB free.
No automatic episode retry: preserve errors and diagnose before any continuation.

Keep atomic gzip episode records (full traces), completion journal, SHA-256 inventory,
resolved inputs, command, plan/tool/source commits, dependency versions, CPU/thread settings,
map/config hashes and environment identity outside worktrees. Validate all 2,700 unique
identities and exact seed/cell/arm cross-product; reject fallback/degraded execution and
invalid records rather than silently dropping them. Run focused tests/lint/format and
recompute tables from preserved records. Full suites use the host suite-slot helper.

Perform exactly one W&B campaign run/upload after acquisition and analysis, using credentials
only in process. Retain artifact version/path and verify uploaded inventory. Do not log tokens.
Commit only plan, tooling/tests and compact summary/tables; raw traces remain in durable W&B
storage and the lane archive. Open a PR, push normally, read back its exact remote head,
record result and limitations in ~/cr_trace_report.md, and remove both owned worktrees after
preserving all inputs/scripts/logs. Never merge. Interrupted work resumes from validated
records only, preserving the original launch identity; missing custody blocks completion.
