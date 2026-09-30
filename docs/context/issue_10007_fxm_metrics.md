# Issue #10007 FXM shared metric correction

Current semantics include the FXM2 refinements and D-055 curvature documented below. Original FXM tables
remain historical diagnostic receipts; they do not serve as current-v2 anchors.

Evidence: **diagnostic-only development episodes**, with no release, held-out,
calibration campaign, ranking or dissertation admission. Base:
`073f492abced5dcff749ddfed01c44dc9fd864b7`. All F4/F5/F7/F8 leads were confirmed
at that HEAD. The clone was branched from `origin/main`; no planner method changed.
Before handoff, the branch was rebased onto refreshed main
`7325fe7985c07a342ed45081b290a37a326f0a9b`. Its added planner-target trace field
is observational and remains independent of the frozen metric reference.

The comparison uses goal, native RVO2-backed ORCA, and
`hybrid_rule_v4_fast_progress_static_escape_s30_h600_release.yaml`.
Three versioned release scenarios run on dev seed **1001**: doorway low, merging
low, frontal approach. A pedestrian-free `planner_sanity_open.svg` diagnostic
runs on **1002** to measure successful time for every planner. Horizon 600, dt 0.1 s;
force recording and simulation trace enabled. Each before/after pair has identical
post-step positions, step count and outcome. All 12 corrected records validate
against the episode schema. Full compact numbers are in
[evidence JSON](evidence/issue_10007_fxm/paired_metrics.json); reset/positions from
one real ORCA episode are checked into `tests/fixtures/benchmark/fxm_merging_orca_dev1001.json`.

## F4: confirmed; final route goal and reset reference

**Original FXM evidence below; the FXM2 refinement supersedes its point-reference and
clipped-efficiency rules.**

Root cause: reset used `simulator.goal_pos[0]` (the handoff waypoint), termination
overwrote it with the active waypoint. Freeze a copy of `robot_navs[0].waypoints[-1]`
at reset; use it for shortest path, path efficiency, ideal-time ratio, failure
progress/distance, static-deadlock evidence and trace/paired-effect denominators.
The shortest path begins at reset and now follows the frozen completion policy:
continuous shortest path to the goal polygon for zone entry; the existing point
reference for waypoint-radius completion. Failed efficiency is NaN/JSON null;
successful efficiency is unclipped, with a reference-violation flag above one.

Merging low, seed 1001 (before → corrected; geometric fields include F8):

| Planner | Shortest path m | Path efficiency | Final distance to metric target m | SNQI-v2 source ideal-time ratio |
|---|---|---|---|---|
| goal | 42.392491 → 57.137143 | 0.949740 → 1.000000 | 6.036148 → 19.799096 | undefined (failure) → undefined (failure) |
| hybrid v4 | 57.137143 → 57.137143 | 0.840130 → 0.840068 | 2.002507 → 2.190519 | 1.781678 → 1.785179 |
| orca | 57.137143 → 57.137143 | 0.697670 → 0.697628 | 1.911517 → 1.911517 | 1.641664 → 1.645165 |

Final distance is independently reconstructed from terminal positions and captured
metric goals; it is not an added scalar output. Undefined goal time on the
colliding goal episode is retained, not zero-imputed.

Trace/reference checks on the same episodes:

| Planner | Initial goal distance m | Final radial progress m | Failure-to-progress windows |
|---|---|---|---|
| goal | 5.213333 → 39.339566 | -12.091767 → 19.540471 | 61.000000 → 114.000000 |
| hybrid v4 | 5.213333 → 39.339566 | -28.154409 → 37.149048 | 147.000000 → 146.000000 |
| orca | 5.213333 → 39.339566 | -28.729661 → 37.428049 | 132.000000 → 132.000000 |

Trace progress is initial Euclidean distance minus current distance to the frozen
terminal goal; it is **radial progress**, not accumulated arclength along the route.
Planner observations continue using the navigator waypoint. This fixes negative
handoff-reference progress after passing the first target.

Tests: `test_final_route_goal_is_captured_at_reset` fails on base with
`assert [2.0, 0.0] == [12.0, 3.0]`; the captured real-coordinate regression also
checks the final route reference. The full step-loop regression also fails on base
with `assert [4.0, 0.0] == [12.0, 3.0]` after a terminal waypoint handoff and
mutated navigator target. No simulator/planner test seam was introduced.

## F5: confirmed; window reduction and collision exclusion

Root cause: all per-step reductions were tested against 0.05 m, effectively a 0.5 m/s
radial-speed cutoff at dt 0.1 s; episode collision status was ignored.
Exact v2 rule: for **15 consecutive samples /14 intervals**, compare first minus
last remaining arclength on the frozen reset route (including reset-to-first-waypoint). A window stalls when reduction **<=0.05 m**;
count overlapping windows only when their last sample is strictly before the
terminal sample. Deadlock requires a stalled internal window and **neither
success nor collision**. Window diagnostics remain available on success/collision
rows. `max_no_progress_run` now counts consecutive stalled window starts.
`deadlock-stall.v2` records these semantics. Reuse the precomputed collision summary;
simulator collision flags override sampled collision detection.

| Planner | Deadlock | Stall windows (F4+F5) |
|---|---|---|
| goal | true → false | 187 → 113 |
| hybrid v4 | false → false | 178 → 151 |
| orca | false → false | 129 → 139 |

These counts include the F4 reference change. ORCA/hybrid complete the route and
keep a false headline flag; this seed does not show a flag change for those arms.
The goal collision formerly counted as deadlock and now does not.
The independent slow-approach regression uses 0.4 m/s: over 15 samples it gains 0.56 m,
so it must not stall. Base fails `assert True is False`. A separate stationary
collision regression fails the same assertion on base. FXM2 now uses closest-polyline projection for this screen; a leg heading away from
the final goal still reduces remaining route length. Historical counts above predate that refinement.

## F7: confirmed; jerk in m/s^3

Root cause: `jerk_mean` averaged acceleration differences without dividing by dt.
Divide those differences by dt, retaining the first T-2 differences and denominator.
The other physical jerk statistics already divided by dt.

| Planner | Mean jerk (old proxy → m/s^3) |
|---|---|
| goal | 0.036067 → 0.360675 |
| hybrid v4 | 0.194903 → 1.949026 |
| orca | 0.136016 → 1.360163 |

Every dt 0.1 s episode increases by exactly 10×. The independent regression has
acceleration increments 2 m/s^2 and asserts 20 m/s^3 at dt 0.1,10 m/s^3 at dt 0.2.
Base reports 2.0 in both cases. Fixed-dt jerk ordering alone would be preserved
with equally rescaled anchors; the combined time/reference corrections can
change SNQI and ranking. Invalid dt yields NaN, with no fabricated physical jerk.

## F8: confirmed; completed-step time and reset-to-first travel

Root cause: completion was a zero-based action index but was used as elapsed step
count. Positions began after action 0, omitting the first geometric segment.
For post-step samples, time=(index+1)*dt and normalized time=(index+1)/H.
The synthetic runner already records reset as sample 0 and completion at sample
index n; its explicit `robot_pos_includes_reset=True` preserves time=n*dt and
existing reset collision evidence. A support regression passes on base and fix.
The older classic producer no longer overwrites elapsed time with index*dt; it
also captures the reset/final route goal after reset and passes reset geometry
to common metrics. Its elapsed-time regression fails on base with `0.1 == 0.2`.
The three-planner successful probes below supply the same completed-step inputs
as this producer: the old overwrite would emit 13.3/7.6/7.6 seconds instead of
13.4/7.7/7.7. No extra simulator run or seed is needed for that arithmetic replay.
Geometric paths prepend the captured reset pose; force, safety and pedestrian
arrays keep their post-step alignment. Cooperative completion time follows the
same index conversion (no episode producer currently supplies the multi-agent map).

Successful open-map diagnostic, seed 1002:

| Planner | Time to goal s | Normalized time | Travelled path m |
|---|---|---|---|
| goal | 13.300000 → 13.400000 | 0.221667 → 0.223333 | 12.882034 → 12.887034 |
| hybrid v4 | 7.600000 → 7.700000 | 0.126667 → 0.128333 | 12.900000 → 12.905000 |
| orca | 7.600000 → 7.700000 | 0.126667 → 0.128333 | 12.895000 → 12.900000 |

Time is independently reconstructed from the captured goal-reaching action index;
metric functions and emitted normalized/ideal-time fields verify the conversion.
All 12 probes restore the omitted first 0.005 m segment. Merging ORCA path length is
81.89706857316321→81.90206857316322 m, verified against an independent `math.dist`
sum of captured reset and simulator coordinates. Base regression failures include
`0.0 !=0.1` for first-action goal time and `3.0 !=8.0` for a 5 m initial segment
followed by 3 m. The real-byte regression fails at 81.89706857316321 vs 81.90206857316322.

## Version migration and consumer inventory

New metric computations and map rows carry **robot-sf-metrics.v2**. Unmarked
historical rows/assets, especially published 0.0.7, are **robot-sf-metrics.v1**.
The episode envelope remains v1. **Do not relabel 0.0.7 rows or old anchors**:
recompute both releases from complete reset/terminal-goal/trajectory bytes for a
shared-definition contrast, or classify changed metrics as non-comparable.

Direct checkout searches followed the retrieval broker's indeterminate result.
Executable consumers were inspected, including:

- `scripts/analysis/compare_release_0_0_7_to_0_0_8.py`: preserves schema identity while
  compacting rows; reports `metric_definition_change` even for equal values,
  suppresses paired deltas and tags summaries `incompatible_definitions`.
  Old/new means remain diagnostic only; analyst rules cannot override the fence.
- `scripts/analysis/_pinned_successor_runtime.py`: resolves campaign/runtime identity,
  not numeric contrasts; metric identity is preserved by the comparator's row loader.
- `scripts/analysis/compare_issue_9431_release.py`: compares only unchanged completion,
  collision and timeout outcomes. Its scope is explicitly documented.
- Hierarchical release adapter/analysis: preserve metric schema through typed ledgers;
  reject mixed path-distance exposure meanings before building paired cells.
- `aggregate.py`: rejects mixed versions, protecting aggregate/CI/paired-contrast,
  canonical tables, figures, campaign reports and benchmark exports using this owner.
- `snqi/compute.py` and `metrics.snqi`: legacy/v0/v1/v2 scoring rejects mismatched
  normalization/anchor schema. This covers optimization, recomputation, sensitivity,
  ablation, figure score injection and publication/consistency recomputation that
  pass episode metric mappings to these owners.
- `snqi/campaign_contract.py`, `snqi/calibration.py`, `snqi/cli.py`: derived baselines
  retain version metadata; fixed-anchor calibration comparisons and planner ordering
  reject mismatch. Sanitization preserves the marker. Scalarization sensitivity
  preserves baseline metadata and rejects mixed definitions/old anchors.
- `snqi/v2_calibration.py`, `v2_spec.py`, `v2_reports.py`: anchor derivation rejects
  mixed definitions and records the source schema; specs preserve anchor identity;
  offline/campaign scoring and compact report projections retain and check it.
  Existing frozen assets remain v1 and **cannot score corrected v2 episodes**.
- `scripts/tools/rebuild_campaign_reports_from_rows.py`, release SNQI field/collision
  checkers, `artifact_publication.py`, weight optimization/recompute/sensitivity
  scripts: their metric dictionaries preserve the marker and their canonical scoring
  calls enforce the compatibility fence; their historical assets were not rewritten.

**0.0.8 execution with the existing camera-ready v3 or SNQI-v2 anchors is now blocked
at scoring until fresh matching-definition normalization is derived and reviewed.**
No recalibration campaign or weight choice was performed in FXM. Existing 101/102
calibration split policy remains unchanged; no such simulation was run here.

Dissertation-facing fields changed: shortest/ideal path reference, path efficiency,
travelled path length/PL, SPL, SocNavBench path length ratio and irregularity,
failure-to-progress and final-goal distance diagnostics; deadlock flags/rate and the
constraints-first screen/endpoint; jerk mean and normalized J; successful elapsed,
normalized and ideal-ratio goal time (SNQI-v2 T), cooperative completion-time
diagnostic; trace radial progress/initial-distance and timeout paired-effect
normalization; derived legacy/v0/v1/v2 SNQI, Social Mini-Game diagnostic composites and
sensitivity/ranking inputs.
Success/collision/timeout outcomes were identical in these probes. No dissertation
source or admitted claim was edited.

## Regression value and validation

- F4 tests protect route target identity/copy and shortest-start geometry; reverting
  to active waypoints is a credible regression. Nearby map-runner characterization
  uses a single-target stub and cannot distinguish final vs handoff waypoints.
  The new full-loop test exercises the actual mutable terminal handoff path.
- F5 tests protect total-window progress and collision exclusion; nearest prior
  deadlock tests cover a stationary/fast endpoint, missing slow steady approach and
  collision episodes. A terminal-window support test passes on base too and is not
  claimed as defect evidence.
- F7 tests protect units across dt; prior jerk tests assert the old per-step number
  or use zero jerk, so they would preserve/miss the defect. Expectations now use
  independent acceleration increments and elapsed time.
- F8 tests protect a first-action success and first geometric segment; prior path
  fixtures start at a position sample without a separately recorded reset. Real
  ORCA coordinates prove wiring with an independent `math.dist` oracle.
- Migration tests use actual JSONL/archive/fixture bytes at consumer entrypoints;
  old code accepts mixed definitions/old anchors or emits a paired delta. Tests
  assert the specific compatibility reason, not a missing unrelated prerequisite.
  SNQI-v2 smoke anchors are synthetic test-only matching-schema values, not relabelled
  publication assets. No test-only production seam, skip, xfail or timeout was added.

All pytest runs used **-n 0**, OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1.
Focused metric/SNQI/helper/calibration suite: 658 passed, 1 existing selector-adapter skip.
Final compatibility/native-command/hierarchical/scalarization suite: 152 passed.
Updated metric/SNQI/helper/native/classic-caller suite: 674 passed, 1 existing skip.
Release comparator equal-value and mixed-successor-schema cases: 2 passed.
After main integration, FXM/helper/native/classic/target-trace suite: 115 passed,
1 existing selector-adapter skip; all 12 real probes rerun with identical corrected
metrics and trajectories, and their recorded source hashes match final code bytes.
These simulation receipts record the pre-amend source commit plus checkout dirty
state and byte hashes; the final amendment changes documentation/evidence only.
Final goal-wiring/helper suite: 66 passed, 1 existing selector-adapter skip.
Caller compatibility/FXM/helper/classic metadata suite: 86 passed, 1 existing skip.
Complete release-comparator/golden/hierarchical run:89 passed, 1 test-fixture filename
failure; that fixture was corrected and its regression passed separately. The
comparator's new equal-value definition-change regression passes at the real entrypoint.
Base runs:15 new checks fail for intended defects, 1 support check passes; separate
SNQI-v2 compatibility, comparator, terminal-handoff and classic-time regressions
also fail on base. Failure snippets
are preserved in `test_failures_base.txt`. Ruff, format, diff and episode schema checks
complete the local proof. Full repository readiness was not run because its broad
simulation lanes contain held-out seed tests and violate this task's seed restriction.
This is a draft for review, not merge/readiness or release admission.
The PR-contract parser also explicitly blocks admission while evidence-bearing
`domain_approval.status` is pending; no approval was fabricated to clear it.

Protocol deviation: an existing campaign smoke test initially ran four steps on
seed 201. It was changed to 1001 and rerun successfully with matching synthetic anchors.
That seed 201 run is excluded from evidence; no seed 111–140 was run. Diagnostic probes
used only 1001/1002 and ran one simulation at a time (at most two processes including
an overlapping four-step dev smoke test).

Reproduce from each checkout (copy this probe script to the base checkout first):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/fast-pysf \
  .venv/bin/python scripts/validation/probe_issue_10007_metrics.py --output output/fxm/probe
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD:$PWD/fast-pysf \
  .venv/bin/python -m pytest -n 0 tests/benchmark/test_fxm_metric_definitions.py -q
```

Full local rows/logs remain in `output/fxm`; compact values, raw fixture, source
identity and regression failure snippets are committed so review does not depend
on ephemeral simulation output. No historical 0.0.2/0.0.7 artifacts or frozen YAML
were modified. No held-out evaluation, release freeze, push to main or merge occurred.


## FXM2 refinement (PR #10014 review follow-up)

Comparator: reviewed head `977378cbdb0d587a65b11cd28bba4c4bab6c2aba` versus the FXM2
implementation. The metric version stays v2: the correction is completed before
merging this one-shot definition, with F7 jerk and F8 time/reset geometry retained.

1. Deadlock/stall uses remaining arclength after projecting positions onto the route
   frozen at reset, including the initial approach to its first waypoint. Closest
   segment wins; earliest breaks ties; retreat stays measurable. The 15-sample/14-interval,
   terminal exclusion and success/collision exclusion rules remain unchanged.
2. Efficiency is defined only for successful episodes. Failures return NaN, serialized
   as explicit JSON null. Success values remain unclipped; values >1 set
   `path_efficiency_reference_violation` for investigation. Finite-value means therefore
   include successful runs only. A reached sample with collision is still a failure.
3. Zone-entry completion uses the exact continuous polygonal robot-centre shortest
   path from reset to the goal set, avoiding obstacle interiors. A visibility graph
   considers obstacle vertices, goal-boundary vertices/intersections and perpendicular
   projections onto goal edges; Dijkstra takes the minimum. No boundary sampling,
   grid rounding or extra footprint inflation is used. This lower reference is no
   longer than a collision-free successful trajectory by construction. Cache keys
   include scenario, seed, reset, polygon, full compound obstacle geometry and bounds.
   Waypoint-radius completion retains the existing Theta* point reference. The same
   scalar feeds efficiency and SNQI-v2 ideal time in map and classic producers.
4. Exclusive-v2 diagnostics/fields without a metric marker are refused, including
   falsely v1-marked payloads. Historical indistinguishable scalar-only mappings remain
   v1; units are never guessed. Both SNQI projection helpers carry the marker,
   including latency raw-input and CSV round trips; historical CSVs remain readable.
5. `simulation-step-trace.v2` and `paired_effect_native_trace.v2` declare the frozen
   final-goal meaning introduced by FXM. Readers retain support for v1/v2 separately,
   with no mixed pairs, native/simulation references, pooled cohorts or reexport arms.
   Trace adapters preserve source-schema identity. Trace progress remains radial;
   the route-arclength change applies to the deadlock/stall detector.

Nine paired episodes replayed the reviewer's exact release-scenario/planner matrix:
merging low at dev 1003/1004, frontal approach at dev 1003, goal/native ORCA/hybrid v4,
horizon 600, dt 0.1, force and trace recording. Positions, reset state, waypoints,
steps, completion index, termination, status, outcomes, spawn validity, safety predicates,
event ledger, failure mechanism and interaction exposure are identical in **9/9**.
Retained paired-effect scalar values also match. The simulation/native trace payloads
are identical apart from their version strings. Only declared metric keys change;
five successes all have efficiency <=1, and four failures have raw NaN / JSON null.
The shortest reference is identical across all arms on each paired scenario/seed.
All nine records validate the episode schema; references recomputed from the final
implementation match exactly (three cache misses, six hits).

The continuous polygon references are lower than the review's sampled Theta*
estimates: those used grid/inflation approximations. The analytic obstacle regression
pins `sqrt(10)+4` m rather than the 9.16 m point reference; the actual producer regression
pins the same 7 m open-map zone reference for efficiency and ideal time.

| Episode | Deadlock before → after | Stall windows | Efficiency before → after | Reference m before → after |
|---|---|---|---|---|
| `classic_merging_low__goal__1003` | true → false | 114 → 0 | 0.987227 → NaN (JSON null) | 55.264411 → 51.964512 |
| `classic_merging_low__goal__1004` | true → false | 113 → 0 | 0.968949 → NaN (JSON null) | 54.083919 → 49.757615 |
| `classic_merging_low__hybrid_rule_local_planner__1003` | false → false | 166 → 0 | 0.804918 → 0.756856 | 55.264411 → 51.964512 |
| `classic_merging_low__hybrid_rule_local_planner__1004` | false → false | 120 → 0 | 0.808819 → 0.744119 | 54.083919 → 49.757615 |
| `classic_merging_low__orca__1003` | false → false | 142 → 3 | 1.000000 → NaN (JSON null) | 55.264411 → 51.964512 |
| `classic_merging_low__orca__1004` | false → false | 125 → 0 | 0.758237 → 0.697584 | 54.083919 → 49.757615 |
| `francis2023_frontal_approach__goal__1003` | false → false | 0 → 0 | 1.000000 → NaN (JSON null) | 32.000000 → 31.804388 |
| `francis2023_frontal_approach__hybrid_rule_local_planner__1003` | false → false | 0 → 0 | 0.993216 → 0.987145 | 32.000000 → 31.804388 |
| `francis2023_frontal_approach__orca__1003` | false → false | 0 → 0 | 0.996109 → 0.990020 | 32.000000 → 31.804388 |

The steady 0.4 m/s approach still has zero stall windows. A 40-sample stop still
has 27 windows and a true deadlock; the same stop with sampled pedestrian collision
or an episode collision flag has a false headline. Cubic motion still gives jerk
6 m/s³ at dt 0.1/0.05; first-step successful time is 0.1 s and reset-inclusive
path geometry is retained.

Compact receipt: [FXM2 paired metrics](evidence/issue_10007_fxm/fxm2_paired_metrics.json).
The real merging goal trajectories are in
`tests/fixtures/benchmark/fxm2_merging_goal_dev1003_1004.json`. Full private replay
rows/logs, probe/analyzer, plan and final report are retained in the caller's home
artifact namespace; no held-out, calibration, release or claim-admission evidence.
The probe is the reviewer-provided script, with additive raw-efficiency capture.

Validation: `uv sync --frozen --all-extras`; every pytest invocation uses
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run pytest -n0`. Final metric/producer/
projection/trace-reader/consumer suite: **692 passed, 1 existing selector-adapter skip**.
The broader reviewer suite initially had 722 passed, 1 skipped and one newly exposed
consumer failure; that consumer was repaired, and all 20 analyzer/native tests plus
the final suite pass. Release-comparison and hierarchical lanes passed in that run.
The reader run's old v2-is-unknown assertion was migrated to v99; both v1/v2 support
and the unknown-version negative control pass. Ruff and formatting pass. Broad PR
readiness remains unrun because its simulator lanes include non-dev seeds.

### FXM2 test-value gate

All new bug assertions have pre-fix proof at the reviewed head: **18 fail at the
intended values/contracts; two v1 controls pass**. The separately exposed #5416
consumer regression also fails on the required finite-efficiency error before repair.
See [compact regression receipt](evidence/issue_10007_fxm/fxm2_regression_failures.txt).
Full failure logs remain in the private artifact namespace.

| Protected behavior | Credible regression / red evidence | Coverage gap | Test-only seam |
|---|---|---|---|
| Actual merging goal timeouts remain moving along the route | Radial reference returns true deadlock on both recorded seeds | Existing steady-approach test has a straight route | None; real coordinate fixture |
| Route/polygon copies remain frozen after navigator mutation | Missing route snapshot / mutable navigation data | Existing reset test freezes only the final goal | None |
| Failed efficiency stays undefined; success >1 stays visible and flagged | Old 1.0 failure value / old clipping hides expected 2.0 | Existing efficiency checks use successful straight paths | None |
| JSON preserves undefined efficiency; reached-with-collision is not efficient | Missing efficiency field / finite collided value | Existing sanitization drops NaN; no success-only collision check | None |
| V2-only payloads require markers | Unmarked row accepted in flat/nested cases | Existing version tests compare marked v1/v2 only | None |
| Both SNQI projections retain identity | Missing metric-schema keys | Existing tests use historical unmarked inputs | None |
| Continuous zone distance and producer ideal time agree | Point distance differs from analytic shortest polygon distance / producer returns 9 instead of 7 m | Existing shortest-path coverage ends at a point | None; pre-fix signature filters only unsupported inputs |
| Paired and pooled trace versions never mix; v1/v2 readers retain identity | Old v2 rejection / absent pair/cohort refusal | Existing fixtures contain only v1 | None |
| Failed v2 rows remain analyzable while successful rows need efficiency | Old missing-or-invalid path-efficiency error | Existing analyzer fixtures give failures a finite legacy value | None |

The historical latency packet test now verifies unchanged values and provenance,
with only its generator hash replaced by an independent SHA-256 of current source
bytes. Editing the source necessarily changes that hash; frozen evidence is not
rewritten. The supported-trace test's old v2 negative control moves to v99. Fake
fidelity rollouts now use dev seed 1001; held-out-looking values elsewhere are
synthetic table keys/config labels, never real simulator steps.

Dissertation-facing metric list for the complete v2 definition: success-only path
efficiency and its denominator/aggregation, shortest-path reference and derived SPL
where produced, normalized goal time (all/success-only), ideal-time ratio/SNQI-v2 T,
aggregated completion time, jerk/SNQI-v2 J, deadlock flags/rates/stall diagnostics and
constraints-first endpoints, failure-to-progress, SocNavBench length/ratio/irregularity,
Social Mini-Game makespan/deadlock/path-deviation rows, legacy SNQI/v1/v2 composites,
and rankings/sensitivity based on them. Trace initial distance/radial progress and
paired timeout normalization retain their separately versioned reference meaning.
Success/collision/timeout outcomes do not change. Matching-definition anchors and
historical recomputation remain downstream review work; no dissertation source or
admitted claim was edited.

## CURVFIX: D-055 bounded arc-length curvature (stacked on PR #10014)

This addition supersedes the pre-D-055 curvature definition in unreleased metric v2.
The lane starts from #10014 head `ca1217c4f9e0ef0570ae172b26722cfc03338702`.
No other metric, planner, environment or frozen anchor changes. Add `curvature_mean`
to `CHANGED_METRICS`; cross-definition comparisons suppress it alongside the earlier
FXM corrections. Existing historical v1 scalar rows remain unchanged.

For consecutive recorded path positions, including reset when supplied, let
`ds_i = ||p_{i+1} - p_i||` and `phi_i = atan2(p_{i+1} - p_i)`.
Discard displacement steps shorter than **1e-3 m**. Over consecutive retained steps:

```text
curvature_mean = sum(abs(wrap(phi_{j+1} - phi_j))) / max(sum(ds_j), 1.0 m)  [rad/m]
```

Wrap to [-pi, pi]. Subthreshold steps add neither turning nor length. Stopping and
rotating in place, then departing in a new direction counts one path turn; reversal
counts pi. Fewer than two retained steps give 0.0. Nonfinite displacements are ignored.
The result is finite and nonnegative; the **1.0 m floor** bounds very short paths.
`CURVATURE_MIN_DISPLACEMENT_M` and `CURVATURE_LENGTH_FLOOR_M` name the thresholds.

The previous time-sampled `|v x a| / |v|^3` formula amplified nearly stationary
creeping and diluted turns with stationary samples. Its body is preserved exactly as
`_legacy_curvature_mean`; public `curvature_mean` dispatches via the explicit
`metric_schema_version` keyword. Historical recomputation passes the row's version
resolved by `metric_definitions.metric_schema_version` (unmarked rows resolve to v1).
New producers continue to emit v2 and use the new definition by default. Historical
v1 recomputation matches the starting function exactly on **672/672** recorded traces
and four analytic controls; see [v1 receipt](evidence/curvfix/v1_parity.json).

### Recorded development data

The requested main rehearsal contains 672 dev1001 scalar rows but disables trace
recording. The trace-enabled companion contains the same 14 arms x 48 scenarios on
seed 1001. All paired identities, step counts, terminations and old curvature scalars
match exactly. Both directories were copied locally; no remote computation or new
simulation was used. Source commit: `ea414933e61ce267389bd3bcbe97fb669a825c6e`.
Input file paths, locations, SHA-256s, arm quantiles and the full-receipt checksum
are in [the compact measurement receipt](evidence/curvfix/measurement.json).
All 672 per-episode values remain in the local full receipt; no public custody
is claimed for that raw diagnostic artifact. These are diagnostic
correctness data, not held-out, calibration, release or admitted research evidence.

Old values are recomputed from post-step positions exactly as before; new values
include the recorded reset pose. Recomputed old scalars are bit-equal to recorded
scalars in 653 episodes; the other 19 differ by at most 3.94527e-16 relatively, consistent
with floating-point reduction roundoff. V1-vs-start-function equality on the same
local arrays remains exact in all 672 episodes.

| Arm (differential_drive; n=48 each) | Old median | Old p95 | Old max | New median | New p95 | New max |
|---|---:|---:|---:|---:|---:|---:|
| goal | 0.0582467588 | 0.192548929 | 0.29020679 | 0.0512666043 | 0.171518231 | 0.277262609 |
| guarded_ppo | 0.26802832 | 9.19554764e+11 | 1.19303259e+13 | 0.181320717 | 0.370619242 | 0.404624581 |
| hybrid_rule_v4_fast_progress_static_escape | 0.0974291029 | 0.290618476 | 0.537732196 | 0.076345651 | 0.186870551 | 0.26431637 |
| hybrid_rule_v4_fast_progress_static_escape_continuous | 0.104042216 | 1.25251414 | 211055.194 | 0.0801537611 | 0.180816456 | 0.220736814 |
| orca | 0.118361024 | 2.60641808 | 539.805964 | 0.0738001038 | 0.361771303 | 0.918282507 |
| ppo | 0.281595747 | 0.545537859 | 0.65096568 | 0.264101994 | 0.404612578 | 0.538529319 |
| prediction_planner | 0.0700293638 | 0.398463081 | 0.751995615 | 0.11259466 | 0.38326374 | 1.01350221 |
| predictive_mppi | 0.0418562955 | 0.194498281 | 1.0046221 | 0.0727027526 | 0.273120713 | 0.424445382 |
| risk_dwa | 0.10392883 | 4.62642729e+09 | 1.81600464e+13 | 0.0767430873 | 0.215268201 | 0.281947131 |
| sacadrl | 0.298265122 | 0.598229083 | 0.625008893 | 0.296773789 | 0.609290466 | 0.7125 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4 | 0.100455425 | 0.191206183 | 0.325599668 | 0.0753285171 | 0.18297371 | 0.212634576 |
| scenario_adaptive_hybrid_orca_v2_collision_guard_v4 | 0.100455425 | 0.190724859 | 0.325599668 | 0.0753285171 | 0.18297371 | 0.212634576 |
| social_force | 0.296872998 | 4.91911541 | 3283.65408 | 0.167980131 | 1.17661929 | 2.25989155 |
| socnav_sampling | 0.140731803 | 1.79374735 | 10.8501668 | 0.109747505 | 0.498993118 | 1.23960009 |

New pooled p95 (NumPy linear quantile): **0.41913666808708483 rad/m**.
This is the future K scale candidate from this development cohort, not an admitted
anchor. **No episodes exceed 10 rad/m**; the largest new value is
2.259891548968766 rad/m (social_force). Therefore the requested extreme-case list is empty.

### Regression and test-value evidence

[Starting-head failures](evidence/curvfix/base_failures.txt): 14 intended failures
(12 geometric cases, changed-metric declaration, updated bent-path characterization),
9 unchanged-behavior controls pass. A fifteenth failure proves the existing mixed-version
fallback fixture problem: its synthetic healthy row lacked the v2 marker. Correct only
that fixture marker so fallback exclusion can be tested on a uniform-definition cohort.
Four historical-version controls are checked separately against the original function
AST because the starting API has no version keyword. Final v1 tests use public dispatch.

The 1e-6 m/s synthetic creep probe changes **5e13 -> 0 rad/m**. Stop/rotate/leave
changes **0 -> pi/4 rad/m** (pi/2 turn over 2 m); reversal changes **0 -> pi/2 rad/m**
(pi over 2 m); the 0.4 m path changes **1.25 -> pi/2 rad/m** (1 m floor).
Straight and circle controls pass both definitions, as expected; circles of radius
0.5, 2 and 10 m give approximately 1/R. The legacy creep stays exactly 5e13.

Test-value gate (no production test seam):

| Behavior protected | Credible regression | Existing coverage gap | Inputs/oracle |
|---|---|---|---|
| Creep suppression | Restore time mean or shrink displacement cutoff | Existing circle/straight tests skip creeping | Synthetic perpendicular 1e-7 m displacement; independent zero-turn oracle; real 672-episode probe corroborates defect |
| Count turns across stops and reversal | Break adjacency at stationary steps or cross-product-only turns | No stationary-to-new-direction case | Hand geometry: pi/2 or pi, divided by 2 m |
| Short path and inclusive cutoff | Divide by actual short length or use strict greater-than | Previous insufficient-point test skips two-step turning | Hand lengths 0.4 m, 0.999/1 mm steps; literal pi/2 |
| Time/angle/reset invariance | Reintroduce dt, unwrapped angles or omit reset | Earlier curvature tests use one dt and no reset | Same geometric path at four dt values; independent angle/length values |
| V1 compatibility | Route historical rows to v2 or change legacy body | No version-specific curvature assertion | Starting-function exact values and 672 identical-array comparisons |
| Changed-field comparison | Omit curvature from CHANGED_METRICS | Prior changed-field assertions cover earlier FXM fields | Public changed_metric_field call, fails on start |
| Bent-path characterization | Retain old time-mean value | Existing test explicitly pins old value 1.0 | Two right-angle turns over three unit legs = pi/3 |
| Fallback exclusion | Drop version marker or include degraded arm | Test failed before reaching exclusion due mixed definitions | Existing row fixture; marker matches producer; excludes degraded row |

### Consumer audit and normalization boundary

`rg -n curvature_mean robot_sf scripts docs configs tests` was inspected by consumer:

- `snqi/compute.py`: K uses `spec.sources["K"] / spec.upper_anchors["K"]`, clipped
  at one. `v2_spec.py` requires an explicit positive frozen calibration-p95 anchor;
  `v2_calibration.py` derives the empirical linear p95 from matching-version rows.
  No historical raw-scale constant appears in the K computation.
- `metrics.py` legacy/optional curvature term and `baseline_stats.py`: supplied
  median/p95 normalization, with existing definition compatibility guards retained.
- Camera-ready reporting/summaries, policy-analysis tables, false-positive replay
  comparisons, collision scenario similarity and release validators consume scalar
  values as names, means, deltas or plots; no rescaling for the old singularity.
- `release_row_anomalies.py` retains an explicit configurable **1.0** raw curvature
  threshold paired with a minimum path length for a diagnostic orbit heuristic.
  This is an absolute geometric diagnostic, not the K anchor; its interpretations
  change with metric semantics. It neither rescales the metric nor assumes the old
  near-stop magnitudes. This audit does not claim every consumer is scale-free.
- `policy_analysis_run.py:: _filtered_curvature_mean` is a separate historical
  speed-filtered diagnostic stored as `curvature_mean_eps0p1`; it is not the robot
  `curvature_mean` field or K source, and is unchanged.
- Existing frozen/published normalization assets stay byte-unchanged. Pre-D-055
  unreleased v2 diagnostic scalars must be recomputed from traces before current
  v2 pooling or calibration; the unchanged version string alone cannot distinguish
  those prerelease snapshots. No old assets are relabeled or calibrated here.

All scored consumers are anchor-driven; none assumes the former exploding scale.
K weights, clipping, sources and frozen anchors are unchanged. Deriving and approving
matching-definition normalization remains a separate calibration action.

### Final lane validation

Every direct importer of either changed module and every test file matching
`curvature` was run by explicit file path: **49 files, 1547 passed, 2 existing skips,
3 held-out simulator nodes deselected**. Pytest ran with `-n0`,
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. The skipped distinct-validator-checkout and optional
0.0.3 publication-bundle checks are named in [validation.json](evidence/curvfix/validation.json),
along with the complete command and excluded nodes. No new skip/xfail or timeout
was added. Synthetic held-out row/config identifiers were inspected without planner
or environment steps; forbidden simulator seeds 111/112 were not executed.
Existing permitted seed-22 group crossing and seed-3971 pedestrian tests remain
existing-test validation, not new development evidence. New recorded-data analysis
uses seed 1001 only. Simulation execution stays serial; native-command child plus
parent is the largest relevant process pair.

Repository-wide `uv run ruff check`, `uv run ruff format --check`,
`scripts/validation/check_seed_holdout_diff.py --base-ref origin/main` and
`git diff --check` pass. Hosted CI is requested by marking the stacked PR ready;
its outcome and domain review remain separate from these local proofs. No merge.

### Evidence hygiene follow-up after orchestrator review

The orchestrator review of PR #10054 at `b87e472d116fc45fec47cfcf669acce84b75373c`
confirmed D-055 implementation and measurement correctness; its remaining findings
were evidence registration, review markers, registry provenance and PR metadata.
All four CURVFIX evidence files now carry the shared writer's AI-GENERATED /
NEEDS-REVIEW marker and have exact catalog entries. The probe uses the shared marked
writer for both summary and local full receipt.

Following the repository artifact policy, the 5,731-line tracked receipt is replaced
with a compact summary. Full episode values remain in the local diagnostic evidence
directory, with their file name, location, size and SHA-256 recorded in the summary.
Source hashes now carry explicit source paths and local locations; all 28 ambiguous
hash findings are remediated instead of accepted into the baseline. The generated
registry baseline and its review companion document the evidence-tree refresh and
absence of new findings. No metric, frozen anchor or measured value changes.

Reproduction uses the existing probe flags plus `--full-output` pointing outside
`docs/context/evidence/`. The local URI locations in the summary identify copies in
`~/curvfix_evidence/`; this is local diagnostic preservation, not public archival custody.
