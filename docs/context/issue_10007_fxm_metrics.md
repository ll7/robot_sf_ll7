# Issue #10007 FXM shared metric correction

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

Root cause: reset used `simulator.goal_pos[0]` (the handoff waypoint), termination
overwrote it with the active waypoint. Freeze a copy of `robot_navs[0].waypoints[-1]`
at reset; use it for shortest path, path efficiency, ideal-time ratio, failure
progress/distance, static-deadlock evidence and trace/paired-effect denominators.
The shortest path begins at the reset pose. Success still uses the declared
completion policy; a point reference to the sampled final target can therefore
be longer than a path ending at the goal-zone boundary (efficiency clips at 1).

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
last distance to the frozen route goal. A window stalls when reduction **<=0.05 m**;
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
collision regression fails the same assertion on base. Radial detours can still
count as no-progress; this repair does not redefine the screen as route arclength.

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
Focused metric/SNQI/helper/calibration suite: 658 passed, 1 existing platform skip.
Final compatibility/native-command/hierarchical/scalarization suite: 152 passed.
Updated metric/SNQI/helper/native/classic-caller suite: 674 passed, 1 existing skip.
Release comparator equal-value and mixed-successor-schema cases: 2 passed.
After main integration, FXM/helper/native/classic/target-trace suite: 115 passed,
1 existing platform skip; all 12 real probes rerun with identical corrected
metrics and trajectories, and their recorded source hashes match final code bytes.
These simulation receipts record the pre-amend source commit plus checkout dirty
state and byte hashes; the final amendment changes documentation/evidence only.
Final goal-wiring/helper suite: 66 passed, 1 existing platform skip.
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
