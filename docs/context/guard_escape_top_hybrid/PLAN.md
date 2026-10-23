# RQ3 top-hybrid development campaign: prospective analysis plan

This plan and all inputs must be committed and pushed before any episode. Record
its 40-hex commit, commit time, push/readback time and first acquisition time in
the report. No pilot, smoke, identity or additional seed run is authorized.

## Question and controller selection

Does the guard/escape result of replication PR #10426 extend to the hybrid with
the highest confinement success point estimates? The 0.0.8 Table 7.2 / Section
7.6 source, `diss/chapters/ceiling/rq3_structure_crosscut.tex`, table label
`tab:rq3-predicate-classification`, gives hybrid v4 places 1 and 2 (tied first
in portal), continuous v4 places 4 and 4. The numeric companion is
`docs/reports/release_group_success.json`: hybrid v4 234/240 static and 206/210
portal, continuous v4 230/240 and 205/210. Source identities and the other hybrids
are recorded in `provenance/controller-selection.json`. These historical rows
select the controller only; no release/sealed seed is executed or pooled.

Use the frozen `hybrid_rule_v4_fast_progress_static_escape` candidate identified
in `inputs/study.yaml`. All four original controls exist. Escape-on retains the
native controller: static clearance escape, recenter and corridor transit on;
corridor-subgoal remains false in every arm because it is off in this candidate.
Escape-off disables the three active switches and leaves that inactive switch
false. This is the original bundle-removal intervention applied to native
settings, not a new corridor-subgoal treatment. The escape bundle therefore
contains three active components here versus four in #10426; no isolated
component attribution or assumption of identical bundle size is permitted.
The canonical runtime resolver was checked without simulation: G1E1 equals the
selected frozen configuration in all 18 cells. No substitution is necessary.

## Design and fixed inputs

Four paired arms: G0E0, G0E1, G1E0, G1E1 (G=hard guard; E=native escape bundle).
G0 sets `ablation_disable_hard_guard=true`; G1 sets it false. Keep speed/turn
limits, forecasts, rollouts and soft scores. E0/E1 switch only the three active
escape flags listed in `inputs/study.yaml`. All nonfactor settings are identical
across arms. Configuration files are complete flattened inputs.

Use development seeds **1061–1090 inclusive**, the same 30 seeds in every cell
and arm: 2,160 episodes, 540 per arm. Never run seeds 111–140, 50036–59019,
release/sealed seeds or any additional seed. Do not pool campaigns/controllers.
Use exactly the 18 #10426 cells and authored budgets, dt=0.1, differential drive:

- Static constriction (8): classic_bottleneck_low, classic_bottleneck_medium,
  classic_bottleneck_high, classic_realworld_double_bottleneck_high,
  classic_cross_trap_low, classic_cross_trap_medium, classic_cross_trap_high,
  francis2023_narrow_hallway.
- Portal (7): classic_doorway_low, classic_doorway_medium, classic_doorway_high,
  francis2023_entering_room, francis2023_exiting_room,
  francis2023_entering_elevator, francis2023_exiting_elevator.
- Crowding control (3): classic_station_platform_medium,
  francis2023_crowd_navigation, francis2023_robot_crowding.

The infeasible narrow doorway remains excluded. All paired scenario inputs,
map identities and budgets must agree with #10426 except seed labels. Source
configuration overrides for blind corner/perpendicular traffic do not intersect
this roster. This design tests a selected configuration, not all hybrids.

## Frozen analysis and five-point rule

Primary endpoint: collision-free route completion (`route_complete` and no
`collision_event`, consistent with `metrics.success`). Also report pedestrian
collision, wall collision and timeout exactly as in #10426. Use its byte-identical
`analyze.py`: equal-cell weighting, 10,000 whole-seed-block draws retaining all
cells and arms, NumPy `default_rng(1001)`, percentile 2.5/97.5 pointwise 95%
intervals. Guard and escape main effects average on-minus-off across the other
factor; interaction is Y11-Y10-Y01+Y00. Report every arm's success and all
contrasts in every group, without outcome-selected subgroups.

The inherited analyzer hard-codes 1001–1030. Use #10426's reversible label adapter
with an explicit acquisition start of 1061: subtract 60 for analysis only. Raw
and compact evidence retain actual seeds. Correct only output identity metadata
and inherited review verdict. The adapter's new explicit-start parameter is
covered by a red-on-main/green regression proving byte-identical numeric CSVs
when only acquisition seed labels change. No numerical analyzer change.

The #10426 decision rule is retained verbatim:

> Pre-registered stop rule (written before the result): outcome = paired seed-block success difference (on minus off) per confinement group, for guard and for escape. EFFECT if |diff| >= 5 pp AND the interval excludes 0 in at least one confinement group -> keep three RQs with a feature answer scoped to that controller. NO EFFECT -> fold RQ3 into RQ2. MIXED, or the control group moves as much -> fold. No rescue reanalysis.

Apply #10426's frozen interpretation: OR across mechanisms. A factor qualifies
when absolute effect >=5.0 pp and its interval excludes zero in at least one
confinement group. Qualifying confinement groups with opposite signs make that
factor MIXED. Control moves as much when its absolute point estimate is >= every
qualifying confinement estimate; that factor cannot supply the feature answer.
Overall EFFECT requires at least one qualifying non-MIXED factor whose control
does not move as much; otherwise NO EFFECT, or MIXED when opposite directions
prevent a consistent answer. Report both groups and directions separately.
Secondary: escape is bounded below five points only if its 95% upper bound is
<5.0 pp in that confinement group. Nonqualification alone is not equivalence.
The RQ-count consequence is a recommendation; manuscript admission is reserved.
Compare descriptively with #10426's already-recorded effects; no cross-controller
causal contrast or seed pooling. Intervals condition on this roster; no unseen
scenario, simultaneous-interval, release, or general safety claim.

## Source, execution and stopping

The original execution commit 0035fba05c9b679e39f58ec286f5d57b96bb9a0a is unavailable
locally and from GitHub. Reconstructed acquisition commit
6a3ac329d25a97202b050edd69190142fd7553e7 consists solely of its archived complete
source/test patch applied to recorded release base
373dbfde4f39667cf9e8732dabe7df5118bdeab1. Patched planner and test Git blobs are
4a1180c6faa1f48b372e9c100dd1d1440db06e1b and
bd58eef51f7a4d6b8cd2678b947d176425c7fbbf, matching archived patch identities.
This is reconstructed source, not recovery of the original commit object.
Preserve the patch and a Git bundle containing the reconstructed commit with the
campaign. Archived pure candidate tests fail on current main (six failures,
one pass) and pass on reconstructed source (seven pass), without episodes.
Do not add the diagnostic runtime to current production code.

Run CPU only, four workers maximum, numerical libraries one thread each. Reuse
Python 3.13.14, NumPy 2.4.6, PyYAML 6.0.3 and the recorded uv.lock digest. Check
load and >=60 GB free disk before launching; leave headroom for other lanes.
Use a supervised process, log outside git and preserve command/environment.
Absolute path resolution in a separate launch-config copy is permitted only for
input paths; preserve both its digest and registered config digest.

Run all 2,160 episodes once regardless of effects. No optional stopping,
replacement seeds or silent reruns/exclusions. Reject duplicate/missing slots,
wrong factors, budgets, seeds/source/config, contradictory endpoint/termination,
fallback/degraded rows or an adapter other than HybridRuleLocalPlannerAdapter.
Any invalid acquisition blocks the primary evidence; report exact failure.

## Preservation, validation and handoff

Keep full producer rows, runner manifest, config/source/environment identities,
logs, all analysis outputs and SHA-256 inventory outside git. Upload once to
W&B with in-process credentials only; verify artifact bytes by download. Keep
only compact aggregate results/provenance and registered inputs in git; compact
endpoint evidence stays in the archived campaign. Preserve source-line digests.
Recompute CSVs byte for byte and independently check effects/intervals with the
unchanged #10426 independent method. Update the context index and open one PR;
never merge. Record SHA, time, results, W&B path and PR in ~/cr_ablation2_report.md.
Preserve reports/logs, verify remote 40-hex head, then remove both owned worktrees.
