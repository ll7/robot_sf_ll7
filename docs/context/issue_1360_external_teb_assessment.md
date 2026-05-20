# Issue #1360 External TEB Reference Assessment

Date: 2026-05-20

Related issue:

- `robot_sf_ll7#1360`: research: evaluate external reference TEB baseline for corridor deadlocks

Related context:

- `docs/context/issue_1318_teb_corridor_deadlock_eval.md`
- `docs/context/external_planner_reuse_checklist.md`
- `configs/scenarios/sets/issue_1318_teb_corridor_deadlock_slice.yaml`
- `robot_sf/planner/teb_commitment.py`

## Goal

Assess whether a maintainable external or reference TEB-style planner can be used as a Robot SF
baseline for the #1318 classic-merging corridor-deadlock slice.

This note is a fail-closed provenance and contract assessment. It does not implement a new adapter
and does not change the #1318 benchmark conclusion.

## #1318 Dependency Summary

Issue #1318 evaluated the existing in-repo `teb` planner on the tracked corridor-deadlock slice:

- `classic_merging_low`, seeds `111` and `113`;
- `classic_merging_medium`, seeds `111`, `112`, and `113`;
- horizon `600`;
- same map-runner episode-record contract for TEB, ORCA, and the hybrid-rule incumbent.

The in-repo TEB commitment planner was runnable but collided with static geometry on all five
selected seeds. ORCA was mixed, and the current hybrid-rule candidate solved four of five. The
useful #1318 result is therefore negative evidence plus a reusable narrow comparison surface, not a
reason to promote the current in-repo TEB approximation.

## Candidate Checked

Candidate: ROS `teb_local_planner`

Source anchors checked on 2026-05-20:

- ROS package index: <https://index.ros.org/p/teb_local_planner/>
- Upstream checkout URI listed by ROS index:
  <https://github.com/rst-tu-dortmund/teb_local_planner.git>
- ROS API docs for the timed elastic band trajectory representation:
  <https://docs.ros.org/en/noetic/api/teb_local_planner/html/classteb__local__planner_1_1TebOptimalPlanner.html>

The ROS package index identifies `teb_local_planner` as a BSD-licensed CATKIN package and describes
it as a `base_local_planner` plugin for the ROS 2D navigation stack. It also lists ROS navigation
dependencies such as `base_local_planner`, `costmap_2d`, `costmap_converter`, `dynamic_reconfigure`,
`geometry_msgs`, `nav_core`, `nav_msgs`, `pluginlib`, `roscpp`, `tf`, `visualization_msgs`, and
`libg2o`.

## Contract Fit

`teb_local_planner` is the right conceptual reference for the planner family, but it is not a small
Robot SF adapter.

Observed contract mismatch:

- Runtime: ROS/catkin plugin architecture, not an importable Python package or direct Robot SF
  planner class.
- Map and obstacle inputs: ROS costmap and costmap-converter surfaces, not Robot SF SVG map,
  scenario, and episode-record contracts.
- Dynamic obstacles: ROS messages and costmap conversion, not direct Robot SF pedestrian state
  arrays.
- Action output: ROS local-planner velocity command path, not a standalone Robot SF action
  dictionary with local diagnostics.
- Optimization dependency: `libg2o` and C++ ROS build/runtime assumptions.
- Benchmark evidence path: faithful use would require launching or embedding a ROS navigation
  stack, bridging Robot SF scenarios into ROS costmaps/messages, then adapting ROS commands back
  into Robot SF while preserving per-step diagnostics.

That integration would be a ROS bridge and parity project, not the "smallest adapter" requested by
Issue #1360.

## Same-Subset Feasibility

No Robot SF benchmark run was attempted for the external reference because no runnable adapter exists
that satisfies the repository planner contract.

A faithful same-subset comparison would first need:

- a ROS runtime environment with the upstream planner built and runnable,
- a Robot SF to ROS costmap/map bridge for the #1318 `classic_merging` scenarios,
- dynamic-pedestrian message conversion with timing semantics matching the Robot SF benchmark step,
- action projection back into Robot SF's command dictionary,
- fail-closed handling for unsupported static geometry or dynamic-obstacle conversion,
- and per-step diagnostics that separate raw planner output, adapted command, projected command,
  runtime status, and fallback/unavailable reasons.

Until those pieces exist, an "external TEB" row would be `not available`, not benchmark evidence.

## Verdict

Current tier: `assessment only / ROS bridge required`.

Do not add an external/reference TEB planner row, config, or benchmark comparison for #1360 now.
The maintained ROS `teb_local_planner` package is useful as a conceptual reference and provenance
anchor, but direct reuse would require a separate ROS bridge/parity issue with its own dependency,
runtime, and diagnostics proof.

The #1318 conclusion remains unchanged: the current in-repo TEB approximation does not resolve the
corridor-deadlock slice, while the external/reference path is unavailable for fair same-subset
comparison in this repository today.

## Follow-Up Boundary

Only reopen implementation if there is explicit maintainer appetite for one of these larger paths:

- a ROS navigation-stack bridge for Robot SF local-planner replay,
- a clean-room TEB optimizer implemented against Robot SF's native planner contract,
- or a narrower motion-primitive/state-lattice baseline under #1361 that avoids ROS integration.

Any future external-TEB comparison should cite this note, use the #1318 scenario slice, and report
unavailable/degraded/fallback states as caveats rather than successful benchmark outcomes.

## Validation

This assessment used source/provenance inspection and repo-local contract review. No benchmark run
was attempted because the selected external reference has no current Robot SF adapter.

Validation commands for this PR:

```bash
rg -n "teb_local_planner|external TEB|issue_1360" docs/context docs/README.md
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
