# Issue 1057 Semantic Blocker Audit

Date: 2026-05-07

Related issues:

* [Issue #1057](https://github.com/ll7/robot_sf_ll7/issues/1057)
* Route handoff: [Issue #730](https://github.com/ll7/robot_sf_ll7/issues/730)
* Invalid SVG repair: [Issue #837](https://github.com/ll7/robot_sf_ll7/issues/837)
* SNQI metric drift: [Issue #455](https://github.com/ll7/robot_sf_ll7/issues/455)
* Follow-up blocker: [Issue #1065](https://github.com/ll7/robot_sf_ll7/issues/1065)

## Goal

Classify known semantic blockers before using affected benchmark failures as planner evidence:
route handoff, invalid SVG geometry, SNQI semantics, metric drift, and fallback/degraded execution.

## Current Verdict

Most named blockers are resolved or have explicit caveats. The live blocker is route-clearance
warnings in current paper and h500 surfaces.

| Blocker | Current status | Evidence | Attribution rule |
|---|---|---|---|
| Route handoff / premature first waypoint completion (`#730`) | resolved | `robot_sf/nav/navigation.py` now has spawn-aware route rebasing; `tests/navigation_test.py` covers spawn-inside-threshold behavior. | Do not reopen unless a current trace shows first-waypoint rebasing causing stalls or skips. |
| Invalid SVG obstacle repair (`#837`) | resolved with logging/tests | `robot_sf/nav/svg_map_parser.py` uses `make_valid`; `tests/test_svg_obstacle_self_intersection.py` checks map-aware invalid-polygon logging and repair behavior. | Invalid-polygon warnings alone are not planner-failure evidence; inspect unrepaired cases if they appear. |
| SNQI semantic drift (`#455`) | resolved for current training/benchmark paths, still caveated for paper interpretation | Training paths use `robot_sf.training.snqi_utils` and record `snqi_formula=robot_sf.benchmark.snqi.compute_snqi`; camera-ready config pins v3 weights/baseline. | SNQI is a fixed synthesis aid, not a universal utility scalar. Use component metrics alongside it. |
| Metric drift / seed sensitivity | caveated | [issue_832_paper_matrix_extended_seed_schedule.md](issue_832_paper_matrix_extended_seed_schedule.md) records S5 ranking stability but scenario-winner and mean-drift sensitivity. | Avoid strengthening scenario-level claims from S3 alone. |
| Fallback/degraded execution | resolved by policy, must remain visible | [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md) and [issue_1054_planner_readiness_fallback_audit.md](issue_1054_planner_readiness_fallback_audit.md). | Never count fallback, degraded, failed, partial-failure, or not-available rows as successful benchmark evidence. |
| Route-clearance warnings | follow-up required | Current preflight/campaign artifacts report 18 route-clearance warnings; [May 4 note](camera_ready_all_planners_slurm_2026-05-04.md) flags negative-clearance classic scenarios. | Do not attribute failures on affected scenarios to planner mechanisms until issue `#1065` classifies the warning. |

## Validation

Named issue inspection:

* `#730` is closed and the current code/tests show spawn-aware waypoint rebasing.
* `#837` is closed and invalid obstacle repair has parser tests.
* `#455` is closed and current PPO training/fine-tuning paths use canonical SNQI utilities.

Search command:

```bash
rtk rg -n "route handoff|handoff|invalid geometry|invalid SVG|SNQI|metric drift|fallback|degraded|route_clearance|clearance|blocked|semantic" \
  docs/context docs/benchmark_spec.md docs/benchmark_planner_family_coverage.md docs/benchmark_camera_ready.md docs/context/evidence
```

The search confirms that fallback/degraded policy is explicit, SNQI caveats are documented, and
route-clearance warnings remain present in tracked evidence.

## Follow-Up Boundary

[Issue #1065](https://github.com/ll7/robot_sf_ll7/issues/1065) now tracks the unresolved
route-clearance warning audit. That issue should classify each current warning before affected
scenario failures are used as policy or planner mechanism evidence. No broad benchmark rerun is
required by this semantic-blocker audit itself.
