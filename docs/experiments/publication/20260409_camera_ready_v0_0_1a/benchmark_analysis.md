# Benchmark Analysis

Date: 2026-04-09

## Scope

This note analyzes the release-rehearsal benchmark campaign before creating a new follow-up tag that
would include the tracked publication snapshot in git.

Analyzed campaign:

* Campaign id:
 `paper_experiment_matrix_v1_release_rehearsal_latest_20260409_20260409_120154`

* Source commit:
 `a60119aa10ed677a55a8b6b69264d4aeabfdb811`

* Scenario matrix:
 `configs/scenarios/classic_interactions_francis2023.yaml`

* Seeds:
 `111, 112, 113`

* Repeats:
`47` scenarios per planner, `141` episodes per planner, `987` total episodes

## Evidence

* External campaign summary:
 `output/benchmarks/camera_ready/paper_experiment_matrix_v1_release_rehearsal_latest_20260409_20260409_120154/reports/campaign_summary.json`

* External camera-ready analyzer outputs:
 `.../reports/campaign_analysis.md`

 `.../reports/campaign_analysis.json`

 `.../reports/scenario_difficulty_analysis.md`

* Tracked paper-facing handoff:
 `paper_results_handoff_snapshot.csv`

## Executive Read

The benchmark run itself looks operationally complete and paper-usable, but it is not clean enough
to treat as a fully healthy new canonical release without caveats.

What is solid:

* The campaign completed successfully with `7/7` successful planner runs.
* The run wrote the expected `141` episodes per planner and `987` episodes total.
* The tracked paper handoff is internally consistent and includes bootstrap confidence intervals for
  the manuscript-facing metrics.

What is not clean:

* The AMV coverage contract remains `warn`.
* The SNQI contract remains `warn`.
* Multiple planners still run in `adapter` mode, with especially high command projection for
`orca` and `socnav_sampling` .
* Several planners remain hard to present as clean comparators because their runtime or adapter
  burden is materially different from the core baselines.

## Findings

### 1. Release-path execution is healthy

* Campaign runtime: `728.73 s`
* Throughput: `1.3544` episodes/sec
* Successful runs: `7/7`
* Per-planner episode count: `141`

This means the benchmark pipeline produced a complete benchmark result, not a partial or
planner-fallback-only artifact.

### 2. Paper-facing headline rows are consistent

From `paper_results_handoff_snapshot.csv` :

* `goal`: success `0.0142` with 95% CI `[0.0000, 0.0213]`, collisions `0.2411` with 95% CI
 `[0.2128, 0.2766]`

* `orca`: success `0.1844` with 95% CI `[0.1277, 0.2270]`, collisions `0.0284` with 95% CI
 `[0.0213, 0.0426]`

* `ppo`: success `0.2411` with 95% CI `[0.1702, 0.2837]`, collisions `0.1135` with 95% CI
 `[0.1064, 0.1206]`

The paper-facing export is credible enough to drive manuscript values.

### 3. AMV coverage remains incomplete

`reports/amv_coverage_summary.md` reports:

* Status: `warn`
* Missing dimensions for the AMV paper profile:
`use_case` , `context` , `speed_regime` , and `maneuver_type`

This does not invalidate the benchmark execution, but it means the benchmark still does not satisfy
the stronger AMV coverage story without qualification.

### 4. SNQI contract is still warning-level, not green

Campaign summary reports:

* `snqi_contract_status: warn`
* Rank-alignment Spearman: `0.3571`
* Outcome separation: `0.2026`
* Dominant component: `time_penalty`

Interpretation: SNQI is still usable as a reported metric, but the current contract evidence is not
strong enough to present the release as fully hardened.

### 5. Adapter burden is still materially present

Notable planner execution details from `campaign_summary.json` :

* `orca`: `adapter` mode,  `projection_rate=0.7738`,  `infeasible_rate=0.7738`
* `socnav_sampling`: `adapter` mode,  `projection_rate=0.9722`,  `infeasible_rate=0.9722`
* `social_force`: `adapter` mode,  `projection_rate=0.1348`
* `prediction_planner`: `adapter` mode but `projection_rate=0.0000`

Interpretation: the campaign is reproducible, but several planners are still benchmarked through an
adapter-heavy command-space boundary. That is a real caveat for release messaging and cross-planner
comparability.

### 6. Analyzer audit path is now fixed

The initial analyzer run misread repo-relative `output/...` artifact paths when the campaign lived
outside the current worktree. That path-resolution bug has now been fixed in
`scripts/tools/analyze_camera_ready_campaign.py` and regression-covered in
`tests/tools/test_analyze_camera_ready_campaign.py` .

After rerunning the analyzer on the same campaign:

* per-planner episode counts resolve correctly at `141`
* scenario difficulty analysis loads correctly from campaign artifacts
* automated consistency findings drop to none

That means the remaining release caveats are benchmark-semantic concerns, not analysis-tooling
breakage.

### 7. Runtime and behavioral hotspots are stable enough to understand

* `prediction_planner` is by far the slowest planner at `309.31 s`
* `ppo` is next at `118.86 s`
* `social_force` is next at `75.38 s`

Behaviorally:

* `ppo` gives the best success among the highlighted rows, but at materially higher collision rate
  than `orca`

* `orca` remains the cleanest low-collision comparator, but only through heavy projection
* `social_force` and `sacadrl` both post `0.0000` success in this rerun

This looks stable enough to interpret, but not strong enough to market as universally healthy.

## Recommendation

Recommendation: do not create a new "clean" canonical tag yet if the tag is meant to signal that
the benchmark and its audit surfaces are fully healthy.

A new tag is reasonable only if the release notes explicitly state:

* benchmark execution completed successfully, 
* the paper-facing handoff is the authoritative metric source, 
* AMV coverage is still `warn`, 
* SNQI contract is still `warn`, 
* several planners remain adapter-mediated.

If the next tag is intended only to include the tracked publication snapshot in source control, that
is fine as a documentation or provenance tag. If the next tag is intended to advertise a stronger
"benchmark healthy and fully hardened" state, the current evidence does not justify that claim.
