# Issue 1054 Planner Readiness And Fallback Audit

Date: 2026-05-07

Related issues:

* `ll7/robot_sf_ll7#1054`
* `ll7/robot_sf_ll7#562`
* `ll7/robot_sf_ll7#691`

## Goal

Audit the paper-facing planner rows for readiness tier, dependency status, and execution mode so
fallback or degraded execution cannot be mistaken for benchmark evidence.

## Evidence Sources

* `configs/benchmarks/paper_experiment_matrix_v1.yaml`
* `docs/benchmark_planner_family_coverage.md`
* `docs/context/issue_691_benchmark_fallback_policy.md`
* `docs/context/camera_ready_all_planners_slurm_2026-05-04.md`
* `docs/context/evidence/camera_ready_all_planners_2026-05-04/`

## Current Paper Matrix

Preflight command:

```bash
rtk uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1.yaml \
  --mode preflight \
  --label issue1054_preflight
```

Preflight produced seven rows in:

```text
output/benchmarks/camera_ready/paper_experiment_matrix_v1_issue1054_preflight_20260507_205442/reports/matrix_summary.csv
```

The current machine has `rvo2` importable, while `social_nav_sim` and `socnav` are not importable.
That supports ORCA-family preflight expectations but keeps the external SocNavBench bridge
dependency-blocked here.

## Planner Readiness Table

| Planner | Group | Expected execution mode | Readiness interpretation | Paper use |
| --- | --- | --- | --- | --- |
| `goal` | core | native | Baseline-ready control. | Valid if run availability is `available`. |
| `social_force` | core | adapter | Baseline-ready force-based comparator through declared adapter metadata. | Valid if run availability is `available`. |
| `orca` | core | adapter | Baseline-ready reciprocal-avoidance comparator, dependency-sensitive to `rvo2`. | Valid if `rvo2` is present and run availability is `available`. |
| `ppo` | learned baseline | native | Paper-facing PPO row when model provenance and benchmark-set claim caveats are satisfied. | Valid as benchmark-set evidence, not OOD/generalization evidence. |
| `prediction_planner` | experimental | adapter | Checkpoint-dependent prediction-aware challenger. | Experimental challenger only. |
| `socnav_sampling` | experimental | adapter | In-repo sampling challenger; not upstream SocNavBench support and not a SocNavBench bridge result. | Experimental challenger only. |
| `sacadrl` | experimental | adapter | Legacy adapter-sensitive challenger. | Implementation-level evidence only. |
| `socnav_bench` | outside frozen paper matrix | unknown/degraded in May 4 all-planners run | Dependency-blocked until issue-562 re-entry succeeds. | Exclude; do not count as fallback/degraded success. |

## Fallback Boundary

The fallback policy remains unchanged:

* `native` and declared `adapter` rows may be benchmark evidence when the run reports
  `availability_status=available`.
* `fallback`, `degraded`, `failed`, `partial-failure`, and `not_available` rows are non-success
  outcomes for benchmark claims.
* Diagnostic fallback remains valid only for explicit probes.

The May 4 all-planners evidence is the current cautionary example. It recorded
`socnav_bench` as `execution_mode=unknown`, `readiness_status=degraded`,
`availability_status=failed`, and `benchmark_success=false`; publication bundle export was skipped.
That run is partial internal evidence for completed rows, not a publication-ready all-planners
bundle.

## Follow-Up Boundary

No new metadata issue is required from this audit. Current configs and reports expose planner group,
benchmark profile, execution mode, readiness status, availability status, and failure reason. The
known SocNavBench dependency blocker is already represented by issue `#562` and the re-entry probe
workflow.
