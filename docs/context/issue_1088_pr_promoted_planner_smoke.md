# Issue 1088: PR Promoted Planner Smoke

Issue: <https://github.com/ll7/robot_sf_ll7/issues/1088>

## Decision

The PR smoke uses a one-scenario, one-seed planner sanity slice rather than the full camera-ready
campaign smoke. The promoted subset is `goal`, `social_force`, and `orca` because those planners are
baseline-safe and do not depend on ignored local model artifacts.

The workflow reports metric deltas against
`configs/benchmarks/pr_promoted_planner_smoke_baseline.json` and fails closed on runner failures,
availability failures, fallback/degraded readiness, missing records, collisions, near misses, or
runtime budget violations.

## Artifacts

- Workflow: `.github/workflows/pr-promoted-planner-smoke.yml`
- Runner: `scripts/validation/run_pr_promoted_planner_smoke.py`
- Scenario: `configs/scenarios/single/pr_promoted_planner_smoke.yaml`
- Baseline: `configs/benchmarks/pr_promoted_planner_smoke_baseline.json`
- User docs: `docs/benchmark_pr_promoted_planner_smoke.md`

Generated `output/benchmarks/pr_promoted_planner_smoke/` files are reproducible workflow artifacts
and should remain ignored unless a compact evidence copy is deliberately promoted for a later
review.
