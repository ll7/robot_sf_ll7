# Issue #2903: Prediction - Repair Horizon/Timestep Denominator Coverage 2026-06-16

> **Status: Historical / superseded by [Issue #2937](../issue_2937_horizon_denominator_health.md).**
> The fixture gaps identified here were repaired in Issue #2937; use the Issue #2937
> evidence for current denominator-health coverage.

## Summary

This note records the denominator-health audit for the horizon x timestep
ablation introduced in issue #2837 / PR #2887. The audit is analysis-only and
does not change forecast defaults.

## Evidence

- Compact summary: [`docs/context/evidence/issue_2903_horizon_denominator_health_2026-06-16/summary.json`](../evidence/issue_2903_horizon_denominator_health_2026-06-16/summary.json)
- Full report: [`docs/context/evidence/issue_2903_horizon_denominator_health_2026-06-16/denominator_report.json`](../evidence/issue_2903_horizon_denominator_health_2026-06-16/denominator_report.json)
- Human-readable report: [`docs/context/evidence/issue_2903_horizon_denominator_health_2026-06-16/denominator_report.md`](../evidence/issue_2903_horizon_denominator_health_2026-06-16/denominator_report.md)

## Key Findings

- Expected matrix size: **180 cells** (5 horizons x 4 timesteps x 9 trace families).
- Currently evaluated: **65 cells (36.1%)**.
- Missingness reasons observed on the durable fixture set:
  - **trace_too_short**: 75 cells - motion-rich traces are too short for the
    requested horizon.
  - **no_pedestrian_motion**: 40 cells - two fixtures contain no pedestrian
    motion.
  - metadata_missing, actor_class_missing, observation_tier_missing, and
    other_explicit_reason: **0 cells** each.
- Category totals sum to the expected 180 cells.
- One missing cell per observed category is spot-checked in the report.

## Minimum Fixture Additions For 90% Coverage

The report estimates that **7 fixture changes** are sufficient to reach at least
90% coverage:

1. Replace the **bottleneck** no-motion fixture with a motion-rich alternative.
2. Replace the **crossing_proxy** no-motion fixture with a motion-rich alternative.
3. Extend the **goal_directed_crossing/goal_directed_crossing_fixture** trace.
4. Extend the **route_conflict_goal/route_conflict_goal_fixture** trace.
5. Extend the **signalized_crossing/signalized_crossing_semantic_metadata** trace.
6. Extend the **waiting_with_intent_change/waiting_intent_change_fixture** trace.
7. Extend the **occluded_emergence/deterministic_occluded_emergence** trace.

Estimated resulting coverage: **~93.9%**.

## Claim Boundary

This is a denominator-health diagnostic. It does not prove navigation benefit,
safety improvement, human realism, benchmark-strength predictor quality, or
closed-loop planner performance. Forecast defaults must not change from this
report alone.

## Reproducible Command

```bash
uv run python scripts/benchmark/build_horizon_timestep_denominator_report.py \
  --issue 2903 \
  --parent-issue 2837 \
  --generated-at-utc 2026-06-16T00:00:00+00:00 \
  --output-dir docs/context/evidence/issue_2903_horizon_denominator_health_2026-06-16
```

## Validation

```bash
uv run pytest tests/benchmark/test_horizon_timestep_denominator.py
uv run ruff check scripts/benchmark/build_horizon_timestep_denominator_report.py
uv run ruff format scripts/benchmark/build_horizon_timestep_denominator_report.py
```
