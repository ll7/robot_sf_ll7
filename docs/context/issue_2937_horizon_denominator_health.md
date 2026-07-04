# Issue #2937: Prediction - Repair Horizon/Timestep Denominator Coverage 2026-06-16

## Summary

This note records the fixture repair and regenerated denominator-health audit
for the horizon x timestep ablation introduced in issue #2837 / PR #2887. It
builds on the gap analysis from [issue #2903](archive/issue_2903_horizon_denominator_health.md)
and is analysis-only: it does not change forecast defaults.

## Evidence

- Compact summary: [`docs/context/evidence/issue_2937_horizon_denominator_health_2026-06-16/summary.json`](evidence/issue_2937_horizon_denominator_health_2026-06-16/summary.json)
- Full report: [`docs/context/evidence/issue_2937_horizon_denominator_health_2026-06-16/denominator_report.json`](evidence/issue_2937_horizon_denominator_health_2026-06-16/denominator_report.json)
- Human-readable report: [`docs/context/evidence/issue_2937_horizon_denominator_health_2026-06-16/denominator_report.md`](evidence/issue_2937_horizon_denominator_health_2026-06-16/denominator_report.md)

## Key Findings

- Expected matrix size: **180 cells** (5 horizons x 4 timesteps x 9 trace families).
- After repairing the seven proposed fixture gaps: **164 cells (91.1%)** evaluated.
- Remaining missingness is `trace_too_short` only, concentrated in the
  `corridor_interaction` default/AMMV Social Force fixtures that were not part
  of the seven proposed gaps.
- `no_pedestrian_motion`, `metadata_missing`, `actor_class_missing`,
  `observation_tier_missing`, and `other_explicit_reason`: **0 cells** each.
- Category totals sum to the expected 180 cells.

## Fixture Changes

The seven gaps from issue #2903 were addressed with new or extended fixtures;
original fixtures were left untouched so existing tests keep their invariants:

1. Replaced `bottleneck/minimal_fixture` with a motion-rich alternative.
2. Replaced `crossing_proxy/synthetic_crossing_proxy_orca` with a motion-rich alternative.
3. Extended `goal_directed_crossing/goal_directed_crossing_fixture`.
4. Extended `route_conflict_goal/route_conflict_goal_fixture`.
5. Extended `signalized_crossing/signalized_crossing_semantic_metadata`.
6. Extended `waiting_with_intent_change/waiting_intent_change_fixture`.
7. Extended `occluded_emergence/deterministic_occluded_emergence`.

The new fixtures live under
`tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/` and
are referenced from `scripts/benchmark/build_horizon_timestep_ablation_report.py`.

## Remaining Blockers

- `corridor_interaction/ammv_social_force` and
  `corridor_interaction/default_social_force` remain `trace_too_short` for some
  long-horizon / fine-dt combinations. They were not in the seven proposed gaps
  and are left as explicitly documented remaining work.

## Claim Boundary

This is a denominator-health diagnostic. It does not prove navigation benefit,
safety improvement, human realism, benchmark-strength predictor quality, or
closed-loop planner performance. Forecast defaults must not change from this
report alone.

## Reproducible Command

```bash
uv run python scripts/benchmark/build_horizon_timestep_denominator_report.py \
  --issue 2937 \
  --parent-issue 2837 \
  --generated-at-utc 2026-06-16T00:00:00+00:00 \
  --output-dir docs/context/evidence/issue_2937_horizon_denominator_health_2026-06-16
```

## Validation

```bash
uv run pytest tests/benchmark/test_horizon_timestep_denominator.py
uv run ruff check scripts/benchmark/build_horizon_timestep_ablation_report.py
uv run ruff check scripts/benchmark/build_horizon_timestep_denominator_report.py
uv run ruff check tests/benchmark/test_horizon_timestep_denominator.py
uv run ruff format scripts/benchmark/build_horizon_timestep_ablation_report.py
uv run ruff format scripts/benchmark/build_horizon_timestep_denominator_report.py
uv run ruff format tests/benchmark/test_horizon_timestep_denominator.py
```
