# Issue #2937 Horizon x Timestep Denominator Health Fixture Repair

## Scope

This analysis-only bundle repairs the seven fixture gaps identified in
[issue #2903](../../archive/issue_2903_horizon_denominator_health.md) and regenerates the
denominator-health audit for the horizon x timestep ablation from issue #2837 /
PR #2887. It classifies each missing `(horizon, dt_s, trace)` cell by reason,
verifies the category totals sum to the expected 180-cell matrix, spot-checks
representative missing cells, and reports the remaining fixture-specific
blockers.

## Evidence status

- `schema`: `HorizonTimestepDenominatorHealth.v1`
- `claim_boundary`: `analysis_only_not_navigation_evidence`
- `forecast_defaults_unchanged`: `true`
- `reason`: the report only diagnoses why ablation cells are unevaluable on the
  repaired durable fixture set. It does not change forecast defaults, simulator
  physics, or claim navigation / closed-loop benefit.

## Files

- [denominator_report.json](denominator_report.json): full denominator-health data, category totals,
  matrix coverage, per-family missingness, spot checks, and fixture proposal
- [denominator_report.md](denominator_report.md): human-readable Markdown report
- [summary.json](summary.json): compact issue coverage summary

## Fixture changes (issue #2937)

The following durable fixtures were added or extended; the original fixtures
were left untouched so existing tests keep their exact invariants:

- `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/bottleneck_motion_rich_fixture.json`
- `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/crossing_proxy_motion_rich_fixture.json`
- `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/issue_2868_goal_directed_crossing_fixture_extended.json`
- `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/issue_2868_route_conflict_goal_fixture_extended.json`
- `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/issue_2868_signalized_crossing_fixture_extended.json`
- `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/issue_2868_waiting_intent_change_fixture_extended.json`
- `tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/occluded_emergence_episode_extended.json`

The ablation report's `TRACE_CANDIDATES` now point at these repaired fixtures.

## Remaining blockers

The `corridor_interaction` default/AMMV Social Force fixtures remain short for
long-horizon / fine-dt combinations and are reported as `trace_too_short`. They
were not part of the seven proposed fixture gaps and are left as documented
remaining work.

## Reproducible command

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

## Claim boundary

This report is useful for planning fixture additions to improve ablation
denominator coverage. It does not prove navigation improvement, safety, human
realism, benchmark-strength predictor quality, or closed-loop planner benefit.
Forecast defaults must not change based on this report alone.
