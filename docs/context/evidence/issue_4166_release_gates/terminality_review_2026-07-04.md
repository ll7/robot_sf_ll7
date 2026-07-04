# Issue #4166 Terminality Review

Issue: [#4166](https://github.com/ll7/robot_sf_ll7/issues/4166) — paired safety and comfort
release-gate reporting.

This review checks whether issue #4166 is terminal after the merged reporting, current-roster,
and evaluability-guard pull requests (PRs). It is a consolidation note only: no benchmark campaign
was run, no historical packet was backfilled, no thresholds were changed, and no planner safety or
release claim is promoted.

## Decision

Issue #4166 is **not terminal yet**.

The reporting contract and future-campaign evaluability path are complete. The retained
current-roster evidence packet still comes from the 2026-05-04 campaign summary, which predates
the new retained fields. The live issue thread explicitly keeps #4166 open until a fresh
current-roster campaign populates `min_clearance_m` and `proxemic_intrusion_rate`.

## Completed

- PR [#4184](https://github.com/ll7/robot_sf_ll7/pull/4184) added the release-gate reporting
  contract: YAML gate specification, fail-closed evaluator, command-line report builder, default
  provisional gate set, fixture report packet, and tests.
- PR [#4313](https://github.com/ll7/robot_sf_ll7/pull/4313) applied that reporter to the
  retained current-roster campaign summary and recorded the current packet under
  `docs/context/evidence/issue_4166_release_gates/current_roster/`.
- PR [#4334](https://github.com/ll7/robot_sf_ll7/pull/4334) made future camera-ready campaign
  summaries retain `min_clearance_m` and `proxemic_intrusion_rate` with fail-closed behavior for
  missing source values.
- PR [#4331](https://github.com/ll7/robot_sf_ll7/pull/4331) retained the evaluator guard that
  proves rows with those two fields become evaluable, while legacy rows stay `not_evaluable`.

## Remaining Blocker

The committed current-roster packet still reports these coverage-gap gates as not evaluable for
the retained 2026-05-04 campaign:

- `min_clearance_floor_coverage_gap` over `min_clearance_m`
- `proxemic_intrusion_rate_coverage_gap` over `proxemic_intrusion_rate`

That is expected and correct for the old packet. PR #4334 changed future campaign-summary
retention; it did not backfill the older campaign or run a fresh campaign. Closing #4166 would
therefore overstate the live evidence.

## Next Empirical Action

Run a fresh current-roster campaign after PR #4334, then rebuild the issue #4166 current-roster
release-gate packet with:

```bash
uv run python scripts/benchmark/build_release_gate_report.py \
  --input-json <fresh-campaign-summary.json> \
  --gate-spec configs/benchmarks/release_gates/camera_ready_current_roster_gates.yaml \
  --output-json docs/context/evidence/issue_4166_release_gates/current_roster/summary.json \
  --output-csv docs/context/evidence/issue_4166_release_gates/current_roster/gate_matrix.csv \
  --output-md docs/context/evidence/issue_4166_release_gates/current_roster/README.md
```

If the fresh packet contains evaluable `min_clearance_m` and `proxemic_intrusion_rate` gate rows,
then #4166 can be closed as reporting-complete without changing metric semantics or thresholds.
