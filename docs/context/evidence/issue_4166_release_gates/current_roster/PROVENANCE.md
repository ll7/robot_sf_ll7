# Current-Roster Release-Gate Evidence — Provenance & Interpretation

Issue: [#4166](https://github.com/ll7/robot_sf_ll7/issues/4166) — paired safety+comfort
release-gate reporting (pass/fail matrix).

This packet is the **real current-roster** application of the release-gate reporting
layer merged in PR #4184. PR #4184 shipped the evaluator, CLI, default gate spec, and a
*fixture* packet; its stated remaining blocker was that "a real campaign summary must be
supplied later to produce release-gate outcomes over nominal benchmark evidence." This
packet supplies that: it runs the merged evaluator over an already-retained benchmark
campaign summary for the current planner roster.

## Claim boundary

- **Reporting only.** This is a pass/fail *rendering* of existing benchmark rows against
  **provisional** thresholds. It is **not** certification, threshold approval, paper-grade
  evidence, planner promotion, or a planner ranking.
- The pass/fail cells are the output of the reporting mechanism over provisional
  thresholds. A "fail" means a planner exceeded a provisional reporting budget, **not**
  that it is unsafe; a "pass" does not imply release approval.
- No new benchmark run, Slurm/GPU job, or metric recomputation was performed. Rows are
  read verbatim from the retained summary below.

## Source

- Input rows: `docs/context/evidence/camera_ready_all_planners_2026-05-04/reports/campaign_summary.json`
  (container key `planner_rows`; 8 planners; campaign date 2026-05-04).
- Gate spec: `configs/benchmarks/release_gates/camera_ready_current_roster_gates.yaml`.
- The retained `planner_rows` are **campaign-aggregated across the scenario matrix**, so
  the scenario-family dimension renders as `all` (campaign aggregate). Per-family
  granularity is available in the campaign's `scenario_family_breakdown` artifacts and is
  left as future work; it is not required by the issue's "planner roster" acceptance.

## Metric mapping and coverage gap

The provisional gate metric names in the *default* spot-check spec
(`default_safety_comfort_gates.yaml`: `collision_rate`, `min_clearance_m`,
`proxemic_intrusion_rate`) do **not** all exist in the retained camera-ready campaign.
This spec targets the field names the campaign actually records, and flags the missing
ones as fail-closed coverage gaps:

| Gate metric (this spec) | Category | Recorded by camera-ready? | Note |
| --- | --- | --- | --- |
| `collisions_mean` | safety | yes | per-episode mean count, **not** a certified rate |
| `near_misses_mean` | safety | yes | per-episode mean count, **not** a certified rate |
| `min_clearance_m` | safety | **no** | `required: false` coverage gap → `not_evaluable` |
| `jerk_mean` | comfort | yes | direct comfort proxy |
| `comfort_exposure_mean` | comfort | yes | closest recorded proxemic-exposure proxy |
| `proxemic_intrusion_rate` | comfort | **no** | `required: false` coverage gap → `not_evaluable` |

The two coverage-gap gates are marked `required: false` so they surface the missing metric
as `not_evaluable` in the detail rows **without** forcing the whole category to
`not_evaluable`. This keeps the real collision/near-miss/jerk/exposure signal visible while
honestly recording that dedicated min-clearance and proxemic-intrusion-rate metrics are not
yet part of this campaign's summary.

### Forward retention (issue [#4326](https://github.com/ll7/robot_sf_ll7/issues/4326))

The camera-ready campaign summary/retention schema now records both `min_clearance_m` and
`proxemic_intrusion_rate` per planner row, aggregated from per-episode values that already
exist in episode rows (`min_clearance` → campaign-wide **worst-case minimum** clearance,
distinct from the mean-of-per-episode-minimums kept as `min_clearance_mean`;
`social_proxemic_intrusion_frac` → mean per-episode personal-space intrusion fraction). So
**future** campaigns make these two gates evaluable with **no evaluator code change** — the
gate spec already targets these exact field names.

This is **not** a backfill. This retained campaign (dated 2026-05-04, produced before the
schema change) does **not** carry the fields, so both gates correctly stay `not_evaluable`
here — fail-closed, no fabricated historical values. Regenerating this packet does not
change these two cells; only a **newly run** campaign under the updated schema will populate
them.

## Fail-closed handling of the degraded planner

`socnav_bench` ran degraded in this campaign (`status: failed`, `benchmark_success: false`,
`readiness_status: degraded`, 0 episodes, all metrics `nan`). Per the repository fail-closed
benchmark policy, its metrics are not treated as evidence: every gate renders
`not_evaluable`, so its safety, comfort, and overall status are all `not_evaluable`. This is
correct fail-closed behavior, not a reporting defect.

## Result summary (provisional thresholds)

`status_counts`: `pass: 2`, `fail: 5`, `not_evaluable: 1` (8 planners).

- `pass` (overall): `orca`, `ppo` — within all provisional required budgets.
- `fail` (overall): `goal`, `prediction_planner`, `sacadrl`, `social_force`,
  `socnav_sampling` — exceeded a provisional safety budget (all 7 evaluable planners pass
  the provisional comfort budgets).
- `not_evaluable` (overall): `socnav_bench` — degraded run, fail-closed.

## Reproduce

```bash
uv run python scripts/benchmark/build_release_gate_report.py \
  --input-json docs/context/evidence/camera_ready_all_planners_2026-05-04/reports/campaign_summary.json \
  --gate-spec configs/benchmarks/release_gates/camera_ready_current_roster_gates.yaml \
  --output-json docs/context/evidence/issue_4166_release_gates/current_roster/summary.json \
  --output-csv docs/context/evidence/issue_4166_release_gates/current_roster/gate_matrix.csv \
  --output-md docs/context/evidence/issue_4166_release_gates/current_roster/README.md \
  --generated-at-utc 2026-07-03T00:00:00+00:00
```

Changing a threshold in the YAML changes the matrix with **no code change** — the acceptance
requirement that thresholds are config, not code.
