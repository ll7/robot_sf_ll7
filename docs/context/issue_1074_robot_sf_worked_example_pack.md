# Issue #1074 Robot-SF Worked-Example Pack

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1074>

Operating envelope follow-up: <https://github.com/ll7/robot_sf_ll7/issues/1075>

H500 mechanism pilot: [Issue #1049 H500 Mechanism Pilot](issue_1049_h500_mechanism_pilot.md)

H500 classification vocabulary:
[Issue #1056 H500 Failure Classification](issue_1056_h500_failure_classification.md)

Exposure-aware tables:
[Issue #1055 Exposure-Aware H500 Tables](issue_1055_exposure_aware_h500_tables.md)

## Purpose

This pack gives dissertation-floor writing a small, concrete set of Robot-SF worked examples. The
examples illustrate framework dimensions and recurring failure-pattern categories without claiming
full empirical generality.

The pack deliberately reuses retained h500 evidence from #1049/#1055/#1056 instead of launching a
new campaign. It is therefore a worked-example pack, not a benchmark ranking, planner promotion, or
statistical study.

## Selection Rule

Examples were selected to satisfy four constraints:

- small pack size: three examples,
- distinct scenario/failure-pattern categories,
- retained durable evidence under `docs/context/evidence/`,
- explicit claim and non-claim boundaries for each example.

## Summary Table

| Example | Scenario class | Actor mix | Metric layer | Failure-pattern tag | Evidence source |
| --- | --- | --- | --- | --- | --- |
| Clean route-budget relief | Bottleneck / route-completion | robot-only trace stream | success, steps, collision, exposure | `time_budget_clean_relief` | #1049 trace summary, #1055 tables |
| Exposure-enabled completion | T-intersection / pedestrian interaction | robot + pedestrian | success, steps, force exposure, comfort exposure, min pedestrian distance | `exposure_enabled_completion` | #1049 trace summary, #1055 tables |
| Safety-regressed long horizon | Merging / pedestrian interaction | robot + pedestrian | collision, steps, force exposure, comfort exposure | `safety_regressed_long_horizon` | #1049 trace summary, #1055 tables |

## Worked Examples

### 1. Clean Route-Budget Relief

- Scenario: `classic_bottleneck_low`
- Planner/seed: ORCA, seed `111`
- Evidence pointer:
  `docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/representative_trace_summary.csv`
- Reporting pointer:
  `docs/context/evidence/issue_1055_exposure_aware_h500_tables_2026-05-07/fixed_vs_h500_outcome_table.csv`
- Scenario class: bottleneck / route-completion.
- Actor mix: robot-only trace stream in the retained summary.
- Metric layer: success, episode steps, collision flag, near-miss/force/comfort exposure.
- Failure-pattern tag: `time_budget_clean_relief`.
- What it supports:
  fixed h100 can hide a route/time-budget artifact when h500 completes cleanly without collision or
  recorded interaction exposure.
- What it does not support:
  it does not prove a general planner improvement, a waiting/yielding strategy, or external-world
  validity.

### 2. Exposure-Enabled Completion

- Scenario: `classic_t_intersection_medium`
- Planner/seed: ORCA, seed `111`
- Evidence pointer:
  `docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/representative_trace_summary.csv`
- Reporting pointer:
  `docs/context/evidence/issue_1055_exposure_aware_h500_tables_2026-05-07/exposure_aware_trace_table.csv`
- Scenario class: T-intersection / pedestrian interaction.
- Actor mix: robot plus pedestrian.
- Metric layer: success, episode steps, force-exposure steps, comfort-exposure sum, minimum
  robot-pedestrian distance.
- Failure-pattern tag: `exposure_enabled_completion`.
- What it supports:
  h500 can convert a fixed-horizon timeout into completion while increasing interaction exposure and
  reducing pedestrian clearance.
- What it does not support:
  it does not support reporting h500 as a pure success gain, and it is not a discrete near-miss
  timing example because the retained `near_misses` counter remains zero.

### 3. Safety-Regressed Long Horizon

- Scenario: `classic_merging_low`
- Planner/seed: ORCA, seed `111`
- Evidence pointer:
  `docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/representative_trace_summary.csv`
- Reporting pointer:
  `docs/context/evidence/issue_1055_exposure_aware_h500_tables_2026-05-07/exposure_aware_trace_table.csv`
- Scenario class: merging / pedestrian interaction.
- Actor mix: robot plus pedestrian.
- Metric layer: collision, episode steps, force exposure, comfort exposure.
- Failure-pattern tag: `safety_regressed_long_horizon`.
- What it supports:
  a longer horizon can reveal unsafe behavior that a fixed horizon would report only as an
  unfinished episode.
- What it does not support:
  it does not prove the scenario is invalid or that ORCA is globally unsafe; recurrence and
  scenario-certification review are required before planner or scenario follow-up claims.

## Reproducible Pointers

The retained evidence can be inspected without rerunning a campaign:

```bash
rtk column -s, -t docs/context/evidence/issue_1055_exposure_aware_h500_tables_2026-05-07/fixed_vs_h500_outcome_table.csv
rtk column -s, -t docs/context/evidence/issue_1055_exposure_aware_h500_tables_2026-05-07/exposure_aware_trace_table.csv
rtk column -s, -t docs/context/evidence/issue_1049_h500_mechanism_pilot_2026-05-07/representative_trace_summary.csv
```

The original trace runner command shape is recorded in
`docs/context/issue_1049_h500_mechanism_pilot.md`. It uses ignored `output/` intermediates, while
the compact trace summaries and exposure tables above are the durable retained evidence for this
worked-example pack.

## Claim Boundary

This pack supports dissertation-floor illustration of a framework workflow:

1. choose a bounded scenario/example,
2. attach metric layers,
3. classify the failure or completion pattern,
4. state the decision relevance,
5. state the non-claim.

It does not support:

- full statistical comparison,
- planner promotion,
- physical AMV validity,
- CARLA transfer or simulator parity,
- OOD or unseen-environment generalization,
- replacing the fixed-horizon paper benchmark with h500.

## Validation

Checked on 2026-05-09:

- Read #1074 in full, including comments.
- Reused retained #1049/#1055/#1056 evidence instead of generating new outputs.
- Confirmed the selected evidence files exist under `docs/context/evidence/`.
- Confirmed the issue-referenced external source path
  `ll7/diss:docs/superpowers/specs/2026-05-08-dissertation-narrative-framework-thesis-floor-design.md`
  was not present in this checkout.

Validation commands for this PR should include:

```bash
rtk rg -n "classic_bottleneck_low|classic_t_intersection_medium|classic_merging_low|time_budget_clean_relief|exposure_enabled_completion|safety_regressed_long_horizon" docs/context/issue_1074_robot_sf_worked_example_pack.md docs/context/evidence
rtk git diff --check
```
