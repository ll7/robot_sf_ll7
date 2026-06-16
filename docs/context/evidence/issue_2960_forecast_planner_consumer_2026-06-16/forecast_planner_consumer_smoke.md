# Issue #2960 Forecast Planner Consumer Smoke

Smoke evidence that forecast_variant can feed a real PredictionPlannerAdapter consumer. This is not nominal or benchmark evidence and does not claim any forecast variant improves safety, success, or runtime.

## Fixture

- Seed: 2960
- Scenario: deterministic_motion_rich_socnav_smoke
- Same seed across variants: True

## Variant Results

| variant | class | mode | collision | near miss | min distance | progress | stop steps | false-positive stops | runtime | changed vs none |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | native | native | False | True | 0.557 | 1.125 | 0 | 0 | 0.007005 | False |
| cv | degraded | native | False | True | 0.557 | 1.125 | 0 | 0 | 0.007928 | False |
| semantic | degraded | native | False | True | 0.557 | 1.125 | 0 | 0 | 0.007727 | False |
| interaction_aware | native | native | False | True | 0.512 | 1.125 | 0 | 0 | 0.008204 | True |
| risk_filtered | degraded | native | False | True | 0.557 | 1.125 | 0 | 0 | 0.007846 | False |

## Limitations
- Single deterministic SocNav observation; not a statistically powered benchmark.
- Planner actions are local one-step commands, not full episode success evidence.
- Rows classify planner-consumption mechanics, not forecast quality or navigation benefit.
