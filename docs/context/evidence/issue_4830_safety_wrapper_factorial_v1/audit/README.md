# Issue #4830 safety-wrapper campaign evidence audit

This audit validates campaign execution and records the boundary between the camera-ready artifact surface and the issue #3501 normalized paired-row report. It does not infer missing metric semantics, promote evidence, or make a safety claim.

- Standard campaign status: `valid`
- Campaign execution: `completed`
- Evidence status: `valid`
- Public commit: `8b5ce0b1ca5b7845ae05b0fdc07761079c75d380`
- Episodes: `864`
- Arms: `6`
- Observed episode records: `864`
- Paired-row contract: `blocked`

## Paired-row gate

The existing issue #3501 report builder requires normalized `metric_values` for every `(planner, scenario_id, seed, wrapper_arm)` row. The camera-ready episode records do not contain that object. Similar fields are listed in `summary.json` as source presence only; this audit does not reinterpret them.

Blocked required metrics:

- `exact_collision_probability`
- `near_miss_probability`
- `min_predicted_separation_m`
- `completion_probability`
- `progress_at_timeout`
- `false_positive_stop_rate`
- `stop_yield_latency_s`
- `wrapper_intervention_rate`

This is an artifact-contract stop, not a failed campaign run. The standard camera-ready campaign artifacts remain separate from any dissertation claim.
