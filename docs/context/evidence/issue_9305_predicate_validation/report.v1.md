# Trace Predicate Validation Report

- Evaluation set: `issue_9305_bounded_predicate_validation` v1
- Evidence status: `diagnostic-only`; retained traces: `unavailable`.
- Source commit: `b8811949e087fabae6c6b57656bd7090d92e43ac`; set SHA-256: `6cc0a06c454142dd7895bac5a68b573dbb30423017608659b4af4e7740d8be71`.
- Claim boundary: Diagnostic-only contract and fixture mechanics; no retained-production precision/recall, grouping, or causal claim.

## Coverage

| field | value |
| --- | ---: |
| `case_count` | 4 |
| `trace_available_count` | 3 |
| `trace_unavailable_count` | 1 |
| `scenario_family_count` | 4 |
| `planner_count` | 3 |
| `map_count` | 3 |
| `seed_count` | 2 |
| `reference_positive_label_count` | 8 |
| `reference_negative_label_count` | 8 |
| `reference_ambiguous_label_count` | 0 |
| `reference_unavailable_or_pending_label_count` | 16 |

## Per-predicate metrics

| predicate | evaluated | TP | TN | FP | FN | precision | recall | unavailable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `collision` | 2 | 1 | 0 | 1 | 0 | 0.500 | 1.000 | 0.250 |
| `late_evasive_reaction` | 2 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 0.250 |
| `oscillatory_local_control` | 2 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 0.250 |
| `occlusion_triggered_near_miss` | 2 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 0.250 |
| `bottleneck_deadlock` | 2 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 0.250 |
| `zero_motion_timeout_behavior` | 2 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 0.250 |
| `low_progress` | 2 | 0 | 1 | 0 | 1 | not available | 0.000 | 0.250 |
| `clearance_critical_interaction` | 2 | 1 | 1 | 0 | 0 | 1.000 | 1.000 | 0.250 |

## Reviewer agreement and effort

| predicate | reviewer pairs | agreement | disagreements | unresolved |
| --- | ---: | ---: | ---: | ---: |
| `collision` | 4 | 1.000 | 0 | 0 |
| `late_evasive_reaction` | 4 | 1.000 | 0 | 0 |
| `oscillatory_local_control` | 4 | 1.000 | 0 | 0 |
| `occlusion_triggered_near_miss` | 4 | 1.000 | 0 | 0 |
| `bottleneck_deadlock` | 4 | 0.750 | 1 | 1 |
| `zero_motion_timeout_behavior` | 4 | 1.000 | 0 | 0 |
| `low_progress` | 4 | 1.000 | 0 | 0 |
| `clearance_critical_interaction` | 4 | 1.000 | 0 | 0 |

Reviewer effort: `8.00` minutes across 2 reviewers and 1 adjudicators.

## Threshold and grouping stability

Threshold variants: `2`; stability rows are label comparisons to the declared baseline.
Grouping identity features: `planner_id, map_id`; same-group pair agreement after ablation: `0.833`; pairwise Jaccard: `0.000`.

## Limitations

- Retained production traces are unavailable; available references are bounded tracked fixtures.
- Fixture detector and adjudication labels are contract plumbing, not an empirical accuracy or prevalence estimate.
- Pending, ambiguous, and unavailable labels are excluded from confusion-matrix metrics; no majority-vote imputation is performed.
- Reviewer agreement is pairwise label agreement, not a reliability or generalization claim; effort is fixture metadata.
- Planner and map identity are recorded separately; map IDs in this fixture are annotations rather than fields in the trace export.
- Grouping assignments are supplied under the collision-similarity schema; this contract runs no campaign or similarity rerun.
- Observed pattern annotations are separate from causal hypotheses, and neither is used to infer mechanism from correlation.
