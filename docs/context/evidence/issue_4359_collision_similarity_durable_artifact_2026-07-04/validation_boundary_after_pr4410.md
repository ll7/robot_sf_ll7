# Issue #4359 Validation Boundary After PR #4410

Plain-language status: collision-scenario similarity is available as a diagnostic
analysis aid on tracked evidence, but the durable issue #1470 artifact still does
not contain the raw per-step trajectory arrays needed for replay-level validation.

## Integrated Slice

| Prior slice | Durable result | Boundary |
| --- | --- | --- |
| PR #4371 | Added the `collision-scenario-similarity` command and `collision_scenario_similarity.v1` report shape with deterministic fixture coverage. | The report is an analysis aid only; it does not change planner, benchmark, metric, or ranking semantics. |
| PR #4386 | Applied the report to a durable issue #1470 artifact and added descriptive validation blocks for external labels and trajectory fields. | External labels and trajectory fields are validation context, not benchmark truth. Missing trajectory fields are reported as unavailable instead of imputed. |
| PR #4410 | Refreshed the durable evidence bundle and added `trajectory_metric_fields` availability for the issue #1470 report. | Trajectory-derived metric fields are diagnostic context only and do not substitute for raw per-step trajectory arrays. |

## Current Contract

- `collision_scenario_similarity.v1` may be cited as diagnostic evidence for
  comparing descriptor-level similarity among selected unsafe or near-unsafe
  records in the input artifact.
- The durable issue #1470 report records `trajectory_metric_fields.status:
  available` and `trajectory_fields.status: unavailable`. That combination means
  summary metric fields are present, while raw replay arrays are not present in
  the tracked source artifact.
- The report must fail open for optional validation context: unavailable raw
  trajectory fields are disclosed in the report and must not be fabricated from
  summary metrics.
- The report must not be used as planner-ranking evidence, paper evidence,
  dissertation evidence, or a substitute benchmark metric without a separate
  validation source.

## Remaining Blockers

| Blocker | Status | Next empirical action |
| --- | --- | --- |
| Raw per-step trajectory-array replay validation | Blocked on durable source. The durable issue #1470 artifact used by `durable_issue1470_similarity_report.*` has zero records with raw trajectory fields for the selected collision-similarity rows. | Identify or publish a durable artifact that contains the original per-step robot and pedestrian trajectory arrays for the same records, then rerun `collision-scenario-similarity` against that source and compare descriptor groups against replay-derived trajectory features. |
| External-label calibration beyond tracked fixture alignment | Diagnostic only. The tracked labeled fixture proves the report can surface labels and raw trajectory fields when present, but it is not a representative calibration set. | Use a representative labeled source only if one is made durable with provenance; otherwise keep label checks descriptive. |

## Explicit Non-Claims

- No full benchmark campaign was run for this boundary record.
- No Slurm, GPU, or compute submission was performed.
- No planner ranking, leaderboard, paper, dissertation, or safety claim is
  changed by this evidence bundle.
- No raw trajectory arrays were reconstructed from summaries.
