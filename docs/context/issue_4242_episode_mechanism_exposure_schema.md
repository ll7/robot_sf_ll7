# Issue #4242 Episode Mechanism And Exposure Schema

Issue #4242 adds native episode-row fields for mechanism labels and interaction exposure. The
fields are additive: older rows remain readable, but current benchmark writers should emit explicit
unknown or not-derivable values instead of omitting the columns.

## Failure-Mechanism Taxonomy

Rows and sidecars use `mechanism_schema_version: failure_mechanism_taxonomy.v1` with these compact
fields:

- `mechanism_label`: taxonomy label from the issue #2220 vocabulary, or `unknown`.
- `mechanism_confidence`: `observed_mechanism`, `supported_hypothesis`, `weak_hypothesis`, or
  `unknown`.
- `mechanism_evidence_mode`: `paired_trace`, `deterministic_replay`, `direct_probe`, `root_cause`,
  `aggregate_summary`, or `unknown`.
- `mechanism_evidence_uri`: tracked evidence or context URI, empty only when unknown.
- `mechanism_case_id`: optional stable case identifier.
- `mechanism_caveat`: free-text caveat, required unless confidence is `observed_mechanism`.

Geometry buckets derived from scenario names are comparison-only metadata. They must not populate
`mechanism_label`, and builders must fail closed when geometry-only labels are the only available
mechanism source.

## Interaction Exposure

Rows and sidecars use `interaction_exposure_schema_version: interaction_exposure.v1`. The minimum
fields consumed by the h600 exposure diagnostics are:

- `interaction_exposure_share`: fraction in `[0, 1]`.
- `robot_motion_share_before_first_clearance`: fraction in `[0, 1]`.
- `first_clearance_step`: first clearance step, or empty when not derivable.
- `low_exposure_success`: true only when the episode succeeded and exposure is below the
  predeclared diagnostic threshold.

Provenance fields should describe how the denominator was computed:

- `interaction_exposure_status`: `computed`, `not_derivable_missing_trace`,
  `not_derivable_no_pedestrians`, `not_applicable`, or `malformed`.
- `interaction_exposure_source`: producer for the flattened fields.
- `interaction_exposure_radius_m`, `interaction_exposure_steps`,
  `interaction_exposure_denominator_steps`, `robot_motion_steps_before_first_clearance`,
  `robot_motion_denominator_steps_before_first_clearance`, and `first_clearance_reason` when
  available.

Missing trace support records `not_derivable_missing_trace`; it does not write zeros or infer
episode-level exposure from aggregate planner metrics.

## Write-Time Instrumentation (map runner)

The map-runner episode writer (`robot_sf/benchmark/map_runner_episode.run_map_episode`) now emits
both schema blocks natively on every episode record it returns (which is what is persisted to
`episodes.jsonl`), closing acceptance criterion #2:

- `record["failure_mechanism"]`: always the fail-closed `unknown` block from
  `unknown_failure_mechanism_record`. A single map-runner episode is not a paired-trace mechanism
  analysis, so no trace-verified label is asserted and geometry or scenario names are never
  substituted. A trace-verified label must come from the mechanism cross-cut path (#4206), not this
  writer.
- `record["interaction_exposure"]`: computed by `compute_interaction_exposure_fields` from the
  episode's own recorded robot/pedestrian trajectory (its real trace, not imputation). When the
  trajectory support is absent (empty trace or no pedestrians), the writer emits the explicit
  `not_derivable_*` block instead of fabricated zeros.

The diagnostic exposure radius defaults to `2.0` m, mirroring the proxemic near/far split used by
`experimental_ped_impact_metrics`, and the low-exposure-success threshold defaults to `0.2`. Both
values are recorded on each row so downstream tooling can override or re-derive them.

These fields are diagnostic-only. This writer does not promote them into F-C4(ii) ranking
conclusions. Teaching the seed-episode-row CSV builder and the #4195/#4206 builders to consume the
native blocks (the other half of AC #5) remains a follow-up slice; the flattened
`seed_episode_rows.csv` consumer still emits its own fail-closed columns until then.
