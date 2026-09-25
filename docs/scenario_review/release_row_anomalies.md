# Release-row anomaly gate

The release-row gate is the row-level companion to the Benchmark Auditor. It
reads the episode summaries that were published with a release bundle. It does
not start a simulator, replay an episode, or require per-step traces. It emits
diagnostic signals for review; a signal does not establish planner causation.

## Bundle source and row contract

The loader admits a publication bundle only after checking
`publication_manifest.json` (`schema_version` `benchmark-publication-bundle.v2`)
and the `size_bytes` and SHA-256 digest for every episode member. The relevant
source mapping is:

| Published source | Loaded value |
| --- | --- |
| `payload/runs/<arm>__differential_drive/episodes.jsonl` | One row per episode; `<arm>` becomes `_release_arm` / the planner identity |
| Episode member path | `_source_member` on each loaded row |
| Archive bytes | `source.bundle_sha256` (available for an archive; an extracted directory has no archive digest) |
| Manifest bytes | `source.manifest_sha256` |
| Verified episode member set | `source.episode_members` |

The aggregate `payload/reports/seed_episode_rows.csv` may accompany the
publication, but the gate's input is the checked episode JSONL rows. Each row
must contain these fields:

```json
{
  "episode_id": "unique-in-its-episode-member",
  "scenario_id": "scenario-name",
  "seed": 112,
  "algo": "planner-name",
  "steps": 2,
  "outcome": {
    "route_complete": false,
    "collision_event": true,
    "timeout_event": false
  },
  "event_ledger": {
    "exact_events": {"invalid_run": false}
  },
  "integrity": {
    "effective_view": {"observation_ped_count": 0}
  },
  "metrics": {}
}
```

`scenario_id`, `seed`, the planner identity (`algo` or `planner_id`),
`steps`, and all three `outcome` values are required. The canonical outcome
keys are `route_complete`, `collision_event`, and `timeout_event`; the loader
also requires a non-empty, member-unique `episode_id`. `metrics` must be an
object. `event_ledger.exact_events.invalid_run` is used when present for
preflight parity. Optional measurements remain unavailable when absent.

Pedestrian-free baseline comparisons use
`integrity.effective_view.observation_ped_count` from both paired rows. A row
is eligible only when this integer is exactly zero, even if its scenario ID is
listed in `pedestrian_free_scenarios`. A positive count excludes that pair.

## Detectors

The configured detectors are:

- `same_step_all_planners`: every sufficiently populated planner cell fails at
  the same terminal step at or below `same_step_max_steps`;
- `short_collision`: a row ends in `collision_event` at or below
  `short_collision_max_steps`;
- `impossible_contact_speed`: a collision contact speed from
  `event_ledger.collision_events[].relative_speed_at_contact`, or the fallback
  `metrics.max_relative_contact_speed_m_s`, exceeds
  `max_contact_speed_m_s`;
- `orbit_zero_progress`: a row has high path curvature, low displacement,
  low progress ratio, or an explicit deadlock flag. Curvature comes from
  `metrics.curvature_mean`; path length from
  `metrics.socnavbench_path_length`; displacement from the first available
  `metrics.robot_displacement_m`, `metrics.net_displacement_m`, or
  `metrics.displacement_m`; and the deadlock flag is `metrics.deadlock`;
- `pedestrian_free_baseline_regression`: a configured pedestrian-aware planner
  has a lower success rate than `baseline_planner` in paired rows whose
  effective observed pedestrian count is zero;
- `universal_failure_unannotated`: every expected planner fails a complete
  scenario-by-seed cell and no matching root-cause annotation exists;
- `invalid_run_preflight_mismatch`: the row's `invalid_run` value disagrees
  with the supplied preflight cell.

For the oscillation signature, the progress ratio is read at the exact path
`safety_predicates.oscillatory_control_predicate.fields.progress_ratio`. It is
used only when the value is a finite number in `[0, 1]`. Missing summary fields
are counted under `missingness`; the gate never fabricates a measurement from
an absent field.

## Configuration, annotations, and preflight

Thresholds and cohort policy come from the JSON configuration. The accepted
keys are:

`short_collision_max_steps`, `same_step_max_steps`,
`max_contact_speed_m_s`, `min_orbit_curvature`,
`min_orbit_path_length_m`, `max_zero_progress_m`, `max_progress_ratio`,
`min_planners_per_cell`, `min_paired_cells`, `min_success_rate_gap`,
`baseline_planner`, `pedestrian_free_scenarios`,
`max_unannotated_findings`, and `require_preflight`.

The preflight input is either a list or an object with schema version
`release-row-preflight.v1` and a `cells` list:

```json
{
  "schema_version": "release-row-preflight.v1",
  "cells": [
    {"scenario_id": "classic_station_platform_medium", "seed": 112, "invalid_run": false}
  ]
}
```

Each cell has a non-empty `scenario_id`, a non-negative integer `seed`, and a
Boolean `invalid_run`. Duplicate cells are rejected. Set `require_preflight`
to `true` when missing or incomplete parity evidence must block the gate.

Annotations are either a list or an object with schema version
`release-row-annotations.v1` and an `annotations` list. Every entry requires a
non-empty `root_cause` and `source_ref`, plus `scenario_id` or `finding_id` as
a selector. Optional selectors are `seed`, `planner_id`, and `detector_id`.
An annotation sets `finding.annotated` on matching detector findings without
rewriting the observed row. For `universal_failure_unannotated`, a matching
annotation removes the missing-root-cause finding and increments
`counts.annotated_universal_cells` instead; the cell's other detector findings
remain visible.

The release gate is evaluated once at report level. `gate.blocked` is true if
the report has more unannotated findings than `max_unannotated_findings`, has
incomplete planner cells, requires unavailable or incomplete preflight, or has
any `invalid_run_preflight_mismatch`. The parity reason remains blocking even
when that finding has an annotation. Detector findings and their annotation
state stay in the JSON report, while the Markdown report summarizes the same
report-level decision.

The JSON report also carries `detector_registry` and its
`detector_registry_digest`. Every entry in `signals` is a canonical BA-03
`signal` audit record, so consumers can validate and deserialize it with
`robot_sf.analysis_workbench.audit_contracts.record_from_dict`. Finding IDs
bind the source digests and detector-registry digest, so a threshold change
produces a new identity for an otherwise matching cell.

## Running the report

Run from the repository root:

```bash
python -m robot_sf.analysis_workbench.release_row_anomalies \
  --bundle /path/to/publication_bundle.tar.gz \
  --config configs/benchmarks/release_row_anomalies_v1.json \
  --output-json output/release-row-anomalies.json \
  --output-markdown output/release-row-anomalies.md
```

To bind an archive run to a separately recorded digest, add
`--expected-bundle-sha256 <64-lowercase-hex-digits>`. This check applies to
archive bytes; an extracted directory does not have `source.bundle_sha256`.
Use `--release-gate` to make the report decision an exit status:

- `0`: report written and the gate passes, or report mode was used without
  `--release-gate`;
- `1`: report written, but `gate.blocked` is true in release-gate mode;
- `2`: the bundle, manifest, rows, config, annotations, preflight, or expected
  archive digest is invalid. Input errors occur before report output is
  written.

The Python API is useful for tests and for already extracted rows:

```python
from robot_sf.analysis_workbench.release_row_anomalies import analyze_release_rows

report = analyze_release_rows(
    rows,
    config=config,
    annotations=annotations,
    preflight=preflight,
    source=source,
)
```

## 0.0.7 retro-validation

The pinned corrected 0.0.7 archive used for retro-validation has SHA-256
`684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`. Run the
gate with that digest and retain the JSON and Markdown reports with the bundle
provenance:

```bash
python -m robot_sf.analysis_workbench.release_row_anomalies \
  --bundle /path/to/0.0.7-publication_bundle.tar.gz \
  --expected-bundle-sha256 684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f \
  --config configs/benchmarks/release_row_anomalies_v1.json \
  --output-json output/0.0.7-release-row-anomalies.json \
  --output-markdown output/0.0.7-release-row-anomalies.md \
  --release-gate
```

The expected diagnostic signals are the 22 same-step early spawn cells from
#9725, short collisions in `classic_station_platform_medium` seeds 112, 117, 125, 131,
133, and 138, the `social_force` pedestrian-free baseline regression from
#9724 (0/30 successes versus 29/30 for the blind goal baseline), and universal
failure of `francis2023_narrow_doorway` from #9728. The observed maximum
contact speed in that run was 731.0746 m/s. These counts identify release-row
patterns for review and do not attribute a root cause without separate
evidence.
