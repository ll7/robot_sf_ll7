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

The published row's top-level `status` is a terminal outcome (`success`,
`collision`, or `failure`), not an execution-availability marker. The gate
applies the Benchmark Auditor's shared execution-admission policy to explicit
execution-status fields, nested planner metadata, and fallback counters. Rows
marked fallback, degraded, unavailable for execution, or with malformed
admission metadata are excluded from detector cohorts and block the release
gate; their scenario/seed cells remain incomplete.

Pedestrian-free baseline comparisons use
`integrity.effective_view.observation_ped_count` from both paired rows. A row
is eligible only when this integer is exactly zero, even if its scenario ID is
listed in `pedestrian_free_scenarios`. A positive count excludes that pair.

## Detectors

The configured detectors are:

- `same_step_all_planners`: every expected planner in a complete cell fails at
  the same terminal step at or below `same_step_max_steps`. The finding includes
  the rows' reported outcome, invalid-run, and collision-event signature, with
  a `consistent`, `mixed`, or `partially_observed` label. It explicitly records
  that root-cause attribution is unavailable from release rows; matching
  signatures are not a causal finding;
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
  effective observed pedestrian count is zero. An empty
  `pedestrian_free_scenarios` list discovers scenario IDs from zero-pedestrian
  rows belonging to the baseline or a configured pedestrian-aware planner; a
  nonempty list narrows the analysis to those IDs. If a candidate reveals a
  zero-pedestrian scenario but the baseline has no eligible zero-pedestrian
  rows, the gate reports the cohort as incomplete. Missing counts on the rows
  used for automatic discovery or conflicting counts between paired rows also
  block the gate. Each actual comparison pair must report zero pedestrians in
  both rows;
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
`baseline_planner`, `pedestrian_free_scenarios`, `pedestrian_aware_planners`,
`max_unannotated_findings`, and `require_preflight`.

`pedestrian_aware_planners` is an explicit release-cohort allowlist. The
detector compares only those planner IDs against the blind baseline; an
allowlisted planner absent from the release roster is reported as unavailable
and blocks the gate, even when no pedestrian-free cohort is present. Keep this
list aligned with planners whose effective observation contract includes
pedestrians. The checked-in 0.0.7 configuration names the observed non-baseline
arms.
When comparisons are enabled, cell coverage requires both the roster observed
in manifest members and the configured baseline/comparison arms. A missing
configured arm is therefore visible even when omitted from every bundle
directory; omissions outside the configured roster cannot be inferred from
release rows alone.
`pedestrian_free_scenarios` is an optional scenario filter and is empty by
default, so the detector discovers applicable scenarios from each release's
effective-view pedestrian counts on the baseline and configured comparison
planners instead of baking a scenario name into the gate configuration. If
those rows disagree about whether pedestrians were observed in a paired cell,
the gate records `pedestrian_free_status_mismatch` and blocks the cohort. If
automatic discovery has no zero-pedestrian candidate and relevant row counts
are missing, the gate records `pedestrian_free_status_unavailable`; fully
observed rows with only positive counts mean no pedestrian-free comparison
cohort was present in that release.

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
a selector. Scenario-scoped entries also require the lowercase SHA-256
`manifest_sha256` of the release bundle they annotate; this prevents an
annotation for one release from clearing the same scenario in another release.
Entries selected by `finding_id` are already source-bound and do not require
that field. Optional selectors are `seed`, `planner_id`, and `detector_id`.
An annotation sets `finding.annotated` on matching detector findings without
rewriting the observed row. For `universal_failure_unannotated`, a matching
annotation removes the missing-root-cause finding and increments
`counts.annotated_universal_cells` instead; the cell's other detector findings
remain visible.

The release gate is evaluated once at report level. It blocks when the report has
more unannotated findings than the configured limit, has incomplete planner
cells or execution admission, requires unavailable or incomplete preflight,
has a configured pedestrian-aware planner or baseline planner missing from the
release roster, has no eligible baseline rows, or has a selected pedestrian-free
scenario without its baseline, enough valid pairs, or required pedestrian-count
evidence. Any invalid-run/preflight mismatch also blocks the gate. An empty
pedestrian-aware planner list disables these comparison checks. The parity reason remains blocking even when that
finding has an annotation. Detector findings and their annotation state stay
in the JSON report, while Markdown summarizes the same report-level decision.

The JSON report also carries `detector_registry` and its
`detector_registry_digest`. Every entry in `signals` is a canonical BA-03
`signal` audit record, so consumers can validate and deserialize it with
`robot_sf.analysis_workbench.audit_contracts.record_from_dict`. Finding IDs
bind the source digests and detector-registry digest, so a threshold change
produces a new identity for an otherwise matching cell. The registry uses a
stable module owner, so API and command-line runs produce the same digest and
signal IDs for the same source and configuration.

These are aggregate cross-planner detector signals, not BA-01 per-episode
detector executions. To persist the typed BA-03 signals in the Benchmark
Auditor's local store, pass `--audit-store <directory>`. Commits are atomic and
idempotent for the same source, registry, and signal set. This handoff does not
fabricate a BA-01 campaign scan, BA-02 queue summary, or human finding; those
remain owned by their respective Auditor contracts. Before committing, it
verifies that every typed signal exactly matches one report finding, that each
finding ID still binds to the report's source and registry digests, and that
each finding's non-empty source-member list belongs to the verified bundle.
Signal ordering does not affect the idempotency key.

## Running the report

Run from the repository root:

```bash
python -m robot_sf.analysis_workbench.release_row_anomalies \
  --bundle /path/to/publication_bundle.tar.gz \
  --config configs/benchmarks/release_row_anomalies_v1.json \
  --output-json output/release-row-anomalies.json \
  --output-markdown output/release-row-anomalies.md
```

Add `--audit-store output/benchmark-auditor` to also commit the produced BA-03
signals to a local Benchmark Auditor store. The optional handoff is summarized
in both reports; with no candidate findings, the store has no release-row
signal transaction.

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

The command below remains a historical **diagnostic**. Its v1 config selects
`collision_metric_contract=legacy_diagnostic` and preserves the old detector
registry. A passing historical report does not satisfy 0.0.8 collision-count
admission; missing historical component fields are not imputed.

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

## 0.0.8 candidate collision-count gate

The checked-in
`configs/benchmarks/release_row_anomalies_0_0_8.template.json` is an
**unfrozen, non-admission template**. It always blocks with
`collision_roster_unfrozen_template`; its pedestrian-aware roster is empty until
#9751 freezes the 14-arm v4 names. A stale historical hybrid roster must never
be copied into the candidate. After that freeze, create a versioned config with
`collision_roster_status=frozen`, `collision_expected_arm_count=14`, and the
exact baseline/aware IDs, then check the pinned bundle:

```bash
python -m robot_sf.analysis_workbench.release_row_anomalies \
  --bundle /path/to/0.0.8-candidate-publication_bundle.tar.gz \
  --config /path/to/frozen-release-row-anomalies-0.0.8.json \
  --preflight /path/to/pinned-0.0.8-preflight.json \
  --output-json output/0.0.8-release-row-anomalies.json \
  --output-markdown output/0.0.8-release-row-anomalies.md \
  --release-gate
```

The candidate manifest and receipt must pin the config hash and show
`collision_metric_contract=release_0_0_8` and
`collision_roster_status=frozen`. That mode checks the configured 14-arm roster
against the verified bundle source and checks every admitted row's
five collision-count fields and blocks missing or inconsistent values. It also
checks equivalent fields in a typed event ledger. Exact contact-event records
and sampled collision counts use different collection semantics, so the gate
does not equate their counts. Missing or unversioned ledgers are reported as
`typed_collision_ledger_unavailable`; a separate provenance gate must establish
ledger completeness. Count arithmetic uses exact integers, including JSON
integers above `2**53`; integral floats above `2**53 - 1` block because
adjacent counts cannot be distinguished. This command is a release gate only
after the bundle and preflight inputs are pinned and verified; the example
paths above are placeholders.
