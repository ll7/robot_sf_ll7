# 0.0.7 → 0.0.8 slot-paired campaign audit

`scripts/analysis/compare_release_0_0_7_to_0_0_8.py` compares the accepted
0.0.7 publication bundle with a 0.0.8 campaign result root. It does not run a
campaign or admit benchmark claims.

## Baseline and slot contract

The accepted 0.0.7 rows are in publication bundle
`issue9431_release_benchmark_data_0_0_7_07f7e8d43084_20260922_publication_bundle.tar.gz`,
preserved as W&B artifact
`ll7/robot_sf/campaign-issue9431_release_benchmark_data_0_0_7_07f7e8d43084_20260922:v0`.
The tool verifies the bundle SHA-256
`684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`,
source commit `07f7e8d43084de748915e1b1eb8b2a1603357c6e`, campaign ID, and
the executed v1 scenario matrix identity. The older checked-in release manifest
is not substituted for these executed rows; see
[`issue_9431_release_0_0_7_closeout.md`](issue_9431_release_0_0_7_closeout.md)
and issue #9668.

The tool also accepts a cold-restored W&B artifact root containing
`campaign_preservation_manifest.json` and its compressed `*.gz` members. It
checks the pinned preservation manifest digest
`5eb0d68e1483f3d82e75c33c3966a1c597330816f911a0caeaecbde119d4379f`
and both stored and uncompressed digests of every consumed file. This is the
direct published-results route when the original publication tarball is not
locally available.

Each slot is `(planner key, kinematics, scenario_id, seed, benchmark_track)`.
The planner key and kinematics come from the campaign run directory
`runs/<planner>__<kinematics>/episodes.jsonl`. `benchmark_track` is empty when
the episode does not declare it. `episode_id`, source hash, and config hash are
provenance, not slot keys: they can change across corrected executions.
Newly named v4 arms occupy different slots from v3 arms, so they appear as
release-only rows, never as paired corrections. Their `replacement_planner`
column points to the other versioned key while still requiring a separately
evidenced `implementation replaced` classification.

## Run and classify

```bash
uv run python scripts/analysis/compare_release_0_0_7_to_0_0_8.py \
  --baseline-root /path/to/restored-0.0.7-artifact \
  --successor-root /path/to/0.0.8-campaign-root \
  --classification-file /path/to/classifications.json \
  --output-dir /path/to/comparison-output
```

Use `--baseline-bundle /path/to/accepted-publication-bundle.tar.gz` instead of
`--baseline-root` when that exact checksummed tarball is available.

The successor root needs `campaign_manifest.json` with `campaign_id` and
`git.commit`, plus `runs/*/episodes.jsonl`. Every 0.0.8 row's source commit
must agree with that manifest. The tool compares every leaf under `outcome`
and `metrics`, including fields present in only one release. Numeric changes
at absolute tolerance greater than `1e-12` become findings; categorical
changes use exact comparison. The outputs are `report.json`, `findings.csv`,
`planner_scenario_metrics.csv`, and `summary.md`. The summary CSV contains
paired means and mean differences for numeric fields for each planner,
kinematics, scenario, track, and field. Findings carry one classification and
explanation per changed field or release-only row. Unexplained findings produce
exit code 1; invalid inputs or ambiguous rules produce exit code 2.
Legacy non-finite metric sentinels compare by value and are omitted from
numeric means; they are represented as strings in JSON findings.

Classification rules are explicit analyst assertions. Omitted selectors match
all values, so scope a rule to the causal evidence it supports. An exact
match wins over a broader match; equally specific matches are an error.
Every rule needs `field`, `planner` or `scenario_id`, and a nonempty
`classification`, `issue`, `explanation`, and `evidence`:

```json
{
  "schema_version": "slot-paired-classifications.v1",
  "rules": [
    {
      "planner": "goal",
      "scenario_id": "classic_doorway_low",
      "field": "metrics.collisions",
      "classification": "route change",
      "issue": "#9870",
      "explanation": "The versioned doorway route moves this slot's obstacle encounter.",
      "evidence": "path/to/paired-trace-or-route-analysis.json"
    },
    {
      "planner": "hybrid_rule_v3_fast_progress_static_escape",
      "field": "__row__",
      "presence": "only_0_0_7",
      "classification": "implementation replaced",
      "issue": "#9874",
      "explanation": "This v3 arm slot was replaced by a separately named v4 arm.",
      "evidence": "path/to/versioned-slot-manifest.json"
    }
  ]
}
```

Likely causes to investigate are route changes #9870/#9887/#9884,
seed/reset fix #9889, Social Force angle wrap #9764/#9878, v4 slots #9874,
collision typing #9867, and SNQI calibration #9667/#9850. A matching issue
number alone is not causal evidence. The audit only records the asserted
classification and fails on missing explanations; release admission still
requires the causal review and impact tables in #9668.

The focused test fixture
`tests/analysis/fixtures/issue_9668_0_0_7_goal_sample.jsonl` projects the
first two `goal__differential_drive` rows of the published artifact onto the
episode identity, outcome, and metrics fields. It keeps the published numeric
sentinels; the test creates a synthetic 0.0.8 root from those rows and injects
one measured-field change. This fixture is implementation proof only.
