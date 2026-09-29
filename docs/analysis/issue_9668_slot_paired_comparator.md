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

The expected 0.0.8 slots come from the pinned campaign's enabled planners,
kinematics matrix, resolved scenarios, seeds, and track values. A configured
planner/kinematics arm with no rows invalidates the input (exit 2). Missing,
extra, and duplicated slots appear as `__slot__` findings in `findings.csv`;
classification rules cannot explain them, so each leaves the audit at exit 1.

## Run and classify

```bash
uv run python scripts/analysis/compare_release_0_0_7_to_0_0_8.py \
  --baseline-root /path/to/restored-0.0.7-artifact \
  --successor-root /path/to/0.0.8-campaign-root \
  --successor-manifest /path/to/reviewed-successor-manifest.json \
  --successor-manifest-sha256 <reviewed-manifest-sha256> \
  --successor-source-root /path/to/successor-source-clone \
  --classification-file /path/to/classifications.json \
  --broad-rule-bound-threshold 1.0 \
  --output-dir /path/to/comparison-output
```

Use `--baseline-bundle /path/to/accepted-publication-bundle.tar.gz` instead of
`--baseline-root` when that exact checksummed tarball is available.

The successor needs a separately reviewed `slot-paired-successor.v1` manifest
and its pinned SHA-256. It names release `0.0.8`, campaign ID, full source
commit, campaign config and scenario matrix paths with raw SHA-256 and campaign
runtime hashes, and the path and raw SHA-256 of each configured v4 planner
input. The tool reads those files from the named Git commit in
`--successor-source-root` and checks their bytes and the config's scenario and
planner bindings. It creates a temporary detached checkout at that commit and
recomputes the runtime hashes and resolves planner configurations in an isolated
Python subprocess whose project imports come from that checkout. It rejects a
source commit without the runtime modules, as well as a manifest and result
root that agree on forged runtime hashes. It rejects a missing or extra v4
binding. The result root's
`campaign_manifest.json` must match the reviewed campaign ID, source commit,
config hash, scenario path, and scenario hash. Each row's run-directory planner
key must occur in the pinned config. Its recorded algorithm, scenario config
hash, algorithm metadata, effective planner config, and run provenance must
match that key. Each row's recorded
`provenance.config_identity.scenario_matrix_hash` must match the pinned runner's
hash of the full scenario list scoped to that planner and kinematics, including
campaign track and observation settings. A row's campaign config hash, when
recorded in its config identity, must match the pinned campaign runtime hash;
the campaign manifest hash is always required and checked. Mixed scoped hashes
invalidate the input (exit 2) before a report is written. Policy-search
candidate base configs and scenario overrides are resolved from files in the
pinned checkout with that commit's production resolver; a referenced config
outside that checkout is rejected. Every row's source commit must also match.
Matrix-owned scenario fields, including the scenario identity
and `map_file`, must match the resolved scenario in the pinned commit. A row's
self-recorded config hash cannot authenticate a changed map. Invalid identity
exits 2 before writing a comparison report. A manifest checksum proves
which reviewed assertion was supplied; it does not independently establish
that a campaign was accepted for release.

Manifest shape (values below are placeholders, not approved identities):

```json
{
  "schema_version": "slot-paired-successor.v1",
  "release": "0.0.8",
  "campaign_id": "<reviewed-campaign-id>",
  "source_commit": "<40-character-source-sha>",
  "campaign_config": {
    "path": "configs/benchmarks/<versioned-campaign>.yaml",
    "sha256": "<raw-file-sha256>",
    "runtime_hash": "<campaign_manifest.config_hash>"
  },
  "scenario_matrix": {
    "path": "configs/scenarios/<versioned-matrix>.yaml",
    "sha256": "<raw-file-sha256>",
    "runtime_hash": "<campaign_manifest.scenario_matrix_hash>"
  },
  "versioned_planner_bindings": {
    "<v4-planner-key>": {
      "path": "configs/policy_search/<versioned-planner>.yaml",
      "sha256": "<raw-file-sha256>"
    }
  }
}
```

The binding map must contain all four v4 keys in the campaign config. Pin the
manifest digest after reviewing these identities, and retain that exact file
with the comparison evidence.

The tool compares every leaf under `outcome`
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

Classification rules are explicit analyst assertions under
`slot-paired-classifications.v2`. Every rule declares its exact planner,
kinematics, scenario, track, and seed set (`"all"` is explicit), fields,
positive `max_findings`, and either a sign plus maximum absolute numeric delta
or exact finding IDs. Finding IDs are SHA-256 values of the slot, field,
presence, and displayed old/new values in the first pass `report.json`.
No finding is explained unless all conditions hold. A rule matching more than
`max_findings` explains none of those findings. Overlapping successful rules
are invalid. Each rule's matching and covered finding IDs appear in
`report.json`; covered IDs also appear in `summary.md`. A `Broad rules` section
precedes rule coverage and lists any rule with `seeds: "all"` or
`max_abs_delta` greater than `--broad-rule-bound-threshold` (default `1.0`).
The threshold changes review visibility only; it never rejects or accepts a
classification:

```json
{
  "schema_version": "slot-paired-classifications.v2",
  "rules": [
    {
      "rule_id": "doorway-goal-111-collision",
      "slots": [{"planner": "goal", "kinematics": "differential_drive", "scenario_id": "classic_doorway_low", "benchmark_track": "", "seeds": [111]}],
      "fields": ["metrics.collisions"],
      "predicate": {"sign": "positive", "max_abs_delta": 1},
      "max_findings": 1,
      "classification": "route change",
      "issue": "#9870",
      "explanation": "The versioned doorway route moves this slot's obstacle encounter.",
      "evidence": "path/to/paired-trace-or-route-analysis.json"
    }
  ]
}
```

Likely causes to investigate are route changes #9870/#9887/#9884,
seed/reset fix #9889, Social Force angle wrap #9764/#9878, v4 slots #9874,
collision typing #9867, and SNQI calibration #9667/#9850. A matching issue
number alone is not causal evidence. For categorical or release-only findings,
use `{"finding_ids": ["<ID from first-pass report>"]}` as the predicate.
The audit only records the asserted
classification and fails on missing explanations; release admission still
requires the causal review and impact tables in #9668.

The focused test fixture
`tests/analysis/fixtures/issue_9668_0_0_7_goal_sample.jsonl` projects the
first two `goal__differential_drive` rows of the published artifact onto the
episode identity, outcome, and metrics fields. It keeps the published numeric
sentinels; the test creates a synthetic 0.0.8 root from those rows and injects
one measured-field change. This fixture is implementation proof only.
