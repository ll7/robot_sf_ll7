# Issue #9748 hybrid v4 tuning protocol

This note defines the development split for hybrid v4 and records the boundary
between a tuning protocol and benchmark evidence. It does not report tuning
results and does not choose the 0.0.8 hybrid roster; that roster is the subject
of #9751.

## Frozen development split

The checked-in scenario manifest is
`configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml`. It starts
from existing scenario families and applies only the author-approved parameter
perturbations:

| Development identity | Source family | `ped_density` | `route_spawn_jitter_frac` |
|---|---|---:|---:|
| `issue_9748_dev_classic_doorway_medium` | `classic_doorway_medium` | 0.065 | 0.30 |
| `issue_9748_dev_classic_group_crossing_medium` | `classic_group_crossing_medium` | 0.10 | source value |
| `issue_9748_dev_francis2023_perpendicular_traffic` | `francis2023_perpendicular_traffic` | 0.12 | source value |
| `issue_9748_dev_francis2023_crowd_navigation` | `francis2023_crowd_navigation` | 0.10 | source value |

The identities are distinct from the frozen release matrix. Reusing the source
maps keeps the tuning split focused on constrained passage, group interaction,
crossing traffic, and dense crowd behavior without introducing an ungoverned
second geometry change. The trade-off is less geometric diversity; this is a
tuning surface, not a generalization benchmark.

The validator compares each resolved row with its named source scenario. It
allows only the approved density/jitter change, the 1001–1030 seed list, the
development-only identity, and development metadata. It checks the map-file
reference, robot settings, and every other simulation setting against the
source row, so these variants retain the existing map geometry and settings.
The referenced v4 candidate files retain their existing release-keyed
`scenario_overrides`. The runtime resolver uses each resolved scenario's
`name` as the key and performs an exact mapping lookup in
`robot_sf/benchmark/policy_search_manifest.py`; it does not alias through the
`source_scenario` metadata. The four issue-specific names therefore select
none of those release-keyed overrides. The focused test resolves every dev row
and confirms the effective config matches an unmatched-name control, and the
validator rejects an override key that exactly matches a dev identity.

The development campaign config
`configs/benchmarks/issue_9748_hybrid_v4_dev_split_v1.yaml` uses the exact
ordered seed list 1001–1030 and only the existing v4 fast-progress and v4
continuous candidate configs. It is explicitly marked development-only and
not release evidence. Release seeds 111–140 remain held out. The config does
not decide which v4 twins replace the four hybrid release slots.

## Freeze point and hashes

The scenario identities, parameter values, and seed list are frozen by the
tracked v1 manifests before tuning begins. At the pre-tuning gate, record the
SHA-256 of both manifests, the source commit, and the exact candidate config
SHAs in the tuning log. Any change to a scenario, seed, candidate config, or
loader after that point requires a new protocol version and a fresh review.
This implementation intentionally records no run hash or tuning result.

The release side is pinned to the accepted 0.0.7 inputs:

| Input | SHA-256 |
| --- | --- |
| Campaign config template `configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml` | `7dc9a2dd9df8585593c9bc8ecc001bed0d2ddff4ebb3803dfb92e8dad8762881` |
| Executed scenario matrix `configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml` | `03fc83302f707dd1b27c0fa81c4e45e36e8354a4413171d09365926f62bb5c2c` |
| Seed-set file `configs/benchmarks/seed_sets_v1.yaml` | `3aaab9171517b8d33bafc679d4a2c740864db0f96650e24d75c4c7e927d239e6` |
| Canonical `paper_eval_s30` schedule (seeds 111–140) | `ecbca1eaa1e3c0615d4d9eec8b3d59ec8432529fd9398482250e6f1c7e0abfe1` |

The checker verifies those bytes and the effective seed-set contents. It also
checks every resolved release scenario row for a development-seed overlap, so
a scenario fixture that introduces seed 1001–1030 fails even when its scenario
IDs do not overlap.

## Structured tuning-log contract

Each tuning log must be YAML or JSON with schema
`issue_9748.tuning_log.v2`. It binds the campaign and scenario files, the
accepted release inputs, and the two approved candidate templates to the
checker output. The source commit must exist in the local Git object database
and contain the exact frozen config, matrix, seed-set, and candidate bytes.
Each trial records its approved candidate, base template hash, complete
effective config snapshot and canonical hash, scenario identities, seeds, and
a checksummed artifact file in a durable location outside the repository
worktree. The checker reads that file and verifies its SHA-256. A remote run
service URI by itself is insufficient because this offline checker cannot
establish that the referenced run or artifact still exists. Candidate names
are required; an arbitrary algorithm or an omitted candidate is rejected.

```yaml
schema_version: issue_9748.tuning_log.v2
source_commit: 0123456789abcdef0123456789abcdef01234567
input_bindings:
  development_campaign_config_path: configs/benchmarks/issue_9748_hybrid_v4_dev_split_v1.yaml
  development_campaign_config_sha256: <checker output>
  development_scenario_matrix_path: configs/scenarios/sets/issue_9748_hybrid_v4_dev_variants_v1.yaml
  development_scenario_matrix_sha256: <checker output>
  release_campaign_config_path: configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_template.yaml
  release_campaign_config_sha256: <checker output>
  release_scenario_matrix_path: configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml
  release_scenario_matrix_sha256: <checker output>
  release_seed_set_path: configs/benchmarks/seed_sets_v1.yaml
  release_seed_set_file_sha256: <checker output>
  release_seed_schedule_sha256: <checker output>
  candidate_template_paths:
    hybrid_rule_v4_fast_progress_static_escape_s30_h600_release: configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_s30_h600_release.yaml
    hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release: configs/policy_search/candidates/hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release.yaml
  candidate_template_sha256:
    hybrid_rule_v4_fast_progress_static_escape_s30_h600_release: <checker output>
    hybrid_rule_v4_fast_progress_static_escape_continuous_s30_h600_release: <checker output>
entries:
  - trial_id: trial-001
    candidate: hybrid_rule_v4_fast_progress_static_escape_s30_h600_release
    base_candidate_config_sha256: <checker output>
    effective_candidate_config:
      # Complete effective candidate config for this trial.
      parameters: {example_parameter: 0.0}
    effective_candidate_config_sha256: <canonical JSON SHA-256>
    scenario_ids:
      - issue_9748_dev_classic_doorway_medium
    seeds: [1001, 1002]
    run_artifact_ref: file:///durable/artifacts/issue-9748/trial-001.tar.zst
    run_artifact_sha256: <exact artifact SHA-256>
    notes: "Free-form rationale may mention the held-out range."
```

The validator inspects typed values under the seed fields (`seed`, `seeds`,
`scenario_seed`, `scenario_seeds`, `seed_range`, `resolved_seeds`, and
`tuning_seeds`) and under the scenario identity fields. It requires typed seeds
to belong to 1001–1030 and rejects any 111–140 value. It requires structured
scenario IDs to belong to the four development identities, and every tuning-log
entry must contain a nonempty string `scenario_id` or a list of string
`scenario_ids`. Release scenario IDs in typed config or log fields fail closed.
Free-form strings such as `notes`, `rationale`, and `claim_boundary` are not
parsed as admissions, so mentioning held-out values does not create a false
violation. Malformed logs, missing typed seed fields, and non-string structured
scenario IDs fail closed.

The effective config snapshot must hash to its declared digest, each base
template digest must match an approved candidate, and the source commit must
contain every frozen input byte. The run artifact URI must resolve to an
existing file outside the checkout, and its recorded SHA-256 must match the
file bytes. Copy remote artifacts to the durable shared artifact root before
validating the log; do not substitute an unverifiable service URI.

Run the checker with:

```bash
uv run python scripts/validation/check_issue_9748_dev_split.py \
  --config configs/benchmarks/issue_9748_hybrid_v4_dev_split_v1.yaml \
  --tuning-log <frozen-log.yaml> --json
```

The checker loads the development and release matrices through the repository
scenario loader and loads the campaign through the camera-ready config loader.
Its success status means that the protocol is structurally valid; it is not
evidence that tuning ran, that a candidate improved, or that a release arm is
accepted.

`load_campaign_config` retains the development-only and claim-boundary markers
without adding them to historical dataclass serialization. The direct campaign
runner refuses a development-only config unless the caller explicitly passes
`--allow-development-tuning` (CLI) or `allow_development_tuning=True` (Python).
That opt-in is for use only after the protocol is reviewed and frozen; it does
not authorize a trial by itself or promote its output to release evidence.

## Held-out process and claim boundary

Tune only on the frozen development split. Do not use release-seed outcomes to
select thresholds, margins, weights, or candidate identities. After tuning,
record every attempted candidate and the approved split in the structured log,
then freeze the selected config before the release campaign. The first run on
seeds 111–140 is held-out evaluation. Any later v4 change is a new candidate
and cannot be treated as the same frozen release arm.

No Slurm job, tuning result, calibration claim, ranking claim, or paper-facing
claim is implied by this protocol document. Development evidence remains
pending until an independently reviewed, hash-bound run and tuning log exist.
