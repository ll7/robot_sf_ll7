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

## Structured tuning-log contract

Each tuning log must be YAML or JSON with the following top-level shape:

```yaml
schema_version: issue_9748.tuning_log.v1
entries:
  - candidate: hybrid_rule_v4_fast_progress_static_escape_s30_h600_release
    scenario_ids:
      - issue_9748_dev_classic_doorway_medium
    seeds: [1001, 1002]
    # Optional typed fields may include seed, scenario_seed, seed_range,
    # resolved_seeds, scenario_id, or scenario_ids.
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
