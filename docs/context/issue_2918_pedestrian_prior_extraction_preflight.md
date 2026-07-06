# Issue #2918 — External Pedestrian-Prior Extraction Staging/Preflight

**Status:** `blocked-external-input` (extraction plan only; no calibrated-prior claim)
**Evidence tier:** `blocked` — staging/intake contract, not benchmark evidence
**Issue:** [#2918](https://github.com/ll7/robot_sf_ll7/issues/2918)
**Related lanes:** #2917 (authored scenario-prior cards), #2919 / #3192 (authored-vs-external
prior comparison/divergence), #2657 / #1498 / #3065 (external-data staging blockers)

"Pedestrian prior" here means the bounded distribution parameters (walking speed, crossing
angle, local density, interaction distance, stop/yield timing) that a scenario-prior card
summarizes. "Dataset-backed" means those parameters were extracted from a license-compatible
external trajectory dataset, as opposed to authored by hand.

## Purpose

Issue #2918 proposes a dataset-backed pedestrian-prior pilot, but extraction is **externally
blocked** until a license-compatible dataset (ETH/UCY, SDD, or SocNavBench-derived) is staged
through the opt-in BYO-dataset path (#3065/#2657/#1498). This slice adds the local,
fixture-testable **staging/preflight contract** — allowed source types, required provenance
fields, the prior parameters the run will emit, and fail-closed blocker reporting — so the
moment real data is staged and reviewed it can be wired in without re-deriving the schema.

It **ingests no external data, stores no raw trajectories in git, and makes no calibrated- or
representative-prior claim** (`evidence_boundary:
prior_extraction_plan_only_no_calibrated_prior_claim`).

The contract is enforced by
`robot_sf/benchmark/pedestrian_prior_extraction_manifest.py`
(`check_pedestrian_prior_extraction_manifest`), with the schema at
`robot_sf/benchmark/schemas/pedestrian_prior_extraction_manifest.v1.json`, a worked example at
`configs/research/pedestrian_prior_extraction_manifest_issue_2918_example.yaml`, and a CLI at
`scripts/tools/check_pedestrian_prior_extraction_manifest.py`.

## Contract

### Allowed source types

| Source type | External-data asset id | Staging contract |
| --- | --- | --- |
| `sdd` | `sdd` | registered (`scripts/tools/manage_external_data.py`) |
| `socnavbench` | `socnavbench-s3dis-eth` | registered |
| `eth_ucy` | _(none yet)_ | BYO-dataset only — cannot back a `dataset-backed` claim |

The source-type → asset-id map is cross-checked against the canonical external-data registry
in the test suite, so it cannot silently drift from `manage_external_data.py`.

### Prior parameters

A complete plan must declare all of: `walking_speed`, `crossing_angle`, `density`,
`interaction_distance`, `stop_yield_timing` (each with `units` and a `value_status`).

### Provenance

A `dataset-backed` manifest must carry `source_id`, `source_uri`, `license`, `citation`,
`access_date`, and `checksum` (plus an optional `staging_manifest` pointer) before any value is
trusted. `checksum` pins the run to a specific staged tree without storing raw trajectories.
Staging flows through `.agents/skills/data-staging-provenance/SKILL.md`.

### Authored-vs-dataset-backed separation

`authored_separation.separation` must be `enforced` so a dataset-backed comparison can never
silently overwrite the authored scenario-prior baseline
(`configs/research/scenario_prior_cards_issue_2917.yaml`); authored and dataset-backed priors
are compared, not merged.

## Lifecycle states and the claim gate

| `extraction_status` | Per-parameter `value_status` | `contract_status` | Claim allowed? |
| --- | --- | --- | --- |
| `blocked-external-input` (default today) | `pending` | `blocked` | no |
| `proxy-only` | `proxy-placeholder` | `proxy-only` | no |
| `dataset-backed` | `dataset-backed` | `ready` (no blockers) | yes |

Fail-closed rules: a `dataset-backed` manifest is downgraded to `blocked` if any prior
parameter is missing, any required provenance field is empty, separation is not enforced, or
the source type has no registered staging contract. A `proxy-only` manifest that declares a
dataset-backed provenance source is rejected as boundary conflation. An unknown source type or
status is rejected at the JSON-schema layer.

## Acceptance status (issue #2918)

- [x] Deterministic preflight contract + manifest schema implemented and fixture-tested; no raw
      trajectories in git.
- [x] Runs/claims only on a `dataset-backed` manifest with accepted provenance; missing data →
      fail-closed `blocked`.
- [x] Boundary states no representativeness/calibration claim beyond the staged source.
- [ ] One **dataset-backed** prior smoke from real staged data — still blocked on
      #3065/#2657/#1498 (external-data staging). This PR delivers the extraction LOGIC +
      contract only; the dataset-backed run is gated on staged data.

**Closure audit (2026-07-06):** the agent-executable slice is now complete across #3754
(staging/preflight contract) and #4566 (fixture extraction pipeline + CLI). Every
agent-executable acceptance criterion is met and reproducibly validated; the only residual is
the external-data-gated dataset-backed smoke, tracked by #3065/#2657/#1498. See the
criterion→PR evidence table and reproduced validation in
[`evidence/issue_2918_closure_audit.md`](evidence/issue_2918_closure_audit.md).
