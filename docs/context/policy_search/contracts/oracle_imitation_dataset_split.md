# Issue #1443 Oracle Imitation Dataset Split Policy (2026-05-22)

## Scope

This contract governs how a planner-oracle dataset is partitioned into train, validation,
and evaluation splits before any imitation-style policy training begins. It applies to
all oracle-trajectory collection, hard-slice recovery augmentation, and relabeling that
feeds the policy-search imitation pipeline.

## Seed Split Rule

- Use the repository canonical seed sets for validation and evaluation. Do not invent or
  regenerate validation/evaluation seeds for this purpose.
- For the first #1397 dataset contract:
  - `validation_seeds`: `configs/benchmarks/seed_sets_v1.yaml` key `dev` (`101`, `102`, `103`).
  - `evaluation_seeds`: `configs/benchmarks/seed_sets_v1.yaml` key `eval` (`111`, `112`, `113`).
  - `train_seeds`: predeclared collection seeds that exclude every seed in `dev` and
    `paper_eval_s20`.
- The three keys in `seeds_by_split` (`train`, `validation`, and `evaluation`) must be
  disjoint. No seed may appear in more than one split. A `no-overlap` invariant is mandatory.
- The split assignment must be recorded before collection starts and must not be changed afterward.

## Hard-Slice Assignment

- Hard-slice recovery examples (e.g., corridor-deadlock, static-escape, or scenario-specific
  failure modes) are assigned to `train` or `validation` by default.
- They may be assigned to `evaluation` only when a dedicated holdout eval slice is predeclared
  in writing before collection begins.
- If no holdout eval slice was predeclared, hard-slice recovery examples must be excluded from
  `evaluation_seeds`.

## Relabeling Policy

- Relabeling of oracle actions or observations is permitted **only** on the `train` split.
- Training-split relabeling must use a documented, deterministic rule and must name the source
  oracle that produced the original label.
- Relabeling on `validation` or `evaluation` is rejected except for metadata-only annotations
  (e.g., tagging an episode with a failure mode label without changing actions or rewards).
- Any relabeling applied to `train` must be declared in the manifest under `relabeling_policy`.

## Manifest Schema

Every generated oracle-imitation dataset must ship a manifest containing at least the following
fields:

| Field | Description |
|-------|-------------|
| `source_candidate` | Name of the planner-oracle candidate that generated the trajectories. |
| `source_candidate_config` | Path or identifier of the config used for the source candidate. |
| `scenario_ids` | List of scenario identifiers included in the dataset. |
| `seeds_by_split` | Mapping with keys `train`, `validation`, `evaluation` listing the exact seeds per split. |
| `episode_ids_by_split` | Mapping with keys `train`, `validation`, `evaluation` listing episode identifiers per split. |
| `hard_slice_assignment` | Per-hard-slice record of which split it was assigned to and whether it was predeclared for evaluation. |
| `relabeling_policy` | Null if no relabeling was performed; otherwise a deterministic rule description and source-oracle reference. |
| `checksums` | Content checksums for the artifact files (e.g., SHA-256). |
| `exclusion_rules` | Description of any episodes or seeds excluded from the dataset and why. |
| `provenance` | Human-readable summary of how the dataset was produced. |
| `created_at` | ISO-8601 timestamp of manifest creation. |
| `generating_commit` | Git commit hash of the repository state used for collection. |
| `artifact_paths` | Relative or absolute paths to the dataset artifacts this manifest describes. |

## Validation Gate

Before training starts, verify:

1. `seeds_by_split` has no overlaps.
2. `hard_slice_assignment` respects the predeclaration rule.
3. If `relabeling_policy` is non-null, it references a training-split-only scope.
4. `checksums` cover every file listed in `artifact_paths`.

## Dependency

This policy is a prerequisite for the oracle-imitation dataset campaign. Do not begin trajectory
collection or hard-slice augmentation until this split contract is in place and the manifest schema
is populated. See the campaign note [SLURM/003_imitation_oracle_dataset_campaign.md](../SLURM/003_imitation_oracle_dataset_campaign.md) for the gated
handoff.
