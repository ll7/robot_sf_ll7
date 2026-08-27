# Config-Family Inventory — issue #7901

**Status:** inventory delivered; disposition **one_family_ready_for_child**.
**Issue:** [#7901](https://github.com/ll7/robot_sf_ll7/issues/7901) (relates #6484).
**Scanner:** `scripts/dev/audit_config_families.py` (`config_family_inventory.v1`).

## Scan

- Roots: `configs/algos`, `configs/training`, `configs/adversarial` (376 files; 372 resolved;
  4 unsupported/error).
- Resolution reuses the canonical resolver
  (`scripts.training.train_ppo._load_expert_training_config_mapping`); no second merge
  algorithm.
- Records per file: raw digest, resolved-mapping digest, inheritance chain, category,
  key/line counts, resolver errors. Fails closed on cycles, missing bases, and unsupported
  categories.
- Full report (SHA-256 in `receipt.json`): `output/config_family_inventory.json`
  (ignored, worktree-local).

## Candidate families (after excluding already-migrated)

| Family | Members | Common resolved paths | Est. reduction |
| --- | --- | --- | --- |
| `hybrid_rule` | 5 | 38 | 0.949 |
| `tentabot` | 4 | — | 0.930 |

Families whose members already declare `base_config` (e.g. `expert_ppo_*`,
`asymmetric_critic_only_*`) are excluded — they are covered by existing #6484 bases.

## Proposed child (at most one)

**`hybrid_rule`** (5 members): introduce ONE new shared base YAML (proposed filename
`hybrid_rule_base`, under `configs/algos/`) containing only byte-identical resolved
key/value paths; freeze pre-change resolved mappings and digests, re-resolve, and diff
after the change; parity tests for missing-base and cycle fail-closed.
No training, simulation, benchmark, external-data, or scheduler work; no runtime/claim change.
The estimated reduction is heuristic and must be confirmed by the frozen-resolved-mapping proof
before the child executes.

## Boundary

No production YAML or resolver file changed; no config was rewritten or migrated. No open issue
or PR owns the `hybrid_rule` or `tentabot` families.
