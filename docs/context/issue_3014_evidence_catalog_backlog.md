# Issue #3014 Evidence Catalog Backlog

Issue: <https://github.com/ll7/robot_sf_ll7/issues/3014>

## Summary

Issue #3014 is the canonical cleanup surface for stale, duplicate, superseded, and unindexed
`docs/context/` material. This note records the current evidence-catalog coverage gap found after
PR #3150 cataloged the recent #3062, #3142, and #3146 context notes.

Current finding on 2026-06-19: the explicit evidence-catalog hygiene check reports 116 tracked
`docs/context/evidence/` bundles or standalone files with no `docs/context/catalog.yaml` coverage.
That is too large for one safe docs-only PR because many entries are benchmark or paper-adjacent
evidence pointers. Future cleanup should split the backlog by evidence family and preserve claim
boundaries rather than adding all paths mechanically.

## Reproduction Command

Run this from the repository root:

```bash
uv run python scripts/validation/check_docs_proof_consistency.py \
  --check-evidence-catalog --json
```

The checker treats each immediate `docs/context/evidence/` child directory as one evidence bundle
and each loose file as a standalone evidence item. A bundle is covered when the catalog contains at
least one path at or below that bundle root.

## Representative First Tranche

The first uncovered entries from the 2026-06-19 scan were:

- `docs/context/evidence/issue_1318_teb_corridor_deadlock_2026-05-20`
- `docs/context/evidence/issue_1344_paired_amv_primary_2026-05-20`
- `docs/context/evidence/issue_1353_broader_amv_2026-05-26`
- `docs/context/evidence/issue_1393_gensafenav_source_harness_2026-05-20`
- `docs/context/evidence/issue_1394_crowdnav_height_source_harness_2026-05-20`
- `docs/context/evidence/issue_1395_learned_risk_launch_packet_2026-05-24`
- `docs/context/evidence/issue_1396_shielded_ppo_launch_packet_2026-05-24`
- `docs/context/evidence/issue_1397_oracle_imitation_launch_packet_2026-05-24`
- `docs/context/evidence/issue_1416_converted_map_cache_2026-05-20`
- `docs/context/evidence/issue_1427_predictive_same_seed_handoff_2026-05-21`
- `docs/context/evidence/issue_1428_orca_residual_lineage_2026-05-24`
- `docs/context/evidence/issue_1430_carla_live_parity_2026-05-21`
- `docs/context/evidence/issue_1437_carla_robot_spawn_2026-05-21`
- `docs/context/evidence/issue_1440_carla_spawn_projection_2026-05-21`
- `docs/context/evidence/issue_1442_carla_native_spawn_probe_2026-05-24`
- `docs/context/evidence/issue_1454_s10_h500_candidates_2026-05-23`
- `docs/context/evidence/issue_1454_s10_preflight_2026-05-22`
- `docs/context/evidence/issue_1454_stage_a_fixed_h100_2026-05-22`
- `docs/context/evidence/issue_1457_adversarial_generation_protocol_2026-05-23`
- `docs/context/evidence/issue_1462_s10_h500_failure_modes_2026-05-24`
- `docs/context/evidence/issue_1467_carla_replay_metrics_2026-05-24`
- `docs/context/evidence/issue_1470_oracle_imitation_traces_12911_2026-06-17`
- `docs/context/evidence/issue_1474_shielded_ppo_repair_2026-06-01`
- `docs/context/evidence/issue_1475_orca_residual_bc_smoke_12749.SHA256SUMS`
- `docs/context/evidence/issue_1475_orca_residual_bc_smoke_12913_2026-06-17`

The highest repeated issue keys in this scan were Issue #852 with four uncovered evidence items,
Issue #1454 with three, and Issue #1475 with two. Most other issue keys appeared once.

## Cleanup Strategy

Use small PRs with explicit evidence-family boundaries:

1. Add catalog coverage for one coherent family at a time, for example learned-policy launch
   packets, CARLA replay evidence, or adversarial-search evidence.
2. Prefer one representative catalog row per evidence bundle when that is enough to satisfy the
   checker and preserve discoverability.
3. Mark rows as `evidence` only when the file content itself is safe for evidence scanning: no
   absolute local paths and no ignored `output/` dependency pointers outside fenced command
   examples.
4. For evidence files that intentionally include local-output provenance, catalog the durable note
   that interprets the evidence instead of forcing the raw file into `catalog.yaml`.
5. Do not delete or rewrite benchmark or paper-facing evidence without checking the linked note,
   issue, PR, and claim boundary.

## Validation

Use the normal diff gate for each bounded cleanup PR:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```

Use the full evidence-catalog check as the backlog meter. It is expected to fail until all
historical evidence bundles are either cataloged, intentionally pruned, or moved behind a documented
replacement surface.

## Current Boundary

This note is durable handoff content, not benchmark evidence. It records a docs/catalog hygiene
backlog and the split strategy for Issue #3014. It does not change any benchmark result, metric
definition, planner ranking, or paper-facing claim.
