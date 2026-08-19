# Job 13512 artifact promotion for Issue #5890

Plain-language summary: the missing compact result bundle for job 13512 was found at the
maintainer-recorded `imech192` path and copied into tracked evidence. This resolves the
missing-artifact blocker for custody, but it does not resolve the bundle's metric and scope
contradictions.

## Status

- Promotion status: `promoted_for_custody_only`
- Issue: #5890; parent campaign issue: #3207
- Job: `13512`
- Source host: `imech192`
- Source result root:
  `/home/luttkule/git/robot_sf_ll7.worktrees/camp5326-192/output/slurm/02b-issue3207-fidelity-full-fixed-scope-job-13512`
- Source access date: 2026-08-19
- Raw episode rows remain remote-only and are not committed.

## What was preserved

The tracked bundle contains the compact CSV, campaign summaries, rank-stability report, run plan,
execution context, environment freeze, and small execution logs. `registration.json` records the
source-relative paths, sizes, and SHA-256 values; `SHA256SUMS` covers this complete tracked bundle.

The raw `campaign/raw/episode_rows.jsonl` was observed remotely at 6,556,734 bytes with SHA-256
`d9efb9fe924eaff126fa5ec5f6b1303baa954d1e4b5400987b350f002ac9e9f3`, but is intentionally not
tracked. It is not required for the custody handoff and must not be recreated from the compact
outputs.

## Observed source facts

- `campaign_outcome.json` reports 5,184 episodes, primary metric `snqi`, and identifiable/stable
  ranking flags.
- `planner_variant_metrics.csv` has 36 rows and contains `success_rate` plus operational metrics,
  but no `snqi` column.
- The source summary classifies the run as a bounded actual slice, records 48 scenarios, seeds
  111–113, and lists `baseline_social_force`, `goal_seek`, and `orca`; the same summary describes
  the result as a bounded two-planner slice.
- The rank report uses `success_rate` as its primary metric, while the outcome and execution
  context name `snqi`. These are preserved as source facts, not normalized by this promotion.

## Reconciliation boundary

`reconciliation.md` compares this bundle with the older tracked
`issue_3207_fidelity_sensitivity_actual_slice_2026-06-23` packet. The two packets are not merged:
the older packet has 30 episodes, 10 rows, two planners, and a non-identifiable zero-variance
`success_rate` result. The new bundle has a different scope and unresolved metric/summary
inconsistencies. No per-axis ranking was recomputed here.

Until a maintainer selects and verifies the canonical metric and scope contract, classify this as
durable internal simulator-fidelity custody evidence only. It is not full #3207 acceptance,
simulator-realism, sim-to-real, safety, planner-superiority, or paper-facing evidence.
