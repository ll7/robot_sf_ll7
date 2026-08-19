# Job 13512 reconciliation record

This record performs the preserve-only comparison required by Issue #5890. It does not recompute
rankings, rename metrics, or decide which source contract is scientifically authoritative.

## Compared packets

| Property | Older tracked packet | Promoted job-13512 bundle |
| --- | --- | --- |
| Repository path | `docs/context/evidence/issue_3207_fidelity_sensitivity_actual_slice_2026-06-23/` | `docs/context/evidence/issue_5890_job13512_promotion_2026-08-19/` |
| Generation date | 2026-06-23 | 2026-07-16 |
| Episodes | 30 | 5,184 |
| CSV rows | 10 | 36 |
| Planners listed by summary | `baseline_social_force`, `goal_seek` | `baseline_social_force`, `goal_seek`, `orca`; summary also says bounded two-planner slice |
| Primary metric in rank report/CSV | `success_rate` | `success_rate` in CSV/rank report |
| Primary metric in campaign outcome/context | not present as a separate outcome field | `snqi` |
| Rank status | non-identifiable: primary-metric zero variance | outcome says identifiable/stable; rank report says identifiable/stable on `success_rate` |
| Scope claim | bounded two-planner internal slice; no benchmark claim | bounded actual slice; not full fixed-scope evidence |

## Resolution

1. The missing-artifact diagnosis for #5890 is resolved: the actual compact job-13512 files are
   now tracked with source pointers and checksums.
2. The promoted bundle must not be treated as a drop-in replacement for the older packet. Its row
   count, planner roster, scope description, and metric naming are not identical.
3. The `snqi` outcome label cannot be independently connected to a column in the promoted CSV,
   which contains `success_rate` instead. The source rank report also uses `success_rate`. This is
   a provenance/schema conflict, not evidence that either metric is wrong.
4. The summary's three-planner roster conflicts with its bounded-two-planner limitation. The
   promotion records both statements and does not infer whether ORCA was included in the claimed
   ranking contract.
5. The honest current disposition is `promotion_complete_pending_metric_scope_reconciliation`.
   The next analysis step requires a maintainer-selected metric/scope contract and an independent
   verifier that binds the chosen field to the complete rows. No SNQI ranking or full #3207 result
   is claimed by this change.

## Evidence boundary

This is internal simulator-fidelity artifact custody and contradiction tracking only. It does not
support simulator realism, sim-to-real transfer, safety, planner superiority, benchmark admission,
publication, or paper-facing claims.
