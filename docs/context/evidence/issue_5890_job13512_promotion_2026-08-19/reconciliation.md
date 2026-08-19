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

## Maintainer ruling and disposition (2026-08-19)

The maintainer ruling for #5890 is recorded in
https://github.com/ll7/robot_sf_ll7/issues/5890 (comment 2026-08-19). This section extends the
reconciliation record with the ruling and the additional provenance contradiction it names.

### Execution-provenance conflict (third contradiction)

The promotion previously recorded two contradictions (metric naming, planner scope). A third
material conflict exists between the execution context and the generated summary:

- `execution_context.txt` records `git_head: ae0130d65cf232e0322cfd4800659a87d481490a` and
  `config: configs/research/issue_3207_fidelity_sensitivity_full_fixed_scope.yaml`
  (`primary_metric: snqi`, `expected_episode_count: 5184`, host `auxme-imech254`,
  `slurm_job_id: 13512`).
- `run_summary_source.txt` pins the same execution commit as `public_commit`
  `ae0130d65cf232e0322cfd4800659a87d481490a` (`EXPECTED_HEAD`).
- `summary.json` records `git_head: c153848d7be2851b5c5e89c11055bf96ea778a84` and
  `config_path: configs/research/fidelity_sensitivity_v1.yaml` (the legacy config name), while its
  `study_id` still reads `issue_3207_fidelity_sensitivity_full_fixed_scope`.

The two git-head values (`ae0130d65…` vs `c153848d7…`) and the two config names do not agree.
The bundle therefore supports **byte custody, checksums, remote source location, and
contradiction preservation**, but not yet **verified execution provenance**. Explaining this
mismatch is a precondition for any reproduction or execution-provenance claim.

### Metric ruling

`success_rate` is the only materialized ranking metric in the compact CSV and the rank report.
`snqi` was the intended pre-run contract field but is absent from the metric rows. The
outcome-generation wrapper hard-codes `"primary_metric": "snqi"` in `campaign_outcome.json` while
copying identifiability/stability flags from the rank report. The outcome therefore does not
establish an SNQI result, and `success_rate` must not be renamed or treated as an alias for SNQI.

### Scope ruling

The materialized scope is unambiguously three planners — `baseline_social_force`, `goal_seek`,
and `orca` — across 12 axis variants, 3 seeds (111–113), and 48 scenarios: 3 × 12 × 3 × 48 =
5,184 episodes, 36 planner–variant rows. Any "two-planner" wording in the source summaries is
stale legacy metadata, not a second plausible execution scope.

### Disposition

For the original SNQI contract, job 13512 receives **`invalid_provenance_or_scope`**. This verdict
does not reject the preserved bytes; it states that the recovered bundle does not answer the
preregistered SNQI question. The existing `success_rate` ranking, ordering, identifiability flags,
and stability flags remain source outputs only and are **not accepted findings**.

Still blocked by this disposition:

- any SNQI ranking, SNQI identifiability, or SNQI stability claim;
- any planner-superiority or accepted `success_rate` ranking claim;
- verified execution reproducibility until the `ae0130d65…` versus `c153848d7…` and configuration
  mismatch is explained;
- full #3207 acceptance, #6151 synthesis, benchmark admission, and paper-facing use;
- any new computation, reconstruction, or campaign without separate authorization (#5890 orders
  promotion and reconciliation before synthesis and does not authorize another campaign).

### Custody review note

PR #7585 merged the custody slice at 2026-08-19T14:59:37Z at head `633428c8` (squash commit
`f54322f2`). The exact-head review evidence on the PR covers head `2d736f27`; the merged head
differs from the reviewed head (review-sidecar `.review.json` files removed, `registration.json`
adjusted), so the review does not cover the merged head byte-for-byte. The evidence bytes are
unchanged by that difference. This note records the gap; the maintainer ruling above governs
interpretation regardless.

## Evidence boundary

This is internal simulator-fidelity artifact custody and contradiction tracking only. It does not
support simulator realism, sim-to-real transfer, safety, planner superiority, benchmark admission,
publication, or paper-facing claims.
