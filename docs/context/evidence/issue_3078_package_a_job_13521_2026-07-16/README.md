<!-- AI-GENERATED (robot_sf#3078, 2026-07-28) - NEEDS-REVIEW -->

# Issue #3078 real held-out-family full pilot (consolidated diagnostic bundle)

Plain-language summary: job 13521 supplied the complete 18-row held-out-family
pilot to the Package A transfer-report path. All six declared held-out cells are
represented once for each of `goal`, `social_force`, and `orca` at the single
evaluation seed `111`. This consolidated bundle replaces the synthetic
seed-sufficiency reference with a real-data seed/rank-stability diagnostic. The
classification is `diagnostic`; no synthetic fixture is used and no
benchmark, ranking, paper, or dissertation claim is promoted.

## Provenance and acceptance

- Job: `13521` (`01e-issue3078-heldout-fullpilot`).
- Execution commit: `9d65072ecd9d04e2f664a4299665dbff718401d9`.
- Expected and observed episode rows: 18.
- Unique `(cell, planner, seed)` identities: 18.
- Cells: 6 across `classic_station_platform` and `francis2023_intersection_wait`.
- Planners: 6 rows each for `goal`, `social_force`, and `orca`.
- Evaluation seed: `111`.
- Row status: 6 native (`goal`), 12 adapter (`social_force`, `orca`).
- Fallback/degraded rows: 0.
- Synthetic fixture used: false.
- Canonical compact source-row store SHA-256:
  `46466cd3db27d6f8a10181a8ec7c4676b24179bb97902aa8eec686d09a53942b`.

The compact source-row store remains outside Git. This bundle registers its
checksum, the exact accepted identity set, aggregated row tables, the real-data
seed/rank-stability diagnostic, deterministic figures, and the transfer-report
outputs. Private source paths are normalized to `private-campaign://job-13521/`
URIs; `registration.json` records source and registered checksums.

## Real-data diagnostics (both not_identifiable)

- **Seed/rank-stability: `not_identifiable`.** There is a single evaluation seed
  (111), so planner-rank stability cannot be estimated. This is a result of the
  diagnostic, not a gap to fill through substitution or a new campaign. See
  `seed_rank_stability_diagnostic.json` and `fig_seed_rank_stability.png`.
- **Held-out transfer-delta: `not_identifiable`.** The eligible comparator is
  frozen as `no_eligible_comparator` (#6150 / merged PR #6166), so the held-out
  transfer delta is empty with `claim_eligible=false`. See
  `no_eligible_comparator.json`, `transfer_delta.csv`, and
  `fig_transfer_delta.png`.

Adapter rows (`social_force`, `orca`) remain labeled adapter and are never
relabeled native-only.

## Package A disposition

- readiness: `ready`;
- decision-packet classification: `diagnostic` (reviewed final classification);
- issue-result classification: `diagnostic`;
- all four decision-packet criteria: satisfied;
- held-out table episode count: 18;
- synthetic fixture marker: absent.

This does not promote a benchmark claim. The report has no benchmark-set baseline
rows for these held-out-only inputs, so `transfer_delta_snqi` remains empty and
`claim_eligible=false`. A separately authorized review and an eligible comparator
are required before any future transfer, ranking, or paper-facing claim; this freeze
does not authorize such a promotion.

## Files

- `row_acceptance.json`: fail-closed 18-row identity and status proof.
- `postrun_acceptance.json`: transfer-report acceptance and classification.
- `fullpilot_plan.json`: predeclared six-cell, three-planner scope.
- `summary.json`: compact result-store summary.
- `heldout_family_table.csv`: real held-out aggregates for all planner/family combinations.
- `baseline_table.csv`: empty by design because this input contains held-out rows only.
- `transfer_delta.csv`: diagnostic output with no eligible comparator delta.
- `seed_rank_stability_diagnostic.json`: real-data seed/rank-stability diagnostic
  (not_identifiable, sole seed 111).
- `fig_seed_rank_stability.png`, `fig_transfer_delta.png`: deterministic figures.
- `no_eligible_comparator.json`: frozen comparator-eligibility receipt.
- `package_a_decision_packet.json`: Package A readiness/classification disposition.
- `claim_card.yaml`: explicit review and promotion boundary.
- `registration.json`: job and checksum lineage.
- `checksums.sha256`: registered bundle integrity manifest.
