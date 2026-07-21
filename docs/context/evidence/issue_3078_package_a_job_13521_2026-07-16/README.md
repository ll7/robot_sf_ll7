<!-- AI-GENERATED (robot_sf#3078, 2026-07-16) - NEEDS-REVIEW -->

# Issue #3078 real held-out-family full pilot

Plain-language summary: job 13521 supplied the complete 18-row held-out-family pilot to the
Package A transfer-report path. All six declared held-out cells are represented once for each of
`goal`, `social_force`, and `orca`. The resulting classification is
`diagnostic_review_ready`; no synthetic fixture is used.

This run supersedes underscaled job 13506. Job 13506 produced only two rows per planner because it
did not expand the two source scenarios into the partition manifest's six cells. It remains smoke
evidence only and must not be treated as the Package A pilot.

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

The compact source-row store remains outside Git. This bundle registers its checksum, the exact
accepted identity set, aggregated row tables, and the transfer-report outputs. Private source paths
are normalized to `private-campaign://job-13521/` URIs; `registration.json` records source and
registered checksums.

## Package A disposition

The existing Package A synthetic held-out classification is replaced for the held-out-transfer
criterion by these real accepted rows:

- readiness: `ready`;
- decision-packet classification: `diagnostic_review_ready`;
- issue-result classification: `diagnostic`;
- all four decision-packet criteria: satisfied;
- held-out table episode count: 18;
- synthetic fixture marker: absent.

This does not promote a benchmark claim. The report has no benchmark-set baseline rows for these
held-out-only inputs, so `transfer_delta_snqi` remains empty and `claim_eligible=false`. Claim-card
review and an eligible comparator remain necessary before any transfer, ranking, or paper-facing
claim.

## Files

- `row_acceptance.json`: fail-closed 18-row identity and status proof.
- `postrun_acceptance.json`: transfer-report acceptance and classification.
- `fullpilot_plan.json`: predeclared six-cell, three-planner scope.
- `summary.json`: compact result-store summary.
- `heldout_family_table.csv`: real held-out aggregates for all planner/family combinations.
- `baseline_table.csv`: empty by design because this input contains held-out rows only.
- `transfer_delta.csv`: diagnostic output with no eligible comparator delta.
- `package_a_decision_packet.json`: Package A readiness/classification disposition.
- `claim_card.yaml`: explicit review and promotion boundary.
- `registration.json`: job and checksum lineage.
- `checksums.sha256`: registered bundle integrity manifest.
