# Issue #5305 certified adversarial failure archive

Plain-language summary: job 13518 produced 17 replay-verified and independently confirmed failure
records whose train/evaluation partitions do not reuse scenario families, seeds, or archive IDs.
The archive is also disjoint on all three keys from the retained #1501 and #1502 archives, removing
the circular-input failure that blocked the next #3275/#2921 stage.

## Evidence boundary

Certified archive input only; no proposal-model, held-out-yield, benchmark, or paper-facing claim.
This registration clears #3275's archive-input blocker and that issue is now Project `In progress`;
it does not execute or classify #3275 or #2921, whose downstream gates remain open.

## Provenance

- Registration status: `issue5305_archive_accepted`.
- Job: `13518` (`05b-issue5305-certified-archive`).
- Campaign ID: `issue5305-certified-archive-13518`.
- Execution commit: `ecf997d392a4f2c1a4fb5a56e8101acb030b7e2f`.
- Config SHA-256: `dfdebd497e19a046e41cb2b1e7d7a7f54cd592ac0a465e4149efff19efa16735`.
- Accepted source archive SHA-256:
  `1318e210bc4771fb0ab4b30d5bc6739f9ecd416e039f61e7da4cd7411f3baa6d`.
- Public-safe registered archive SHA-256:
  `79e022587b35c1c42bc07cfefaf882af473e96841a99ef57f98a4cee26636445`.
- Entries: 17 across `classic_cross_trap_medium` and `classic_group_crossing_medium`.

The registered JSON is a content-preserving public projection of the accepted source archive.
Only private path prefixes were replaced with `private-campaign://job-13518/` URIs; candidates,
metrics, certification decisions, families, seeds, archive IDs, and checksums of underlying records
were not changed. `registration.json` records both archive checksums and the transformation.

## Certification

All 17 entries have:

- passed scenario certification;
- exact deterministic-replay signature agreement;
- at least three independent-seed confirmations from five attempts;
- stable failure-mechanism attribution;
- no fallback or degraded classification.

The current repository readiness checker reports `ready`, 17 entries, two distinct families, no
missing certification/attribution/seed/ID fields, and no blocking reason.

## Disjointness proof

The archive's explicit split is:

- train: 5 `classic_cross_trap_medium` entries;
- evaluation: 12 `classic_group_crossing_medium` entries;
- train/evaluation family overlap: 0;
- train/evaluation scenario-seed overlap: 0;
- train/evaluation archive-ID overlap: 0.

The same shape test compares the new archive with each retained source archive and their union:

| Reference input | Reference entries | Family overlap | Scenario-seed overlap | Archive-ID overlap | Result |
|---|---:|---:|---:|---:|---|
| Issue #1501 one-family smoke | 15 | 0 | 0 | 0 | disjoint |
| Issue #1502 two-family comparison | 60 | 0 | 0 | 0 | disjoint |
| Issue #1501 + Issue #1502 union | 75 | 0 | 0 | 0 | disjoint |

This is the essential non-circularity result: neither the family labels nor the scenario seeds used
by the new train/evaluation archive appear in the older archives. See
`reference_disjointness.json` for the exact compared sets, source checksums, and overlap output.

## Files

- `archive.json`: public-safe complete certified archive projection.
- `registration.json`: source/registered checksums and artifact classification.
- `acceptance_report.json`: acceptance, certification, and split summary.
- `reference_disjointness.json`: explicit #1501/#1502 family/seed/ID comparison.
- `generation_summary.json`: campaign generation summary.
- `repository_readiness.json`: fresh current-repository readiness output.
- `SHA256SUMS`: checksums for the registered bundle.
