<!-- AI-GENERATED (robot_sf#6156, 2026-07-28) - NEEDS-REVIEW -->

# Reproduction

This bundle is consolidated diagnostic evidence for the 18 real job-13521
held-out identities. The seed/rank-stability diagnostic and figures are emitted
directly under this evidence directory from the committed 18-identity
aggregates; the core transfer report remains reproducible through the existing
Package A renderer.

## Real-data seed/rank-stability diagnostic + figures

The diagnostic (`seed_rank_stability_diagnostic.json`) and the deterministic
figures (`fig_seed_rank_stability.png`, `fig_transfer_delta.png`) are produced
from the already-committed aggregates only:

- `row_acceptance.json` (18 identities, adapter=12, native=6, seed 111);
- `heldout_family_table.csv` and `transfer_delta.csv` (per-planner held-out means);
- `no_eligible_comparator.json` (no eligible benchmark-set comparator).

No private `episodes.parquet` is required for the diagnostic or figures, and no
new campaign, compute submission, or `configs/benchmarks/` change is involved.
Both diagnostics record `not_identifiable`: seed/rank-stability because there is
a single evaluation seed (111), and held-out transfer-delta because no eligible
comparator exists (#6150 / merged PR #6166).

Rebuild the diagnostic JSON and both figures from those tracked compact inputs,
then compare their exact bytes with the committed artifacts:

```bash
uv run python scripts/analysis/build_issue_3078_job_13521_diagnostic.py --check
```

The command does not read the private episode store or submit compute. To write
the three generated outputs to a scratch directory for inspection, replace
`--check` with `--output-dir /tmp/issue3078-job13521-diagnostic`.

## Core Package A transfer report (local-Path renderer)

The renderer accepts local filesystem paths. First hydrate the registered
private source URI `private-campaign://job-13521/result_store` to a local
result-store directory, then set that directory in `JOB_13521_RESULT_STORE`:

```bash
export JOB_13521_RESULT_STORE=/path/to/hydrated/job-13521/result_store
```

Renderer command (now pointing at the real-data diagnostic, not the synthetic
2026-07-08 seed analysis):

```bash
uv run python scripts/tools/build_package_a_transfer_report.py \
  --output-dir output/issue_3078_package_a_job_13521_transfer_report \
  --readiness-manifest configs/benchmarks/issue_3078_package_a_readiness.yaml \
  --heldout-partition-manifest configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml \
  --result-store "$JOB_13521_RESULT_STORE" \
  --seed-analysis-report docs/context/evidence/issue_3078_package_a_job_13521_2026-07-16/seed_rank_stability_diagnostic.json
```

The private campaign URI remains the durable provenance pointer; it is not passed
directly to this local-`Path` CLI. This command renders compact evidence only. It
does not run Package A campaigns.

## Domain decision overlay

The renderer-produced `claim_card.yaml`, `package_a_decision_packet.json`,
`postrun_acceptance.json`, and `registration.json` retain their
`diagnostic_review_ready` / `NEEDS-REVIEW` state. The human decision is recorded
separately in `package_a_domain_decision_overlay.json`, which binds the renderer
outputs and the `no_eligible_comparator.json` receipt (including its review sidecar)
by SHA-256. The overlay narrows the claim boundary to diagnostic-only evidence; it
does not turn the generated renderer output into benchmark or ranking evidence.

Validate the overlay, renderer state, receipt binding, and full checksum inventory with:

```bash
uv run pytest tests/benchmark/test_issue_3078_job_13521_registration.py -q
```

## Integrity check

From the repository root, verify every primary bundle artifact against the
tracked checksum manifest:

```bash
sha256sum -c docs/context/evidence/issue_3078_package_a_job_13521_2026-07-16/checksums.sha256
```
