# Reproduction

The renderer accepts local filesystem paths. First hydrate the registered private source
URI `private-campaign://job-13521/result_store` to a local result-store directory, then set
that directory in `JOB_13521_RESULT_STORE`:

```bash
export JOB_13521_RESULT_STORE=/path/to/hydrated/job-13521/result_store
```

Renderer command:

```bash
uv run python scripts/tools/build_package_a_transfer_report.py \
  --output-dir output/issue_3078_package_a_job_13521_transfer_report \
  --readiness-manifest configs/benchmarks/issue_3078_package_a_readiness.yaml \
  --heldout-partition-manifest configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml \
  --result-store "$JOB_13521_RESULT_STORE" \
  --seed-analysis-report docs/context/evidence/issue_3078_package_a_2026-07-08/seed_sufficiency_analysis.json
```

The private campaign URI remains the durable provenance pointer; it is not passed directly to
this local-`Path` CLI. This command renders compact evidence only. It does not run Package A
campaigns.
