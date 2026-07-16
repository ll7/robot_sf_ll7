# Reproduction

Renderer command:

```bash
uv run python scripts/tools/build_package_a_transfer_report.py \
  --output-dir private-campaign://job-13521/package_a_transfer_report \
  --readiness-manifest configs/benchmarks/issue_3078_package_a_readiness.yaml \
  --heldout-partition-manifest configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml \
  --result-store private-campaign://job-13521/result_store \
  --seed-analysis-report docs/context/evidence/issue_3078_package_a_2026-07-08/seed_sufficiency_analysis.json
```

This command renders compact evidence only. It does not run Package A campaigns.
