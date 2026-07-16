# Leakage Audit

- Partition manifest: `configs/benchmarks/issue_2128_heldout_family_transfer_partitions.yaml`
- Benchmark-set labeling mode: `inferred_by_excluding_heldout_families`
- Surface-labeling warning: benchmark-set families are not declared; benchmark_set rows are inferred as the complement of heldout_family_evaluation.scenario_families
- Benchmark-set families: none declared
- Held-out families: classic_station_platform, francis2023_intersection_wait
- Status: manifest validation delegated to the Package A decision packet.
