# Issue 1023 Candidate-Augmented Local Full Evidence

Date: 2026-05-06

Source command:

```bash
.venv/bin/python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml \
  --output-root output/benchmarks/issue_1023 \
  --campaign-id issue1023_scenario_horizons_candidates_local_2026-05-06 \
  --mode run \
  --log-level INFO
```

Summary:

- Campaign ID: `issue1023_scenario_horizons_candidates_local_2026-05-06`.
- Config: `configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500.yaml`.
- Scenarios: 48.
- Planners: 9.
- Episodes: 1296.
- Successful planner runs: 9 of 9.
- Runtime: 1966.588 seconds.
- Benchmark success: true.
- Publication bundle: not exported.
- SNQI contract status: `fail` with warn enforcement.
- Analyzer finding: no failed, partial, not-available, fallback, or degraded planners.
- Analyzer warning: `scenario_adaptive_hybrid_orca_v1` has a small SNQI mean mismatch between
  summary row and episode recomputation.

Experimental candidate outcomes:

| planner | success | collisions | near misses | SNQI | runtime(s) |
|---|---:|---:|---:|---:|---:|
| `scenario_adaptive_hybrid_orca_v1` | 0.9097 | 0.0278 | 18.7447 | -0.0795 | 570.8285 |
| `hybrid_rule_v3_fast_progress_static_escape` | 0.9028 | 0.0278 | 20.7778 | -0.0874 | 576.6053 |

Release interpretation:

This is valid local non-Slurm evidence that the candidate-augmented long-horizon benchmark runs
end-to-end. It is not clean release evidence. The SNQI contract fails, the comparison against the
May 4 fixed-horizon evidence reports drift, and the candidates remain experimental challenger rows.

Included files:

- `campaign/`: campaign manifest, run metadata, and preflight validation JSON.
- `reports/`: compact tables, analyzer output, SNQI diagnostics, and fixed-vs-scenario comparison.
- `runs/*/summary.json`: per-planner summary files only.

Omitted intentionally:

- Raw episode JSONL files.
- Videos.
- Large local output trees.
