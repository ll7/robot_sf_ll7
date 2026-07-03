# Issue #4249 Recovery Evidence

This directory records the recovered #3216 rank-stability report from completed job 13274
artifacts. It is diagnostic recovery evidence only: no benchmark campaign was rerun, no
Slurm or GPU job was submitted, and no paper or dissertation claim is promoted here.

Source artifacts were retrieved from `imech192`:

`~/git/robot_sf_ll7.worktrees/issue-3216-refresh-hash/output/benchmarks/issue_3216_headline_ci/issue3216_s20_headline_ci/`

Recovered locally with:

```bash
uv run python scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py \
  --campaign output/issue_4249_recovery/issue3216_s20_headline_ci \
  --rank-metric snqi \
  --invalid-rank-metric-reason 'SNQI contract status=fail@enforcement=warn in completed job 13274 artifacts; expected pre-#4247' \
  --expected-planners-from-config configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml \
  --output-dir docs/context/evidence/issue_4249_recovery_3216_rank_stability
```

Generated input rows were written to ignored local output:

- `output/issue_4249_recovery/issue3216_s20_headline_ci/reports/headline_rows.json`
- SHA-256: `5ad462f169a634d82764b22732610f10d2be42e1fe29fc03e9ca31afe74d21d8`

Recovered report outputs:

- `result.json`: `blocked_until_run`, 315 counted cells, 0 excluded cells, complete configured
  planner grid, SNQI warning preserved as an invalid-rank-metric reason.
  SHA-256: `d80af874e5826e5dad6d34008d5cf00ea9bd266336d2e0601c04c0815b5a430a`.
- `report.md`: human-readable report generated from the same `result.json`.
  SHA-256: `090d840b55a2e468191bacd5c2fcbb3993b94c51e31b09ab1321d82ca0391927`.
