# Quickstart Validation – Feature 142

- Ran smoke episode generation sequentially for `sf`, `ppo`, `random` using `run_full_benchmark` with `workers=1`, `smoke=True`, headless display vars, and custom `MPLCONFIGDIR` to satisfy sandbox permissions. Outputs saved under `results/smoke_alg_grouping/<algo>/episodes/episodes.jsonl`.
- Simulated a missing baseline by removing the PPO directory before aggregation.
- Aggregated surviving episodes via `compute_aggregates_with_ci` with `expected_algorithms={"sf", "ppo", "random"}`. The run emitted the expected warning (`event="aggregation_missing_algorithms"`) and produced `results/smoke_alg_grouping/aggregated_results.json` with `_meta.missing_algorithms == ["ppo"]` and `_meta.effective_group_key == "scenario_params.algo | algo | scenario_id"`.
- Verified first episode record mirrors algorithm metadata (`algo` == `scenario_params["algo"]`) and captured `_meta.warnings` reflecting the absent PPO baseline.
