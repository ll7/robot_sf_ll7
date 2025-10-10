# Feature 142 – Preserve Algorithm Separation in Benchmark Aggregation

## What Changed
- `robot_sf.benchmark.full_classic.orchestrator` now mirrors the `algo` slug into `scenario_params["algo"]`, corrects mismatches with a warning (`event="episode_metadata_mismatch"`), and raises `AggregationMetadataError` on malformed episode payloads.
- `robot_sf.benchmark.aggregate` promotes algorithm-aware grouping with the fallback chain `scenario_params.algo → algo → scenario_id`, publishes `_meta` diagnostics (`group_by`, `effective_group_key`, `missing_algorithms`, `warnings`), and emits `aggregation_missing_algorithms` warnings when baselines are absent.
- `scripts/validation/test_classic_benchmark_full.sh` asserts nested metadata in `episodes.jsonl` and `_meta.effective_group_key` within the aggregated summary.
- Documentation (`docs/benchmark.md`, `docs/README.md`) records the new metadata contract and troubleshooting flow.

## Regression Matrix
| Area | Scenario | Verification |
|------|----------|--------------|
| Orchestrator metadata | Missing nested key | `tests/benchmark/test_orchestrator_metadata.py::test_injects_nested_algo_metadata` ensures the nested field is injected and top-level value set. |
| Orchestrator validation | No `algo` provided | `tests/benchmark/test_orchestrator_metadata.py::test_raises_on_missing_algo` expects `AggregationMetadataError`. |
| Aggregation fallback | Legacy records with/without nested key | `tests/benchmark/test_aggregation_algorithms.py::test_grouping_prefers_nested_algo` verifies nested-first, top-level fallback behavior. |
| Aggregation observability | Absent baseline | `tests/benchmark/test_aggregation_algorithms.py::test_warns_and_flags_missing_algorithms` + integration smoke `tests/integration/test_classic_benchmark_alg_grouping.py` confirm warning emission and `_meta.missing_algorithms`. |
| CLI/Docs workflow | Validation script & quickstart | `scripts/validation/test_classic_benchmark_full.sh` checks mirrored metadata; `progress/142-aggregation-mixes-algorithms.md` captures quickstart smoke run with missing PPO baseline warning. |

## Operational Guidance
- Treat `AggregationMetadataError` as a hard failure: regenerate episodes when either `algo` or `scenario_params.algo` is missing.
- When `_meta.missing_algorithms` is non-empty, surface the warning in release notes / dashboards and investigate benchmark worker failures before publishing metrics.
- For sandboxed environments, export `MPLCONFIGDIR` and headless display variables before running `run_full_benchmark` to avoid non-writable cache issues.

## QA / Manual Checklist
- [x] Run smoke episodes for `sf`, `ppo`, `random` with `workers=1` and confirm mirrored metadata in `results/smoke_alg_grouping/<algo>/episodes/episodes.jsonl`.
- [x] Delete one baseline (`ppo`) and aggregate remaining episodes expecting `aggregation_missing_algorithms` warning and `_meta.missing_algorithms == ["ppo"]` in `aggregated_results.json`.
- [x] Execute `uv run pytest tests/benchmark/test_orchestrator_metadata.py tests/benchmark/test_aggregation_algorithms.py tests/integration/test_classic_benchmark_alg_grouping.py`.
- [x] Run `scripts/validation/test_classic_benchmark_full.sh` to verify the automated metadata assertion gate.

## References
- Spec: `specs/142-aggregation-mixes-algorithms/spec.md`
- Research & design: `specs/142-aggregation-mixes-algorithms/research.md`, `plan.md`, `data-model.md`
- Progress notes: `progress/142-aggregation-mixes-algorithms.md`
