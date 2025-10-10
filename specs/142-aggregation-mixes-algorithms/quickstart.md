# Quickstart – Verifying Algorithm Separation in Aggregation

This quickstart validates the new per-algorithm grouping guarantees and missing-algorithm warnings in the classic benchmark pipeline.

## 1. Environment Setup
1. Ensure submodules are initialized and dependencies installed:
   ```bash
   git submodule update --init --recursive
   uv sync && source .venv/bin/activate
   ```
2. (Optional) Activate headless display settings if running on CI:
   ```bash
   export DISPLAY=
   export MPLBACKEND=Agg
   export SDL_VIDEODRIVER=dummy
   ```

## 2. Generate Sample Episodes
Run a smoke-sized classic benchmark to produce JSONL episodes for all baselines:
```bash
uv run python scripts/run_social_navigation_benchmark.py --smoke --output-root results/smoke_alg_grouping
```
Key expectations:
- Each baseline (`sf`, `ppo`, `random`) writes to `results/smoke_alg_grouping/<algo>/episodes/episodes.jsonl`.
- Opening a JSONL line should now show both `"algo": "sf"` and `"scenario_params": { ..., "algo": "sf" }`.

## 3. Aggregate Episodes With Warning Coverage
1. Delete the episodes for one algorithm to simulate a missing baseline (e.g., remove the PPO directory).
2. Aggregate the remaining episodes using the benchmark script:
   ```bash
   uv run python scripts/run_social_navigation_benchmark.py --aggregate-only --episodes-root results/smoke_alg_grouping
   ```
3. Observe the Loguru warning:
   - Message should contain `aggregation_missing_algorithms` with `missing=['ppo']`.
   - Exit status remains zero.

## 4. Inspect Aggregated Output
Open `results/smoke_alg_grouping/aggregated_results.json`:
- Verify groups exist for each algorithm present.
- Confirm the `_meta.missing_algorithms` list matches the warning.
- Ensure `_meta.group_by` reports `scenario_params.algo` and `_meta.effective_group_key` includes the fallback chain.

## 5. Regression Tests
Run the targeted pytest module (to be added in implementation) that asserts grouping and warning semantics:
```bash
uv run pytest tests/benchmark/test_aggregation_algorithms.py
```

## 6. Documentation & Changelog
After validation, update:
- `docs/benchmark.md` with a note on algorithm grouping keys.
- `CHANGELOG.md` summarizing the fix and warning behaviour.

Following these steps confirms the feature end-to-end: metadata injection, aggregation fallback, warnings, and doc updates.
