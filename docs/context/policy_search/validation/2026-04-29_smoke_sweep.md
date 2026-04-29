# Policy Search Smoke Sweep (2026-04-29)

## Scope

Run every currently implemented non-training candidate through the local `smoke` stage and capture the outcome in reproducible markdown artifacts.

## Commands

```bash
source .venv/bin/activate && uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_orca_sampler_v1 --stage smoke
source .venv/bin/activate && uv run python scripts/validation/run_policy_search_candidate.py --candidate risk_guarded_ppo_v1 --stage smoke
source .venv/bin/activate && uv run python scripts/validation/run_policy_search_candidate.py --candidate scenario_adaptive_orca_v1 --stage smoke
source .venv/bin/activate && uv run python scripts/validation/run_policy_search_candidate.py --candidate planner_selector_v1 --stage smoke
source .venv/bin/activate && uv run python scripts/validation/run_policy_search_candidate.py --candidate mpc_clearance_sampler_v1 --stage smoke
source .venv/bin/activate && uv run python scripts/tools/compare_policy_search_candidates.py output/policy_search/hybrid_orca_sampler_v1/smoke/latest/summary.json output/policy_search/risk_guarded_ppo_v1/smoke/latest/summary.json output/policy_search/scenario_adaptive_orca_v1/smoke/latest/summary.json output/policy_search/planner_selector_v1/smoke/latest/summary.json output/policy_search/mpc_clearance_sampler_v1/smoke/latest/summary.json --output docs/context/policy_search/validation/smoke_sweep_comparison
```

## Results

- `risk_guarded_ppo_v1`: `3/3` smoke successes, `0/3` collisions.
- `scenario_adaptive_orca_v1`: `3/3` smoke successes, `0/3` collisions.
- `hybrid_orca_sampler_v1`: `0/3` smoke successes, `0/3` collisions, all failures classified as `timeout_low_progress`.
- `planner_selector_v1`: `0/3` smoke successes, `0/3` collisions, all failures classified as `timeout_low_progress`.
- `mpc_clearance_sampler_v1`: `0/3` smoke successes in the original sweep, superseded by
  `2026-04-29_mpc_clearance_sampler_v1_speed_cap_retune.md`.

## Sweep Artifact

- Consolidated comparison: `docs/context/policy_search/validation/smoke_sweep_comparison/comparison.md`

## Important Local Fix During Sweep

`scenario_adaptive_orca_v1` initially failed because family-override runs were handed to the benchmark as in-memory scenario dicts without preserving a usable scenario-path base for relative `map_file` resolution. The runner now materializes those paths before list-based execution, and the candidate clears the smoke stage after that fix.

## Interpretation

Smoke is only an execution gate. It tells us whether a candidate can run cheaply and whether it obviously stalls or collides on the simplest planner-sanity slice.

The current smoke leader set in this sweep was `risk_guarded_ppo_v1` and `scenario_adaptive_orca_v1`. The mpc candidate was later retuned and should be read from the follow-up note rather than this original sweep snapshot.

## Next Local Priority

Promising local follow-up should start with `nominal_sanity` for `risk_guarded_ppo_v1` and `scenario_adaptive_orca_v1`. The low-progress candidates need tuning before broader local evaluation is worth the runtime.
