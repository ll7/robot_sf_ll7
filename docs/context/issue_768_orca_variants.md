# Issue 768: Benchmark ORCA variants on release surface

## Objective

Evaluate whether ORCA variants improve over current ORCA baseline on `configs/scenarios/classic_interactions_francis2023.yaml` (eval seed set), and decide: `keep current ORCA` , `replace ORCA` , or `do not pursue` .

## What was changed

### Script updates

* Updated `scripts/tools/policy_analysis_run.py` to include new policy options:
  + `socnav_orca_nonholonomic`
  + `socnav_orca_dd`
  + `socnav_orca_relaxed`
  + `socnav_hrvo`
* Added policy selection logic in `_build_socnav_policy` for these new names.
* Added `make_hrvo_policy` import to policy analysis script.
* Existing `socnav_orca` remains unchanged.

### Policy variant behavior

* `socnav_orca_nonholonomic`: ORCA config tuned for stronger heading-commit, stall guidance
* `socnav_orca_dd`: ORCA config tuned shorter horizon, lower neighbor count, lower stall-threshold
* `socnav_orca_relaxed`: ORCA config tuned less strict/relaxed obstacle and head-on bias
* `socnav_hrvo`: uses hardcoded `HRVOPlannerAdapter` path

## What was run

1. Baseline ORCA run (quick partial) with:
   - `--scenario configs/scenarios/classic_interactions_francis2023.yaml`

   - `--policy socnav_orca`

   - `--seed-set eval`

   - `--max-seeds 2`

   - output: `output/experiments/768_orca_baseline/summary.json`

2. Variant runs (same scenario/seed-set, same output layout):
   - `socnav_orca_nonholonomic` -> `output/experiments/768_orca_nonholonomic`

   - `socnav_orca_dd` -> `output/experiments/768_orca_dd`

   - `socnav_orca_relaxed` -> `output/experiments/768_orca_relaxed`

   - `socnav_hrvo` -> `output/experiments/768_hrvo`

## Summary results (replicated exactly from experiment outputs)

| policy | success | collision | ped collision | obstacle collision |
|---|---|---|---|---|
| socnav_orca | 0.6170 | 0.3830 | 0.1170 | 0.2659 |
| socnav_orca_nonholonomic | 0.6277 | 0.3723 | 0.1064 | 0.2659 |
| socnav_orca_dd | 0.6808 | 0.3191 | 0.0745 | 0.2447 |
| socnav_orca_relaxed | 0.5851 | 0.4149 | 0.1170 | 0.2979 |
| socnav_hrvo | 0.6170 | 0.3830 | 0.0851 | 0.2979 |

### Primary decision

* `replace ORCA` with ORCA-DD-style variant (`socnav_orca_dd`) because it had best success and best collision reduction in this evaluation pass.

## Verification performed

1. Code review
   - `git diff -- scripts/tools/policy_analysis_run.py` assessed manually.
2. Lint check
   - `uv run ruff check scripts/tools/policy_analysis_run.py` passed.
3. Unit tests
   - `uv run pytest -q tests/tools/test_policy_analysis_run.py` passed (23 tests).
4. Runtime benchmark check
   - All policy variants executed successfully.
   - final summary files are generated in `output/experiments/768_*` .
5. Data consistency
   - `episodes=94` for each run and expected metric structure in `summary.json` .

## Notes on algorithmic status

* This is effectively a policy-variant tuning experiment reading to existing ORCA implementation (and in-repo HRVO). 
* It is not a fully new `ORCA-DD` paper-precise algorithm implementation; it is a configuration-based variant of existing `ORCAPlannerAdapter` to behave like differential-drive configured ORCA.
* For stronger paper-precise claims, a dedicated adapter would need to implement nonholonomic ORCA (**future work**).

## Status

* Implementation complete. Verified by tests and metrics.  
* Next step: promote `socnav_orca_dd` with full seed coverage and documentation/CHANGELOG changes.
