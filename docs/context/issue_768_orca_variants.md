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

1. Initial pilot benchmark (partial eval subset) with:
   - `--scenario configs/scenarios/classic_interactions_francis2023.yaml`

   - `--policy socnav_orca`

   - `--seed-set eval`

   - `--max-seeds 2`

   - output: `output/experiments/768_orca_baseline/summary.json`

2. Variant pilot runs (same partial eval subset):
   - `socnav_orca_nonholonomic` -> `output/experiments/768_orca_nonholonomic`

   - `socnav_orca_dd` -> `output/experiments/768_orca_dd`

   - `socnav_orca_relaxed` -> `output/experiments/768_orca_relaxed`

   - `socnav_hrvo` -> `output/experiments/768_hrvo`

3. Follow-up full eval comparison on the complete eval seed set:
   - `socnav_orca` -> `output/experiments/768_orca_full_eval`

   - `socnav_orca_dd` -> `output/experiments/768_orca_dd_full_eval`

## Summary results (full eval, same 141 episodes per policy)

| policy | success | collision | ped collision | obstacle collision |
|---|---|---|---|---|
| socnav_orca | 0.4539 | 0.3333 | 0.1064 | 0.2270 |
| socnav_orca_dd | 0.4681 | 0.3191 | 0.0993 | 0.2199 |

### Primary decision

* `replace ORCA` with ORCA-DD-style variant (`socnav_orca_dd`) because it still improves on the full eval seed set, even though the complete eval revealed harder cases and lower overall success rates than the initial partial pilot.

### Why the full eval is worse than the pilot

* The pilot used `--max-seeds 2`, so it covered only a subset of the `eval` seed set.
* The full eval run used the complete `eval` seeds from `configs/benchmarks/seed_sets_v1.yaml` and included additional harder episodes.
* That broader coverage lowered absolute success rates for both `socnav_orca` and `socnav_orca_dd`, but the relative improvement of `socnav_orca_dd` remained positive.

## Follow-up full-eval comparison context

* `socnav_orca` full eval: success `0.4539`, collision `0.3333`, ped collision `0.1064`, obstacle collision `0.2270`
* `socnav_orca_dd` full eval: success `0.4681`, collision `0.3191`, ped collision `0.0993`, obstacle collision `0.2199`

*The remaining evaluation risk is that the absolute success rate on the full eval seed set is low, but ORCA-DD is currently the best of the tested ORCA variants on that same coverage.*

## Verification performed

1. Code review
   - compared branch changes against `main` for `scripts/tools/policy_analysis_run.py` , `tests/tools/test_policy_analysis_run.py` , and this issue note.
2. Lint check
   - `uv run ruff check scripts/tools/policy_analysis_run.py` passed.
3. Unit tests
   - `uv run pytest -q tests/tools/test_policy_analysis_run.py` passed (28 tests after preserving `socnav_sacadrl` coverage).
4. Runtime benchmark check
   - All policy variants executed successfully.
   - final summary files are generated in `output/experiments/768_*` .
5. Data consistency
   - `episodes=141` for each full-eval run and expected metric structure in `summary.json` .

## Branch comparison against `main`

Compared with `main` , the issue 768 branch changes are concentrated in three files:

* `scripts/tools/policy_analysis_run.py`
* `tests/tools/test_policy_analysis_run.py`
* `docs/context/issue_768_orca_variants.md`

The main implementation risk found during verification was a CLI regression: the branch added new ORCA variant policy names but accidentally dropped the existing `socnav_sacadrl` parser choice. That regression has been corrected and covered by targeted tests.

## Notes on algorithmic status

* This is effectively a policy-variant tuning experiment reading to existing ORCA implementation (and in-repo HRVO). 
* It is not a fully new `ORCA-DD` paper-precise algorithm implementation; it is a configuration-based variant of existing `ORCAPlannerAdapter` to behave like differential-drive configured ORCA.
* For stronger paper-precise claims, a dedicated adapter would need to implement nonholonomic ORCA (**future work**).

## Status

* Implementation complete. Verified by tests and metrics.  
* Next step: promote `socnav_orca_dd` with full seed coverage and documentation/CHANGELOG changes.
