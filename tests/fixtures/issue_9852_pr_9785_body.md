## Summary

`pr_ready_check.sh` selected tests from three sources: the fixed `core_test_paths` list in `run_tests_parallel.sh`, the shared optional-test allowlist, and the PR's changed test files. Two test roots were in none of them, so a green readiness report silently omitted them and a PR body could claim coverage that no lane produced.

This adds an explicit extended lane for those roots, and makes every readiness run state which lanes ran and which roots it did not cover.

## Linked Issues

- Closes #9754

## Issue Relationship Mirror

- Parent: none
- Blocked by: none
- Blocking: none

## Stack / Dependency

- Base dependency: `origin/main` at `46875697bafcfcf5defc0c24f2552695f919d8d5`.
- Safe to review independently: yes; CI/readiness tooling plus one test module and the shared allowlist.

## Gap this closes

Verified against current `main`, the roots named in #9754 resolve as:

| Root | core list | optional allowlist | before this PR |
| --- | --- | --- | --- |
| `tests/validation` (150 test files) | no | one file only | **never run** |
| `tests/maps` (11 test files) | no | no | **never run** |
| `tests/benchmark` (583) | no | yes (`tests/benchmark/`) | optional lane |
| `tests/training` (109) | no | yes (`tests/training/`) | optional lane |

So the two roots that no lane reached were `tests/validation` and `tests/maps`.

## What Changed

- **`scripts/dev/pr_ready_check.sh`**
  - Computes the uncovered test roots explicitly instead of letting them fall through.
  - Marks the extended lane required when the diff touches `robot_sf/`, `configs/`, `maps/`, or one of the uncovered roots.
  - Runs the extended lane over those roots when required.
  - Prints a lane-coverage summary, and writes it to `output/validation/pr_ready/lane_coverage.txt`, naming the roots the run did **not** cover so a PR body cannot imply they were proven.
- **`tests/support/optional_test_allowlist.txt`** — eight files inside those roots need optional extras (`pandas`, `pygame`, `stable_baselines3`, `optuna`). They move to the optional lane so the extended core-style run stays valid in a no-extras profile. The allowlist stays the single source of truth for all three consumers.
- **`tests/dev/test_readiness_lane_coverage.py`** — encodes the coverage property so future lane drift fails here.

## Why the allowlist had to change with it

Running the new lane over the roots verbatim failed closed in a no-extras profile, which is the profile the core lane is designed for:

- collection errors: `No module named 'pandas'`, `'pygame'`
- 4 test failures: `No module named 'stable_baselines3'`, `'optuna'`

`pandas` and `scipy` are in the `benchmark` extra and `stable-baselines3`/`optuna` in the `training` extra (`pyproject.toml`), not base dependencies. Pulling those files into a core-style lane would have turned unrelated readiness runs red, so they are allowlisted instead of being forced in. The extended lane therefore runs the remaining ~145 validation and ~10 maps tests.

## Validation / Proof

- `tests/dev/test_readiness_lane_coverage.py` — **4 passed**.
- Same module against **unmodified** `main` `pr_ready_check.sh` — **2 failed**, with the actionable message: `these test roots belong to no pr_ready_check lane, so a green readiness run never executes them: ['tests/maps', 'tests/validation']`. The guard is therefore not a tautology.
- `tests/dev/test_readiness_lane_coverage.py tests/test_ci_script_contract.py` — **361 passed**.
- `tests/dev/test_readiness_lane_coverage.py tests/test_ci_script_contract.py tests/dev/test_pr_ready_preflight.py tests/dev/test_pr_ready_artifact_contract.py` — **445 passed**.
- Extended lane end to end, no-extras profile, 8 workers:
  `ROBOT_SF_TEST_LANE=core PYTEST_FAST_FAIL=0 scripts/dev/run_tests_parallel.sh --lane core tests/validation tests/maps` — **exit 0**, 2065 tests, `real 1m36s`.
- `ruff check`, `ruff format --check`, `bash -n scripts/dev/pr_ready_check.sh`, `git diff --check` all clean.

### Pre-existing failure, not caused by this change

`tests/dev/test_check_worktree_optional_deps.py` reports **3 failed, 14 passed** in this worktree. Verified pre-existing: with this branch's changes stashed, unmodified `main` reports the identical 3 failed / 14 passed. The cause is this local venv's reduced extras profile, not the allowlist edit.

## Risks / Rollback

- Readiness runs that touch `robot_sf/`, `configs/` or `maps/` get about 1m36s slower with 8 workers. Runs with no such change are unaffected because the lane does not trigger.
- The new lane is additive; no existing lane, path list or allowlist entry is removed, so no previously passing test stops running.
- Revert restores the previous behavior, including the silent omission.

## Docs / Provenance

No production or benchmark code changed. No benchmark artifacts, seeds, or evaluation runs were produced or claimed.

## Downstream Propagation

None: readiness tooling and its test allowlist. No schema, metric, or released-arm behavior is touched.

## Follow-Up / Residual Scope

- The third item in #9754 (rejecting "full suite" PR-body claims backed only by the core lane) is **not** done here. The summary and the `lane_coverage.txt` receipt make the omission visible in readiness output, but no contract checker consumes the receipt yet. That is a separate change and should not be bundled silently into this one.
- Whether `tests/validation` and `tests/maps` should instead become permanent core or optional lanes is a maintainer call; this PR keeps the change reversible and path-triggered rather than re-baselining the whole suite.

## Reviewer Notes

- Confirm the extended lane's trigger set is right: it fires on `robot_sf/`, `configs/`, `maps/` and the uncovered roots, not on docs-only changes.
- Confirm the eight allowlist additions are genuinely optional-extra dependent; each was observed failing to import or failing on a missing module in a no-extras run, and the failing module names are recorded above.
- The coverage test asserts the four roots named in #9754, not every on-disk root. Asserting all roots would claim a readiness-lane contract the repository does not make, since other roots run in separate CI phases.

<!-- pr-contract:v2
change_class: tooling
linked_issues:
  closes: [9754]
  relates: []
deferred_work:
  status: partial
  issues: [9754]
  reason: "PR-body contract rejection of core-only full-suite claims is not implemented here."
evidence:
  applicability: na
  tier: null
  result: na
domain_approval:
  required: false
  status: not_required
  domains: []
  note: "NA - CI/readiness tooling; no experimental or benchmark claim."
performance:
  claimed: false
-->

