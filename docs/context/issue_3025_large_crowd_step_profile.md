# Issue #3025 Large-Crowd Step Profile

Date: 2026-06-19  
Status: diagnostic synthesis; no speedup claim.

## Scope

Parent issue: <https://github.com/ll7/robot_sf_ll7/issues/3025>  
Supporting PRs: <https://github.com/ll7/robot_sf_ll7/pull/3121>, <https://github.com/ll7/robot_sf_ll7/pull/3122>, <https://github.com/ll7/robot_sf_ll7/pull/3124>, <https://github.com/ll7/robot_sf_ll7/pull/3125>

The merged PRs added a dedicated profiling surface in
`scripts/validation/performance_smoke_test.py` for issue #3025 and documented the contract in tests.
No optimizer changes were included in this slice.

## Evidence Limit

This note summarizes post-merge tooling shape only:

- command-contract changes in `scripts/validation/performance_smoke_test.py`,
- contract tests in `tests/perf/test_large_crowd_step_profile_contract.py` and
  `tests/validation/test_performance_smoke_test.py`,
- reproducible runs using the issue-scoped presets/modes.

It does **not** include benchmark-strength comparative runs or before/after runtime
claims.

## Current post-merge state

1. `--large-crowd-profile` preset is available and sets:
   - scenario: `configs/scenarios/single/dense_pedestrian_stress.yaml`
   - measured steps: `20` (unless `--scenario` or `--step-samples` is explicitly set).
2. Large-crowd metadata uses `dense_pedestrian_stress` and resolves to 17 pedestrians from
   the sampled profile path.
3. Step-profile support is now:
   - optional (`--step-profile`)
   - includes bounded hotspots (`--step-profile-limit`, default 10)
   - portable file labels in hotspot rows (`file`, `line`, `name`, `ncalls`, `tottime`, `cumtime`).
4. Warmup semantics now split by profile mode:
   - default `--step-profile-mode steady` injects one warmup step when `--step-profile` is used.
   - explicit `--step-profile-mode cold-start` runs with zero warmup and captures full startup/JIT path.

## Diagnostic observation from merged PR runs

- `--large-crowd-profile --step-profile` steady path (explicit warmup routing) showed cold-path warmup
  dominated by the added warmup measurement and then steady step behavior around
  `0.011–0.012 s` for the steady loop.
- `--large-crowd-profile --step-profile --step-profile-mode cold-start` showed startup-dominant
  first-step timing around `~5.5 s` with steady loop still around `~0.011 s`.

The issue is therefore still at profiling/diagnostic stage; no runtime optimization has been validated
as a safe speedup yet.

## Follow-up profiling scout on current main (2026-06-19)

After PR #3140, a follow-up scout reran the steady large-crowd profile on current
`origin/main` with the same diagnostic contract:

```bash
uv run python scripts/validation/performance_smoke_test.py \
  --large-crowd-profile \
  --step-profile \
  --json-output <artifact-dir>/steady-profile.json \
  --telemetry-output <artifact-dir>/steady-profile-telemetry.jsonl \
  --num-resets 2 \
  --step-samples 20 \
  --step-profile-mode steady \
  --step-profile-limit 8
```

Observed diagnostic summary:

- scenario: `configs/scenarios/single/dense_pedestrian_stress.yaml`
- pedestrian count: `17`
- measured step samples: `20`
- steady step loop: `0.011061429977416992 s`
- steady throughput: `1717.68026727018 steps/sec`
- top cumulative hotspots:
  - `robot_sf/gym_env/robot_env.py:819` `step`: `0.011843816 s`
  - `robot_sf/sim/simulator.py:279` `step_once`: `0.008182197 s`
  - `fast-pysf/pysocialforce/simulator.py:246` `compute_forces`: `0.006406373 s`
  - `fast-pysf/pysocialforce/simulator.py:57` `_sum_forces_explicitly`: `0.006397025 s`
  - `fast-pysf/pysocialforce/forces.py:609` `__call__`: `0.00304015 s`

This confirms the previous diagnostic: the dominant steady-step cost remains inside the fast-pysf
force-computation chain, not in a clear low-risk Robot SF wrapper-only path. The wrapper-visible
overhead should not be treated as a speedup target unless a same-contract before/after comparison
first proves a semantics-preserving change.

## Follow-up boundary

The next optimization boundary must be measured with a fresh baseline/target comparison on the same
profiling contract:

1. fix a single optimization target, preferably under the fast-pysf force-computation path identified
   above, with `--step-profile` command and explicit mode,
2. collect a baseline snapshot and a post-change snapshot with the same mode/warmup values,
3. run correctness checks that the issue already uses in CI-contract tests,
4. only then interpret direction of change.

Current optimization mode should be chosen after correctness validation, and only as a diagnostic-to-action
step under this issue workflow.

## Related commands and contracts

Suggested reproducible entry command:

```bash
scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/validation/performance_smoke_test.py --large-crowd-profile --step-profile
```

Cold-start-only snapshot:

```bash
scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/validation/performance_smoke_test.py \
  --large-crowd-profile --step-profile --step-profile-mode cold-start
```

Contract proofs are in:

- <https://github.com/ll7/robot_sf_ll7/pull/3121>
- <https://github.com/ll7/robot_sf_ll7/pull/3122>
- <https://github.com/ll7/robot_sf_ll7/pull/3124>
- <https://github.com/ll7/robot_sf_ll7/pull/3125>
