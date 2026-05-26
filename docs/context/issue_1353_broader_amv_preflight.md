# Issue #1353 Broader AMV Baseline Preflight 2026-05-20

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1353>

## Decision

Issue #1353 should reuse the paired nominal/stress protocol proven by Issue #1344, but expand the
planner rows before campaign execution. This slice records the broader row list in versioned configs
and validates that the benchmark preflight path can enumerate the nominal and stress campaigns
without running the full broader evidence bundle.

This is preparatory work only. It does not close #1353 because the issue still requires a clean
broader-baseline campaign, a delta report against the #1344 primary rows, compact evidence under
`docs/context/evidence/`, and follow-up issues for any planner-specific failures.

## Configs

The broader preflight configs are stacked on the Issue #1344 primary protocol:

- `configs/benchmarks/issue_1353_paired_nominal_v1_broader_baselines.yaml`
- `configs/benchmarks/issue_1353_paired_stress_broader_baselines.yaml`

Both preserve the #1344 protocol constants:

- `paper_facing: false`
- `paper_profile_version: paper-matrix-v1`
- `amv_profile: amv-paper-v1` with warn-mode coverage enforcement
- S3 `eval` seed policy from `configs/benchmarks/seed_sets_v1.yaml`
- SNQI v3 weights, baseline, and warn-mode contract diagnostics
- differential-drive kinematics
- `horizon: 100`, `dt: 0.1`, single-worker execution, and `stop_on_failure: false`

The intentional changes are planner coverage and the non-paper-facing
`paper_interpretation_profile`, which is renamed to `issue-1353-broader-amv-preflight` so generated
artifacts are traceable to this preparatory scope.

## Row List

The broader configs include the #1344 primary rows plus currently configured experimental baseline
families from the all-planners camera-ready matrix:

| Planner key | Source role | Benchmark profile | Notes |
| --- | --- | --- | --- |
| `goal` | #1344 primary row | `baseline-safe` | Goal-following control baseline. |
| `social_force` | #1344 primary row | `baseline-safe` | Local Social Force baseline. |
| `orca` | #1344 primary row | `baseline-safe` | Uses `socnav_missing_prereq_policy: fallback`, as in #1344. |
| `ppo` | broader row | `experimental` | Uses `configs/baselines/ppo_15m_grid_socnav.yaml`; adapter impact evaluation remains enabled. |
| `prediction_planner` | broader row | `experimental` | Uses `configs/algos/prediction_planner_camera_ready.yaml`. |
| `socnav_sampling` | broader row | `experimental` | `skip-with-warning` SocNav prerequisite policy; missing assets are recorded as unavailable rather than blocking the campaign. |
| `sacadrl` | broader row | `experimental` | Fallback SocNav prerequisite policy. |
| `socnav_bench` | broader row | `experimental` | `skip-with-warning` SocNav prerequisite policy; missing assets are recorded as unavailable rather than blocking the campaign. |

2026-05-25 implementation update: campaign-level `benchmark_success` is now anchored on `core`
planner rows whenever a campaign has explicit core rows. Experimental rows such as `socnav_bench`
still report `not_available` when their external prerequisites are absent, and they remain visible
in warnings/tables, but they no longer make an otherwise core-successful campaign exit as failed.
Campaign summaries expose `benchmark_success_basis`, `core_successful_runs`, and `core_total_runs`
so downstream reports can distinguish core success from all-row availability.

## Caveats

- These configs lock the row list and preflight contract, but they do not prove planner execution.
- Unsupported, degraded, fallback-only, or unavailable rows must remain separated in the final
  Issue #1353 report rather than being treated as successful evidence.
- The Issue #1344 primary campaigns reported `amv_coverage_status=warn` and
  `snqi_contract_status=fail`; Issue #1353 must carry those interpretation boundaries forward
  unless a later claim-scope review changes them.
- Raw preflight output under `output/benchmarks/issue_1353/` is disposable and worktree-local. The
  durable artifact for this slice is this note plus the tracked config files.

## Validation

Both configs were validated in preflight mode on 2026-05-20 before any full Issue #1353 campaign:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1353_paired_nominal_v1_broader_baselines.yaml \
  --mode preflight \
  --output-root output/benchmarks/issue_1353 \
  --campaign-id issue_1353_nominal_broader_preflight

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1353_paired_stress_broader_baselines.yaml \
  --mode preflight \
  --output-root output/benchmarks/issue_1353 \
  --campaign-id issue_1353_stress_broader_preflight
```

The preflight proof is sufficient for row-list readiness. The final Issue #1353 proof still requires
running the broader paired campaign bundle and writing the delta report against Issue #1344.

Additional local checks for this slice:

```bash
uv run pytest tests/benchmark/test_issue_1353_broader_amv_configs.py \
  tests/benchmark/test_camera_ready_campaign.py -k 'preflight or issue_1353' -q

BASE_REF=origin/issue-1344-paired-amv-report scripts/dev/check_docs_proof_consistency_diff.sh
```

## Issue #1353: 2026-05-25 Row-Contract Update and Slurm Submission

Issue #1353 accepted a row-contract revision on 2026-05-25: SocNavBench-family rows should be
reported as unavailable/excluded while #1456 remains blocked, not kept as fail-fast participants.
Branch `issue-1353-broader-amv-row-contract` updates the nominal and stress configs accordingly:

- `socnav_sampling`: `socnav_missing_prereq_policy: skip-with-warning`
- `socnav_bench`: `socnav_missing_prereq_policy: skip-with-warning`

Validation from the worktree:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1353_paired_nominal_v1_broader_baselines.yaml \
  --mode preflight \
  --output-root output/benchmarks/issue_1353_preflight_20260525 \
  --campaign-id issue_1353_nominal_pref_20260525 \
  --log-level INFO

uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_1353_paired_stress_broader_baselines.yaml \
  --mode preflight \
  --output-root output/benchmarks/issue_1353_preflight_20260525 \
  --campaign-id issue_1353_stress_pref_20260525 \
  --log-level INFO

git diff --check
uv run ruff check tests/benchmark/test_issue_1353_broader_amv_configs.py
uv run pytest -q tests/benchmark/test_issue_1353_broader_amv_configs.py --no-cov
```

Results: both preflights exited 0 and wrote eight-row matrix summaries. The focused config test
passed (`2 passed`), and `git diff --check` plus Ruff on the Python test passed.

Submitted Slurm runs from commit `20c25d6d`:

- Job `12619`, nominal, label `issue1353-nominal-rowcontract`, config
  `configs/benchmarks/issue_1353_paired_nominal_v1_broader_baselines.yaml`
- Job `12620`, stress, label `issue1353-stress-rowcontract`, config
  `configs/benchmarks/issue_1353_paired_stress_broader_baselines.yaml`

Both jobs started at `2026-05-25T06:03:03` on `l40s` and printed the expected config paths and
labels to:

- `output/slurm/12619-camera-ready-benchmark.out`
- `output/slurm/12620-camera-ready-benchmark.out`

Job `12619` finished `FAILED 2:0` after producing a complete nominal campaign bundle. The failure
is the current workflow's nonzero exit for `benchmark_success=false`, triggered by the accepted
`socnav_bench` `not_available` row:

- Campaign root:
  `output/benchmarks/issue_1353/issue_1353_paired_nominal_v1_broader_baselines_issue1353-nominal-rowcontract_20260525_060323`
- Summary: `total_runs=8`, `successful_runs=7`, `total_episodes=84`, `benchmark_success=false`
- Warning: `socnav_bench` `not_available` because SocNavBench control-pipeline assets are missing.

This is not a config-path or missing-output failure, but it does make the SLURM job state ambiguous.
Follow-up issue #1487 tracks separating accepted unavailable/excluded rows from unexpected failed
campaign exits.

Follow-up checks:

```bash
squeue -j 12619,12620 --format='%i %j %T %P %Q %y %b %M %l %S %R'
sacct -j 12619,12620 --format=JobID,JobName%30,State,ExitCode,Partition,Elapsed,Start,End -P
tail -n 120 output/slurm/12619-camera-ready-benchmark.out
tail -n 120 output/slurm/12620-camera-ready-benchmark.out
```

After completion, analyze the campaign roots under `output/benchmarks/issue_1353/`, preserve compact
summaries under `docs/context/evidence/`, and write the #1353 delta report against the closed #1344
primary paired AMV result. Fallback, degraded, and unavailable SocNav rows must stay caveated and
must not be counted as successful benchmark evidence.

## Issue #1353: 2026-05-26 Current-Main Result

After PR #1492 merged, current `main` (`469d40c9`) includes core-anchored campaign success
semantics. The 2026-05-26 SLURM sweep also exposed that the shared virtualenv was missing the ORCA
`rvo2` extra; `uv sync --all-extras` restored the dependency and
`.venv/bin/python -c 'import rvo2'` passed before the final submissions. Follow-up issue #1517
tracks making this optional-extra check fail fast before future camera-ready SLURM campaigns.

Fresh current-main jobs:

- `12626`, nominal, `COMPLETED 0:0`, `a30`, runtime `00:03:16`
- `12625`, stress, `COMPLETED 0:0`, `l40s`, runtime `00:21:10`
- Related compact cross-kinematics closure for #1354: `12624`, `COMPLETED 0:0`, `a30`

Campaign roots:

- nominal:
  `output/benchmarks/issue_1353/issue_1353_paired_nominal_v1_broader_baselines_issue1353-nominal-main-rvo2-20260526_20260526_062603`
- stress:
  `output/benchmarks/issue_1353/issue_1353_paired_stress_broader_baselines_issue1353-stress-main-rvo2-20260526_20260526_062603`
- #1354 compact cross-kinematics:
  `output/benchmarks/issue_1354_20260526/paper_cross_kinematics_v1_issue1354-paper-compact-main-rvo2-20260526_20260526_062604`

Results summary:

| Surface | Total runs | Successful rows | Core rows | Episodes | Campaign success | AMV coverage | SNQI contract |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| #1353 nominal | 8 | 7 | 3/3 | 84 | true, `basis=core` | warn | warn |
| #1353 stress | 8 | 7 | 3/3 | 1008 | true, `basis=core` | warn | fail |
| #1354 compact cross-kinematics | 9 | 9 | 9/9 | 9 | true, `basis=core` | pass | warn |

The analyzer found no consistency findings for all three fresh campaign roots. Compact evidence is
preserved in `docs/context/evidence/issue_1353_broader_amv_2026-05-26/`.

Planner-level interpretation boundaries:

- `socnav_bench` remains `not_available` in both #1353 surfaces because SocNavBench
  control-pipeline assets are missing. This is an accepted unavailable row, not successful evidence.
- #1353 nominal adds runnable rows beyond #1344, but success remains low: `ppo` and
  `prediction_planner` reach `0.3333` success; `goal`, `orca`, `sacadrl`, and `socnav_sampling`
  are at `0.2500`; `social_force` remains `0.0000`.
- #1353 stress also adds runnable rows, but the strongest stress success is still modest:
  `ppo=0.2222`, `orca=0.1667`, `socnav_sampling=0.1528`, `prediction_planner=0.0694`,
  `sacadrl=0.0208`, and `goal/social_force=0.0000`.
- `prediction_planner` is the main runtime hotspot: `59.1s` nominal and `613.4s` stress.
- The stress `snqi_contract_status=fail` preserves the #1344 caution: these results are useful
  benchmark evidence, but SNQI should not be promoted into paper-facing claims without an explicit
  claim-scope decision.

Validation commands:

```bash
sacct -j 12624,12625,12626 \
  --format=JobID,JobName%35,Partition,State,ExitCode,Elapsed,Timelimit,NodeList%30

uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1353/issue_1353_paired_nominal_v1_broader_baselines_issue1353-nominal-main-rvo2-20260526_20260526_062603

uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1353/issue_1353_paired_stress_broader_baselines_issue1353-stress-main-rvo2-20260526_20260526_062603

uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/issue_1354_20260526/paper_cross_kinematics_v1_issue1354-paper-compact-main-rvo2-20260526_20260526_062604
```
