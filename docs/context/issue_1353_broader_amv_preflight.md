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
| `socnav_sampling` | broader row | `experimental` | Fail-fast SocNav prerequisite policy. |
| `sacadrl` | broader row | `experimental` | Fallback SocNav prerequisite policy. |
| `socnav_bench` | broader row | `experimental` | Fail-fast SocNav prerequisite policy. |

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
