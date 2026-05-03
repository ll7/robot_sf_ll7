# Policy Search Nominal-Sanity Follow-up (2026-04-29)

## Scope

Advance the two smoke-passing candidates into `nominal_sanity`, probe one concrete `scenario_adaptive_orca_v1` retune, and add per-step diagnostics for `hybrid_orca_sampler_v1` before any further hybrid tuning.

## Commands

```bash
source .venv/bin/activate && uv run python scripts/validation/run_policy_search_candidate.py --candidate risk_guarded_ppo_v1 --stage nominal_sanity
source .venv/bin/activate && LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate scenario_adaptive_orca_v1 --stage nominal_sanity --workers 1
source .venv/bin/activate && uv run pytest tests/planner/test_risk_dwa_mppi_hybrid.py
source .venv/bin/activate && LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_candidate.py --candidate hybrid_orca_sampler_v1 --stage smoke
source .venv/bin/activate && LOGURU_LEVEL=WARNING uv run python scripts/validation/run_policy_search_step_diagnostics.py --candidate hybrid_orca_sampler_v1 --stage smoke
```

## Results

- `risk_guarded_ppo_v1` nominal sanity:
  - episodes: `18`
  - success: `0.1667`
  - collision: `0.2778`
  - decision: `revise`
  - dominant failures: `timeout_low_progress=10`, `static_collision=5`

- `scenario_adaptive_orca_v1` nominal sanity:
  - episodes: `18`
  - success: `0.1667`
  - collision: `0.0000`
  - near miss: `0.1111`
  - decision: `revise`
  - dominant failures: `timeout_low_progress=13`, `near_miss_intrusive=2`

- `scenario_adaptive_orca_v1` aggressive family-override retune:
  - outcome: regressed and reverted
  - temporary nominal result: `0.1667` success, `0.0000` collision, `0.2778` near miss
  - main regression: intrusive near misses rose from `2` to `5` without improving either overall success or Francis-family completions
  - interpretation: pushing speed and reducing slowdown/margins was not enough to break the bottleneck, and it worsened close-pass behavior instead

- `hybrid_orca_sampler_v1` tuned smoke rerun:
  - episodes: `3`
  - success: `0.0000`
  - collision: `0.0000`
  - decision: `pass` on execution only
  - dominant failures: `timeout_low_progress=3`

- `hybrid_orca_sampler_v1` step diagnostics on `planner_sanity_simple`, seed `101`:
  - trace: `output/policy_search/hybrid_orca_sampler_v1/step_diagnostics/smoke/latest/trace.json`
  - report: `output/policy_search/hybrid_orca_sampler_v1/step_diagnostics/smoke/latest/report.md`
  - decision counts: `orca_clear_scene=80`
  - selected heads: `orca=80`, `sampler=0`, `stop=0`
  - key finding: the sampler never activated on the smoke trace because the current short-horizon proxy kept rating ORCA as safe and high-progress on every step

## Important Execution Note

`scenario_adaptive_orca_v1` nominal sanity was interrupted repeatedly when run through the default multi-worker path on this local machine. The same evaluation completed successfully with `--workers 1`, which was sufficient for collecting the stage result.

## Hybrid Tuning Attempt

The hybrid ORCA sampler selection logic was patched so clear-scene ORCA commands no longer bypass the sampler when ORCA is safe but clearly low-progress. A regression test was added in `tests/planner/test_risk_dwa_mppi_hybrid.py` to lock that behavior.

That fix did not improve the current smoke artifact. The rerun remained `0/3` success with the same `timeout_low_progress` classification, which means the next hybrid iteration should focus on ORCA-head command quality or richer progress instrumentation rather than only the clear-scene handoff rule.

The new step trace sharpens that conclusion: the current handoff rule is not failing because the sampler loses a comparison, but because the sampler never enters the comparison path at all on the smoke slice. The adapter records `orca_clear_scene` for every step, so the next hybrid change should target the progress proxy or clear-scene bypass contract rather than sampler weights.

## Interpretation

Neither smoke leader passed nominal sanity.

`risk_guarded_ppo_v1` is currently not viable for promotion because it regresses badly on classic interactions and introduces a high collision rate.

`scenario_adaptive_orca_v1` is still the safer nominal-sanity candidate because it stays collision-free, but the first local retune attempt shows that simply making the family overrides more aggressive is not enough. The current bottleneck appears to be the quality of the bypass/commit behavior in classic scenes and unresolved Francis progress, not raw speed alone.

## Next Step

Keep `scenario_adaptive_orca_v1` on its safer pre-retune settings until a more targeted classic-scene hypothesis is available.

For `hybrid_orca_sampler_v1`, use the new per-step diagnostics as the main tuning surface. The immediate next code change should force the sampler path to engage when route-level goal distance is not improving, even if the local short-horizon ORCA rollout still reports positive progress.