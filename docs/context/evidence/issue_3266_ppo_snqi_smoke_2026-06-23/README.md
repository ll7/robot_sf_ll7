# Issue 3266 PPO/SNQI Smoke Evidence

Date: 2026-06-23

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/3266>

This bundle stores compact, reviewable evidence from the smallest post-repair
scenario-horizon smoke rerun for the PPO/SocNav quality index (SNQI) validity
blocker. The run is valid blocker-resolution evidence for this one-scenario smoke
slice. It is not paper-facing Results evidence or a full scenario-horizon promotion.

## Source Config

- `configs/benchmarks/issue_3266_scenario_horizon_ppo_snqi_smoke.yaml`
- Scenario matrix: `configs/scenarios/classic_interactions_francis2023.yaml`
- Scenario candidate: `francis2023_blind_corner`
- Planners: `goal`, `ppo`
- Seed: `219`

## Source Command

```bash
rtk scripts/dev/run_worktree_shared_venv.sh -- python scripts/dev/run_compact_validation.py \
  --timeout-seconds 1800 \
  -- scripts/dev/run_worktree_shared_venv.sh -- python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_3266_scenario_horizon_ppo_snqi_smoke.yaml \
  --output-root output/benchmarks/issue_3266 \
  --campaign-id issue3266_ppo_snqi_smoke_live \
  --mode run \
  --skip-publication-bundle \
  --log-level WARNING
```

Raw reports remain ignored under `output/benchmarks/issue_3266/` because they are reproducible from
the tracked config, commit, scenario matrix, seed, and command above.

## Result

- Exit code: `0`
- Campaign status: `benchmark_success`
- Evidence status: `valid`
- Successful runs: `2` / `2`
- Unexpected failed runs: `0`
- Unexpected failed rows: `0`
- Fallback or degraded rows: `0`
- PPO execution mode: `native`
- PPO learned policy contract status: `pass`
- SNQI contract status: `pass`
- SNQI positioning recommendation: `downgrade_to_appendix_or_implementation_aid`

## Interpretation Boundary

This evidence resolves the immediate PPO adapter/SNQI smoke blocker that prevented a
valid post-repair rerun. It should be cited as blocker-resolution evidence only until a broader,
paper-facing scenario-horizon run is executed and promoted with its own durable evidence packet.
