# Issue 857 — horizon-alignment setup and job launch

## Goal

Prepare and launch the horizon-matched retrain requested by issue #857 so the issue-791 PPO
leader can be evaluated under the same nominal 100-step budget as the camera-ready benchmark.

## Status

- Phase A complete: the repo now has a horizon-100 scenario surface, a matching training config,
  and a horizon-400 benchmark probe config.
- Phase B submitted: Slurm job `12178` (`robot-sf-issue857-horizon100`) via
  `scripts/dev/sbatch_auxme_issue791.sh`.
- Phase E submitted: Slurm job `12179` (`robot-sf-issue791-benchmark`) via
  `scripts/dev/sbatch_use_max_time.sh` with
  `ISSUE791_BENCHMARK_CONFIG=configs/benchmarks/paper_experiment_matrix_v1_issue_791_horizon400_probe.yaml`.
- Phase C/D remain pending until those jobs finish.

Initial queue state captured immediately after submit:

- `12178`: `PENDING (Resources)` on `l40s`
- `12179`: `PENDING (Priority)` on `l40s`

## Implemented surfaces

- `configs/scenarios/sets/ppo_full_maintained_eval_v1_horizon100.yaml`
  - New horizon-matched eval/training manifest.
  - Uses the new manifest-level `scenario_overrides` support to force
    `simulation_config.max_episode_steps: 100` across the expanded 70-scenario surface.
- `configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_horizon100.yaml`
  - Seed-123, 10M-step, env22 clone of the issue-791 large-capacity leader.
  - Training and evaluation both point at the new horizon-100 manifest.
- `configs/benchmarks/paper_experiment_matrix_v1_issue_791_horizon400_probe.yaml`
  - Optional diagnostic probe for the existing issue-791 leader with `horizon: 400`.
  - Marked `paper_facing: false` and `export_publication_bundle: false` because it is a
    diagnostic attribution run, not a release artifact.

## Runtime fix discovered during Phase A validation

The horizon smoke initially failed even though the new manifest loaded correctly. Root cause:
`RobotState.step()` set `is_timeout` from floating elapsed time with `sim_time_elapsed > sim_time_limit`,
which made a nominal 100-step limit expire on step 101.

This was corrected in `robot_sf/robot/robot_state.py` by switching to the discrete contract:

- timeout now triggers on `timestep >= max_sim_steps`

This keeps `max_episode_steps` aligned with the configured step budget and removes a one-step drift
from all scenario-driven timeouts.

## Validation

Focused pytest slice:

```bash
source .venv/bin/activate
python -m pytest \
  tests/integration/test_train_expert_ppo.py \
  tests/benchmark/test_camera_ready_campaign.py \
  -k 'horizon100_eval_manifest_overrides_all_episode_limits or issue_857_horizon100_training_config_uses_horizon_matched_surface or issue_857_horizon100_surface_truncates_empty_map_at_step_100 or issue_857_horizon400_probe_only_changes_horizon_and_bundle_export'
```

Result on 2026-04-29: `4 passed, 82 deselected`.

Additional submission preflight:

```bash
source .venv/bin/activate
python -c 'import rvo2; print(rvo2.__file__)'
```

Result: `rvo2` import succeeded on this workstation, so the optional benchmark probe is not blocked
by the known ORCA prerequisite failure mode.

## Related surfaces

- Issue body: GitHub issue `#857`
- Decision context: `docs/context/issue_791_best_policy_verdict_2026_04_21.md`
- Prior benchmark gap note: `docs/context/issue_791_wave6_results_and_benchmark_orca_block.md`
- Reusable experiment memory: `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`
- Fallback policy: `docs/context/issue_691_benchmark_fallback_policy.md`
- Submitted logs (once Slurm opens them): `output/slurm/12178-issue791-reward-curriculum.out`,
  `output/slurm/12179-issue791-benchmark.out`

## Open boundary

This note only captures setup and launch. It does **not** claim any benchmark or training outcome yet.
Update this note after jobs `12178` and `12179` finish with:

- seed-123 in-distribution eval metrics,
- horizon-400 probe result for job 11724,
- benchmark termination breakdown,
- Phase D decision (replicas vs mixed vs no-go).