# Issue 857 — horizon-alignment setup and job launch

## Goal

Prepare and launch the horizon-matched retrain requested by issue #857 so the issue-791 PPO
leader can be evaluated under the same nominal 100-step budget as the camera-ready benchmark.

## Status

- Phase A complete: the repo now has a horizon-100 scenario surface, a matching training config,
  and a horizon-400 benchmark probe config.
- Phase B **complete**: Slurm job `12178` (`robot-sf-issue857-horizon100`) finished
  `2026-04-30T02:17` after 8h23m on `l40s` (auxme-imech093). Best in-distribution eval
  (70 episodes, horizon-100 manifest) at step 7,864,320 / 10M:
  `success_rate=0.6429`, `collision_rate=0.0429`, `comfort_exposure=0.0190`,
  `path_efficiency=0.3264`, `snqi=-0.3298`.
  WandB: `ll7/robot_sf/h746lfsd`.
- Phase E (initial submission) **failed at 0s** (Slurm job `12179`): the worktree's
  `output/model_cache/ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417/model.zip`
  was missing because each git worktree owns its own gitignored cache. Resolved by
  creating a relative symlink to the artifact already cached in the parent repo
  (`/home/luttkule/git/robot_sf_ll7/output/model_cache/...`).
- Phase C **submitted on 2026-04-30**: Slurm job `12205`
  (`robot-sf-issue791-benchmark`, label `issue857-phase-c-horizon100-12178`) running
  `configs/benchmarks/paper_experiment_matrix_v1_issue_857_horizon100.yaml` with the
  PPO row pointing at `configs/baselines/ppo_issue_791_horizon100_12178.yaml`
  (the job-12178 candidate).
- Phase E **resubmitted on 2026-04-30**: Slurm job `12206`
  (label `issue857-phase-e-horizon400-probe-leader-11724`) running
  `configs/benchmarks/paper_experiment_matrix_v1_issue_791_horizon400_probe.yaml`.

Queue state captured immediately after the 2026-04-30 submissions:

- `12205`: `PENDING (QOSMaxJobsPerUserLimit)` on `l40s`
- `12206`: `PENDING (QOSMaxJobsPerUserLimit)` on `l40s`

Local preflight (`scripts/tools/run_camera_ready_benchmark.py --mode preflight`) on the
Phase C config validated 47 scenarios × 7 planners × 3 seeds (eval seed-set
`[111, 112, 113]`) at horizon 100 before submission.

## Phase D gate (issue #857)

Promotion gate is on the camera-ready matrix, not on training-eval:

- `success_mean ≥ 0.50` AND `max_steps_share ≤ 0.30` → seed replicas at 231/1337,
  promote candidate as canonical PPO baseline.
- `success_mean ∈ [0.30, 0.50)` → mixed; investigate per-scenario before more GPU.
- `success_mean < 0.30` → negative result; fall back to 11724 leader.

Phase B in-distribution `success_rate=0.6429` clears the threshold at training-eval
distribution but does NOT decide Phase D. The 11724 leader's training-eval was 0.929
yet collapsed to 0.255 on the camera-ready matrix; the horizon-matched candidate's
camera-ready row is the actual go/no-go signal.

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
- Phase B training log: `output/slurm/12178-issue791-reward-curriculum.out`
- Phase B best summary: `output/slurm/issue791-reward-curriculum-job-12178/benchmarks/expert_policies/checkpoints/ppo_expert_issue_791_reward_curriculum_promotion_10m_env22_horizon100/ppo_expert_issue_791_reward_curriculum_promotion_10m_env22_horizon100_best.summary.json`
- Phase C adapter: `configs/baselines/ppo_issue_791_horizon100_12178.yaml`
- Phase C benchmark config: `configs/benchmarks/paper_experiment_matrix_v1_issue_857_horizon100.yaml`
- Phase C job log (once opened): `output/slurm/12205-issue791-benchmark.out`
- Phase E (failed) log: `output/slurm/12179-issue791-benchmark.out`
- Phase E (resubmitted) log (once opened): `output/slurm/12206-issue791-benchmark.out`

## Open boundary

This note now covers Phase A (complete), Phase B (complete with results), and the
launch of Phase C (job 12205) and Phase E (job 12206). It does **not** claim any
camera-ready benchmark outcome yet. Update this note after jobs `12205` and `12206`
finish with:

- horizon-matched candidate camera-ready metrics (success_mean, collision, snqi,
  termination histogram),
- horizon-400 probe result for the Wave-5 leader (attribution: horizon vs policy),
- Phase D decision (replicas vs mixed vs no-go) per the gate above,
- registry / canonical-baseline pointer changes if the candidate is promoted.