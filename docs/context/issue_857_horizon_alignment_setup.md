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
- Phase C **complete**: Slurm job `12205` finished
  `2026-04-30T10:07` after 30m22s on `l40s` (label
  `issue857-phase-c-horizon100-12178`, 7 planners × 47 scenarios × 3 seeds = 987
  episodes). Campaign:
  `output/benchmarks/issue_791/paper_experiment_matrix_v1_issue857-phase-c-horizon100-12178_20260430_093726/`.
  PPO row (horizon-matched candidate, 141 episodes):
  `success_mean=0.1489`, `collisions_mean=0.0993` (per-step normalization;
  binary collision share=0.000), `snqi_mean=-0.2867`, `path_efficiency_mean=0.9526`,
  `time_to_goal_norm_mean=1.7163`. Termination histogram from per-episode rows:
  21/141 success (14.9%), 0/141 binary collision (0.0%), 120/141 timeouts (85.1%).
- Phase E **still pending** at the time of Phase C readout: Slurm job `12206`
  on `l40s` waiting for the QOS slot.

Local preflight (`scripts/tools/run_camera_ready_benchmark.py --mode preflight`) on the
Phase C config validated 47 scenarios × 7 planners × 3 seeds (eval seed-set
`[111, 112, 113]`) at horizon 100 before submission.

## Phase D gate (issue #857)

Promotion gate is on the camera-ready matrix, not on training-eval:

- `success_mean ≥ 0.50` AND `max_steps_share ≤ 0.30` → seed replicas at 231/1337,
  promote candidate as canonical PPO baseline.
- `success_mean ∈ [0.30, 0.50)` → mixed; investigate per-scenario before more GPU.
- `success_mean < 0.30` → negative result; fall back to 11724 leader.

### Phase D verdict (2026-04-30): NEGATIVE RESULT, no-go

| Metric | Phase B in-dist (job 12178 horizon-100 manifest) | Phase C camera-ready (job 12205) | 11724 leader on the same camera-ready matrix |
|---|---:|---:|---:|
| success_mean | 0.6429 | **0.1489** | 0.2553 |
| binary collision share | 0.0429 | 0.000 | ~0.085 |
| max_steps share | (training horizon=100) | **0.851** | ~0.65 |
| snqi_mean | -0.330 | -0.287 | (per leader compare) |

Conclusions:

1. **Horizon hypothesis is rejected.** Forcing training-time `max_episode_steps: 100`
   did not lift the camera-ready PPO row. It actually fell from 0.255 (leader 11724)
   to 0.149 — about 0.106 absolute and ~42% relative degradation, well outside
   bootstrap noise on a 141-episode row.
2. **Conservative-policy collapse.** Binary collision share dropped to 0.000 while
   timeouts climbed to 0.851. The horizon-matched policy effectively learned to never
   commit to motion that could collide and instead time out — a degenerate solution
   to the shorter episode budget.
3. **Distribution gap persists across the in-dist / camera-ready boundary even at
   matched horizon.** The candidate scored 0.643 on the horizon-100 in-distribution
   manifold but 0.149 on the camera-ready matrix — a 0.494-absolute collapse at
   identical nominal step budget. This rules out the "training horizon vs benchmark
   horizon" attribution as the dominant cause of the 0.929 → 0.255 leader gap;
   the gap is dominated by something else (per-scenario reachability at 10s,
   training-vs-benchmark scenario mix, or interaction with the camera-ready
   pedestrian-density / kinematics contract).

Decisions:

- **Do not** queue seed replicas at 231 / 1337 — the gate is failed at seed 123.
- **Do not** repoint `configs/baselines/ppo_15m_grid_socnav.yaml` — the canonical
  PPO baseline remains the issue-791 Wave-5 leader (artifact 11724).
- **Do not** mark `configs/baselines/ppo_issue_791_horizon100_12178.yaml` as a
  promotion candidate; keep it as an archived ablation so Phase E and any later
  attribution work can still cite the artifact.
- Keep Phase E in flight regardless (job 12206) — even with Phase C as a no-go,
  the horizon=400 probe of the Wave-5 leader is the cleanest attribution evidence
  for whether the camera-ready matrix is benchmark-horizon-bound at all.

Phase B in-distribution `success_rate=0.6429` clears the threshold at training-eval
distribution but does NOT decide Phase D. The 11724 leader's training-eval was 0.929
yet collapsed to 0.255 on the camera-ready matrix; the horizon-matched candidate's
camera-ready row is the actual go/no-go signal — and it failed the gate.

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
- Phase C job log: `output/slurm/12205-issue791-benchmark.out`
- Phase C campaign artifacts: `output/benchmarks/issue_791/paper_experiment_matrix_v1_issue857-phase-c-horizon100-12178_20260430_093726/`
  (campaign_table.md, seed_episode_rows.csv, snqi_diagnostics.md, publication bundle).
- Phase E (failed) log: `output/slurm/12179-issue791-benchmark.out`
- Phase E (resubmitted) log (once opened): `output/slurm/12206-issue791-benchmark.out`

## Open boundary

Phases covered as of 2026-04-30:

- Phase A: complete.
- Phase B (training): complete; in-distribution metrics recorded above.
- Phase C (camera-ready benchmark): complete; **negative result**, Phase D no-go.
- Phase D (decision): recorded above — fall back to leader 11724, no replicas.
- Phase E (horizon=400 probe of leader 11724): job 12206 still pending; will be
  appended once it finishes. The Phase D verdict is independent of Phase E and
  does not block on it.

Still to do once Phase E lands:

- record horizon-400 probe metrics (success_mean, collision, max_steps_share),
- decide whether the horizon mismatch contributes anything detectable on the
  leader side, or whether the 0.929 → 0.255 leader gap is entirely
  distribution / scenario-mix driven,
- append the horizon-attribution outcome to
  `memory/experiments/2026-04-20_issue_791_distribution_alignment_dominates.md`.