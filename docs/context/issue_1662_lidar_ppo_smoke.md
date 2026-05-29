# Issue #1662 LiDAR PPO MLP Smoke

Date: 2026-05-29

## Goal

Issue #1662 asked for the first bounded LiDAR learned-policy training smoke after the #1615 launch
packet. The target candidate was `ppo_lidar_mlp_gate_v1`, with runtime observations limited to
`ObservationMode.DEFAULT_GYM` `drive_state` and `rays`, and no benchmark claim from the smoke.

## Fixed Smoke Config

The #1615 launch packet included an Optuna command shape, but a one-trial Optuna run can sample a
non-MLP extractor. This issue therefore materializes a fixed config:

- `configs/training/lidar/lidar_ppo_mlp_smoke_issue_1662.yaml`
- `policy_id`: `ppo_lidar_mlp_gate_v1_issue_1662_smoke`
- `total_timesteps`: 32,000 requested, 32,768 executed by PPO rollout chunking
- `seed`: 123
- `num_envs`: 4, `worker_mode`: `subproc`
- `feature_extractor`: `mlp`
- `feature_extractor_kwargs`: `ray_hidden_dims=[64, 32]`,
  `drive_hidden_dims=[16, 8]`, `dropout_rate=0.1`
- `policy_net_arch`: `[64, 64]`
- tracking disabled for both TensorBoard and W&B

The config is deliberately a training smoke config, not a benchmark adapter config and not a model
registry entry.

## Validation

Metadata preflight:

```bash
uv run python scripts/validation/check_learned_policy_eligibility.py \
  configs/training/lidar/lidar_ppo_mlp_eligibility_issue_1615.yaml \
  configs/training/lidar/lidar_perception_adapter_eligibility_issue_1615.yaml
```

Result: both specs passed.

Focused tests:

```bash
uv run pytest -q \
  tests/training/test_lidar_ppo_smoke_issue_1662.py \
  tests/training/test_lidar_learned_policy_launch_packet.py \
  tests/validation/test_check_learned_policy_eligibility.py \
  tests/test_feature_extractors.py
```

Result: `39 passed`.

Smoke command:

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python scripts/training/train_ppo.py \
  --config configs/training/lidar/lidar_ppo_mlp_smoke_issue_1662.yaml \
  --log-level INFO \
  --log-file output/training/lidar_learned_policy/issue_1662/ppo_lidar_mlp_gate_v1_smoke.log
```

Result: completed successfully with run id
`ppo_lidar_mlp_gate_v1_issue_1662_smoke_20260529T105828`.

Compact evidence:

- `docs/context/evidence/issue_1662_lidar_ppo_smoke_summary.json`

## Observed Smoke Metrics

The smoke proved that the fixed LiDAR PPO MLP path trains and evaluates end to end on this
repository checkout. It did not prove benchmark readiness.

- Wall clock: 59.74 seconds
- Mean train env steps/sec: 956.23
- Best checkpoint metric: `success_rate=0.0` at step 16,000
- Final aggregate success rate: 0.0
- Final aggregate collision rate: 0.0
- Final aggregate SNQI: -0.8000
- Convergence: not met

The zero success rate is acceptable for this issue because the command is a 32k-step smoke. It is
not evidence that the candidate should be promoted to benchmark comparison or model registry use.

## Artifact Decision

Generated outputs remain ignored under `output/`:

- checkpoints and config manifests under `output/benchmarks/expert_policies/`
- run, perf, timeline, per-scenario, and episode records under
  `output/benchmarks/ppo_imitation/`
- the smoke log under `output/training/lidar_learned_policy/issue_1662/`
- coverage artifacts from the focused pytest run under `output/coverage/`

No checkpoint is committed. A future checkpoint intended for reuse must be published to W&B or
another durable artifact store and represented by a compact tracked manifest.

## Qwen Scout

Qwen Code was run as a read-only scout:

```bash
codex-agent-worker --provider qwen --model Qwen3.6-27B --timeout 900 \
  --slug issue-1662-qwen-scout --task-file /tmp/qwen_issue1662_scout.md
```

Run directory:
`.git/worktrees/issue-1662-lidar-training-smoke/codex-agent-runs/20260529T105401Z_qwen_issue-1662-qwen-scout`.

Qwen agreed that the fixed config is preferable to the one-trial Optuna command for this issue,
because it pins `ppo_lidar_mlp_gate_v1`. The scout overlapped with local edits, so its
`changed_files` summary includes orchestrator-created files; local validation above is the source
of truth.

## Follow-Up Boundary

Do not open a long training issue solely from this smoke. The next useful step is to add or identify
a deployment adapter and rerun a stronger local smoke or short train with a durable checkpoint
publication plan. The current smoke is evidence of executable training plumbing, not benchmark
performance.
