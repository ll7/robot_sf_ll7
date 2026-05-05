# Local Policy Search And Training Launch

Date: 2026-05-05

## Goal

Continue the policy-search work locally now that this machine is allowed to run local policy
search and training jobs. This is a local execution note, not a benchmark promotion claim.

## Machine Boundary

Local machine context:

- machine: `auxme-imech036`
- GPU: NVIDIA RTX 3080, 10 GiB VRAM
- local policy: long training must run in `tmux`
- CPU guidance: keep routine worker counts bounded; avoid unbounded xdist/Ray/vector-env fanout
- SLURM submission: not allowed from this machine context

Pre-launch GPU check on 2026-05-05 around 22:25 showed only desktop processes using the GPU
(`91 MiB`, `0%` utilization).

## Local Policy Evidence

The current tracked policy-search evidence already identifies the strongest local non-learning
candidates:

- Raw h500 leader: `scenario_adaptive_hybrid_orca_v1`
  (`0.9097` success, `0.0208` collision in
  `docs/context/policy_search/reports/2026-05-05_full_matrix_h500_analysis.md`).
- Strict-gate h500 candidate: `hybrid_rule_v3_fast_progress`
  (`0.8264` success, `0.0139` collision in the same note).

For this turn, the local search action was a fresh portfolio-summary export to ignored scratch:

```bash
uv run python scripts/tools/summarize_policy_search_portfolio.py \
  --output-md output/ai/autoresearch/local_policy_training/portfolio_overview_2026-05-05.md \
  --output-json output/ai/autoresearch/local_policy_training/portfolio_overview_2026-05-05.json
```

## Training Probe

Selected config:

```text
configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_128k.yaml
```

Rationale:

- bounded `131072` timestep PPO probe rather than env22/10M+ training,
- `num_envs: 2`, which stays within the local CPU limit,
- same reward-curriculum / predictive-foresight family as the stronger issue-791 PPO line,
- W&B forced offline so local training does not depend on network credentials.

Dry-run validation:

```bash
WANDB_MODE=offline uv run python scripts/training/train_ppo.py \
  --config configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_128k.yaml \
  --dry-run \
  --log-level WARNING \
  --log-file output/ai/autoresearch/local_policy_training/ppo_128k_dry_run.log
```

The dry run completed and wrote local expert-policy placeholder artifacts under
`output/benchmarks/expert_policies/`.

Launched training:

```bash
tmux new -d -s ppo128k_local_20260505 -- bash -lc 'cd /home/luttkule/git/robot_sf_ll7 && source .venv/bin/activate && mkdir -p output/ai/autoresearch/local_policy_training && export WANDB_MODE=offline PYTHONUNBUFFERED=1 LOGURU_LEVEL=WARNING CUDA_VISIBLE_DEVICES=0 && uv run python scripts/training/train_ppo.py --config configs/training/ppo/ablations/expert_ppo_issue_791_reward_curriculum_promotion_128k.yaml --log-level WARNING 2>&1 | tee output/ai/autoresearch/local_policy_training/ppo_128k_train_20260505_2229.log'
```

Observed startup evidence:

- tmux session: `ppo128k_local_20260505`
- W&B mode: offline, local run under `output/wandb/wandb/offline-run-20260505_222910-z6rljd2f`
- training device: `cuda`
- early GPU snapshot: about `446 MiB` used, `1%` utilization, `45C`

## Follow-Up Evaluation

When the tmux run exits, inspect:

```bash
tmux capture-pane -pt ppo128k_local_20260505 -S -200
tail -200 output/ai/autoresearch/local_policy_training/ppo_128k_train_20260505_2229.log
find output/benchmarks/expert_policies/checkpoints/ppo_expert_issue_791_reward_curriculum_promotion_128k -maxdepth 1 -type f -print
```

If a real checkpoint exists, evaluate it before treating it as useful:

```bash
LOGURU_LEVEL=WARNING uv run python scripts/tools/policy_analysis_run.py \
  --scenario configs/scenarios/classic_interactions_francis2023.yaml \
  --policy ppo \
  --model-path output/benchmarks/expert_policies/checkpoints/ppo_expert_issue_791_reward_curriculum_promotion_128k/ppo_expert_issue_791_reward_curriculum_promotion_128k_best.zip \
  --seed-set eval \
  --output output/ai/autoresearch/local_policy_training/ppo_128k_policy_analysis_eval
```

If the best checkpoint file is absent, use the final checkpoint emitted by the training script
instead and record the exact path before evaluating.

## Current Interpretation

This run is only a bounded local training probe. It does not replace the current non-learning
policy-search leaders unless the follow-up evaluation beats the tracked h500/full-matrix evidence
without increasing collision rate.
