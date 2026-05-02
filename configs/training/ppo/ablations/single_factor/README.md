# Issue-791 single-factor PPO ablations

Clean fixed-seed counterfactuals for the issue-791 promotion features. The
historical promotion configs under `configs/training/ppo/ablations/` stack
multiple changes per file because the campaign treated attribution at the
campaign layer. Those files remain untouched as run-history artifacts; this
directory is the corrections pass spawned by issue #852.

Each config here flips exactly one feature on top of the eval-aligned
no-feature baseline `../expert_ppo_issue_791_baseline_promotion_10m_env22_eval_aligned.yaml`:

- `attention_only_*` enables only `feature_extractor_kwargs.use_pedestrian_attention`.
- `asymmetric_critic_only_*` enables only `env_factory_kwargs.asymmetric_critic`.
- `reward_curriculum_only_*` enables only `env_factory_kwargs.reward_curriculum.stages`.

For each feature, this suite provides 1M smoke and 10M promotion configs for
fixed seeds 123, 231, and 1337. `randomize_seeds` is deliberately `false` so
the declared seed is honored by `scripts/training/train_ppo.py`.

## Wave plan

1. Submit the 1M smoke configs first with the matching `SLURM/Auxme/issue_791_*.sl` wrapper.
2. Submit the 10M promotion configs only after the corresponding 1M smoke is green.
3. Record SLURM job IDs, W&B run IDs, and results in
   `docs/context/issue_791_promotion_campaign_128k_256k.md`.

All configs tag W&B with `issue-852`, `issue-791-followup`, `fixed-seed`, and
`single-factor-<feature>` so correction-pass runs are separable from the
historical campaign artifacts.
