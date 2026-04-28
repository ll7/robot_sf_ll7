---
name: WandB rejects runs when job_type exceeds 64 characters
description: `wandb.init()` returns 400 "64 limit exceeded for JobType" and the training run exits immediately; keep `tracking.wandb.job_type` ≤ 64 chars in PPO ablation configs.
type: failure
---

Jobs 11869 and 11870 (seed 231/1337 10M replicas of the issue-791 leader) crashed
within 2 minutes of launch with:

```
wandb.errors.errors.CommError: Error uploading run: returned error 400:
{"errors":[{"message":"invalid parameters: 64 limit exceeded for JobType"}]}
```

Both configs set `tracking.wandb.job_type` to strings of 66–67 characters:

- `expert-ppo-10m-promotion-env22-eval-aligned-large-capacity-seed231` (66 chars)
- `expert-ppo-10m-promotion-env22-eval-aligned-large-capacity-seed1337` (67 chars)

WandB enforces a hard 64-character limit on this field; the training script fails at
`_init_wandb()` before any environment is built.

**How to apply:**

- Before submitting a new ablation config, count the length of `tracking.wandb.job_type`:
  `echo -n "<job_type>" | wc -c` must be ≤ 64.
- Prefer abbreviated forms that drop redundant tokens (e.g. drop `promotion-env22`
  when the tag list already carries it). Tags (`tracking.wandb.tags`) have no such
  limit and are the right place for long labels.
- This rule applies to all `expert_ppo_*` configs in
  `configs/training/ppo/ablations/`. Existing configs that already run are fine;
  watch this when cloning a long-named recipe.

Resolved 2026-04-20 by shortening to
`expert-ppo-10m-eval-aligned-large-capacity-seed{231,1337}` (50–51 chars);
resubmitted as jobs 11872 / 11873.
