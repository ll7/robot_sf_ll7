# PPO Imitation Pipeline Configurations

This directory hosts configuration files used by the expert-policy, trajectory collection, and imitation pre-training workflows. Each config captures the exact seed lists, scenario setups, and hyperparameters required for reproducible runs.

## Layout

* `expert_ppo.yaml` – Expert training defaults, including convergence thresholds and evaluation schedule.
* `expert_ppo_issue_576_br06_v2_sanity_500k_all_maps.yaml` – BR-06 sanity pass (0.5M steps) before full retraining.
* `expert_ppo_issue_576_br06_v2_15m_all_maps.yaml` – BR-06 retraining v2 run plan (15M steps, all classic + Francis maps, route-completion v2 reward, full tracking).
* `optuna_expert_ppo.yaml` – Config-first launcher settings for Optuna sweeps on top of `expert_ppo.yaml`.
* `optuna_expert_ppo_issue_4019_demo.yaml` – Smoke-scale automated hyperparameter optimization (HPO) demo with explicit search space, cost bounds, and provenance output.
* `bc_pretrain.yaml` – Behavioural cloning phase settings (optimizer, batch sizes, dataset references).
* `ppo_finetune.yaml` – PPO fine-tuning schedule that resumes from a pre-trained checkpoint.
* `bc_pretrain_issue_749_v10_warm_start.yaml` and
  `ppo_finetune_issue_749_v10_warm_start.yaml` – Issue #749 BC warm-start PPO challenger launch
  packet; see `docs/context/issue_749_bc_preinit_ppo_launch_packet.md` for the dataset collection,
  artifact-persistence, and evaluation boundary.
* `oracle_dataset_issue_1397_launch_packet.yaml` – pre-Slurm oracle-imitation dataset launch
  packet. Validate with
  `uv run python scripts/validation/validate_oracle_imitation_launch_packet.py --config configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml --json`.
  Downstream imitation training must use the stricter gate:
  `uv run python scripts/validation/validate_oracle_imitation_launch_packet.py --config configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml --json --require-training-ready`.
  That gate intentionally fails while Issue #2620 remains `artifact_retrieval_blocked`.
* `oracle_trace_uri_registry_issue_1470.yaml` – issue #1470 artifact-promotion readiness packet
  for oracle-imitation trace evidence. Validate
  `uv run python scripts/validation/validate_oracle_trace_uri_registry.py --config configs/training/ppo_imitation/oracle_trace_uri_registry_issue_1470.yaml --json`.
  The strict downstream gate intentionally fails closed until every split has a concrete durable
  raw-trace URI:
  `uv run python scripts/validation/validate_oracle_trace_uri_registry.py --config configs/training/ppo_imitation/oracle_trace_uri_registry_issue_1470.yaml --json --require-training-ready`.
* Additional files describe scenario coverage manifests referenced by trajectory collection commands.

## Usage Notes

1. Do not treat worktree-local `output/` files as durable training inputs; promote required
   manifests or datasets before downstream launch packets consume them.
2. Keep seed lists consistent across configs when running comparative studies.
3. Document any temporary overrides in the run manifest to preserve reproducibility.
4. For Optuna launcher configs, prefer `log_level: WARNING` by default and override only when debugging.
5. W&B is enabled by default for Optuna launcher runs; set `disable_wandb: true` only for offline/reliability scenarios.
6. Safety-gated Optuna sweeps can set `constraint_collision_rate_max` (and optional
`constraint_comfort_exposure_max` ) plus `constraint_handling: penalize|prune` .
7. Optuna launcher configs use `schema_version: robot_sf.optuna_expert_ppo_launcher.v1` or
`robot_sf.optuna_expert_ppo_launcher.v2`. Version 2 adds an allowlisted search space, explicit cost
bounds, and reviewable study/trial provenance. Unsupported versions fail at
`scripts/training/launch_optuna_expert_ppo.py --config <path>` before any trials are launched.

## Issue #4019 HPO Demo

The issue #4019 demo validates the workflow harness only. It does not run a full benchmark campaign,
submit Slurm or GPU jobs, or support a paper/dissertation claim.

Dry-run the generated command:

```bash
uv run python scripts/training/launch_optuna_expert_ppo.py \
  --config configs/training/ppo_imitation/optuna_expert_ppo_issue_4019_demo.yaml \
  --dry-run
```

After a completed smoke run, export the selected best config:

```bash
uv run python scripts/tools/export_optuna_best_config.py \
  --db output/optuna/issue_4019_hpo_demo.db \
  --study-name issue_4019_hpo_demo \
  --base-config configs/training/ppo_imitation/expert_ppo.yaml \
  --out output/optuna/issue_4019_hpo_demo/best_config.yaml \
  --report output/optuna/issue_4019_hpo_demo/selection_report.md
```
