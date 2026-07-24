# Issue #6095 S10 ORCA/PPO Discriminability Calibration Evidence (2026-07-22)

Compact preflight evidence for the issue #6095 benchmark campaign. The configs are
ready for SLURM execution; this evidence captures config validation and scenario
preview outputs only.

## Source

Preflight outputs generated via `scripts/tools/run_camera_ready_benchmark.py --mode preflight`.

## Contents

- `manifest.sha256`: checksums for all evidence files.
- `nominal/`: preflight and reports for `issue_6095_nominal_discriminability_v1.yaml`.
- `stress/`: preflight and reports for `issue_6095_stress_discriminability_v1.yaml`.

## Campaign Design

| Property | Nominal | Stress |
|---|---|---|
| Scenario matrix | `configs/scenarios/nominal_v1.yaml` | `configs/scenarios/classic_interactions_francis2023.yaml` |
| Scenario count | 4 | 48 |
| Planners | ORCA, PPO | ORCA, PPO |
| Seeds | 111-120 (paper_eval_s10) | 111-120 (paper_eval_s10) |
| Horizon | 100 | 100 |
| dt | 0.1 | 0.1 |
| Kinematics | differential_drive | differential_drive |
| PPO checkpoint | `ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417` | same |
| Expected rows | 80 (4x10x2) | 960 (48x10x2) |

## Execution Status

Preflight passed for both configs. Full execution requires SLURM or equivalent.

## Storage Decision

This bundle keeps compact preflight evidence in git. Raw benchmark results
(episode JSONL, videos, logs) belong in durable storage outside git, tracked via
manifest references.
