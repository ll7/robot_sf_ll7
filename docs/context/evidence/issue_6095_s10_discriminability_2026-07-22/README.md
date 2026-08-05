# Issue #6095 S10 ORCA/PPO Discriminability Calibration Evidence (2026-07-22)

Compact preflight evidence for the issue #6095 benchmark campaign. The configs are
validated for a future staged execution; this evidence captures config validation
and scenario preview outputs only. It is not a completed benchmark or a
discriminability result. The metadata-only checkpoint check is not submit-safe.
S10 denotes the frozen ten-seed schedule (111-120).

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

## Exact Preflight Provenance

Both packets were regenerated together from source revision
`d791c08f70b9af20f93babd5f1f17b06d581a185` with the canonical metadata-only
preflight command after rebasing onto current `main`. The nominal matrix hash
continues to normalize repository-resident scenario file references; the stress
packet has no route override and its matrix hash is unchanged. Each preflight artifact records the
repository-relative source config and its full source-file SHA-256.

| Property | Nominal | Stress |
|---|---|---|
| Campaign ID | `issue_6095_nominal_discriminability_v1_20260725_final` | `issue_6095_stress_discriminability_v1_20260725_final` |
| Config hash | `60448a7228d1a450` | `0375e182d186a8bc` |
| Scenario matrix hash | `e5fc81d3eef3` | `6b1f3a702703` |
| Source config SHA-256 | `3bf27cc362055e6874125f93b793c70f099ce6049641b60c1cb69974b3a55df7` | `e8f8b56097964568da4784054d23e1c590c14d32634c8ec6d465f735d1208dc6` |
| Clearance warnings | 2 total; 1 certified, 1 unresolved | 15 total; all 15 certified |
| ORCA native prerequisite | `rvo2` import passed | `rvo2` import passed |
| PPO checkpoint mode | metadata-only; stageable remote; not submit-safe | metadata-only; stageable remote; not submit-safe |

The stress certifications retain their Issue #1105 planner-attribution caveats.
The remaining nominal `empty_map_8_directions_east` map-level warning is
explicitly unresolved and must remain a caveat in any later campaign report.

## Execution Status

Preflight passed for both configs. Full execution requires SLURM or equivalent.

## Storage Decision

This bundle keeps compact preflight evidence in git. Raw benchmark results
(episode JSONL, videos, logs) belong in durable storage outside git, tracked via
manifest references.

## Portability

Provenance paths in the tracked preflight packet are repository-relative, and
the nominal scenario-matrix hash normalizes repository-resident scenario file
references before hashing. CSV artifacts use LF line endings so their checksums
and review sidecars are stable across repository checks.
