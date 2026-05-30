# Learned Local-Navigation Policy Registry

Related issues:

- Issue #1657: <https://github.com/ll7/robot_sf_ll7/issues/1657>
- Issue #1618 learned-policy adapter interface:
  <https://github.com/ll7/robot_sf_ll7/issues/1618>
- Issue #1363 learned-policy eligibility checklist:
  <https://github.com/ll7/robot_sf_ll7/issues/1363>

## Purpose

This registry records learned local-navigation policy families that Robot SF has implemented,
staged, rejected, or is monitoring. It is metadata for planning and review, not benchmark evidence.

`docs/context/policy_search/candidate_registry.yaml` remains the canonical registry for implemented
or concrete runnable Robot SF policy-search candidates with config pointers. This note is broader:
it includes proposal, launch-packet, monitor-only, and blocked learned-policy families so future
agents do not recreate one-off learned-policy assessments without comparable metadata.

## Registry Schema

```yaml
policy_id: stable snake_case identifier
policy_family: learned_baseline | guarded_policy | residual_policy | auxiliary_risk |
  predictive_model | lidar_policy | external_graph_policy | external_learned_policy |
  external_visual_policy | external_world_model
paper_or_source: paper, source repository, local issue, or local design anchor
upstream_implementation: URL or local path when available
license: known license, local-only, unknown, or not_applicable
observation_schema: planner-facing observation contract or required adapter
action_interface: velocity_command | residual_command | score_or_cost | trajectory |
  waypoint | unavailable
checkpoint_availability: local_registry | wandb_artifact | pending | source_claimed |
  missing | not_required
expected_dependencies: local_repo | slurm | external_legacy_env | source_harness |
  unknown
reproducibility_status: smoke_proven | launch_packet | source_harness_required |
  comparison_available | proposal | prototype_only | monitor_only | blocked | rejected
integration_status: implemented | staged | adapter_needed | monitor_only | rejected
benchmark_status: smoke_only | comparison_available | not_benchmark_evidence |
  blocked | rejected | rejected_for_current_adapter
evidence_boundary: short statement of what current evidence can and cannot support
integration_effort: small | medium | large | slurm_campaign | not_currently_actionable
local_anchors: local docs, configs, tests, issues, or model registry paths
```

Passing this registry screen does not make a policy benchmark-ready. Any learned policy still needs
the checklist in `docs/context/policy_search/contracts/learned_local_policy_eligibility.md`, adapter
metadata from Issue #1618, smoke proof, and fail-closed handling before benchmark claims.

## Entries

| `policy_id` | `policy_family` | `integration_status` | `reproducibility_status` | `benchmark_status` | Boundary |
| --- | --- | --- | --- | --- | --- |
| `ppo_issue791_best_v1` | `learned_baseline` | `implemented` | `comparison_available` | `comparison_available` | Best current learned-only baseline for success-oriented comparison; not a safety promotion because paper-matrix collision rate is worse than ORCA. |
| `guarded_ppo_orca_prior` | `guarded_policy` | `implemented` | `smoke_proven` | `not_benchmark_evidence` | Inference-only guarded variants are exhausted as a tuning lane; further value requires training a residual or learned risk component. |
| `orca_residual_guarded_ppo_v0` | `residual_policy` | `staged` | `launch_packet` | `smoke_only` | Runtime residual surface exists, but learned residual training/checkpoint lineage is pending and fallback/degraded rows remain caveats. |
| `learned_risk_model_v1` | `auxiliary_risk` | `staged` | `launch_packet` | `not_benchmark_evidence` | Pre-SLURM launch packet only; hard guards remain authoritative and learned risk may only add auxiliary candidate cost. |
| `predictive_planner_v1` | `predictive_model` | `implemented` | `comparison_available` | `comparison_available` | Uses a learned pedestrian predictor inside a planner stack; evidence applies to the configured predictive planner, not a general learned local policy. |
| `predictive_mppi` | `predictive_model` | `implemented` | `comparison_available` | `comparison_available` | Learned prediction informs MPPI-style rollout scoring; integration depends on predictive model/config provenance. |
| `lidar_ppo_mlp_gate_v1` | `lidar_policy` | `adapter_needed` | `proposal` | `blocked` | Planned LiDAR learned-policy smoke from Issues #1615/#1662; not available on `main` until launch-packet work lands and smoke training runs. |
| `crowdnav_height_igat_family` | `external_graph_policy` | `monitor_only` | `source_harness_required` | `blocked` | Source/checkpoint and graph-observation parity must be proven before a Robot SF adapter or benchmark row. |
| `drl_vo_family` | `external_learned_policy` | `monitor_only` | `prototype_only` | `blocked` | Tracked-agent diagnostic/prototype boundary only; not main-table ready and not a leakage-free benchmark policy. |
| `navdp_nomad_visual_family` | `external_visual_policy` | `monitor_only` | `monitor_only` | `rejected_for_current_adapter` | RGB-D/topomap/visual-goal assumptions do not reduce cleanly to the current 2D local-planner contract. |
| `dreamerv3_navigation_family` | `external_world_model` | `monitor_only` | `blocked` | `blocked` | Checkpoint import and Robot SF observation-contract boundaries remain fail-closed. |

## Entry Details

### `ppo_issue791_best_v1`

- `paper_or_source`: internal PPO policy from Issue #791 promotion lineage.
- `upstream_implementation`: local Robot SF PPO baseline.
- `license`: local repository.
- `observation_schema`: configured PPO observation pipeline, including predictive foresight where
  enabled by the base config.
- `action_interface`: velocity command through the PPO planner wrapper.
- `checkpoint_availability`: W&B artifact via `model/registry.yaml`.
- `expected_dependencies`: local repo plus artifact hydration.
- `reproducibility_status`: smoke, nominal, and paper-matrix comparison recorded.
- `integration_effort`: small for reruns, medium for new training.
- `local_anchors`:
  - `docs/context/policy_search/reports/2026-05-05_best_learning_policy.md`
  - `configs/policy_search/candidates/ppo_issue791_best_v1.yaml`
  - `model/registry.yaml`

### `guarded_ppo_orca_prior`

- `paper_or_source`: internal guarded PPO and ORCA-prior experiments.
- `upstream_implementation`: `robot_sf/planner/guarded_ppo.py`.
- `license`: local repository.
- `observation_schema`: PPO observation plus guard/fallback planner observations.
- `action_interface`: guarded velocity command with raw/adapted/post-guard distinction required.
- `checkpoint_availability`: local/W&B depending on PPO model id.
- `expected_dependencies`: local repo plus artifact hydration.
- `reproducibility_status`: smoke and policy-search reports exist.
- `integration_status`: implemented but rejected as another inference-only tuning lane.
- `local_anchors`:
  - `docs/context/issue_602_guarded_ppo_profile.md`
  - `docs/context/policy_search/reports/2026-05-05_learning_hybrid_policy_search.md`

### `orca_residual_guarded_ppo_v0`

- `paper_or_source`: internal ORCA-residual behavior-cloning plan.
- `upstream_implementation`: local guarded PPO residual surface.
- `license`: local repository.
- `observation_schema`: runtime-only observation plus ORCA command/risk context.
- `action_interface`: bounded residual command added to ORCA, then hard-guarded.
- `checkpoint_availability`: pending.
- `expected_dependencies`: SLURM training campaign after durable dataset/materialized artifacts.
- `reproducibility_status`: launch packet and runtime-surface smoke, not trained residual evidence.
- `local_anchors`:
  - `docs/context/issue_1428_orca_residual_lineage.md`
  - `docs/context/policy_search/SLURM/005_orca_residual_bc_lineage.md`
  - `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml`

### `learned_risk_model_v1`

- `paper_or_source`: internal learned auxiliary risk scorer plan.
- `upstream_implementation`: local launch packet only.
- `license`: local repository.
- `observation_schema`: trajectory features and labels from frozen baseline trace contract.
- `action_interface`: score or auxiliary cost, never an unguarded command.
- `checkpoint_availability`: pending.
- `expected_dependencies`: SLURM training campaign and durable trace artifacts.
- `reproducibility_status`: launch packet validates, no trained model.
- `local_anchors`:
  - `docs/context/issue_1395_learned_risk_launch_packet.md`
  - `configs/training/learned_risk_model_issue_1395_launch_packet.yaml`
  - `scripts/validation/validate_learned_risk_launch_packet.py`

### `predictive_planner_v1` And `predictive_mppi`

- `paper_or_source`: internal predictive planner and predictive MPPI configs.
- `upstream_implementation`: local planner stack.
- `license`: local repository.
- `observation_schema`: predictive planner features, obstacle features, and model/config metadata.
- `action_interface`: velocity command or rollout-selected command from planner scoring.
- `checkpoint_availability`: model/config dependent.
- `expected_dependencies`: local repo plus configured predictive model artifact.
- `reproducibility_status`: benchmark reports exist for specific configured variants.
- `evidence_boundary`: predictive evidence belongs to the full planner configuration, not to a
  standalone learned policy claim.
- `local_anchors`:
  - `configs/algos/prediction_planner_camera_ready.yaml`
  - `configs/algos/predictive_mppi_camera_ready.yaml`
  - `docs/context/prediction_planner_literature_audit.md`
  - `docs/context/issue_675_predictive_mppi_benchmark.md`

### External Monitor Entries

External learned-policy families remain registry entries only to prevent duplicate work. They need
source-side harnesses, license/checkpoint proof, observation/action-contract mapping, and eligibility
metadata before entering the runnable candidate registry.

- CrowdNav HEIGHT / IGAT / CrowdNav-family graph policies:
  `docs/context/policy_search/issue_1367_crowdnav_family_verdict.md`,
  `docs/context/policy_search/issue_1394_crowdnav_height_source_harness.md`.
- DRL-VO family: `docs/context/issue_769_drl_vo_assessment.md`.
- NavDP / NoMaD visual navigation:
  `docs/context/policy_search/2026-05-20_navdp_nomad_diffusion_assessment.md`.
- Foundation-model / VLA / multimodal navigation:
  `docs/context/policy_search/2026-05-30_foundation_model_readiness_issue_1626.md`.
- DreamerV3/world-model navigation:
  `docs/context/issue_1190_dreamerv3_checkpoint_import_boundary.md`.

## Intake Rules

1. Add implemented or concrete runnable Robot SF candidates to
   `docs/context/policy_search/candidate_registry.yaml` only after this registry and the eligibility
   checklist agree that a runnable local contract exists.
2. Keep source-only, monitor-only, and blocked candidates in this registry or their family verdict
   notes until source/checkpoint/adapter proof exists.
3. Every new learned-policy candidate should name its `observation_schema`, `action_interface`,
   `checkpoint_availability`, `integration_status`, `reproducibility_status`, and
   `benchmark_status`.
4. Fallback, degraded, or guard-dominated runs must be reported as caveats unless the issue is
   explicitly about measuring that mode.
5. Any benchmark row must record raw model action, adapted action, post-guard action, guard/fallback
   reason, observation level, and action projection metadata as required by the eligibility
   checklist.

## Validation

This note is documentation and registry metadata only. Validate changes with:

```bash
git diff --check origin/main...HEAD
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```
