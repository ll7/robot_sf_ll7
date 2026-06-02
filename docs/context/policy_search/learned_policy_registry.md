# Learned Local-Navigation Policy Registry

Related issues:

- Issue #1657: <https://github.com/ll7/robot_sf_ll7/issues/1657>
- Issue #1618 learned-policy adapter interface:
  <https://github.com/ll7/robot_sf_ll7/issues/1618>
- Issue #1363 learned-policy eligibility checklist:
  <https://github.com/ll7/robot_sf_ll7/issues/1363>
- Issue #1870 external learned-policy intake:
  <https://github.com/ll7/robot_sf_ll7/issues/1870>

## Purpose

This registry records learned local-navigation policy families that Robot SF has implemented,
staged, rejected, or is monitoring. It is metadata for planning and review, not benchmark evidence.

`docs/context/policy_search/candidate_registry.yaml` remains the canonical registry for implemented
or concrete runnable Robot SF policy-search candidates with config pointers. This note is broader:
it includes proposal, launch-packet, monitor-only, and blocked learned-policy families so future
agents do not recreate one-off learned-policy assessments without comparable metadata.

Ownership boundary:

- This learned-policy registry owns the current planning state for learned-policy families:
  whether a family is implemented, staged, adapter-needed, monitor-only, or rejected for the current
  Robot SF adapter contract.
- `docs/context/policy_search/reject_monitor_registry.md` owns historical negative, deferred,
  monitor-only, and source-side-first rationale plus reopen criteria. When both files mention the
  same family, this registry should keep one current-state row and link to the reject/monitor entry
  for detailed source-side evidence.
- `docs/context/policy_search/candidate_registry.yaml` owns runnable Robot SF candidate configs and
  validation gates. A learned-policy row here is not runnable or benchmark-ready until the
  candidate registry and eligibility checklist say so.

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
  source_smoke_proven | comparison_available | proposal | prototype_only | monitor_only |
  blocked | rejected
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

External learned-policy families must also follow
`docs/context/policy_search/contracts/external_policy_intake.md`. That contract defines the
source-screen, license, checkpoint, observation/action, source-side-smoke, Robot SF adapter,
Robot SF smoke, and benchmark-suite stages. This registry owns only the durable roll-up status; do
not maintain a second current-state table in the intake contract or family notes.

## Status Crosswalk

Use this crosswalk when moving between the current-state learned-policy registry and the
reject/monitor registry. The mapping is intentionally conservative: when statuses disagree, the
stricter benchmark-readiness interpretation wins.

| Learned-policy current state | Reject/monitor vocabulary | Benchmark-readiness interpretation |
| --- | --- | --- |
| `implemented` with `comparison_available` | Usually not a reject/monitor row | Runnable only for the specific config and proof named in `candidate_registry.yaml`; not a general learned-policy promotion. |
| `staged` or `adapter_needed` with `launch_packet` or `proposal` | `defer` until prerequisite lands | Not benchmark evidence; implement only the named prerequisite or smoke path. |
| `monitor_only` with `source_harness_required` | `source-side reproduction first` | No Robot SF adapter or benchmark row until an upstream/source command, checkpoint, and observation/action contract are proven. |
| `monitor_only` with `prototype_only` | `prototype only` | Prototype or metadata surface only; keep out of main benchmark tables unless a follow-up proves adapter eligibility. |
| `monitor_only` with `monitor_only` | `monitor only` or `defer` | Track for future review; do not open implementation work without materially new source or contract evidence. |
| `blocked` benchmark status | `defer`, `source-side reproduction first`, or `prototype only` | Treat as blocked for benchmark use; the row may still guide a narrow prerequisite issue. |
| `rejected` or `rejected_for_current_adapter` | `reject for now` | Not compatible with the current local-planner adapter contract; reopen only with new public assets or a narrower reduction proof. |

## External Intake Crosswalk

Use this crosswalk with `contracts/external_policy_intake.md` when an external family moves through
source-side and Robot SF adapter intake. The mapping is fail-closed: source-side proof and
adapter-only proof are useful planning evidence, but they are not benchmark evidence.

| External intake roll-up | Registry fields | Benchmark-readiness interpretation |
| --- | --- | --- |
| `source_screened` | `integration_status: monitor_only`; `reproducibility_status: monitor_only` or `source_harness_required`; `benchmark_status: blocked` or `not_benchmark_evidence` | Interesting enough to track, not enough to implement or benchmark. |
| `license_or_checkpoint_blocked` | `integration_status: monitor_only` or `rejected`; `reproducibility_status: blocked` or `source_harness_required`; `benchmark_status: blocked` or `rejected_for_current_adapter` | Do not start adapter work until license and durable artifact blockers are cleared. |
| `contract_blocked` | `integration_status: monitor_only` or `adapter_needed`; `reproducibility_status: source_harness_required` or `monitor_only`; `benchmark_status: blocked` or `rejected_for_current_adapter` | Observation/action mismatch blocks Robot SF benchmark use even if the source looks promising. |
| `source_smoke_only` | `integration_status: monitor_only` or `adapter_needed`; `reproducibility_status: source_smoke_proven`; `benchmark_status: not_benchmark_evidence` | Native upstream/source-harness inference worked, but no Robot SF benchmark claim is allowed. |
| `adapter_only` | `integration_status: staged` or `adapter_needed`; `reproducibility_status: launch_packet` or `source_smoke_proven`; `benchmark_status: not_benchmark_evidence` | Adapter import or metadata exists, but the Robot SF runtime path has not produced smoke evidence. |
| `robot_sf_smoke_only` | `integration_status: staged` or `implemented`; `reproducibility_status: smoke_proven`; `benchmark_status: smoke_only` | Local smoke supports adapter viability only, not ranking or paper-facing comparison. |
| `benchmark_suite_complete` | `integration_status: implemented`; `reproducibility_status: comparison_available`; `benchmark_status: comparison_available` | Benchmark evidence is available only for the named config, checkpoint, stage, and promotion gate. |

## Entries

| `policy_id` | `policy_family` | `integration_status` | `reproducibility_status` | `benchmark_status` | Boundary |
| --- | --- | --- | --- | --- | --- |
| `ppo_issue791_best_v1` | `learned_baseline` | `implemented` | `comparison_available` | `comparison_available` | Best current learned-only baseline for success-oriented comparison; not a safety promotion because paper-matrix collision rate is worse than ORCA. |
| `guarded_ppo_orca_prior` | `guarded_policy` | `implemented` | `smoke_proven` | `not_benchmark_evidence` | Inference-only guarded variants are exhausted as a tuning lane; further value requires training a residual or learned risk component. |
| `shielded_ppo_issue1474_collision20_v1` | `guarded_policy` | `staged` | `smoke_proven` | `smoke_only` | Issue #1474 retrained the PPO proposal policy with the launch-packet collision-20 reward delta and W&B artifact provenance; issue #2006 repaired the zero-motion handoff, and #2029 replayed the smoke gate successfully on SLURM. Nominal-sanity remains required before any promotion or stress/full-matrix escalation. |
| `orca_residual_guarded_ppo_v0` | `residual_policy` | `staged` | `launch_packet` | `smoke_only` | Runtime residual surface exists, but learned residual training/checkpoint lineage is pending and fallback/degraded rows remain caveats. |
| `learned_risk_model_v1` | `auxiliary_risk` | `staged` | `launch_packet` | `not_benchmark_evidence` | Pre-SLURM launch packet only; hard guards remain authoritative and learned risk may only add auxiliary candidate cost. |
| `predictive_planner_v1` | `predictive_model` | `implemented` | `comparison_available` | `comparison_available` | Uses a learned pedestrian predictor inside a planner stack; evidence applies to the configured predictive planner, not a general learned local policy. |
| `predictive_mppi` | `predictive_model` | `implemented` | `comparison_available` | `comparison_available` | Learned prediction informs MPPI-style rollout scoring; integration depends on predictive model/config provenance. |
| `lidar_ppo_mlp_gate_v1` | `lidar_policy` | `adapter_needed` | `proposal` | `blocked` | Planned LiDAR learned-policy smoke from Issues #1615/#1662; not available on `main` until launch-packet work lands and smoke training runs. |
| `tentabot_value_scorer_v0` | `external_learned_policy` | `staged` | `smoke_and_nominal_diagnostic` | `not_benchmark_evidence` | Clean-room Robot SF scorer spike is executable, but the hand-authored recovery lane is stopped after Issues #1832, #1877, and #1908: scalar and trace-rule changes did not preserve the Issue #1832 collision baseline when reducing low-progress timeouts. Keep only as diagnostic baseline unless a separate learned-value-estimator contract is opened. |
| `tentabot_value_scorer_v1_static_gated` | `external_learned_policy` | `staged` | `smoke_and_nominal_diagnostic` | `not_benchmark_evidence` | Clean-room v1 static-gate lane is stopped: nominal sanity stayed at 4/18 success while collisions rose to 2/18 and near misses to 4/18. Do not extend this hand static-gate mechanism. |
| `tentabot_value_scorer_v2_route_arc` | `external_learned_policy` | `staged` | `smoke_and_nominal_diagnostic` | `not_benchmark_evidence` | Clean-room v2 route-arc lane is stopped: nominal sanity reduced low-progress to 8/18 but kept 2/18 static collisions and raised near misses to 5/18. Do not extend scalar route-progress weighting. |
| `tentabot_value_scorer_v3_trace_recovery` | `external_learned_policy` | `staged` | `smoke_and_nominal_diagnostic` | `not_benchmark_evidence` | Clean-room v3 trace-recovery lane is stopped: the smoke route low-progress timed out, and nominal sanity kept 2/18 static collisions with 9/18 low-progress timeouts. A future Tentabot-style attempt must be a learned-value/data-derived ranking experiment with explicit provenance, not another hand-authored recovery retune. |
| `crowdnav_height_igat_family` | `external_graph_policy` | `monitor_only` | `source_harness_required` | `blocked` | Source/checkpoint and graph-observation parity must be proven before a Robot SF adapter or benchmark row. |
| `arena_rosnav_stack` | `external_learned_policy` | `monitor_only` | `source_harness_required` | `blocked` | ROS Noetic/Gazebo/Flatland stack is a source-side reproduction target only; no single Robot SF-compatible policy checkpoint or adapter contract is claimed, and a named Rosnav agent must run from durable source assets before adapter work. |
| `drl_vo_family` | `external_learned_policy` | `monitor_only` | `prototype_only` | `blocked` | Tracked-agent diagnostic/prototype boundary only; not main-table ready and not a leakage-free benchmark policy. |
| `gensafenav_sonic_family` | `external_graph_policy` | `monitor_only` | `source_harness_required` | `blocked` | Safety-aware crowd-navigation source family remains behind source-harness and conformal-contract gates; no Robot SF benchmark parity claim. |
| `neupan_family` | `external_learned_policy` | `monitor_only` | `source_harness_required` | `blocked` | Point-obstacle model-based learning source is not social-navigation benchmark evidence without source-side proof and adapter contract. |
| `sage_mpc_transfer_family` | `external_graph_policy` | `monitor_only` | `source_harness_required` | `blocked` | MPC-transfer/GNN source lane remains blocked after legacy dependency smoke; checkpoint/inference path is not proven. |
| `navdp_nomad_visual_family` | `external_visual_policy` | `monitor_only` | `monitor_only` | `rejected_for_current_adapter` | RGB-D/topomap/visual-goal assumptions do not reduce cleanly to the current 2D local-planner contract. |
| `diffusion_policy_family` | `external_learned_policy` | `monitor_only` | `monitor_only` | `rejected_for_current_adapter` | Diffusion/consistency/diffuser sources are design references, not current Robot SF local-navigation adapters. |
| `decision_transformer_local_nav_family` | `external_learned_policy` | `monitor_only` | `proposal` | `blocked` | Local trajectory-data preflight exists, but no external local-navigation checkpoint or runnable adapter is selected. |
| `foundation_vla_navigation_family` | `external_visual_policy` | `rejected` | `monitor_only` | `rejected_for_current_adapter` | VLA/foundation-model sources require missing RGB/RGB-D, language-task, semantic-map, and action-interface contracts. |
| `dwa_rl_family` | `external_learned_policy` | `monitor_only` | `monitor_only` | `blocked` | Learned dynamic-window route remains source/checkpoint-first; no public source/checkpoint path is currently selected. |
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

### `shielded_ppo_issue1474_collision20_v1`

- `paper_or_source`: internal shielded-PPO repair campaign from Issues #1396 and #1474.
- `upstream_implementation`: local Robot SF PPO plus `robot_sf/planner/guarded_ppo.py`.
- `license`: local repository.
- `observation_schema`: BR-06-style SocNav/occupancy-grid PPO observation from
  `configs/training/ppo/ablations/expert_ppo_issue_1474_shielded_repair_collision20_5m.yaml`.
- `action_interface`: raw PPO velocity command, then the frozen `risk_guarded_ppo_v1` runtime guard.
- `checkpoint_availability`: W&B artifact
  `ll7/robot_sf/ppo_expert_issue_1474_shielded_repair_collision20_5m-best-success:v5`.
- `expected_dependencies`: local repo plus W&B artifact hydration.
- `reproducibility_status`: training complete; issue #2006 local guarded smoke and issue #2029
  SLURM replay pass the launch-packet smoke gate after restoring the SocNav observation handoff.
- `integration_status`: staged candidate config, prototype-gated until nominal-sanity is explicitly
  submitted and interpreted.
- `benchmark_status`: smoke-only until nominal-sanity stop gates pass with guard diagnostics
  present.
- `local_anchors`:
  - `configs/policy_search/candidates/shielded_ppo_issue1474_collision20_v1.yaml`
  - `configs/training/shielded_ppo_issue_1396_launch_packet.yaml`
  - `docs/context/issue_1396_shielded_ppo_launch_packet.md`
  - `docs/context/issue_1474_shielded_ppo_repair_closeout.md`
  - `docs/context/issue_2006_guarded_ppo_zero_motion_repair.md`
  - `docs/context/policy_search/reports/2026-06-02_shielded_ppo_issue1474_collision20_v1_smoke.md`

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

- Tentabot-style motion-primitive value policies:
  `docs/context/policy_search/2026-05-30_external_learned_policy_ranking_issue_1620.md`,
  `docs/context/policy_search/candidate_registry.yaml`.
- CrowdNav HEIGHT / IGAT / CrowdNav-family graph policies:
  `docs/context/policy_search/2026-05-30_external_learned_policy_ranking_issue_1620.md`,
  `docs/context/policy_search/issue_1367_crowdnav_family_verdict.md`,
  `docs/context/policy_search/issue_1394_crowdnav_height_source_harness.md`.
- Arena-Rosnav stack / Rosnav learned navigation:
  `docs/context/policy_search/2026-05-30_external_learned_policy_ranking_issue_1620.md`,
  `docs/context/policy_search/issue_1758_arena_rosnav_source_assessment.md`.
- DRL-VO family: `docs/context/issue_769_drl_vo_assessment.md`.
- GenSafeNav / SoNIC, NeuPAN, SAGE / MPC-transfer, DWA-RL, Decision Transformer,
  and Foundation/VLA families:
  `docs/context/policy_search/2026-05-30_external_learned_policy_ranking_issue_1620.md`.
- NavDP / NoMaD visual navigation:
  `docs/context/policy_search/2026-05-30_external_learned_policy_ranking_issue_1620.md`,
  `docs/context/policy_search/2026-05-20_navdp_nomad_diffusion_assessment.md`.
- Diffusion Policy / Consistency Policy / Diffuser families:
  `docs/context/policy_search/2026-05-30_external_learned_policy_ranking_issue_1620.md`,
- Foundation-model / VLA / multimodal navigation:
  `docs/context/policy_search/2026-05-30_foundation_model_readiness_issue_1626.md`.
  `docs/context/policy_search/2026-05-20_navdp_nomad_diffusion_assessment.md` and
  `docs/context/policy_search/2026-05-30_diffusion_policy_feasibility_issue_1621.md`.
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

This note is documentation and registry metadata only. Validate candidate and learned-policy
registry consistency with:

```bash
uv run python scripts/validation/validate_policy_search_registry.py
```
