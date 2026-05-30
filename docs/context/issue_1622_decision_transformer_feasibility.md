# Issue #1622 Decision Transformer Feasibility

Date: 2026-05-30

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1622>

External method anchors:

- Decision Transformer: <https://arxiv.org/abs/2106.01345>
- Trajectory Transformer: <https://arxiv.org/abs/2106.02039>

## Scope

This note assesses whether a Decision Transformer-style policy is a credible Robot SF local
navigation baseline and whether existing Robot SF data can support it. It does not train a model,
add a DT implementation, create a benchmark row, or replace the PPO, BC, or oracle-imitation
workflows.

Recommendation: `monitor_then_preflight`.

Decision Transformer is plausible as a later offline baseline, but Robot SF does not currently have
a durable sequence dataset with the full `(state, action, reward, return-to-go, terminal/truncated)`
contract needed for a fair DT smoke. The existing imitation trajectory path is useful, but it is
primarily behavior-cloning oriented and does not persist reward or continuation labels.

## Data Source Inventory

| Source | Current status | State/observation | Action | Reward/return | Durable provenance | DT reuse verdict |
| --- | --- | --- | --- | --- | --- | --- |
| PPO checkpoints / registry | Implemented learned baseline, especially `ppo_issue791_best_v1`. | Checkpoints do not store rollouts; replay through collectors is required. | Inferred during replay. | Not in checkpoint. | `configs/policy_search/candidates/ppo_issue791_best_v1.yaml`, `model/registry.yaml`, and `docs/context/policy_search/reports/2026-05-05_best_learning_policy.md`. | Good teacher source, but not a DT dataset by itself. |
| PPO trajectory export | Implemented for imitation workflows. | `observations` object array in `.npz`. | `actions` object array in `.npz`. | Not stored by the current collector. | Manifest from `robot_sf.benchmark.imitation_manifest`; validation by `TrajectoryDatasetValidator`. | Good BC source; incomplete for DT until rewards, dones, and returns are added. |
| BC pretraining pipeline | Implemented. | Flattens stored observations into imitation trajectories. | Uses stored action arrays. | BC does not require returns. | Config-first BC/PPO warm-start manifests. | Reusable loader patterns, but objective differs from DT. |
| Oracle imitation launch packet | Staged, no dataset collected yet. | Planned manifest/split contract. | Planned oracle actions. | Not collected yet. | `configs/training/ppo_imitation/oracle_dataset_issue_1397_launch_packet.yaml` and split policy. | Best future source if extended with reward/return labels before collection. |
| ORCA-residual BC packet | Staged, no residual dataset/checkpoint yet. | Runtime-only observation plus ORCA command/risk context. | Bounded residual target over ORCA. | Not a return-conditioned dataset. | `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml`. | Better suited to supervised residual BC than DT today. |
| Manual-control BC samples | Implemented compact JSONL export for training-marked manual records. | Per-sample observation payload. | Human mapped action. | No rewards/returns; demonstrations may be sparse. | Manual session metadata and rewind-safe sample filtering. | Useful for BC or qualitative augmentation, not first DT baseline. |
| Benchmark episode JSONL | Implemented episode-level metrics and outcome schema. | Episode summaries, not dense DT sequences by default. | Final algorithm metadata, not aligned per-step actions by default. | Metrics and outcome labels, not per-step reward stream. | Schema-backed benchmark artifacts. | Useful for evaluation labels, not direct DT training input. |
| Simulation trace export schema | Analysis-workbench trace fixture and loader. | Robot/pedestrian frame state plus planner event/action block. | Planner action/event block. | No governed reward/return contract. | `docs/context/issue_1689_simulation_trace_export_schema.md`; analysis-only evidence boundary. | Useful for visualization or trace-shape inspiration, not DT training evidence. |
| Step diagnostics scripts | Existing diagnostics can record per-step reward in targeted tools such as `scripts/validation/run_policy_search_step_diagnostics.py`. | Tool-specific records. | Tool-specific actions. | Some scripts log reward per step. | Output-local unless promoted. | Useful prototype reference, not a governed dataset surface. |

The most relevant prior local finding is in
`docs/context/issue_782_dreamerv3_pretraining_design.md`: the current PPO trajectory export path is
repo-native and reproducible, but reward-free. That is acceptable for imitation-style warm starts
and incomplete for sequence-model offline RL.

## Candidate DT Schema

A first Robot SF DT dataset should be explicit and small:

| Field | Proposed first contract |
| --- | --- |
| `state_t` | Flattened adapter-compatible observation from the same contract used by `ppo_issue791_best_v1` or a narrower LiDAR/drive-state smoke contract. |
| `action_t` | Robot SF unicycle command after policy action adaptation, with raw model action stored only if the source policy exposes it. |
| `reward_t` | Environment reward from the configured reward function, plus optional reward-component metadata when available. |
| `return_to_go_t` | Discounted or undiscounted future sum computed inside the dataset builder, with the convention recorded in the manifest. |
| `terminal_t` / `truncated_t` | Explicit episode end flags so low-progress timeouts are not confused with successful completion. |
| `episode_id`, `scenario_id`, `seed`, `step_idx` | Required provenance keys for split, leakage, and reproducibility checks. |
| `source_policy_id` | Teacher policy or planner that generated the action. |
| `availability_status`, `readiness_status` | Canonical fallback-policy fields for the source action, following `docs/context/issue_691_benchmark_fallback_policy.md`: availability uses values such as `available`, `failed`, or `not_available`, while readiness carries `fallback` or `degraded` caveats. |
| `split` | Train/validation/evaluation split assigned before collection. |

Initial horizon should be short enough to keep the baseline falsifiable: use context windows of
`20-50` decision steps before considering longer social-navigation histories. Start with one
observation contract and one source policy; mixing PPO, ORCA, manual, and oracle data should wait
until normalization and source tags are proven.

## Comparison Against Existing Paths

| Path | Strength | Weakness | Decision relative to DT |
| --- | --- | --- | --- |
| Behavior cloning | Already supported by trajectory collection and BC pretraining. | Imitates source actions without return conditioning. | Best first offline baseline because data contract exists. |
| PPO warm-start / fine-tune | Existing config-first workflow and checkpoints. | Requires online training; safety still needs benchmark proof. | Stronger near-term route than DT when a checkpoint exists. |
| Oracle imitation | Has split/leakage policy and launch packet. | Dataset not collected; future artifacts pending. | Best future teacher for DT only if reward/return fields are added before collection. |
| Decision Transformer | Uses offline sequence modeling and return conditioning; can test whether high-return behavior is recoverable from mixed-quality data. | Needs dense, durable, reward-labeled sequences and careful return normalization. | Feasible later, not implement-now. |
| Trajectory Transformer-style model | Models broader state/action/reward trajectories and can support planning-style decoding. | Higher data/schema burden and more inference complexity than DT. | Defer until DT data preflight exists. |

## Risks

- **Return-conditioning ambiguity:** Robot SF rewards include sparse success, low-progress, collision,
  comfort, and route terms. A target return may encode unsafe shortcuts unless safety/status labels
  stay visible.
- **Dataset leakage:** Evaluation seeds, hard-slice relabeling, and oracle recovery examples must
  follow the split policy before any offline training.
- **Source-policy mixture:** Mixing PPO, guarded PPO, oracle, manual, and deterministic planner data
  without source tags and normalization could train source identity rather than local navigation.
- **Fallback contamination:** Fallback or degraded actions must not become positive labels unless
  the task is explicitly to model fallback behavior.
- **Small-data overfit:** A small local DT smoke could memorize scenario IDs or seed-specific
  trajectories unless scenario and seed splits are enforced.

## Minimal Follow-Up Shape

Do not open a DT training campaign yet. The next concrete child should be:

`data: define Decision Transformer trajectory dataset preflight`

Acceptance criteria for that child:

- extend or wrap the existing trajectory collector to emit rewards, terminal/truncated flags, and
  return-to-go values without changing benchmark metrics,
- define a manifest schema with source policy, scenario IDs, seeds by split, episode IDs by split,
  reward convention, return convention, checksums, generating commit, and durable artifact URI
  policy,
- validate that rows with `readiness_status` `fallback`/`degraded` or `availability_status`
  `not_available` are excluded or explicitly labeled,
- prove the schema on a tiny dry-run fixture or fail closed with missing durable inputs,
- avoid model training until the dataset validator passes and a fair BC/PPO/oracle comparison plan
  exists.

Possible first source policy: `ppo_issue791_best_v1`, because
`configs/policy_search/candidates/ppo_issue791_best_v1.yaml` and
`docs/context/policy_search/reports/2026-05-05_best_learning_policy.md` provide the cleanest
current learned-policy mapping and registry provenance. The referenced registry model id is
`ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`. Use
`hybrid_rule_v3_static_margin0_waypoint2` only if the oracle-imitation dataset collection packet is
already being executed and can include DT fields from the start.

## Recommendation

`monitor_then_preflight`

Decision Transformer should remain a monitored offline-RL candidate until Robot SF has a governed
reward-labeled trajectory dataset. It is more expensive and less immediately useful than BC or PPO
warm-start today, but it is worth preserving as a future comparison once the oracle-imitation or PPO
trajectory export path grows reward, done, and return-to-go fields.

No DT smoke-training issue should be opened from this assessment. Open the dataset-preflight child
only when someone is ready to modify and validate the trajectory dataset contract.

## Validation

This is a documentation-only assessment. Validate with:

```bash
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
