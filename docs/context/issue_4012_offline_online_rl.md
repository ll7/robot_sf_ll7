# Issue #4012 Offline-to-Online Reinforcement Learning Fine-Tuning

Plain-language summary: issue #4012 now has a CPU-only diagnostic smoke path that loads a local
`RLTrajectoryDataset.v1` input, runs the offline-to-online Soft Actor-Critic arm, and runs the
matched from-scratch Soft Actor-Critic arm. This is workflow evidence only; it is not benchmark
evidence and makes no paper-facing performance or robustness claim.

Claim boundary: diagnostic implementation lane only; not benchmark evidence and not paper-facing.
Evidence status: smoke/diagnostic-only until a predeclared benchmark campaign runs.
Major caveats: fallback, degraded, or off-contract offline rows are fail-closed exclusions;
worktree-local datasets and checkpoints are not durable evidence.
Uncertainty: high performance effects; the current evidence only proves the workflow can initialize
and complete the paired smoke run.

## What Landed

- `RLTrajectoryDataset.v1` train-split rows can be loaded into the Soft Actor-Critic offline
  transition batch.
- Observation and action rows are checked against online environment spaces before replay use.
- Soft Actor-Critic config has an opt-in `offline_online` block; disabled configs keep existing
  behavior.
- The diagnostic orchestrator runs an offline-to-online Soft Actor-Critic arm and a from-scratch
  Soft Actor-Critic arm with matching seed and timestep budget.

## Commands

Generate or provide a compatible `RLTrajectoryDataset.v1` JSONL manifest under
`output/issue_4012_offline_online_rl_smoke/`, then run:

```bash
uv run --extra training python scripts/training/run_offline_online_rl.py \
  --config configs/training/offline_online_rl/issue_4012_sac_smoke.yaml
```

## Artifact Disposition

The default config writes checkpoints and summaries under `output/`, which is worktree-local and
ignored. Those files are diagnostic smoke artifacts only unless promoted through durable artifact
store provenance.

## Closure Audit 2026-07-05

The live issue thread's acceptance criteria map to merged implementation PRs plus the local
diagnostic smoke evidence captured in
[`docs/context/evidence/issue_4012_offline_online_smoke_2026-07-05/summary.json`](evidence/issue_4012_offline_online_smoke_2026-07-05/summary.json).

| Criterion | Evidence | Status |
| --- | --- | --- |
| Offline-pretrained policy initializes online fine-tuning end-to-end on smoke scenario. | PR #4170 added the config-gated offline-to-online Soft Actor-Critic lane; the 2026-07-05 smoke completed the `offline_online` arm with `offline_online_enabled=true` and 256 online timesteps. | Met at diagnostic-smoke tier. |
| Comparison vs from-scratch online training, diagnostic tier first. | PR #4170 added the matched scratch arm and PR #4472 added paired-arm integration reporting; the 2026-07-05 smoke completed both arms under seed 4012 and 256 online timesteps. | Met at diagnostic-smoke tier. |
| Claim boundary and any degraded mode documented. | This note, PR #4170, and PR #4472 state diagnostic-only boundaries, fail-closed exclusions, no full benchmark campaign, no Slurm/GPU submission, and no paper-facing performance claim. | Met. |
| `RLTrajectoryDataset.v1` train-split rows load into an offline transition batch. | PR #4170 added `load_offline_transition_batch`; focused tests passed on 2026-07-05. | Met. |
| Transition construction derives next observations without crossing episode boundaries. | PR #4170 added transition derivation and terminal-row handling; `tests/training/test_offline_online_rl.py` passed on 2026-07-05. | Met. |
| Dataset preflight validates observation/action compatibility against the online SAC environment. | PR #4170 added environment-space validation and Soft Actor-Critic seeding tests; focused tests passed on 2026-07-05. | Met. |
| Incompatible action semantics fail closed. | PR #4170 rejects out-of-space offline actions; focused tests passed on 2026-07-05. | Met. |
| Hybrid replay buffer stores offline and online partitions separately. | PR #4170 added `HybridReplayBuffer`; `tests/training/test_hybrid_replay_buffer.py` passed on 2026-07-05. | Met. |
| SAC config supports an opt-in `offline_online` block. | PR #4170 added the config block and parser tests; focused tests passed on 2026-07-05. | Met. |
| Existing SAC behavior unchanged when `offline_online.enabled` is false. | PR #4170 added disabled-default tests and the paired scratch arm completed with `offline_online_enabled=false` on 2026-07-05. | Met. |
| Offline replay warm-up runs or fails closed with documented reason. | PR #4170 added offline replay seeding and warm-up; PR #4472 makes arm blockers explicit in the report; the 2026-07-05 smoke had no blockers. | Met at diagnostic-smoke tier. |
| Online SAC fine-tuning continues after offline warm-up. | The 2026-07-05 smoke completed the offline-to-online arm after the offline stage and produced an ignored local checkpoint. | Met at diagnostic-smoke tier. |
| Smoke summary records both arms, artifact paths, dataset checksum, caveats, diagnostic-only evidence tier. | PR #4472 emits the paired-arm summary; the promoted evidence summary records both arms and caveats, with checksums for the small generated dataset, manifest, summary, and report. | Met. |
| Documentation states claim boundary before interpretation. | This note leads with the diagnostic-only claim boundary and the promoted summary has `eligible_for_claim=false`. | Met. |
| No paper-facing robustness or performance claim is made. | The implementation, smoke report, and this note explicitly exclude benchmark and paper-facing claims. | Met. |

Residual risk: the completed smoke used a generated local compatibility dataset and ignored local
checkpoints. A durable dataset/checkpoint publication or full benchmark campaign remains outside
this issue closure audit and must be separately scoped before any research-result claim.

## Known Limitations

No full benchmark campaign, Slurm/GPU submission, Decision Transformer policy, or paper-facing
claim is included in this slice.

## Next Empirical Action

Use the diagnostic evidence to decide whether a separate durable benchmark campaign is justified.
