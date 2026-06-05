# Issue #2274 Hybrid-Learning Component Evidence Status Matrix

Date: 2026-06-05
Issue: <https://github.com/ll7/robot_sf_ll7/issues/2274>
Parent: <https://github.com/ll7/robot_sf_ll7/issues/1489>

## Scope

This note builds a component status matrix for the hard-guarded hybrid-learning synthesis parent
Issue #1489. It does not launch training, promote a learned policy, or make a paper-facing
comparative claim. The matrix answers a narrower question: which component lanes have usable
evidence now, what blocks each lane, and whether Issue #1489 can safely synthesize them.

The validator-readable rows live at
`docs/context/evidence/issue_2274_hybrid_component_matrix_2026-06-05/matrix.yaml`. A compact
reader table lives beside it as `status_matrix.csv`. The BC warm-start row uses the #1499
`not_run`/`launch_packet` shape in YAML because no deployable navigation-evaluation row exists; the
CSV and interpretation classify the same run shape as `not_available` for continuation.

## Evidence Sources

- `docs/context/issue_1499_hybrid_evidence_matrix_schema.md` for required matrix fields, evidence
  tiers, guard authority, and Issue #1489 consumer rules.
- `docs/context/issue_1624_hybrid_learning_architecture.md` for the component boundary and hard
  guard architecture.
- `docs/context/issue_2225_learned_policy_failure_synthesis.md` for the recent learned-policy
  failure synthesis.
- `docs/context/issue_1395_learned_risk_launch_packet.md` and issue #1472 comments for the learned
  risk launch-packet boundary.
- `docs/context/issue_1428_orca_residual_lineage.md`,
  `docs/context/evidence/issue_1967_orca_residual_bc_smoke_adapter_summary.json`, and issue #1475
  comments for ORCA-residual BC smoke status.
- `docs/context/issue_1397_oracle_imitation_launch_packet.md` and issue #1470 comments for oracle
  trace collection and artifact-promotion status.
- `docs/context/issue_1474_shielded_ppo_repair_closeout.md`,
  `docs/context/issue_2006_guarded_ppo_zero_motion_repair.md`, and
  `docs/context/policy_search/reports/2026-06-02_shielded_ppo_issue1474_collision20_v1_smoke.md`
  for the shielded PPO repair checkpoint and guarded smoke replay.
- `docs/context/issue_1961_bc_warm_start_recoverability.md` and
  `docs/context/evidence/issue_1977_bc_warm_start_cancelled_2026-06-02/artifact_manifest.md` for
  BC warm-start PPO recoverability and the cancelled rerun.

## Component Matrix

| Component | Evidence tier | Verdict | Blocker | Next action |
| --- | --- | --- | --- | --- |
| Learned risk model v1 | `launch_packet` | `pending` | No materialized durable trace inputs, trained checkpoint, or downstream evaluation summary. | Run #2273 to materialize/validate trace-manifest inputs before training. |
| ORCA-residual BC v1 | `failed` smoke | `revise` | The bounded smoke path reached runnable evidence but timed out with low progress and did not allow nominal escalation. | Revise the residual candidate or smoke target before another #1475 rerun. |
| Oracle imitation v1 | `launch_packet` / dataset-prep | `pending` | Train traces were collected and checksummed in issue comments, but durable artifact promotion remains the gate. | Run #2271/final preflight and promote dataset/manifest pointers before imitation training. |
| Shielded PPO repair v1 | `smoke_only` | `continue` | The repaired guarded smoke replay passed one episode, but nominal-sanity, stress, and full-matrix evidence are missing. | Continue only to nominal-sanity with guard diagnostics preserved; do not escalate broadly from smoke alone. |
| BC warm-start PPO v10 | `not_available` | `stop` for the same run shape | The repaired continuation was cancelled after 5,361,664 timesteps with tail success rate 0 and no final checkpoint or policy-analysis comparison. | Do not resubmit the same continuation; reopen only through objective redesign or intermediate-checkpoint analysis. |

## Issue #1489 Readiness Decision

Issue #1489 should remain blocked from comparative synthesis.

Current confidence: `0.85`.

Rationale:

- No component has `stress` or `full_matrix` evidence that the #1499 consumer rules allow for
  comparative synthesis.
- Shielded PPO is the only lane with a positive guarded runtime signal, but it is a single smoke
  row and therefore diagnostic only.
- ORCA-residual BC produced the most mechanism-specific negative runtime signal: adapter blockers
  are repaired, but the bounded smoke still times out and should be revised before nominal.
- Learned risk and oracle imitation remain useful research directions, but their current evidence
  is launch-packet/dataset-prep, not downstream navigation evidence.
- BC warm-start PPO should not receive another long allocation in the same shape without redesign.

## Claim Boundary

This matrix is synthesis-readiness evidence only. It does not show that learned hybrid components
improve Robot SF navigation, and it does not show that learned methods are generally ineffective.
It says the current durable evidence stack supports targeted continuation for shielded PPO nominal
sanity and artifact/preflight work for learned risk/oracle imitation, while revising ORCA-residual
BC and stopping the unchanged BC warm-start continuation.

## Validation

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/validate_hybrid_evidence_matrix.py \
  --input docs/context/evidence/issue_2274_hybrid_component_matrix_2026-06-05/matrix.yaml
bash scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check origin/main...HEAD
```
