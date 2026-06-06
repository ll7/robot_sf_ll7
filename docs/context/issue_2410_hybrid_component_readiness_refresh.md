# Issue #2410 Hybrid-Learning Component Readiness Refresh

Date: 2026-06-06
Issue: <https://github.com/ll7/robot_sf_ll7/issues/2410>
Parent: <https://github.com/ll7/robot_sf_ll7/issues/1489>
Previous matrix: [issue_2274_hybrid_component_matrix.md](issue_2274_hybrid_component_matrix.md)

## Scope

This refresh maintains the hard-guarded hybrid-learning evidence matrix after the Issue 2274
matrix and the Issue 2390/Issue 2398 ORCA-residual progress-probe launch-packet revision. It does
not launch training, submit SLURM jobs, promote checkpoints, or claim learned-component navigation
improvement.

The validator-readable rows live at
`docs/context/evidence/issue_2410_hybrid_component_readiness_refresh_2026-06-06/matrix.yaml`. A
reader-oriented classification table lives beside it as `readiness_matrix.csv`, and
`summary.json` records the conservative Issue 1489 recommendation.

## Evidence Sources

- `docs/context/issue_2274_hybrid_component_matrix.md` and
  `docs/context/evidence/issue_2274_hybrid_component_matrix_2026-06-05/matrix.yaml` for the prior
  component baseline.
- `docs/context/issue_1499_hybrid_evidence_matrix_schema.md` for the validator schema and Issue
  1489 consumer rules.
- `docs/context/issue_2225_learned_policy_failure_synthesis.md` for the learned-policy failure
  boundary.
- `docs/context/issue_2273_learned_risk_trace_preflight.md` for learned-risk trace-input status.
- `docs/context/issue_2311_orca_residual_lane_decision.md`, PR 2398, and
  `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml` for the ORCA-residual
  progress-probe revision.
- `docs/context/issue_2271_oracle_imitation_trace_preflight.md` for oracle-imitation trace
  promotion status.
- `docs/context/issue_1474_shielded_ppo_repair_closeout.md`,
  `docs/context/issue_2006_guarded_ppo_zero_motion_repair.md`, and
  `docs/context/policy_search/reports/2026-06-02_shielded_ppo_issue1474_collision20_v1_smoke.md`
  for shielded PPO repair status.
- `docs/context/issue_1961_bc_warm_start_recoverability.md` and
  `docs/context/evidence/issue_1977_bc_warm_start_cancelled_2026-06-02/artifact_manifest.md` for
  BC warm-start PPO status.

## Readiness Matrix

| Component | Current classification | Validator tier | Verdict | Blocker | Next action |
| --- | --- | --- | --- | --- | --- |
| Learned risk model v1 | blocked | `launch_packet` | `pending` | Trace inputs still lack durable artifact URIs/checksums suitable for training and downstream evaluation. | Complete trace-manifest preflight/promotion before training. |
| ORCA-residual BC v1 | diagnostic-only | `failed` | `revise` | The v0 smoke produced useful fail-closed low-progress evidence; Issue 2398 only created a v1 progress-probe launch packet and did not produce runtime success. | Run the v1 bounded smoke on a SLURM-capable host; keep nominal escalation blocked until durable smoke success exists. |
| Oracle imitation v1 | blocked | `launch_packet` | `pending` | Trace collection/preflight remains dataset-prep evidence, not promoted downstream navigation evidence. | Promote durable dataset/manifest pointers before imitation training or evaluation. |
| Shielded PPO repair v1 | diagnostic-only | `smoke_only` | `continue` | The repaired guarded smoke is positive but single-row diagnostic evidence only. | Continue only to nominal-sanity with guard diagnostics preserved. |
| BC warm-start PPO v10 | negative | `launch_packet` | `stop` | The continuation was cancelled with tail success rate 0 and no deployable final checkpoint or policy-analysis comparison. | Do not rerun the same continuation; reopen only through objective redesign or intermediate-checkpoint analysis. |

## Issue #1489 Recommendation

Issue #1489 remains blocked from comparative hybrid-learning synthesis.

Current confidence: `0.88`.

Rationale:

- No component has stress or full-matrix evidence, the only tiers that the Issue 1499 consumer rules make
  synthesis-eligible.
- Issue #2398 improves ORCA-residual rerun readiness, but it is launch-packet/config preflight evidence,
  not learned-residual success evidence.
- Shielded PPO remains the only positive runtime learned-component signal, and it is still a
  one-row smoke.
- Learned risk and oracle imitation remain blocked on durable input/artifact promotion before
  downstream navigation evidence can exist.
- BC warm-start PPO remains a negative result for the same continuation shape.

## Claim Boundary

This is synthesis-readiness maintenance. It preserves the conclusion that current hybrid-learning
evidence is useful for component-specific next actions but insufficient for a comparative
hard-guarded hybrid-learning claim.

## Validation

```bash
uv run python scripts/validation/validate_hybrid_evidence_matrix.py \
  --input docs/context/evidence/issue_2410_hybrid_component_readiness_refresh_2026-06-06/matrix.yaml \
  --check-git-history
uv run python scripts/validation/check_docs_proof_consistency.py \
  --path docs/context/issue_2410_hybrid_component_readiness_refresh.md \
  --path docs/context/evidence/issue_2410_hybrid_component_readiness_refresh_2026-06-06/summary.json \
  --path docs/context/catalog.yaml
git diff --check
```
