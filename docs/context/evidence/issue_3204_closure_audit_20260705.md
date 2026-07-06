# Issue #3204 Closure Audit Evidence

Issue #3204 asks whether rollout-derived proxy checkpoint selection maps to predictive-planner
hard-case success better than validation average displacement error (ADE) or final displacement
error (FDE). This audit rechecks the original acceptance criteria against merged pull requests and
the current fail-closed readiness checker.

**Issue:** https://github.com/ll7/robot_sf_ll7/issues/3204
**Audit date:** 2026-07-05
**Conclusion:** keep open, blocked. The analyzer and readiness contract exist, but the empirical
acceptance criteria remain unmet because the required checkpoint artifacts and a non-degenerate
proxy-enabled training summary are not available.

## Current Checker Evidence

Command:

```bash
scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/research/check_predictive_checkpoint_proxy_readiness.py \
    --config configs/research/predictive_checkpoint_proxy_v1.yaml \
    --json
```

Observed result on 2026-07-05:

- Exit code: `2`
- Status: `blocked`
- Readiness schema: `predictive-checkpoint-proxy-readiness-report.v1`
- Hard-seed fixture: `passed`
- Predictive checkpoint candidates: `8`
- Locally resolvable predictive checkpoints: `0`
- Minimum required locally resolvable checkpoints: `6`
- Candidate checkpoints with `proxy_training_run_id`: `0`
- Remaining known blockers:
  - `missing_durable_predictive_checkpoints`
  - `degenerate_hardcase_proxy_probe_v1`

The checker's own integration report states the next empirical action: hydrate or promote at least
six predictive checkpoints from one real training run, provide a proxy-enabled training summary with
non-degenerate hard-seed success spread, then rerun the readiness preflight.

## Acceptance Criteria Map

| Acceptance criterion from #3204 | Evidence found in merged PRs | Audit status |
| --- | --- | --- |
| Proxy scoring config plus script path produce per-checkpoint proxy scores on the hard-seed fixture. | #3307 merged `scripts/research/analyze_predictive_checkpoint_proxy.py` and tests. #3759 merged `configs/research/predictive_checkpoint_proxy_v1.yaml` plus fail-closed readiness preflight. #3990, #4026, #4066, #4086, #4132, and #4483 tightened the readiness contract and blocker reporting. | Partially met. The code path exists, but the current checker reports `0/8` predictive checkpoints resolve locally, so it cannot produce valid per-checkpoint proxy scores for the issue evidence contract. |
| Spearman rank correlations reported for candidate proxy and `val_ade` or `val_fde` versus hard-set success across at least six checkpoints. | #3307 added Spearman analysis for proxy history. The live issue thread and current checker both report no usable input set: fewer than six locally resolvable checkpoints and no non-degenerate proxy-enabled training summary. | Not met. No valid rank-correlation evidence across at least six checkpoints from one real training run is present. |
| A/B proxy-selected versus ADE-selected checkpoint on `predictive_hard_seeds_v1`. | No merged PR or current readiness output contains the hard-seed A/B benchmark result. #4483 explicitly keeps this as remaining work after input hydration. | Not met. The A/B remains blocked by missing durable or hydrated checkpoints and missing non-degenerate proxy history. |
| Outcome classified as adopt, negative result, or diagnostic with next step recorded. | Current merged tooling classifies the state fail-closed as diagnostic and blocked. #4483 added `integration_report` with delivered contract, remaining blockers, intentional boundaries, and next empirical action. | Met for diagnostic blocked state only. It is not a final proxy-selection outcome because the empirical contract has not run. |
| Durable artifact policy followed; do not rely on worktree-local `output/tmp`. | Current checker reports all eight predictive candidates as `missing_local_path`, with `blocked_by_artifact_scope.worktree_local_output = 8`. The tracked readiness contract names durable hydration or promotion as the revival condition. | Met for blocker detection. Not met for final issue completion because the durable or hydrated checkpoint set is still missing. |

## Relevant Merged Pull Requests

- #3307: proxy-vs-ADE checkpoint-selection analyzer and tests.
- #3759: fail-closed readiness preflight and issue #3204 readiness context note.
- #3839, #3848, #3861, #3956, #3990, #4026, #4066, #4086, #4132: incremental readiness mapping, provenance, schema, and blocker reporting slices.
- #4483: machine-readable readiness `integration_report` consolidating delivered contract, remaining blockers, intentional boundaries, and next empirical action.

## Closure Decision

Do not close #3204 yet. The merged code covers the local analyzer and readiness surfaces, but the
original empirical criteria still require inputs that are outside this CPU-only closure audit:

1. At least six durable or locally hydrated predictive checkpoints from one real training run.
2. A proxy-enabled training summary with non-degenerate hard-seed success spread.
3. The resulting Spearman proxy-versus-success comparison against ADE/FDE.
4. The hard-seed A/B benchmark for proxy-selected versus ADE-selected checkpoints.

No full benchmark campaign, Slurm or GPU submission, checkpoint hydration, model training, release
operation, or paper/dissertation claim edit was performed for this audit.
