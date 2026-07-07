# Issue #1358 Closure Audit

Plain-language summary: Issue #1358 is the parent question for the bounded ORCA
(Optimal Reciprocal Collision Avoidance)-residual learned local policy lane. The local
implementation and handoff-readiness surfaces are now in place, but the parent should stay open
because the child Slurm smoke/nominal evidence that would classify the lane has not been produced.

**Issue:** <https://github.com/ll7/robot_sf_ll7/issues/1358>
**Audit date:** 2026-07-07
**Closure call:** keep open; reference Issue #1358, do not close it.

## Source Thread

- Full issue body and comments were reviewed via GitHub REST API because `gh issue view 1358
  --comments` failed on the deprecated classic Projects GraphQL field.
- Latest issue-thread guidance reviewed: 2026-07-06 comment states the parent is blocked, must not
  execute training directly, and is waiting on child Issue #1475 durable smoke/nominal evidence.
- Live child states checked during audit:
  - Issue #1475: open, `resource:slurm`, `evidence:launch-packet`; still the empirical execution gate.
  - Issue #2445: closed; classifies the earlier progress-probe lane decision, but does not replace
    the missing Issue #1475 durable smoke/nominal evidence.

## Current Readiness Evidence

Command run during this audit:

```bash
LOGURU_LEVEL=WARNING ./scripts/dev/run_worktree_shared_venv.sh -- \
  uv run python scripts/tools/orca_residual_lane_readiness.py --json
```

Observed result on 2026-07-07:

- Exit code: `0`
- Readiness schema: `orca-residual-lane-readiness.v1`
- `overall_status`: `blocked_on_followup`
- `integration_report.integration_status`: `local_handoff_ready_parent_blocked`
- Local prerequisites: `9/9` ready
- Remaining blocker keys:
  - `child_classification_gate`
  - `slurm_training_required`
  - `durable_artifacts_pending`

## Acceptance Criteria Map

| Parent acceptance criterion | Evidence found in merged PRs and tracked artifacts | Audit status |
| --- | --- | --- |
| Candidate design records exact observation additions and residual action bounds. | Pull request 1409 added the bounded residual guarded-PPO surface. Pull request 1875 added the Issue #1475 Slurm/lineage launch packet, including `configs/training/orca_residual/orca_residual_bc_issue_1428.yaml` and `configs/training/orca_residual/orca_residual_bc_issue_1475_smoke_pretrain.yaml`. Pull request 3770 added `robot_sf/benchmark/orca_residual_lane_readiness.py`, which checks the local lane contract and reports the canonical routes. | Met for local handoff readiness; not learned-residual success evidence. |
| Training config versioned and runnable by single command. | Pull request 1875 added the versioned ORCA-residual behavior-cloning launch packet and `scripts/dev/sbatch_orca_residual_bc_issue1475.sh`. The readiness report names the validation and Slurm handoff command shapes without executing them. | Met for launch-packet readiness. |
| Trained checkpoint durable lineage explicit artifact pointer. | Issue #1475 tracked smoke artifacts exist, but `docs/context/evidence/issue_1475_acceptance_audit_2026-07-06.json` records durable dataset/checkpoint pointer criteria as only partially met, and current parent readiness still reports `durable_artifacts_pending`. | Not met for parent closure. |
| Candidate is registered in the policy-search registry. | Pull request 3770's readiness checker requires `docs/context/policy_search/candidate_registry.yaml` and the registered ORCA-residual candidate ids; the audit-time readiness run reported all local prerequisites ready. | Met for local handoff readiness. |
| Smoke and nominal-sanity policy-search stages run without fallback/degraded success. | Issue #1475 smoke evidence exists but failed closed. The Issue #1475 closure audit records `status: failed_closed`, `success_rate: 0.0`, `nominal_escalation_allowed: false`, and missing post-fix smoke diagnostic fields. No nominal run exists. | Not met. Do not count fallback, degraded, or failed-closed smoke as success evidence. |
| Report compares ORCA, current PPO leader, failed guarded-PPO variants, and new residual policy. | The original Issue #1358 body points to the 2026-05-05 learning-hybrid report for prior ORCA/PPO/guarded-PPO evidence. The new learned-residual comparison cannot be completed until Issue #1475 produces a valid smoke/nominal artifact set. | Not met for the new residual policy. |
| Result is classified promote, revise, or reject with scenario-stratified evidence. | Pull request 3458 / Issue #2445 recorded a stop decision for the earlier progress-probe target. The parent thread still requires Issue #1475 durable evidence before Issue #1358 receives a final continue/revise/stop classification for the learned-residual lane. | Not met for parent closure. |
| Parent stays open until Issue #1475 classifies lane continue/revise/stop durable evidence. | Latest Issue #1358 comment on 2026-07-06 and the readiness report both say Issue #1358 remains blocked on Issue #1475 durable smoke/nominal training evidence and classification. | Met as a guardrail; requires keeping issue open. |
| No new residual-policy training children are added before Issue #1475 reports durable evidence or a fail-closed blocker. | This audit adds only a tracked evidence note. It does not add a training child, submit Slurm, run training, or mutate planner behavior. | Met. |

## Relevant Merged Pull Requests

- Pull request 1409: bounded residual guarded-PPO runtime surface.
- Pull request 1875: Issue #1475 ORCA-residual Slurm prep and launch packet.
- Pull request 3458: Issue #2445 progress-probe lane decision.
- Pull request 3770: read-only Issue #1358 lane readiness/preflight checker and CLI.
- Pull request 4555: parent integration status report in the readiness checker.
- Pull request 4561: Issue #1475 smoke-to-nominal gate and acceptance audit evidence.

## Closure Decision

Do not close Issue #1358 yet. The smallest remaining action that would move the issue toward
closure is not local code: child Issue #1475 needs one approved Slurm smoke rerun that satisfies the
smoke-to-nominal gate, then a nominal run and durable artifact pointers if the gate passes. After
that evidence exists, Issue #1358 can be classified `continue`, `revise`, or `stop`.

No full benchmark campaign, Slurm or GPU submission, model training, release operation, or
paper/dissertation claim edit was performed for this audit.
