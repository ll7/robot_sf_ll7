# Issue #4142 dense DPCBF comparison — pipeline consolidation & state

**Status:** diagnostic-only. The packet → readiness → run-plan → summary pipeline is
schema-complete and fully wired end to end. No campaign run, no episodes, no Slurm/GPU
submission, no safety-performance or collision-reduction claim has been made.

**Claim boundary:** bounded diagnostic comparison inputs and contract only; not a safety
certificate and not paper-facing collision-reduction evidence. Fallback, degraded, failed, or
ineligible rows are caveats and are never success evidence.

This note is the consolidation capstone for issue #4142. The dense-comparison work landed as
several small slices in quick succession; this is the single place that records where the
pipeline stands as a whole, what remains, and the one remaining empirical action.

## Pipeline at a glance

The three CBF arms — unfiltered (`cbf_off`), collision-cone (`cbf_collision_cone_on`), and the
Dynamic Parabolic CBF variant (`cbf_dynamic_parabolic_v1_on`) — flow through a three-stage,
fail-closed, execution-gated pipeline built from one predeclared packet.

| Stage | Module | Consumes | Produces (schema) | Landed |
| --- | --- | --- | --- | --- |
| Packet | `configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml` | — | `robot_sf.issue_4142_dpcbf_dense_comparison.v1` | #4142 |
| Readiness | `robot_sf/benchmark/issue_4142_dpcbf_dense_readiness.py` | packet | `issue-4142-dpcbf-dense-readiness.v1` | #4299 |
| Run planner | `robot_sf/benchmark/issue_4142_dpcbf_dense_runner.py` | packet (gated on readiness) | `robot_sf.issue_4142_dpcbf_dense_comparison_plan.v1` | #4318 |
| Executor | `robot_sf/benchmark/issue_4142_dpcbf_dense_runner.py` (`execute_run_plan`) | resolved plan + authorization ID | `robot_sf.issue_4142_dpcbf_dense_comparison_execution.v1` | #5419 |
| Summarizer | `robot_sf/benchmark/issue_4142_dpcbf_dense_summary.py` | resolved run plan | `robot_sf.issue_4142_dpcbf_dense_comparison_summary.v1` | #4345 |

Per-stage context notes: [`issue_4142_dpcbf_dense_readiness.md`](issue_4142_dpcbf_dense_readiness.md),
[`issue_4142_dpcbf_dense_runner.md`](issue_4142_dpcbf_dense_runner.md),
[`issue_4142_dpcbf_dense_summary.md`](issue_4142_dpcbf_dense_summary.md).

The required-arms tuple and the fail-closed excluded-row statuses (`fallback`, `degraded`,
`failed`, `ineligible`) have a single canonical owner — the readiness module — which the
runner and summarizer import rather than re-declare.

## What this consolidation slice adds

The "packet-consuming runner schema" is already delivered by PR #4318 (`build_run_plan`
consumes schema `robot_sf.issue_4142_dpcbf_dense_comparison.v1`). Rather than duplicate it,
this slice binds the three merged modules with the one contract they were missing:

- `tests/benchmark/test_issue_4142_dpcbf_dense_pipeline_contract.py` — the only test that
  drives all three public entry points (`evaluate_readiness` → `build_run_plan` →
  `summarize_dense_comparison`) as one unit. It pins the packet → plan → summary schema
  lineage, asserts the runner/summarizer reuse readiness's arm/exclusion vocabulary *objects*
  (guarding against a future re-hardcoded copy that drifts), and confirms execution stays
  authorization-gated across the whole pipeline.

## What #5419 adds: the authorization-gated local executor

PR #5419 adds a bounded local executor gated by `RSF-DPCBF-DENSE-20260712`. It runs each fixed
arm through `run_batch`, checkpoints before/after every arm, and binds effective inputs plus
completed JSONL bytes. Dirty, orphaned, malformed, duplicate, or mismatched resume state fails
closed.

## Integration status

- **Delivered:** authorization gate, checkpointed manifest, artifact-bound resume, and smoke.
- **Remaining:** the tracked packet is not run here; summary stays incomplete until authorized.

Next empirical action: run the resolved plan with the exact authorization ID and summarize its
per-arm JSONL as bounded diagnostic evidence, never a paper-facing collision-reduction claim.
