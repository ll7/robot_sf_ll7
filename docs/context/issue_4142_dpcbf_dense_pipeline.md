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

PR #5419 turns `execute_run_plan()` from an always-raise gate into a bounded, explicitly
authorized **local episode executor**:

- **Fixed bounded execution inputs.** The packet gained an optional `execution` block (base
  seed, repeats, horizon, `dt`, worker cap, video-off, resume policy) with fixed defaults;
  out-of-bounds overrides fail closed in `build_run_plan`. This keeps the run a bounded
  diagnostic comparison, never a benchmark-grade campaign.
- **In-process executor.** With the exact public authorization ID, the executor reuses the
  canonical `robot_sf.benchmark.runner.run_batch` once per resolved arm, in packet order, each
  pinned to the shared scenario manifest and that arm's distinct algorithm config, writing the
  planned per-arm JSONL.
- **Authorization gate.** `execute_run_plan(plan, authorization=...)` fails closed with
  `DenseComparisonExecutionGatedError` unless `authorization == "RSF-DPCBF-DENSE-20260712"`. A
  missing/boolean/env/TTY value or a bare `--execute` flag is insufficient and **no output
  files are created**.
- **Execution manifest** (`robot_sf.issue_4142_dpcbf_dense_comparison_execution.v1`): packet/
  plan schema versions, authorization ID, git SHA + dirty flag, effective arguments, per-arm
  output paths/statuses, start/end timestamps, and overall completeness. A failing arm stays a
  visible caveat and blocks an overall `complete` status; later arms keep their true status.
- **Provenance-safe resume.** A repeated invocation resumes only when the on-disk manifest's
  provenance key (packet/plan schema, algorithm, per-arm configs, execution inputs, git SHA)
  matches; otherwise it fails closed with `DenseComparisonProvenanceMismatchError` rather than
  mixing incompatible artifacts.

The executor is local-only: it submits no Slurm/GPU job and knows nothing about `sbatch`, SSH,
tmux, or queue tooling. The packet's canonical command now points at the real issue-scoped CLI
(`scripts/tools/run_issue_4142_dpcbf_dense_comparison.py --execute --authorization ...`); the
generic `benchmark.cli run` never accepted the packet's `--config` form.

## Integration status

- **Delivered (new):** the authorization-gated local executor + execution manifest + provenance
  -safe resume (#5419), proven by fail-closed contract tests and one real-`run_batch` smoke on a
  reduced, test-only fixture (no performance claim).
- **Remaining (intentional):** an *authorized, full-scale* local or campaign run of the tracked
  `prediction_mpc_cv` packet is not performed here; the summarizer stays `results_incomplete`
  until authorized per-arm JSONL exists under `output/issue_4142_dpcbf_dense/`.
- **Blocked-on:** a maintainer decision to run the bounded local comparison (with the exact
  authorization ID) or a benchmark-grade campaign; either remains out of scope for this PR.

## Next empirical action

Run the resolved three-arm plan locally with the exact authorization ID (or under an authorized
campaign), land the per-arm JSONL under `output/issue_4142_dpcbf_dense/`, then re-run
`summarize_dense_comparison` to move the summary from `results_incomplete` to `complete`. Only
then may bounded, caveated comparison evidence be reported — and still not as a paper-facing
collision-reduction claim without a predeclared, fully reviewed benchmark campaign.

## Validation

```bash
uv run pytest tests/benchmark/test_issue_4142_dpcbf_dense_pipeline_contract.py \
  tests/benchmark/test_issue_4142_dpcbf_dense_executor.py -q
```
