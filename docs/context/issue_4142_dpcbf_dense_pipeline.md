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

## Integration status

- **Remaining (intentional, fail-closed):** running the three-arm dense comparison — episodes,
  Slurm/GPU — is out of scope for every slice. `execute_run_plan()` always raises
  `DenseComparisonExecutionGatedError`; the summarizer reports `results_incomplete` while no
  authorized artifacts exist. This is the designed gated state, not a defect.
- **Remaining (new):** none. This slice adds no new runtime behavior and no new gate.
- **Blocked-on:** explicit human/Slurm authorization for a benchmark-grade campaign.

## Next empirical action

The only forward step is the authorized campaign itself (out of scope here): run the resolved
three-arm plan under the canonical command, land the per-arm JSONL under
`output/issue_4142_dpcbf_dense/`, then re-run `summarize_dense_comparison` to move the summary
from `results_incomplete` to `complete`. Only then may bounded, caveated comparison evidence be
reported — and still not as a paper-facing collision-reduction claim without a predeclared,
fully reviewed benchmark campaign.

## Validation

```bash
uv run pytest tests/benchmark/test_issue_4142_dpcbf_dense_pipeline_contract.py -q
```
