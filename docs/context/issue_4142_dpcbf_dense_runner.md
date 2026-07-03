# Issue #4142 dense DPCBF comparison run planner

**Status:** diagnostic-only packet-consuming run *planner*. Resolves the predeclared
comparison packet into a concrete three-arm run plan. No campaign run, no episodes, no
Slurm/GPU submission, no safety-performance or collision-reduction claim.

## What this adds

PR #4299 added the read-only *readiness* surface
(`robot_sf/benchmark/issue_4142_dpcbf_dense_readiness.py`) that validates the predeclared
packet `configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml`, and it explicitly left
one downstream gate open:

> no packet-consuming runner is wired to schema
> `robot_sf.issue_4142_dpcbf_dense_comparison.v1`; the canonical command cannot yet execute
> the three-arm comparison.

This slice closes that first gate at the **planning** level. It does not run the comparison.

- `robot_sf/benchmark/issue_4142_dpcbf_dense_runner.py` — consumes the packet schema and
  resolves it into an ordered, per-arm run plan (`cbf_off`, `cbf_collision_cone_on`,
  `cbf_dynamic_parabolic_v1_on`). Each arm resolves to one benchmark job pinned to the
  packet's shared algorithm (`prediction_mpc_cv`), that arm's validated adapter config, the
  shared scenario manifest, and a per-arm output JSONL path under the git-ignored
  `output/issue_4142_dpcbf_dense/`. It reuses the canonical readiness validator as the single
  source of truth for arm semantics (no parallel validation path).
- `scripts/tools/run_issue_4142_dpcbf_dense_comparison.py` — a thin CLI
  (`--format markdown|json`, `--fail-on-blocked`, `--execute`) over that planner. The default
  and only supported mode is a dry-run that prints the plan and writes nothing.
- `tests/benchmark/test_issue_4142_dpcbf_dense_runner.py` — pins the fail-closed contract.

New output-contract schema: `robot_sf.issue_4142_dpcbf_dense_comparison_plan.v1`.

## Fail-closed contract

- **Readiness gate.** A plan resolves executable arm jobs only when the canonical readiness
  validator reports `inputs_ready_campaign_gated`. Any structural gap — missing/duplicated
  arm, a fallback exclusion weakened below `{fallback, degraded, failed, ineligible}`, a
  `fallback_rows_are_success_evidence: true` flag, or a missing shared `algorithm` — yields
  status `prerequisites_incomplete` with **no** executable arm jobs. There is no path that
  resolves runnable jobs from an invalid packet.
- **Fail-closed exclusion carried forward.** The packet's
  `summary_contract.excluded_row_statuses` and `fallback_rows_are_success_evidence: false`
  are copied verbatim into the resolved plan, so a downstream summarizer inherits the
  exclusion from the plan instead of re-deriving it. Fallback/degraded/failed/ineligible rows
  stay caveats, never success evidence.
- **Execution gate.** `execute_run_plan()` always raises
  `DenseComparisonExecutionGatedError`; the planner never runs episodes. Running the dense
  comparison requires explicit human/Slurm authorization and a benchmark-grade campaign,
  which stays out of scope. The CLI `--execute` flag exits non-zero (code 3) with that
  message.

## Status semantics

- `prerequisites_incomplete` — readiness failed or a runner precondition is unmet; no
  executable arm jobs were resolved.
- `plan_ready_campaign_gated` — every input is valid; the three-arm plan is resolved and
  reviewable, but execution stays gated. This is the expected healthy state; it confirms the
  plan is inspectable, **not** that the comparison may run.

## Reproduce

```bash
# Markdown run plan against the current checkout (dry-run; writes nothing).
uv run python scripts/tools/run_issue_4142_dpcbf_dense_comparison.py

# JSON plan, non-zero exit unless plan_ready_campaign_gated (CI/preflight gate).
uv run python scripts/tools/run_issue_4142_dpcbf_dense_comparison.py --format json --fail-on-blocked

# Attempting execution fails closed (exit 3).
uv run python scripts/tools/run_issue_4142_dpcbf_dense_comparison.py --execute

# Focused tests.
uv run pytest tests/benchmark/test_issue_4142_dpcbf_dense_runner.py -q
```

At the commit that introduced this planner the tracked packet resolves to
`plan_ready_campaign_gated`: all three arms resolve one benchmark job each (pinned to the
shared algorithm, distinct adapter configs, and distinct output paths), the fail-closed row
exclusion is carried into the plan, and execution stays gated.

## Artifact disposition

The CLI writes nothing to disk (report to stdout). No `output/` artifacts are produced or
promoted by this slice; the per-arm JSONL paths are recorded in the plan only, for a future
authorized executor.

## Remaining work toward issue #4142

- An authorized executor that runs the resolved plan (the second declared gate: explicit
  human/Slurm authorization) — out of scope here.
- A dense-comparison summarizer that consumes the per-arm outputs under the plan's fail-closed
  exclusion now exists (`robot_sf/benchmark/issue_4142_dpcbf_dense_summary.py`, PR #4142
  summary slice; see `docs/context/issue_4142_dpcbf_dense_summary.md`). The bounded dense
  comparison itself remains out of scope until execution is authorized.

## Related

- Readiness surface: `robot_sf/benchmark/issue_4142_dpcbf_dense_readiness.py`
  (`docs/context/issue_4142_dpcbf_dense_readiness.md`)
- Packet: `configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml`
- Runtime arm contract: `robot_sf/benchmark/cbf_safety_filter_runtime.py`
- Prior slices: DPCBF arm (PR #4168), passthrough gate hardening (PR #4231), readiness
  preflight (PR #4299)
- Parent: issue #3948; first CBF slice PR #4139
