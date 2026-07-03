# Issue #4142 dense DPCBF comparison summarizer

**Status:** diagnostic-only plan-consuming result *summarizer*. Reads each planned arm's
per-episode JSONL output into a fail-closed comparison summary. No campaign run, no
episodes, no Slurm/GPU submission, no safety-performance or collision-reduction claim.

## What this adds

PR #4318 added the packet-consuming *run planner*
(`robot_sf/benchmark/issue_4142_dpcbf_dense_runner.py`) that resolves the predeclared packet
`configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml` into an ordered three-arm run
plan (`cbf_off`, `cbf_collision_cone_on`, `cbf_dynamic_parabolic_v1_on`), each with a per-arm
output JSONL path. That planner's context note listed the next downstream gate as its own
remaining work:

> A dense-comparison summarizer that consumes the per-arm outputs under the plan's
> fail-closed exclusion.

This slice closes that gate at the **summary** level. It does not run the comparison.

- `robot_sf/benchmark/issue_4142_dpcbf_dense_summary.py` — consumes the resolved run plan and,
  for each planned arm, reads that arm's per-episode JSONL output (if present) into a
  fail-closed comparison summary. It reuses
  `robot_sf.benchmark.issue_4142_dpcbf_dense_runner.build_run_plan` as the single source of
  truth for arms, output paths, the fail-closed row-status exclusion, and the plan/readiness
  gates (no re-derivation from the packet).
- `scripts/tools/summarize_issue_4142_dpcbf_dense_comparison.py` — a thin CLI
  (`--format markdown|json`, `--fail-on-incomplete`) over that summarizer. It runs no episodes
  and writes nothing to disk.
- `tests/benchmark/test_issue_4142_dpcbf_dense_summary.py` — pins the fail-closed contract.

New output-contract schema: `robot_sf.issue_4142_dpcbf_dense_comparison_summary.v1`.

## Fail-closed contract

- **Plan gate.** The summary consumes artifacts only when the run plan is
  `plan_ready_campaign_gated`. An invalid/blocked plan yields status `plan_blocked` with no
  artifacts consumed and the plan's blockers surfaced.
- **Artifact gate.** A required arm whose artifact is missing, empty, or unparseable keeps
  the comparison out of `complete`. A single malformed JSON line marks the whole arm artifact
  `unparseable` (a partially parseable file is never summarized as complete evidence).
- **Caveat separation.** Each row's `status` is compared against the plan's declared
  `excluded_row_statuses` (`fallback`, `degraded`, `failed`, `ineligible`). Excluded rows are
  counted as caveats, broken out by status, and are **never** added to success-evidence
  counts. An unrecognized status also fails closed to a caveat. `complete` reports artifact
  *coverage*, not a safety conclusion; `all_arms_have_success_evidence` is surfaced separately
  so coverage is never conflated with comparable success rows.

## Status semantics

- `plan_blocked` — the underlying run plan did not resolve; no artifacts were consumed.
- `results_incomplete` — the plan is ready but at least one required arm artifact is
  missing/empty/unparseable. This is the expected state while execution stays gated (no arm
  output exists yet).
- `complete` — the plan is ready and all three required arms have a present, parseable
  artifact with at least one row.

## Reproduce

```bash
# Markdown comparison summary against the current checkout (reads output/, writes nothing).
uv run python scripts/tools/summarize_issue_4142_dpcbf_dense_comparison.py

# JSON summary, non-zero exit unless every required arm artifact is present (CI gate).
uv run python scripts/tools/summarize_issue_4142_dpcbf_dense_comparison.py \
    --format json --fail-on-incomplete

# Focused tests.
uv run pytest tests/benchmark/test_issue_4142_dpcbf_dense_summary.py -q
```

At the commit that introduced this summarizer the tracked packet resolves a
`plan_ready_campaign_gated` plan, but because execution stays authorization-gated no arm
output exists, so the summary reports `results_incomplete` with every arm artifact recorded
as `missing`. This is the correct, honest state: the surface is ready to consume a future
authorized campaign's outputs; until then it reports incomplete.

## Artifact disposition

The CLI writes nothing to disk (report to stdout). Per-arm JSONL inputs are read from the
git-ignored `output/issue_4142_dpcbf_dense/` directory recorded in the plan; none are
produced or promoted by this slice.

## Remaining work toward issue #4142

- An authorized executor that runs the resolved plan and produces the per-arm JSONL outputs
  (the second declared gate: explicit human/Slurm authorization) — out of scope here.
- The bounded dense comparison itself, after which this summarizer consumes the outputs.

## Related

- Run planner: `robot_sf/benchmark/issue_4142_dpcbf_dense_runner.py`
  (`docs/context/issue_4142_dpcbf_dense_runner.md`)
- Readiness surface: `robot_sf/benchmark/issue_4142_dpcbf_dense_readiness.py`
  (`docs/context/issue_4142_dpcbf_dense_readiness.md`)
- Packet: `configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml`
- Prior slices: DPCBF arm (PR #4168), passthrough gate hardening (PR #4231), readiness
  preflight (PR #4299), run planner (PR #4318)
- Parent: issue #3948; first CBF slice PR #4139
