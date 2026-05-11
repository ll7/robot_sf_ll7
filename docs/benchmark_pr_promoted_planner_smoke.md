# PR Promoted Planner Smoke

The PR promoted-planner smoke workflow gives pull requests a small benchmark-facing signal without
running a full campaign. It runs one scenario and one seed from
`configs/scenarios/single/pr_promoted_planner_smoke.yaml` over a documented core subset:
`goal`, `social_force`, and `orca`.

This subset is intentionally limited to baseline-safe planners that do not require large local model
artifacts. It is not a leaderboard and should not be used as publication evidence.

## Workflow

GitHub Actions runs `.github/workflows/pr-promoted-planner-smoke.yml` on `pull_request` and
`workflow_dispatch`.

The workflow command is:

```bash
uv run python scripts/validation/run_pr_promoted_planner_smoke.py \
  --output-root output/benchmarks/pr_promoted_planner_smoke \
  --runtime-budget-seconds 90 \
  --github-step-summary "$GITHUB_STEP_SUMMARY"
```

The runner writes:

- `output/benchmarks/pr_promoted_planner_smoke/summary.md`
- `output/benchmarks/pr_promoted_planner_smoke/summary.json`
- `output/benchmarks/pr_promoted_planner_smoke/episodes.jsonl`
- per-planner logs under `output/benchmarks/pr_promoted_planner_smoke/logs/`

The Markdown summary is appended to the GitHub step summary and the same files are uploaded as the
`pr-promoted-planner-smoke` artifact.

## Failure Conditions

The workflow exits non-zero when any planner:

- runner command exits non-zero,
- emits anything other than exactly one episode record,
- is unavailable,
- reports a readiness mode outside `native` or `adapter`,
- reports failed benchmark availability,
- falls below the tracked minimum success rate,
- exceeds the tracked maximum collision or near-miss rate,
- exceeds the per-planner runtime budget.

The tracked minimums, maximums, and reference metrics live in
`configs/benchmarks/pr_promoted_planner_smoke_baseline.json`. The summary reports deltas against the
reference metrics so reviewers can spot drift without interpreting the smoke as a full benchmark.

## Runtime Target

The workflow timeout is 20 minutes. The intended steady-state runtime is under 10 minutes on GitHub
Actions and under 2 minutes on a warmed local checkout. Each planner has a default 90 second runtime
budget enforced by the script.
