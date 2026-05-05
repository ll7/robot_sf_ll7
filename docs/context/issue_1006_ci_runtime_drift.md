# Issue #1006 GitHub CI Runtime Drift Diagnosis

## Scope

Issue #1006 diagnoses why GitHub CI takes much longer than local PR readiness on recent small PRs.
This slice adds `scripts/dev/ci_timing_summary.py`, a lightweight reporting command that turns
`gh run view --json ...` output into a compact timing summary for issue comments and PR triage.

No CI cache, shard, or test-selection policy is changed in this PR. The evidence below supports a
reporting-first fix because the slow phase is observable but not yet isolated to one test subset or
cache key.

## Command

For a completed GitHub Actions run:

```bash
rtk uv run python scripts/dev/ci_timing_summary.py --run-id <run-id> --top 10
```

For saved JSON from `gh run view`:

```bash
rtk gh run view <run-id> --json databaseId,displayTitle,headBranch,status,conclusion,createdAt,updatedAt,jobs > output/tmp/ci_run.json
rtk uv run python scripts/dev/ci_timing_summary.py --run-json output/tmp/ci_run.json --top 10
```

The saved JSON/log path is disposable input. Do not promote it unless a later diagnosis needs a
durable artifact.

## Evidence

Sampled completed PR CI runs:

| PR | run | local proof | CI total | queue | slowest CI steps |
| --- | --- | --- | --- | --- | --- |
| #1007 | `25373308385` | `BASE_REF=origin/main rtk scripts/dev/pr_ready_check.sh` completed with `3191 passed, 14 skipped, 3 warnings in 247.69s` | 1477s | 3s | Unit tests 714s; System packages 489s; Validation smoke 125s |
| #1008 | `25374732844` | post-main-merge readiness completed with `3194 passed, 14 skipped, 3 warnings in 296.02s` | 1042s | 34s | Unit tests 710s; Validation smoke 127s; Cache `.venv` 44s |

Interpretation:

- Queue time is not the bottleneck for these runs.
- The repeatable dominant cost is the CI `Unit tests` step at about 11m50s, versus roughly 4-5m
  local full readiness on the 16-worker Linux workstation.
- `Validation smoke tests` consistently adds about 2m05s.
- #1007 had an additional setup outlier: `System packages for headless` took 8m09s; #1008 reduced
  that same step to 20s, so this appears intermittent mirror/package pressure rather than the main
  repeatable drift.

## Recommended Next Actions

Use the new timing command before changing CI structure. The next evidence-bearing issue should:

- compare `Unit tests` logs with the local slow-test report from `BASE_REF=origin/main
  scripts/dev/pr_ready_check.sh`,
- decide whether example/image integration tests need a separate CI phase or summary,
- avoid cache-key changes unless multiple runs show setup/cache phases are the repeatable
  bottleneck.

This slice does not recommend test removal. Constitution Principle XIII still applies: optimize or
split high-value tests only after verifying their contract value.

## Validation

RED proof:

```bash
rtk uv run pytest tests/dev/test_ci_timing_summary.py -q
```

Failed before implementation with `ModuleNotFoundError` because `scripts.dev.ci_timing_summary`
did not exist.

GREEN proof:

```bash
rtk uv run pytest tests/dev/test_ci_timing_summary.py -q
rtk uv run ruff check scripts/dev/ci_timing_summary.py tests/dev/test_ci_timing_summary.py
rtk uv run python scripts/dev/ci_timing_summary.py --run-id 25373308385 --top 6
rtk uv run python scripts/dev/ci_timing_summary.py --run-id 25374732844 --top 6
rtk uv run python scripts/dev/ci_timing_summary.py --run-id 25374732844 --top 3 --json
```

The targeted tests passed with `3 passed in 17.96s`; Ruff passed on the touched Python files.

## Artifact Decision

The timing command writes no files unless the caller chooses to save `gh run view` JSON separately.
Targeted pytest refreshed ignored coverage output under `output/coverage/`; those files are
disposable validation artifacts and are not promoted.
