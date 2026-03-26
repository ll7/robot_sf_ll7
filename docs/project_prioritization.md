# Project Prioritization

[Back to Documentation Index](./README.md)

This note defines the lightweight task-scoring model used to rank GitHub
Project #5 issues for the AMV benchmark program.

## Score Model

The derived numeric score is:

```text
Priority Score =
  Improvement * Success Probability * Time Criticality * Unlock Factor
  / Effort Hours^alpha
```

The current default is:

```text
alpha = 0.8
```

This preserves the simple expected-value-per-effort intuition while fixing the
two structural gaps that matter most in this repository:

- time-critical work is elevated through `Time Criticality`
- dependency-unlocking work is elevated through `Unlock Factor`

## Project Fields

The workflow expects these Project #5 number fields:

- `Improvement`
- `Success Probability`
- `Time Criticality`
- `Unlock Factor`
- `Priority Score`

It reuses the existing Project #5 field:

- `Expected Duration in Hours`
  This is the effort input. No duplicate `Effort Hours` field is created.

## Defaults and Clamping

When fields are missing or malformed, the sync helper uses conservative defaults
and clamps values into stable ranges:

| Field | Default | Range / clamp |
| --- | --- | --- |
| Improvement | `1.0` | `>= 0.0` |
| Success Probability | `0.7` | `0.0 .. 1.0` |
| Expected Duration in Hours | `1.0` | `>= 0.1` |
| Time Criticality | `1.0` | `0.5 .. 2.0` |
| Unlock Factor | `1.0` | `1.0 .. 3.0` |

The helper writes `Priority Score` rounded to 6 decimal places.

## Calibration Anchors

Use coarse estimates, not false precision.

### Improvement

- `5.0` required for benchmark or paper correctness
- `3.0` significant planner / benchmark improvement
- `1.0` normal useful task
- `0.2` cleanup or low-impact polish

### Success Probability

- `0.9` straightforward extension
- `0.7` normal implementation path
- `0.5` uncertain integration
- `0.3` research-heavy or highly uncertain

Avoid `0.0` unless the task is effectively impossible under current
constraints.

### Time Criticality

- `2.0` milestone-critical or actively blocking a deadline
- `1.0` normal
- `0.5` can wait

### Unlock Factor

- `3.0` unlocks many downstream tasks
- `1.5` partial enabler
- `1.0` standalone work

## Workflow

### Local or manual sync

```bash
uv run python scripts/tools/project_priority_score.py sync \
  --owner ll7 \
  --project-number 5 \
  --ensure-fields \
  --summary-file output/project_priority_score/sync_summary.json
```

Useful flags:

- `--dry-run` computes scores without writing them back
- `--issue-number <n>` updates one issue only
- `--skip-status Done` skips completed work

### GitHub Actions sync

The repository workflow
`/.github/workflows/project-priority-score-sync.yml` runs:

- manually via `workflow_dispatch`
- nightly on a schedule
- on issue metadata changes such as `opened`, `edited`, `labeled`, and
  `unlabeled`

## GitHub Actions Limitation

Projects v2 field edits do not currently provide a simple repository-local
workflow trigger equivalent to normal issue events. That means this repository
cannot do true immediate recomputation on every Project field edit with a
purely in-repo workflow alone.

The current implementation is intentionally pragmatic:

- do now: manual, scheduled, and issue-event sync
- do later if needed: webhook- or GitHub-App-driven sync on
  `projects_v2_item`

## Authentication

The workflow uses a PAT-style secret:

- `PROJECT_AUTOMATION_TOKEN`

It must have at least:

- `repo`
- `project`
- `workflow`

`GITHUB_TOKEN` is not sufficient for this user-owned Projects v2 automation
path.

## Worked Example

Example infrastructure task:

- Improvement = `3.0`
- Success Probability = `0.8`
- Expected Duration in Hours = `4`
- Time Criticality = `1.5`
- Unlock Factor = `2.0`

This scores higher than a small standalone cleanup task because it is both more
time-critical and more dependency-relevant, even if the direct local
improvement is not dramatic.
