# Project Prioritization

[Back to Documentation Index](./README.md)

This note defines the lightweight task-scoring model used to rank GitHub
Project #5 issues for the AMV benchmark program.

The LLM-backed priority-assessment skill uses this document as the static
rubric for plausibility checks and proposed field values.

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

## Batch-First Workflow

When processing a batch of issues, keep the GitHub work in three separate passes:

1. Clean up issue text, labels, and comments first.
2. Update Project #5 fields second, after resolving the project and field IDs once for the batch.
3. Run the derived score sync last, once per batch, rather than after every individual issue.

This keeps issue cleanup independent from project routing and makes it easier to avoid GraphQL
quota exhaustion mid-batch.

Canonical workflow note:

- `docs/context/issue_713_batch_first_issue_workflow.md`

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

This measures the expected project-level gain, not just the amount of code
changed.

- `5.0` required for benchmark or paper correctness
- `3.0` significant planner / benchmark improvement
- `1.0` normal useful task
- `0.2` cleanup or low-impact polish

### Success Probability

This is the probability that the scoped work can be completed and validated in
this repository, not the probability that the idea is generally good.

- `0.9` straightforward extension
- `0.7` normal implementation path
- `0.5` uncertain integration
- `0.3` research-heavy or highly uncertain

Avoid `0.0` unless the task is effectively impossible under current
constraints.

### Time Criticality

This reflects cost of delay, blockers, or deadline pressure.

- `2.0` milestone-critical or actively blocking a deadline
- `1.0` normal
- `0.5` can wait

### Unlock Factor

This captures how much downstream work the issue unblocks.

- `3.0` unlocks many downstream tasks
- `1.5` partial enabler
- `1.0` standalone work

### Expected Duration in Hours

Use coarse buckets rather than false precision:

- `1` to `2` for very small, bounded follow-up work
- `3` to `4` for a narrow but non-trivial implementation or documentation task
- `6` to `8` for a medium spike with tests and review
- `12` to `20` for a larger integration or evidence pass
- `20+` only when the issue is clearly multi-stage and the body says so

Prefer the best honest estimate over exact arithmetic. The field is a planning
input, not a promise.

## Plausibility Checks

Use the issue body and linked context to sanity-check the proposed values before
they are written back to Project #5.

- High `Improvement` should be backed by a visible project-level gain, not just a
  local cleanup.
- High `Success Probability` should match the actual risk in the issue. Research
  spikes and uncertain integrations should usually stay near the default range.
- High `Time Criticality` should be reserved for deadline-linked or blocking
  work.
- High `Unlock Factor` should only be used when the issue clearly unblocks
  follow-up work or removes a shared bottleneck.
- Large `Expected Duration in Hours` should be justified by clear scope,
  dependencies, or multi-step validation.
- If the issue body itself says "stretch", "follow-up", or "experimental", the
  score inputs should usually reflect that conservatively.

## Assessment Workflow

The recommended workflow is review-first:

1. Read the issue description, linked notes, and current Project #5 values.
2. Compare the text against the calibration anchors above.
3. Record a short rationale for each proposed value and note the uncertainty.
4. Mark anything that looks inconsistent with the body as a plausibility risk.
5. Only write fields back to Project #5 if the result has been reviewed.

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

### GitHub Actions Sync

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
