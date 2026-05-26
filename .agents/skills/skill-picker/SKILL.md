---
name: skill-picker
description: Choose the most appropriate repo-local skill for an ambiguous task by consulting .agents/skills/README.md.
category: context-docs
kind: atomic
phase: context
requires_write: false
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# Skill Picker

## When to use

Use this skill when the user asks which repo-local skill applies, gives a broad or ambiguous
workflow request, or asks for routing across issue, PR, benchmark, SLURM, artifact, or docs work.

## Purpose

Select the smallest useful repo-local skill stack when the user request is ambiguous.

## Workflow

1. Read `.agents/skills/README.md` and `.agents/skills/skills.yaml`.
2. Classify request type with the routing matrix below.
3. Choose one primary skill; add secondary skills only when phases are distinct.
4. Prefer registry metadata over prose when routing conflicts:
   `category`, `kind`, `phase`, `requires_write`, `requires_slurm`,
   `requires_benchmark_artifacts`, and `delegates_to`.
5. If a specific user choice conflicts and is clearly wrong, override with rationale.
6. Report selected skills and continue execution if requested.

## Routing matrix

| User intent | Primary skill | Secondary skill |
| --- | --- | --- |
| "Take next issue" | `goal-issue-implementation` | `gh-issue-autopilot` |
| One selected issue to PR | `gh-issue-autopilot` | `implementation-verification`, `gh-pr-opener` |
| Ambiguous or underspecified issue | `issue-contract-maintainer` | `gh-issue-sequencer` |
| Create new issue | `gh-issue-creator` | `gh-issue-sequencer` |
| Sequence issue batch | `gh-issue-sequencer` | `gh-issue-priority-assessor` |
| Fix PR comments | `gh-pr-comment-fixer` | `pr-ready-check` |
| Open PR | `gh-pr-opener` | `artifact-provenance` |
| Verify implementation | `implementation-verification` | `pr-ready-check` |
| Branch cleanup | `clean-up` | `pr-ready-check` |
| Benchmark-sensitive review | `review-benchmark-change` | `benchmark-row-status` |
| Camera-ready benchmark audit | `analyze-camera-ready-benchmark` | `artifact-provenance` |
| Submit training or benchmark job | `slurm-campaign-submit` | `artifact-provenance` |
| Submit issue-791 Auxme job | `auxme-issue791-submit` | `slurm-campaign-submit` |
| Stage external data | `data-staging-provenance` | `artifact-provenance` |
| Synthesize multiple evidence sources | `evidence-synthesis` | `paper-facing-docs` |
| Durable context note | `context-note-maintainer` | `paper-facing-docs` when claims matter |

## Negative routing

- Do not use `autoresearch` for ordinary cleanup.
- Do not use `paper-facing-docs` for non-claim docs.
- Do not use `gh-issue-autopilot` for ambiguous issues; route to `issue-contract-maintainer`.
- Do not use `auxme-issue791-submit` for non-issue-791 campaigns.
- Do not use `benchmark-overview` as validation proof; it is an orientation skill.
- Do not combine multiple skills that own the same phase unless one is a legacy compatibility entry.

## Guardrails

- Do not treat skill-picker as mandatory pre-step for explicit single-skill requests.
- Avoid proposing overly broad bundles that overlap the same phase.
- Use exact repository-specific terminology from AGENTS/skills docs.
- Match mutating skills to the user's permission and the current worktree safety constraints.

## Output

Compact routing note:

- Selected skill(s)
- Why they fit
- Skipped alternatives
- Next action
