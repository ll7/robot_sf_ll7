---
name: agent-workflow-capture
description: Capture private candidate lessons from agent execution into `.git/codex-agent-runs/notes/inbox/`
  when a repeatable workflow, routing, validation, tooling, or instruction improvement is noticed.
category: research-iteration
kind: atomic
phase: analysis
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
writes:
  git: false
  github_issue: false
  github_project: false
  github_pr: false
  filesystem: true
requires:
- git
delegates_to: []
output_schema: skill_run_summary.v1
aliases:
- agent-improvement-capture
---

# Agent Workflow Capture

Use this skill to record a compact private candidate lesson while the execution context is still
fresh. It writes gitignored notes under the repository Git directory; it does not change tracked
instructions, skills, prompts, docs, or benchmark evidence.

## When to use

Use this when normal work reveals a reusable process lesson, for example:

- routing: a model, subagent, or delegation path was clearly too expensive, too weak, or blocked;
- prompt-contract: a worker needed clearer ownership, validation, output shape, or boundaries;
- skill-overhead: a skill caused avoidable context load or duplicated instructions;
- validation: a check was missing, misleading, stale, or too expensive for the risk tier;
- file-scope: an agent read or edited too broadly or missed a necessary surface;
- tooling: a wrapper, CLI, artifact path, or local environment convention caused avoidable friction;
- documentation drift: repo instructions no longer match current maintainer direction.

Also use it after delegated worker runs when no richer worker artifact bundle exists. In that case,
record that confidence is lower and include the local validation that confirmed or overrode the
worker summary.

Do not use this for:

- mandatory note creation after every task;
- benchmark, metric, schema, model-provenance, or paper-facing claims;
- storing raw prompts, full logs, secrets, tokens, quota details, or machine-specific paths;
- durable instruction changes; use `agent-workflow-promotion` for that.

## Workflow

1. Pick one observation class: `routing`, `prompt-contract`, `skill-overhead`, `validation`,
   `file-scope`, `tooling`, or `documentation-drift`.
2. Create the private inbox if needed:
   `mkdir -p "$(git rev-parse --path-format=absolute --git-path codex-agent-runs)/notes/inbox"`.
3. Write one small Markdown note per lesson. Prefer filenames like
   `YYYYMMDD-HHMMSS-<class>-<short-slug>.md`.
4. Keep the note evidence-first and compact. Mention commands and changed-file classes, not private
   transcripts or full logs.
5. If the lesson is weak, mark it low confidence and explicitly say `do not promote yet`.

Preferred note shape:

```markdown
---
observation_class:
confidence:
source:
promote:
---

# Short title

## Scope

- task:
- allowed_edits:
- affected_surfaces:

## Evidence

- observed_failure_or_cost:
- local_validation:
- diff_or_artifact_review:

## Candidate Lesson

One reusable lesson, or "do not promote yet" with the reason.
```

## Guardrails

- Keep notes private in `.git/codex-agent-runs/notes/inbox/`; never add a tracked parallel
  agent-run note convention.
- Do not include secrets, raw logs, prompt transcripts, local quota details, or machine-specific
  absolute paths.
- Do not treat one low-confidence note as permission to rewrite durable instructions.
- Do not weaken benchmark, metric, schema, model-provenance, or paper-facing evidence rules.
- Do not cite worktree-local `output/` artifacts as durable evidence.
- If a lesson came from a subagent without compact worker artifacts, say so and require local
  validation before promotion.

## Output

Report:

- note path under the Git-dir inbox,
- observation class and confidence,
- whether the note is promotable now or only a candidate,
- any validation or diff evidence recorded,
- any private details intentionally omitted.
