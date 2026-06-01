---
name: agent-workflow-promotion
description: Promote accumulated private `.git/codex-agent-runs/notes/inbox/` workflow lessons into
  small, evidence-backed repository instruction, skill, docs, or tooling changes with validation.
category: research-iteration
kind: orchestrator
phase: implementation
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
writes:
  git: true
  github_issue: true
  github_project: false
  github_pr: false
  filesystem: true
requires:
- git
- uv
delegates_to:
- review-and-refactor
- update-docs-on-code-change
- context-note-maintainer
- gh-issue-creator
output_schema: skill_run_summary.v1
aliases:
- agent-improvement-promotion
---

# Agent Workflow Promotion

Use this skill when private workflow-capture notes already exist and the user wants durable
repository improvements. It turns candidate lessons into small tracked changes only when the
evidence warrants promotion.

## When to use

Use this to collect accumulated notes from:

```bash
git rev-parse --path-format=absolute --git-path codex-agent-runs
```

Then inspect `notes/inbox/` under that Git-dir root.

Good promotion targets include:

- `.agents/skills/...` skill instructions and registry metadata,
- `AGENTS.md` or focused `docs/ai/...` workflow guidance,
- lightweight helper scripts or validation wrappers,
- issue templates, PR guidance, or context-note conventions,
- follow-up GitHub issues when the lesson is real but too large for the current PR.

Do not use this for:

- recording a fresh candidate lesson; use `agent-workflow-capture`;
- importing raw private notes, prompts, full logs, secrets, local quota details, or machine-specific
  paths into tracked files;
- broad instruction rewrites from sparse or low-confidence notes;
- relaxing benchmark, metric, schema, model-provenance, or paper-facing evidence requirements.

## Promotion Gate

Promote a lesson only when at least one condition is true:

- repeated evidence appears across multiple notes or runs;
- the lesson directly explains a costly failure or blocked workflow;
- the fix is low-risk and clearly improves an existing instruction;
- the user explicitly asked to promote that specific lesson.

Leave low-confidence, sparse, or single-use observations in the inbox. When promotion is not safe,
create a follow-up issue only if the remaining problem is actionable and bounded.

## Workflow

1. List inbox notes:
   `find "$(git rev-parse --path-format=absolute --git-path codex-agent-runs)/notes/inbox" -maxdepth 2 -type f -name '*.md' -print | sort`.
2. Summarize each candidate in one line: class, confidence, evidence, and likely target.
3. Cluster related notes and choose the smallest durable change for the strongest evidence.
4. Edit only the relevant tracked files. Prefer updating existing repo-local skills or docs over
   creating new instruction surfaces.
5. Keep private note contents summarized. Do not copy raw logs, prompt text, secrets, local paths,
   or quota details into the repository.
6. If a skill changes, update `.agents/skills/skills.yaml` and regenerate `.agents/skills/README.md`
   when routing metadata changes.
7. Run validation proportional to the touched surface.

Suggested validation for skill or docs-only promotion:

```bash
uv run python scripts/dev/generate_skills_readme.py --check
uv run python scripts/dev/check_skills.py
git diff --check
```

If helper scripts or repo tooling change, add targeted shell syntax, unit, or smoke checks for the
changed script.

## Guardrails

- Do not delete, move, or commit inbox notes unless the user explicitly asks.
- Do not promote private local artifact paths as durable evidence.
- Do not create a tracked agent-run docs directory or similar replacement for the Git-dir inbox.
- Keep durable changes short, reviewable, and tied to evidence.
- Preserve existing benchmark fail-closed, fallback, artifact-provenance, and paper-facing proof
  rules.
- For unresolved but actionable deferred work, open a dedicated GitHub issue before closing out.

## Output

Report:

- notes reviewed and which lessons were promoted,
- files changed and why,
- validation commands and results,
- candidate lessons intentionally left private,
- follow-up issues opened or blockers that remain.
