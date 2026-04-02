# Awesome Copilot Adaptation

Date: 2026-04-02

## Decision

Adopt only the repo-local pieces that improve discoverability and repeatable workflow guidance.

The repository does **not** adopt a new marketplace, retrieval layer, or Copilot-specific runtime
assumption. The existing repo-native stack remains the source of truth:

- `AGENTS.md`
- `.specify/memory/constitution.md`
- `docs/dev_guide.md`
- `.codex/skills/`
- `docs/ai/`

## Concepts Adopted

### Skill-pack pattern

Keep reusable workflow instructions in repo-local skills so agents can discover them without
hunting through issue comments or ad hoc chat history.

### `autoresearch`

Use this for measurable, iterative improvement loops:

- define a metric,
- establish a baseline,
- try one change at a time,
- keep or discard each experiment,
- report the final trade-off clearly.

### `auto-improvement`

Use this for smaller refinement loops where the goal is clarity, robustness, or validation
tightening rather than broad experimentation.

### Discoverability links

Surface the new skills from:

- `docs/README.md`
- `docs/ai/repo_overview.md`
- `.github/copilot-instructions.md`

### Additional adopted skills

These upstream ideas were kept because they add direct value to this repository's workflow:

- `context-map` for focused file and validation discovery,
- `what-context-needed` for underspecified tasks,
- `quality-playbook` for proof-first non-trivial changes,
- `agentic-eval` for improving AI workflow outputs with small rubrics,
- `review-and-refactor` for narrow review-then-refactor passes,
- `update-docs-on-code-change` for keeping workflow docs synchronized with code.

## Concepts Rejected

- a new retrieval database or MCP-backed memory layer,
- a public Copilot marketplace clone,
- prompt copies that depend on GitHub-specific runtime behavior,
- copying upstream text verbatim when a shorter repo-native version is enough.

## Skill Selection Guide

- Use `.codex/skills/autoresearch/SKILL.md` when there is a measurable metric and room for
  multiple experiments.
- Use `.codex/skills/auto-improvement/SKILL.md` when the task is a targeted refinement pass.
- Use `context-map` before multi-file changes when you need a compact file/validation map.
- Use `what-context-needed` when the task is underspecified and you need the minimum missing
  context before proceeding.
- Use `quality-playbook` when the task needs proof-first planning, risk classification, and a
  validation choice.
- Use `agentic-eval` when the artifact being improved is a skill, instruction, prompt-like text, or
  other AI workflow output.
- Use `review-and-refactor` when the change should be reviewed first and then refactored
  surgically.
- Use `update-docs-on-code-change` when code changes would otherwise leave docs stale.
- Use `clean-up` when the task is a straightforward cleanup, formatting, or lint pass.
- Use `gh-issue-clarifier` when the task is underspecified and needs scope decisions first.

## Maintenance Rule

Keep the skills short, repo-specific, and conservative. If a workflow stops being measurable or
stops fitting the current repo validation gates, move it back to docs instead of expanding the
skill into a generic agent framework.
