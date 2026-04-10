# Issue 796 Agent Knowledge Capture Policy

- GitHub issue: [#796](https://github.com/ll7/robot_sf_ll7/issues/796)
- Canonical workflow: [docs/context/README.md](README.md)
- Related repo guidance:
  - [AGENTS.md](../../AGENTS.md)
  - [.github/copilot-instructions.md](../../.github/copilot-instructions.md)
  - [docs/dev_guide.md](../dev_guide.md)
  - [docs/README.md](../README.md)
  - [docs/ai/repo_overview.md](../ai/repo_overview.md)
  - [.agents/skills/context-note-maintainer/SKILL.md](../../.agents/skills/context-note-maintainer/SKILL.md)

## Goal

Make linked Markdown knowledge capture an explicit repository contract for non-trivial agent work so
important context does not remain trapped in chat, PR text, or issue comments.

## Decisions Implemented

- `docs/context/` is now treated as the repository's Markdown knowledge base for durable handoff
  context, not just as a loose collection of execution notes.
- Agents should persist non-trivial reusable insights, decisions, reasoning, and validation notes in
  Markdown when that context would otherwise be lost.
- Existing canonical notes should be updated in preference to creating duplicates.
- Touched stale or superseded notes must be updated, removed, or clearly marked with a pointer to
  the current source of truth.
- Notes should cross-link to issues, PRs, canonical docs, validation commands, and successor notes
  to keep the repository discoverable without adding a database.

## Rollout Surface

- Policy rule in [AGENTS.md](../../AGENTS.md)
- Copilot-facing mirror in [.github/copilot-instructions.md](../../.github/copilot-instructions.md)
- Contributor workflow pointer in [docs/dev_guide.md](../dev_guide.md)
- Docs index entry in [docs/README.md](../README.md)
- AI-facing discoverability update in [docs/ai/repo_overview.md](../ai/repo_overview.md)
- Dedicated note-maintenance workflow in
  [.agents/skills/context-note-maintainer/SKILL.md](../../.agents/skills/context-note-maintainer/SKILL.md)

## Validation

- Verify all referenced files exist.
- Run `uv run python scripts/dev/check_skills.py`.
- Run `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`.

## Current Boundary

This rollout defines the rule, the discoverability surfaces, and one dedicated skill. It does not
attempt a full historical cleanup of every old note in `docs/context/`; retroactive cleanup remains
incremental and should happen when older notes are touched or when a high-value stale surface is
identified.
