---
name: context-note-maintainer
description: "Create or refresh linked docs/context notes so reusable agent knowledge stays discoverable, current, and easy to hand off."
---

# Context Note Maintainer

## Overview

Use this skill when work produces non-trivial reusable insights, decisions, reasoning, or
validation context that should be preserved in Markdown, or when an existing `docs/context/` note
has become stale, superseded, or hard to discover.

This skill is for durable handoff knowledge, not transient scratch notes.

## Read First

- `AGENTS.md`
- `docs/context/README.md`
- `docs/dev_guide.md`
- `docs/README.md`
- `docs/ai/repo_overview.md`

## Workflow

1. Identify the canonical note surface
   - Prefer updating the existing note when it already covers the same issue, workflow, planner
     family, or decision boundary.
   - Create a new note only when the topic is distinct enough that merging would blur the source of
     truth.

2. Capture the reusable context
   - Persist the goal, key decisions, reasoning, validation commands, current conclusion, and
     follow-up boundary.
   - Keep the note concise, but include enough context that another agent can resume work without
     re-deriving the same conclusions.

3. Clean up stale content
   - If touched content is outdated, update it, remove it, or mark it clearly as superseded with a
     pointer to the replacement note.
   - Do not leave contradictory statements in place without a status marker.

4. Link for discoverability
   - Link the note to the related issue or PR, the canonical docs/configs, and the proof artifacts
     or commands that justify the conclusion.
   - If the note changes workflow guidance, make sure a normal entry point such as `docs/README.md`,
     `docs/dev_guide.md`, `AGENTS.md`, or `docs/ai/repo_overview.md` points to it.

5. Validate references
   - Check that referenced paths and commands exist.
   - Run `uv run python scripts/dev/check_skills.py` if skill docs were changed.

## Output

Always report which note was updated or created, why that was the canonical surface, what stale
content was cleaned up, and which entry points now link to it.

## Guardrails

- Do not create duplicate notes when an existing canonical note can be updated.
- Do not leave touched stale content ambiguous.
- Do not treat `docs/context/` as a scratchpad or dump of raw terminal output.
- Do not rely on chat-only context for decisions that future agents will need to reuse.
