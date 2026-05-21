---
name: context-note-maintainer
description: "Create or refresh linked docs/context notes so reusable agent knowledge stays discoverable, current, and easy to hand off."
---

# Context Note Maintainer

## When to use

Use this skill when reusable repo knowledge needs durable Markdown capture, or when `docs/context/` notes are
stale, conflicting, or hard to discover.

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

## Proof and Guardrails

- Record what changed and why it is the canonical note.
- Remove or clearly mark superseded claims.
- Confirm links and commands resolve to existing files/entries.
- For durable benchmark evidence, prefer a short manifest over raw logs in `output/`.
- Run `uv run python scripts/dev/check_skills.py` only after finishing skill doc edits.

## Output

Always report:
- which note was created/updated,
- why this surface is canonical,
- stale content addressed,
- updated discoverability links.

Also include:
- confirmation that the note is durable handoff content, not a temporary scratchpad.
