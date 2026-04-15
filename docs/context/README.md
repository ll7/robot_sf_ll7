# Context Notes Workflow

`docs/context/` is the repository's Markdown knowledge base for issue execution history, durable
agent handoff, and reusable reasoning that should not be trapped in chat or PR text.

Use this directory for non-trivial insights, decisions, tradeoffs, validation notes, and execution
context that future contributors or agents are likely to need again.

## When To Update An Existing Note

Prefer updating an existing note when:

- the same issue, planner family, workflow, or benchmark surface is already documented there,
- the new work changes or clarifies an existing conclusion,
- or splitting the note would make the decision trail harder to follow.

When you update a note, preserve the current source of truth and remove ambiguity:

- replace outdated statements when the old wording is no longer useful,
- add dated outcome updates when historical context still matters,
- and link to the validation commands, artifacts, or follow-up notes that justify the new state.

## When To Create A New Note

Create a new note when the subject is distinct enough that merging it into an existing document
would blur ownership or make the reasoning harder to locate.

Prefer these naming patterns:

- `issue_<number>_<topic>.md` for issue-scoped notes,
- `<topic>_<date>.md` using `YYYY-MM-DD` for cross-issue audits, release notes, or bounded
  investigations.

## Required Linking

Every durable context note should link to the smallest useful set of related surfaces:

- the GitHub issue or PR that motivated the work,
- canonical docs or configs that define the contract,
- validation commands, artifacts, or output paths that support the conclusion,
- predecessor or successor notes when a document is continued or superseded.

If the note changes repository guidance, also link it from a normal entry point such as
`docs/README.md`, `docs/dev_guide.md`, `AGENTS.md`, or `docs/ai/repo_overview.md`.

## Outdated And Superseded Content

Touched notes must not leave stale conclusions ambiguous.

If a note is still the canonical surface, update it in place.

If a note should remain for history but is no longer current, mark that clearly near the top:

```md
> Status: superseded by `docs/context/issue_999_new_note.md` on 2026-04-09.
> Keep this note only for historical context.
```

If the note is no longer useful even as history, remove the outdated statement instead of stacking
contradictory prose.

## Lightweight Structure

Use the smallest structure that keeps the note reusable. Most notes should include:

- the goal or decision,
- the assumptions made and why they matter,
- the key evidence or reasoning,
- the validation path,
- the current conclusion or follow-up boundary.

Avoid turning `docs/context/` into a scratchpad. Capture what future readers need to reuse the
knowledge, not every transient iteration detail.

## Skills And Entry Points

- Repository rule: [AGENTS.md](../../AGENTS.md)
- Contributor workflow: [docs/dev_guide.md](../dev_guide.md)
- Docs index entry: [docs/README.md](../README.md)
- AI-facing orientation: [docs/ai/repo_overview.md](../ai/repo_overview.md)
- Note-maintenance skill:
  [.agents/skills/context-note-maintainer/SKILL.md](../../.agents/skills/context-note-maintainer/SKILL.md)

## Example

- [docs/context/issue_796_agent_knowledge_capture_policy.md](issue_796_agent_knowledge_capture_policy.md)
- [docs/context/issue_805_teb_corridor_commitment_iteration.md](issue_805_teb_corridor_commitment_iteration.md)

## Reasoning Notes

Design and decision rationale notes live in `docs/context/reasoning/` when the goal is to preserve
why a change was made rather than a full issue execution transcript.
