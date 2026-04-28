# Failure Patterns: Memory Misuse And Staleness

Date: 2026-04-13

## Anti-Patterns

- Storing issue-specific execution logs in `memory/` instead of `docs/context/`
- Letting `memory/MEMORY.md` grow into a second `AGENTS.md`
- Copying benchmark claims into memory without linking the validating artifact or note
- Leaving stale summaries in place after the underlying workflow changes
- Creating multiple near-identical notes for the same stable fact

## Mitigations

- Keep `memory/MEMORY.md` as an index, not a dump.
- Put validation-heavy narratives in `docs/context/`.
- Update the canonical note instead of adding another one.
- Link benchmark-sensitive memory notes to the proof surface that justifies them.

## Retrieval Warning

Agents should treat `memory/` as helpful context, not as an override for newer canonical docs or
freshly validated benchmark evidence.
