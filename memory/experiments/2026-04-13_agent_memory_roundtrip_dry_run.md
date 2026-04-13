# Experiment: Agent Memory Round-Trip Dry Run

Date: 2026-04-13
Status: documented dry run

## Goal

Show the intended write -> index -> recall loop for repo-local memory entries.

## Steps

1. Write or update a topic note in `memory/experiments/`.
2. Add a one-line entry for it in `memory/MEMORY.md`.
3. Ensure `CLAUDE.md` imports `memory/MEMORY.md`.
4. In a fresh agent session, start from `CLAUDE.md` or `memory/MEMORY.md`, then open the linked
   experiment note on demand.

## Example Recall Target

This note records that the repository memory layer is designed around a concise index plus
on-demand topic files rather than a monolithic instruction file.

## Validation Notes

- `CLAUDE.md` imports `memory/MEMORY.md`, so the index is part of Claude Code startup context.
- The index links this file directly, so a fresh session has a deterministic path to the note.

## Boundary

This is a documented dry run, not an automated live Claude Code session test. If future work needs
tool-specific proof, run a manual `/memory` check in Claude Code and record the result here or in a
linked `docs/context/` note. Follow-up tracker: issue `#816`.
