# Decision: Adopt A Repo-Local Markdown Memory Layer

Date: 2026-04-13

## Decision

Adopt a repo-local `memory/` tree with a concise `memory/MEMORY.md` index and topic files for
architecture, decisions, experiments, failures, and benchmark-specific memory guidance.

## Why

- Agents in this repository need stable cross-session recall for design and experiment context.
- The repository already prefers auditable Markdown knowledge over opaque external state.
- A Markdown tree is easy to review, diff, and update alongside code and docs.

## What This Includes

- `CLAUDE.md` importing both `AGENTS.md` and `memory/MEMORY.md`
- a stable `memory/` taxonomy
- one example note per memory type
- contributor guidance in `docs/dev_guide.md`

## What This Excludes

- vector databases
- embedding pipelines
- knowledge graphs
- automatic summarization daemons
- benchmark or training code changes

## Status

Current repository policy allows optional MCP exposure of these Markdown files, but the files
themselves remain the source of truth.
