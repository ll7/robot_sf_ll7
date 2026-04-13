# Layered Agent Memory Architecture

## Goal

Provide a passive, versioned memory layer for coding agents without introducing a retrieval
database, vector store, or background service that needs operational ownership.

## Layers

1. `AGENTS.md`, `docs/dev_guide.md`, and other repo instructions define workflow rules.
2. `CLAUDE.md` imports `AGENTS.md` and `memory/MEMORY.md` so Claude Code gets both startup
   instructions and the memory index.
3. `memory/` stores reusable cross-session facts:
   - architecture summaries,
   - stable decisions,
   - reusable experiment outcomes,
   - known failure patterns,
   - benchmark memory boundaries.
4. `docs/context/` remains the broader execution-note knowledge base for issue history, validation
   detail, and handoff context.

## Optional MCP Integration Path

The supported integration path is file exposure, not a new database:

- Use a filesystem-oriented or Markdown-aware MCP server to expose `memory/` read-only to agents.
- A server such as `gnosis-mcp` or an equivalent local-Markdown MCP is acceptable only when it
  serves the repo files directly.
- Prefer exposing `memory/` first and add `docs/context/` only if the agent needs wider issue
  history.
- Keep writeback manual unless the team deliberately opts into automated note generation later.

## Hybrid Retrieval Boundary

Hybrid retrieval is optional and constrained:

- Default flow: read `memory/MEMORY.md`, then open the few linked files that match the task.
- Optional enhancement: let an MCP server help locate the right Markdown file quickly.
- Still out of scope: embeddings, vector search, knowledge graphs, or any memory source that is
  not plainly auditable from versioned Markdown in this repository.

## Authoring Rules

- Keep `memory/MEMORY.md` concise and index-like.
- Put durable details into topic files with stable filenames.
- Prefer dated experiment and failure notes when the content is chronological.
- Link out to canonical docs or context notes instead of copying large validation narratives here.
