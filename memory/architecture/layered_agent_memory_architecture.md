# Layered Agent Memory Architecture

## Goal

Provide a passive, versioned memory layer for coding agents without introducing a retrieval
database, vector store, or background service that needs operational ownership.

## Layers

1. `AGENTS.md`, `docs/dev_guide.md`, and other repo instructions define workflow rules.
   These are human-readable first; agents are secondary consumers.
2. `AGENTS.md` lists `memory/MEMORY.md` in the repo-native context stack so supported agents can
   load the memory index on demand after the top-level workflow guide.
3. `memory/` stores reusable cross-session facts:
   - architecture summaries,
   - stable decisions,
   - reusable experiment outcomes,
   - known failure patterns,
   - benchmark memory boundaries.
4. `docs/context/` remains the broader execution-note knowledge base for issue history, validation
   detail, and handoff context.

## Multi-Agent Startup Map

| Agent         | Primary instruction entrypoint             | memory/MEMORY.md path                        | Status             |
|---------------|--------------------------------------------|----------------------------------------------|--------------------|
| Copilot       | `.github/copilot-instructions.md`          | explicit pointer in instructions file        | ✅ documented       |
| Codex         | `AGENTS.md` (Codex Context Stack section)  | listed in stack; agent reads on demand       | ✅ documented       |
| GitHub agents | `.agents/agents/github/` agent files       | not directly referenced; reads `AGENTS.md`   | ⚠️ indirect only   |
| Opencode      | `opencode.json` (imports `AGENTS.md`)     | listed in "First Files To Read"              | ⚠️ indirect only   |
| Gemini        | `.gemini/commands` and `AGENTS.md`         | listed in stack; agent reads on demand       | ✅ documented       |

**Opencode gap**: Opencode uses `opencode.json` (repository root) as its configuration
entrypoint, which currently imports `AGENTS.md` but does not explicitly list
`memory/MEMORY.md` in its `instructions` array. It can still discover the memory index via
`AGENTS.md` or `docs/ai/repo_overview.md`. Adding `memory/MEMORY.md` explicitly to
`opencode.json`'s instructions would make the startup path explicit. Tracked as a known gap,
not a critical failure.

## Local Memory Distinction

Repo-local memory (`memory/`) is committed to the repository, shared across supported agents and
contributors, and authoritative for stable cross-session facts. User- or tool-local memories may
exist outside the repository, but they are private caches and must not replace repo-local decisions
or be treated as reviewable project evidence.

## Optional MCP Integration Path

The supported integration path is file exposure, not a new database:

- Use a filesystem-oriented or Markdown-aware MCP server to expose `memory/` read-only to agents.
- A server such as `gnosis-mcp` or an equivalent local-Markdown MCP is acceptable only when it
  serves the repo files directly.
- Prefer exposing `memory/` first and add `docs/context/` only if the agent needs wider issue
  history.
- Keep writeback manual; automated note generation is opt-in and requires explicit team decision.

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
- Most context should live in human-readable docs (`AGENTS.md`, `docs/`) rather than agent-only
  memory files. `memory/` is for stable facts that agents benefit from recalling across sessions
  but that also serve as project documentation for contributors.
