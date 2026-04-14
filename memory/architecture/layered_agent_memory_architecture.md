# Layered Agent Memory Architecture

## Goal

Provide a passive, versioned memory layer for coding agents without introducing a retrieval
database, vector store, or background service that needs operational ownership.

## Layers

1. `AGENTS.md`, `docs/dev_guide.md`, and other repo instructions define workflow rules.
   These are human-readable first; agents are secondary consumers.
2. `CLAUDE.md` imports `AGENTS.md` and `memory/MEMORY.md` so Claude Code gets both startup
   instructions and the memory index at session start.
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
| Claude Code   | `CLAUDE.md` (`@AGENTS.md`, `@memory/MEMORY.md`) | direct import at startup                 | ✅ live validated   |
| Copilot       | `.github/copilot-instructions.md`          | explicit pointer in instructions file        | ✅ documented       |
| Codex         | `AGENTS.md` (Codex Context Stack section)  | listed in stack; agent reads on demand       | ✅ documented       |
| GitHub agents | `.agents/agents/github/` agent files       | not directly referenced; reads `AGENTS.md`   | ⚠️ indirect only   |
| Opencode      | `AGENTS.md` via `docs/ai/repo_overview.md` | listed in "First Files To Read"              | ⚠️ no entrypoint   |

**Opencode gap**: Opencode has `.opencode/skills/` (mirrored from `.agents/skills/`) and
`.opencode/tools/` but no dedicated instruction entrypoint that imports `memory/MEMORY.md`.
It relies on reading `AGENTS.md` and `docs/ai/repo_overview.md`, which list the memory index.
This is sufficient for agents that read those files; a dedicated `opencode.md` or similar config
would make the path explicit. Tracked as a known gap, not a critical failure.

## Auto-Memory Distinction

Claude Code's user-level auto-memory (written to `~/.claude/projects/<repo-slug>/memory/`) is
**separate** from this repo-local `memory/` layer:

- **User-level auto-memory**: private to the user's local Claude Code install; contains session
  notes, preferences, and per-user context. Not committed to the repository.
- **Repo-local memory** (`memory/`): committed to the repository; shared across all agents and
  contributors; authoritative for stable cross-session facts.

Both layers can coexist. The repo-local layer is the canonical shared source; user-level memory
adds personal context on top. Do not duplicate repo-level decisions in user-level memory.

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
