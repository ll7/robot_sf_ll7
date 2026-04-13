# Issue 812 Layered Agent Memory Rollout

- GitHub issue: [#812](https://github.com/ll7/robot_sf_ll7/issues/812)
- Related guidance:
  - [AGENTS.md](../../AGENTS.md)
  - [CLAUDE.md](../../CLAUDE.md)
  - [docs/dev_guide.md](../dev_guide.md)
  - [docs/README.md](../README.md)
  - [docs/ai/repo_overview.md](../ai/repo_overview.md)
  - [docs/ai/retrieval_deferral.md](../ai/retrieval_deferral.md)
  - [memory/MEMORY.md](../../memory/MEMORY.md)

## Goal

Add a repo-local Markdown memory layer that agents can reuse across sessions while keeping the
existing no-database retrieval policy intact.

## Key Decisions

- The canonical startup pattern for Claude Code is now `CLAUDE.md` importing `AGENTS.md` and
  `memory/MEMORY.md`.
- `memory/` is reserved for stable cross-session facts; `docs/context/` remains the issue and
  validation note knowledge base.
- Optional MCP integration is limited to exposing repository Markdown files, not adding a separate
  semantic-memory store.
- `memory/MEMORY.md` stays concise and points to topic files for detail, following the same
  index-plus-topic-file shape Claude Code documents for its own auto memory.

## Validation

Commands and checks used for this rollout:

- Verify Anthropic's current Claude Code memory model and `AGENTS.md` import guidance from the
  official docs page on project memory.
- Confirm the new file surfaces exist and cross-link correctly:
  - `CLAUDE.md`
  - `memory/MEMORY.md`
  - `memory/architecture/layered_agent_memory_architecture.md`
  - `memory/decisions/2026-04-13_repo_local_memory_layer.md`
  - `memory/experiments/2026-04-13_agent_memory_roundtrip_dry_run.md`
  - `memory/failures/2026-04-13_memory_misuse_and_staleness.md`
  - `memory/benchmarks/2026-04-13_benchmark_memory_scope.md`
- Run the repository readiness gate after syncing docs changes.

## Current Boundary

This rollout adds the Markdown taxonomy, startup wiring, and documentation surfaces. It does not
introduce automated memory capture or a live retrieval backend. The round-trip example is a
documented dry run; live Claude Code `/memory` verification is tracked separately in
[#816](https://github.com/ll7/robot_sf_ll7/issues/816).
