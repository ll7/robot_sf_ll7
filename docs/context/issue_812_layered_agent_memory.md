# Issue 812 Layered Agent Memory Rollout

- GitHub issue: [#812](https://github.com/ll7/robot_sf_ll7/issues/812)
- Related guidance:
  - [AGENTS.md](../../AGENTS.md)
  - [docs/dev_guide.md](../dev_guide.md)
  - [docs/README.md](../README.md)
  - [docs/ai/repo_overview.md](../ai/repo_overview.md)
  - [docs/ai/retrieval_deferral.md](../ai/retrieval_deferral.md)
  - [memory/MEMORY.md](../../memory/MEMORY.md)

## Goal

Add a repo-local Markdown memory layer that agents can reuse across sessions while keeping the
existing no-database retrieval policy intact.

## Key Decisions

- The current canonical startup pattern is `AGENTS.md` as the top-level guide, with
  `memory/MEMORY.md` listed in the repo-native context stack for supported agents to load on
  demand.
- `memory/` is reserved for stable cross-session facts; `docs/context/` remains the issue and
  validation note knowledge base.
- Optional MCP integration is limited to exposing repository Markdown files, not adding a separate
  semantic-memory store.
- `memory/MEMORY.md` stays concise and points to topic files for detail.

Historical note: the original issue-812 rollout used a Claude-specific startup pointer. Issue #1728
retired that compatibility surface; the memory-layer decision remains active through `AGENTS.md`.

## Validation

Commands and checks used for this rollout:

- Confirm the new file surfaces exist and cross-link correctly:
  - `AGENTS.md`
  - `memory/MEMORY.md`
  - `memory/architecture/layered_agent_memory_architecture.md`
  - `memory/decisions/2026-04-13_repo_local_memory_layer.md`
  - `memory/experiments/2026-04-13_agent_memory_roundtrip_dry_run.md`
  - `memory/failures/2026-04-13_memory_misuse_and_staleness.md`
  - `memory/benchmarks/2026-04-13_benchmark_memory_scope.md`
- Run the repository readiness gate after syncing docs changes.

## Live Round-Trip Validation (2026-04-14, issue #816)

Issue #816 performed the live agent round-trip that this rollout deferred. Findings:

- **Historical Claude Code result** (model `claude-sonnet-4-6`): the retired `CLAUDE.md` startup
  pointer loaded both `@AGENTS.md` and `@memory/MEMORY.md` at session startup. All five typed topic
  files were reachable. This remains historical validation of the memory layout, not a current
  supported entrypoint.
- **Multi-agent coverage**: Copilot via `.github/copilot-instructions.md`, Codex via `AGENTS.md`
  Codex Context Stack section, and GitHub agents via `.agents/agents/github/` all have structural
  pointers or references. Opencode uses `opencode.json` as its configuration entrypoint; `docs/ai/repo_overview.md`
  lists `memory/MEMORY.md` in "First Files To Read". The gap is that `opencode.json` does not
  explicitly import `memory/MEMORY.md` in its `instructions` array; tracked as a known limitation.
- **Local-memory distinction clarified**: tool-local memories are private caches outside this
  repository. The committed `memory/` tree remains the shared, reviewable memory layer.

Full validation record: `memory/experiments/2026-04-13_agent_memory_roundtrip_dry_run.md`
Updated architecture note: `memory/architecture/layered_agent_memory_architecture.md`

## Current Boundary

This rollout adds the Markdown taxonomy, startup wiring, and documentation surfaces. It does not
introduce automated memory capture or a live retrieval backend. Opencode lacks a dedicated
instruction entrypoint that explicitly imports `memory/MEMORY.md`; tracked as a known gap.
