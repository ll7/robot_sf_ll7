# Experiment: Agent Memory Round-Trip Dry Run → Live Validation

Date: 2026-04-13 (dry run), 2026-04-14 (live validation)
Status: **live validated** — issue #816

## Goal

Show the intended write → index → recall loop for repo-local memory entries, then confirm the
loop with a real Claude Code session.

## Steps

1. Write or update a topic note in `memory/experiments/`.
2. Add a one-line entry for it in `memory/MEMORY.md`.
3. Ensure `CLAUDE.md` imports `memory/MEMORY.md`.
4. In a fresh agent session, start from `CLAUDE.md` or `memory/MEMORY.md`, then open the linked
   experiment note on demand.

## Live Round-Trip Result (2026-04-14, issue #816)

Session: Claude Code, model `claude-sonnet-4-6`, invoked via Codex worktree
Branch: `816-validate-live-agent-round-trip-for-repo-local-memory-layer`

**What was loaded at startup:**
- `CLAUDE.md` — visible in `system-reminder` startup context
- `@AGENTS.md` — imported via CLAUDE.md; content confirmed in session context
- `@memory/MEMORY.md` — imported via CLAUDE.md; index confirmed readable

**What was validated during the session:**
- All five typed topic files (`architecture/`, `decisions/`, `experiments/`, `failures/`,
  `benchmarks/`) were reachable via direct `Read` calls
- `docs/context/issue_812_layered_agent_memory.md` was reachable and used to understand prior
  issue-812 rollout scope
- Linked memory entries in the index were followed without any broken paths
- The memory files remained concise and did not replicate content already in `AGENTS.md`

**What was NOT validated in this session:**
- Automated memory writeback (repo-local `memory/` files are written manually; Claude Code's
  user-level auto-memory writes to `~/.claude/projects/`, a separate store)
- Opencode and raw-Codex live sessions — no direct evidence that those agents loaded the index;
  only structural coverage was confirmed (see architecture note)

## Example Recall Target

This note records that the repository memory layer is designed around a concise index plus
on-demand topic files rather than a monolithic instruction file.

## Boundary

The live round-trip proves that the Claude Code startup path correctly loads `memory/MEMORY.md`
and that topic files are reachable. Automated memory capture and non-Claude agent live tests
remain future work. See `memory/architecture/layered_agent_memory_architecture.md` for the
multi-agent coverage map.
