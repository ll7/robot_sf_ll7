@AGENTS.md
@memory/MEMORY.md

## Claude Code

- Treat `AGENTS.md` as the canonical repository workflow guide.
- Treat `memory/MEMORY.md` as the concise repo-local memory index; open linked topic files on
  demand instead of expanding the whole tree into startup context.
- Before coding, read the relevant entries under `memory/architecture/`,
  `memory/decisions/`, and `memory/failures/` when the task touches workflow, experiments, or
  benchmark behavior.
- Use `docs/context/` for issue-specific execution notes and `memory/` for stable cross-session
  facts, reusable experiment summaries, and durable failure patterns.
