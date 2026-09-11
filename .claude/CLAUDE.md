# Claude Code Project Configuration

**Canonical repository instruction sources**: `AGENTS.md` owns the boot contract and the
`Instruction Precedence` contract, `docs/maintainer_values.md` records maintainer principles, and
`docs/dev_guide.md` is the contributor workflow reference. This file contains only Claude-specific
mechanics not already covered there.

## Tool-Specific Entry Points

- **GitHub (Copilot, PR agents)**: [`.github/copilot-instructions.md`](../.github/copilot-instructions.md)
- **Codex / VS Code**: [`AGENTS.md`](../AGENTS.md) plus Codex-compatible skill mirrors under
  [`.codex/skills`](../.codex/skills)
- **Cline / Claude Extension**: This file
- **Manual usage**: Start with [`AGENTS.md`](../AGENTS.md)

## Machine Context

- **Local machine config**: optional `local.machine.md` at the repository root when present; follow
  its limits before expensive commands.
- **Claude skills mirror**: `.claude/skills` is a symlink mirror of the canonical
  [`.agents/skills`](../.agents/skills) tree; edit the canonical tree.

## Provider Mechanics

- Claude Code discovers this file as project instructions; keep it to pointers and mechanics only.
- Command and model selection follow the canonical owners: the route table in
  `docs/ai/agent_workflow_entrypoints.md` and the shared routing pointer in `.agents/README.md`.
  Do not hard-code model identifiers or mode policy here.
- Long-running shell jobs run under `tmux` when the local machine context requires it.

**Last Updated**: 2026-09-10
