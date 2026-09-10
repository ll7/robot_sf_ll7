# Claude Code Project Configuration

**Canonical repository instruction sources**: `AGENTS.md` owns the boot contract and the
`Instruction Precedence` contract, `docs/maintainer_values.md` records maintainer principles, and
`docs/dev_guide.md` is the contributor workflow reference. This file is the Claude-facing entrypoint
and contains only Claude-specific mechanics not already covered there.

## Tool-Specific Entry Points

When using a specific tool or agent, read these first:

- **GitHub (Copilot, PR agents)**: [`.github/copilot-instructions.md`](../.github/copilot-instructions.md)
- **Codex / VS Code**: [`AGENTS.md`](../AGENTS.md) plus Codex-compatible skill mirrors under
  [`.codex/skills`](../.codex/skills)
- **Cline / Claude Extension**: This file
- **Manual usage**: Start with [`AGENTS.md`](../AGENTS.md)

## Machine Context

- **Local machine config**: optional `local.machine.md` at the repository root when present
- **Disk artifacts**: generated output goes to git-ignored `output/`; small, durable evidence may
  be promoted to `docs/context/evidence/`

## Claude Code Model and Mode Selection

- **Default**: Claude Opus 4.8 (sufficient for most tasks)
- **Fast mode** (`/fast`): rapid iteration on low-risk tasks (docs, tests, refactors)
- **Benchmark/metric work**: Opus without fast mode; full verification required
- **Research/exploratory**: fast mode with clear `exploratory` status labels

## Preferred Command Interfaces

- **Testing**: `scripts/dev/run_tests_parallel.sh`
- **Formatting**: `scripts/dev/ruff_fix_format.sh`
- **Long jobs**: wrap in `tmux new-session -d -s <name>` (survives SSH disconnect)
- **Entry points**: prefer scripts under `scripts/dev/` over direct CLI

All workflow policy, validation tiers, evidence grading, and publication rules live in the
canonical sources above — do not duplicate them here.

**Last Updated**: 2026-09-10
