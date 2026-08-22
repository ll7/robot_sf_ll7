# Claude Code Project Configuration

**Canonical repository instruction sources**: `AGENTS.md` is the top-level instruction source for
repository rules, `docs/maintainer_values.md` is the compact source for current values and hard
contracts, and `docs/dev_guide.md` is the contributor workflow reference. This file is the
Claude-facing entrypoint and contains only Claude-specific pointers not already covered there.

## Instruction Hierarchy

For all work in this repository, consult sources in this order:

1. **Current maintainer direction** in an active issue, PR, or thread — supersedes all else.
2. **[`docs/maintainer_values.md`](../docs/maintainer_values.md)** — compact values, hard rules,
   and validation hierarchy.
3. **[`AGENTS.md`](../AGENTS.md)** — repository execution rules, structure, workflow defaults,
   and conflict precedence.
4. **[`docs/dev_guide.md`](../docs/dev_guide.md)** — contributor workflow and validation.
5. **[`.agents/skills/README.md`](../.agents/skills/README.md)** — skill landscape and decision tree.

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

**Last Updated**: 2026-08-22
