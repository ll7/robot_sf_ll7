# Claude Code Project Configuration

**Canonical repository instruction sources**: This file points to the authoritative sources for this
repository. Use the listed resources for detailed guidance on repository conventions, validation
approaches, and AI assistant workflow.

## Instruction Hierarchy

For all work in this repository, consult sources in this order:

1. **Current maintainer direction** in an active issue, PR, or thread — supersedes all else.
2. **[`docs/maintainer_values.md`](../docs/maintainer_values.md)** — compact source for current
   values, hard rules (be honest, transparent, reproducible), and validation hierarchy.
3. **[`AGENTS.md`](../AGENTS.md)** — repository execution rules, structure, workflow defaults, and
   precedence order for resolving conflicts.
4. **[`docs/dev_guide.md`](../docs/dev_guide.md)** — contributor workflow, testing strategy, and
   validation checklist.
5. **[`.agents/PLANS.md`](../.agents/PLANS.md)** — plan-writing convention for non-trivial work.
6. **[`.agents/skills/README.md`](../.agents/skills/README.md)** — skill landscape, maturity levels,
   and decision tree for common Claude Code tasks.
7. **[`memory/MEMORY.md`](../memory/MEMORY.md)** — project-local cross-session memory index (stable
   facts, research hypotheses, blocked work).

## Tool-Specific Entry Points

When using a specific tool or agent, read these first:

- **GitHub (Copilot, PR agents)**: [`.github/copilot-instructions.md`](../.github/copilot-instructions.md)
- **Codex / VS Code**: [`AGENTS.md`](../AGENTS.md) plus Codex-compatible skill mirrors under
  [`.codex/skills`](../.codex/skills)
- **Cline / Claude Extension**: This file
- **Manual usage**: Start with [`AGENTS.md`](../AGENTS.md)

## Machine Context

- **Local machine config**: optional `local.machine.md` at the repository root when present
- **Disk artifacts**: All generated output must go to git-ignored `output/` directory; small,
  durable evidence may be promoted to `docs/context/evidence/`
- **Shell commands**: use ordinary repository commands directly unless current maintainer
  direction names a specific wrapper.

## Quick Start for Claude Code Tasks

### For exploration or triage:
1. Check [`memory/MEMORY.md`](../memory/MEMORY.md) to ground context in prior work.
2. Use [`.agents/skills/context-map/SKILL.md`](../.agents/skills/context-map/SKILL.md) for
   multi-file navigation.
3. Use [`.agents/skills/what-context-needed/SKILL.md`](../.agents/skills/what-context-needed/SKILL.md)
   if the task is underspecified.

### For implementation:
1. Consult [`AGENTS.md`](../AGENTS.md) and [`docs/dev_guide.md`](../docs/dev_guide.md) for
   validation approach (docs-only, code-focused, benchmark-facing, paper-grade).
2. Use [`.agents/skills/quality-playbook/SKILL.md`](../.agents/skills/quality-playbook/SKILL.md) to
   pick validation style.
3. Use [`.agents/skills/auto-improvement/SKILL.md`](../.agents/skills/auto-improvement/SKILL.md) for
   refinement loops.
4. Use [`.agents/skills/autoresearch/SKILL.md`](../.agents/skills/autoresearch/SKILL.md) for
   measurement-driven improvement.

### Before opening a PR:
1. Run `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` (or cheaper path from
   `docs/ai/ai-workflow.md` for low-risk branches).
2. Verify AI config drift with `uv run python scripts/tools/sync_ai_config.py --check`.
3. Update `CHANGELOG.md` for user-facing changes.

### For code review:
- Use [`.agents/skills/review-and-refactor/SKILL.md`](../.agents/skills/review-and-refactor/SKILL.md)
  for narrow code review, or
  [`.agents/skills/review-benchmark-change/SKILL.md`](../.agents/skills/review-benchmark-change/SKILL.md)
  for benchmark-facing changes.
- For benchmark/metric changes, always require `paper-grade` evidence per
  [`docs/maintainer_values.md`](../docs/maintainer_values.md).

## Claude Code Model & Mode Selection

- **Default**: Claude Opus 4.8 (sufficient for most tasks)
- **Fast mode** (`/fast`): Use for rapid iteration on low-risk tasks (docs, tests, refactors)
- **Benchmark/metric work**: Use Opus without fast mode; requires full verification
- **Research/exploratory**: Use fast mode with clear `exploratory` status labels

## Preferred Command Interfaces

- **Testing**: `scripts/dev/run_tests_parallel.sh` (16 workers on this machine)
- **Formatting**: `scripts/dev/ruff_fix_format.sh`
- **Pre-commit**: `uv run pre-commit install && uv run pre-commit run --all-files`
- **Long jobs**: Wrap in `tmux new-session -d -s <name>` (survives SSH disconnect)
- **All entry points**: Prefer scripts under `scripts/dev/` over direct CLI

## Test Failure Evaluation

When you encounter test failures, follow the protocol in
[`.github/copilot-instructions.md`](../.github/copilot-instructions.md) under "Test Failure
Evaluation": classify test value first (core feature, regression, edge case, brittle), then decide
whether to fix immediately, defer with tracking, or remove with documented rationale.

## Evidence Grading for Claims

Use the ladder from [`docs/maintainer_values.md`](../docs/maintainer_values.md):

- `diagnostic-only`: Debugging or contract probes; no semantic claim
- `smoke evidence`: Narrow execution proof; good for "does it run"
- `nominal benchmark evidence`: Predeclared benchmark-matrix results
- `paper-grade`: Fully reproducible; suitable for manuscript-facing claims

Always open with claim boundary, evidence status, major caveats, and uncertainty < ~95%.

## Skills Quick Reference

See [`.agents/skills/README.md`](../.agents/skills/README.md) for full matrix. Common patterns:

| Task | Recommended Skill |
| --- | --- |
| Multi-file navigation | `.agents/skills/context-map/` |
| Validation planning | `.agents/skills/quality-playbook/` |
| Benchmark changes | Full PR + `.agents/skills/review-benchmark-change/` |
| Docs sync on code change | `.agents/skills/update-docs-on-code-change/` |
| Measurement loop | `.agents/skills/autoresearch/` |
| Quick refinement | `.agents/skills/auto-improvement/` |
| Memory housekeeping | `.agents/skills/context-note-maintainer/` plus targeted memory edits |

## Troubleshooting & Help

- **Claude Code features**: `/help` or <https://github.com/anthropics/claude-code/issues>
- **Repo-specific blocks**: Check `AGENTS.md` and active issues in `docs/context/`
- **Skill questions**: See [`.agents/skills/README.md`](../.agents/skills/README.md) or use
  `skill-creator` to improve
- **Memory questions**: See [`memory/MEMORY.md`](../memory/MEMORY.md) and
  [`docs/context/INDEX.md`](../docs/context/INDEX.md)

---

**Last Updated**: 2026-06-19
**Maintainer**: See [`docs/maintainer_values.md`](../docs/maintainer_values.md)
**Changelog**: See [`CHANGELOG.md`](../CHANGELOG.md) for recent project updates
