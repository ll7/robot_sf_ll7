# Agent Configuration

This directory is the canonical repository-owned source tree for AI assistant workflow content.
Tool-specific directories should point here when their formats allow it.

## Canonical Surfaces

| Canonical path | Compatibility path | Purpose |
| --- | --- | --- |
| `.agents/skills/` | `.codex/skills/`, `.opencode/skills/` | Repo-local workflow skills. |
| `.agents/prompts/codex/` | `.codex/prompts/` | Codex prompt files. |
| `.agents/prompts/github/` | `.github/prompts/` | GitHub prompt wrappers. |
| `.agents/agents/github/` | `.github/agents/` | GitHub agent definitions. |
| `.agents/commands/gemini/` | `.gemini/commands/` | Gemini command definitions. |

`AGENTS.md` remains the top-level instruction source for repository rules and task-scoped context
entrypoints, and `docs/maintainer_values.md` is the compact source for current values and hard
contracts. Use `docs/ai/agent_workflow_entrypoints.md` for task route selection (read-only observation,
documentation edit, runtime change, scientific interpretation, or environment repair), correct `uv run`
command entrypoints, model registry lookup, shared routing handoff format, and targeted large-file
navigation. Branch synchronization is mode-specific: implementation branches merge `origin/main` early,
while read-only review worktrees never merge or push to implementation branches (enforced by
`scripts/dev/review_worktree_guard.py` / issue #8321).
Tool-specific instruction files, such as `.github/copilot-instructions.md`, `.claude/CLAUDE.md`,
and `.cursorrules`, are thin pointers to those sources plus only the tool-specific details that
cannot live there; `scripts/tools/sync_ai_config.py` enforces their line and section budgets.

## Shared model routing

The shared dual-tier resolver is the sole source of truth for delegated model and provider
selection: see the [`ai-delegation-routing` skill](https://github.com/ll7/codex-personal-skills/blob/main/skills/system/ai-delegation-routing/SKILL.md)
and the [shared route planner](https://github.com/ll7/codex-personal-skills/blob/main/scripts/resolve-route.py).
The accepted handoff input and the checkout-based dispatch command are documented with a complete
`handoff.v2` example in `docs/ai/agent_workflow_entrypoints.md` (search for "handoff.v2").
Route output never substitutes for repository-local artifact, diff, validation, benchmark,
evidence-admission, or paper-facing acceptance proof.

When canonical and compatibility surfaces disagree, follow the precedence rule in `AGENTS.md`.
Patch the canonical source first, then update generated or mirrored compatibility surfaces when a
sync command exists. If a broad mirror update would be risky, keep the canonical change bounded and
open a follow-up issue that names the affected compatibility entry points.

Stale compatibility surfaces should be removed when they no longer provide value. Claude cleanup is
tracked in issue #1728.

## Maintenance

Run the drift check after changing AI assistant surfaces:

```bash
uv run python scripts/tools/sync_ai_config.py --check
```

For skill edits, also run the relevant skill preflight when one exists:

```bash
uv run python scripts/dev/check_skills.py --preflight <skill-name>
```

If a supported compatibility symlink is missing or stale, repair it with:

```bash
uv run python scripts/tools/sync_ai_config.py --fix
```
