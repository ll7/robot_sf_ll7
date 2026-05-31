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

`AGENTS.md` remains the top-level instruction source for repository rules, and
`docs/maintainer_values.md` is the compact source for current values and hard contracts.
Tool-specific instruction files, such as `.github/copilot-instructions.md` and `.cursorrules`,
should be thin pointers to those sources plus only the tool-specific details that cannot live there.

Stale compatibility surfaces should be removed when they no longer provide value. Claude cleanup is
tracked in issue #1728.

## Maintenance

Run the drift check after changing AI assistant surfaces:

```bash
uv run python scripts/tools/sync_ai_config.py --check
```

If a supported compatibility symlink is missing or stale, repair it with:

```bash
uv run python scripts/tools/sync_ai_config.py --fix
```
