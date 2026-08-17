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
Use `docs/ai/agent_workflow_entrypoints.md` for correct `uv run` command entrypoints,
model registry lookup, and targeted large-file navigation.
Tool-specific instruction files, such as `.github/copilot-instructions.md` and `.cursorrules`,
should be thin pointers to those sources plus only the tool-specific details that cannot live there.

## Shared model routing

The shared dual-tier resolver is the sole source of truth for delegated model and provider
selection: see the [`ai-delegation-routing` skill](https://github.com/ll7/codex-personal-skills/blob/main/skills/system/ai-delegation-routing/SKILL.md)
and the [shared route planner](https://github.com/ll7/codex-personal-skills/blob/main/scripts/resolve-route.py).
For a production `--out` plan, pass the explicit identity/risk/head contract
`--task-id`, `--task-class`, `--risk`, `--handoff-file`, `--frozen-head`, `--target-repo`, and
`--out`. Keep ownership and validation in the handoff file; do not duplicate them with
`--owned-paths`, `--validation`, or `--prompt` flags.

The accepted handoff input is a flat `handoff.v2` request (there is no nested `packet`):

<!-- handoff.v2-example:start -->

```yaml
schema_version: handoff.v2
handoff_type: request
task_id: ROBOTSF-EXAMPLE
provider: opencode_go
mode: issue_implementation
goal: Implement the bounded Robot SF packet and return frozen-head evidence.
owned_paths:
  - .agents/README.md
forbidden_actions:
  - push
  - open_pr
  - mutate_remote
required_context:
  - target repository frozen HEAD
  - accepted route-plan contract
required_output:
  - changed_files
  - validation_evidence
  - final_status
acceptance_gate:
  - all declared validation commands pass
  - changed files stay within owned_paths
validation_commands:
  - uv run pytest -q tests/dev/test_check_skills.py
execution_mode: external_runtime
dependencies: []
budget:
  runtime_minutes: 30
stop_conditions:
  - scope expands beyond owned_paths
  - a forbidden action is requested
side_effect_policy:
  remote_mutation: false
  local_edits: true
max_depth: 0
sync_barrier: null
```

<!-- handoff.v2-example:end -->

The equivalent checkout-based command below can be run from this repository without assuming
that the shared checkout is the current directory:

```bash
TARGET_REPO="$(pwd)"
TARGET_HEAD="$(git -C "$TARGET_REPO" rev-parse HEAD)"
ROUTING_REPO="${CODEX_ROUTING_REPO:?set CODEX_ROUTING_REPO to a codex-personal-skills checkout}"
HANDOFF_FILE="${HANDOFF_FILE:?set HANDOFF_FILE to the flat handoff.v2 YAML above}"
python3 "$ROUTING_REPO/scripts/resolve-route.py" \
  --task-id ROBOTSF-EXAMPLE --task-class issue_implementation --risk R1 \
  --handoff-file "$HANDOFF_FILE" \
  --frozen-head "$TARGET_HEAD" --target-repo "$TARGET_REPO" \
  --out "${TMPDIR:-/tmp}/robotsf-route-plan.json"
```

Read back the private plan's `selected_route`, `forbidden_actions`, and `acceptance_gate` before
dispatch. The resolver owns native-tier selection, evidence-gated escalation, and external-provider
budget alternatives. Do not copy a volatile model inventory or add local legacy routes here. For
[Robot SF](../docs/glossary.md), route output never substitutes for repository-local artifact,
diff, validation, benchmark, evidence-admission, or paper-facing acceptance proof.

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
