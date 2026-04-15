# Coding-Agents Compatibility Note

**Issue**: [#728](https://github.com/ll7/robot_sf_ll7/issues/728)
**Date**: 2026-04-12
**Status**: Adopted

## Purpose

This note records which concepts from the external `vultuk/coding-agents` reference repository Robot
SF adopts, why each decision was made, and which ideas are explicitly rejected. It serves as the
single canonical source for cross-agent portability decisions in this repository.

A contributor or AI assistant can read this note plus the linked entry points (`AGENTS.md`,
`.github/copilot-instructions.md`, `docs/dev_guide.md`) to understand the full cross-agent
compatibility stance without having to reconstruct it from issue comments or PR text.

## Concepts Adopted

### 1. Retrieval → Planning → Execution → Verification discipline

`vultuk/coding-agents` makes explicit a four-phase agent loop that is easy to skip under time
pressure. Robot SF maps each phase to concrete, repo-local skills so the discipline is enforced
through tooling rather than informal convention:

| Phase | Robot SF mapping |
|---|---|
| **Retrieval** | `context-map`, `what-context-needed`, `benchmark-overview`, `experiment-context` |
| **Planning** | `quality-playbook`, `gh-issue-clarifier`, `.agent/PLANS.md` convention |
| **Execution** | `gh-issue-autopilot`, `autoresearch`, `auto-improvement` |
| **Verification** | `implementation-verification`, `pr-ready-check`, `review-benchmark-change` |

No phase is optional for non-trivial work. Skipping retrieval produces misscoped implementations;
skipping verification produces unproven claims. The `quality-playbook` skill enforces this sequence
end-to-end for high-risk changes.

### 2. Per-source compatibility notes

For each significant external source of workflow inspiration (Awesome Copilot, coding-agents, etc.),
a dedicated note in `docs/context/` captures the accept/reject decisions. This prevents drift where
agent instructions absorb upstream phrasing without explicit rationale.

- `docs/ai/awesome_copilot_adaptation.md` covers Awesome Copilot List adaptation decisions.
- This note (`docs/context/issue_728_coding_agents_compatibility.md`) covers `vultuk/coding-agents`.

### 3. Separation of prompts / skills / agents in a canonical directory

`vultuk/coding-agents` separates these three artifact types cleanly so tool-specific compatibility
paths can be derived from a single canonical source. Robot SF already adopted this via issue `#705`:
`.agents/` is the canonical source tree, and symlinks project it into `.codex/`, `.opencode/`,
`.claude/`, `.github/`, and `.gemini/` compatibility paths.

Validation and repair:

```bash
uv run python scripts/tools/sync_ai_config.py --check   # detect stale mirrors
uv run python scripts/tools/sync_ai_config.py --fix     # repair symlinks
```

### 4. Explicit entry-point linking

The compatibility note is only useful if agents can discover it. `vultuk/coding-agents` emphasizes
surfacing canonical docs from every agent entry point. For this note, the required links are:

- `AGENTS.md` — top-level execution rules entry point
- `.github/copilot-instructions.md` — Copilot-facing pointer
- `docs/dev_guide.md` — contributor workflow reference

## Concepts Rejected

### Wholesale migration to the `coding-agents` directory layout

Robot SF has an established `.agents/` canonical structure with functioning symlinks. Migrating to a
different layout would invalidate existing compatibility paths and produce churn without net gain.

### Generator / compiler framework for syncing

`vultuk/coding-agents` uses an external generator or build step to produce tool-specific files.
Robot SF uses lightweight symlinks managed by `scripts/tools/sync_ai_config.py`. The symlink
approach has fewer moving parts and does not introduce an external dependency.

### Cross-agent runtime assumptions

Robot SF does not assume any agent runtime beyond what `AGENTS.md` and tool-specific thin-pointer
files convey. Concepts that require a specific runtime (prompt injection hooks, context-window
managers, session memory) are out of scope for the compatibility layer.

### Verbatim upstream text

Where an upstream concept is worth adopting, the preferred form is a concise repo-native restatement
rather than a verbatim copy. Verbatim copies create maintenance debt and obscure where the original
concept came from.

## Maintenance Rule

When a new external agent-workflow source is evaluated for adoption, create a parallel note in
`docs/context/` before changing any instruction file. Keep one note per source. Link the new note
from `AGENTS.md`, `.github/copilot-instructions.md`, and `docs/dev_guide.md`. Update this note if
the accept/reject decisions change.

## Related

- Issue `#705` — Unify AI assistant rules, prompts, and sync workflow (canonical `.agents/` structure)
- Issue `#713` — Batch-first issue workflow note (`docs/context/issue_713_batch_first_issue_workflow.md`)
- `docs/ai/awesome_copilot_adaptation.md` — prior per-source adaptation note
- `AGENTS.md` — top-level execution rules
- `.agents/README.md` — canonical surface registry and maintenance commands
