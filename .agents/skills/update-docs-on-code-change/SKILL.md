---
name: update-docs-on-code-change
description: Keep docs aligned with code changes that affect workflows, contracts, or user-facing behavior.
category: context-docs
kind: atomic
phase: context
requires_write: true
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# Update Docs On Code Change

## Purpose

Keep repository docs accurate when code, commands, or contracts change so docs stale against the
implementation are repaired. Focus on minimal, discoverable workflow docs updates, not broad
documentation rewrites.

## Workflow

1. Identify affected docs from the change surface (workflows, commands, contracts).
2. Apply only minimal corrections in existing docs.
3. Keep terminology aligned with existing repo naming.
4. Validate references and examples point to existing files/commands.
5. For discoverability-impacting updates, add or update relevant docs tests/checks if present.

## Guardrails

- Prefer updating existing docs over creating new documents.
- Do not document behavior that no longer exists.
- For user-facing behavior changes, update docs in the same change set.

## Output

- Updated docs list and specific sections changed.
- Reference validation result.
- Remaining documentation risks not yet covered.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
