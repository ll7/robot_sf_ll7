# Retrieval And MCP Deferral Note

Date: 2026-03-19

## Decision

Do **not** add retrieval/database infrastructure yet.

Specifically deferred:

- `ast-grep MCP`
- `Qdrant MCP`
- `Chroma MCP`
- any persistent semantic-memory or vector-database layer

## Rationale

The repository already has a strong local-context baseline:

- `AGENTS.md`
- `.specify/memory/constitution.md`
- `docs/dev_guide.md`
- execution-oriented `.codex/skills/`
- the new repo-local context stack added in this issue

That is enough to improve onboarding and long-horizon work without adding infrastructure that would
need operational ownership, update discipline, and benchmark-safety review.

## Trigger For Reconsideration

Revisit retrieval only if a real boundary appears, such as:

- repeated failure to fit the required subsystem context into normal agent workflows,
- repeated planner-zoo or benchmark tasks that require high-friction structural lookup,
- evidence that manual context packs are too costly or too stale.

Until then, prefer:

1. repo-local instructions,
2. focused docs under `docs/ai/`,
3. `.agents/skills/` context packs,
4. manual or scripted context bundles via the chosen packer.

## What This Prevents

Deferring retrieval now avoids:

- premature infrastructure ownership,
- stale semantic-memory layers with unclear trust boundaries,
- overcomplicating a repository whose main need is clearer local guidance, not another service.
