# Repository Memory Index

This directory is the repo-local memory layer for stable cross-session agent context in
`robot_sf_ll7`.

It mirrors Claude Code's auto-memory shape on purpose: keep this `MEMORY.md` file short and use it
as the startup index, then put details in linked topic files that agents read on demand.

## Retrieval Order

1. Read this file first.
2. Open only the linked topic files relevant to the current task.
3. Prefer `architecture/` and `decisions/` before making workflow or design changes.
4. Consult `failures/`, `benchmarks/`, and `experiments/` when the task depends on prior outcomes
   or known pitfalls.

## Write Rules

- Use `memory/` for reusable knowledge that should survive across sessions.
- Use `docs/context/` for issue execution history, validation notes, and handoff details that are
  specific to one change or investigation.
- Keep this file under Claude-style startup limits in practice: concise, under roughly 200 lines,
  and free of duplicated detail.
- Prefer updating an existing memory note over creating a near-duplicate.
- When a memory note becomes stale, update it or remove its index entry instead of stacking
  contradictory summaries.

## Directory Map

### `architecture/`

- [Layered Agent Memory Architecture](architecture/layered_agent_memory_architecture.md)
  Layer boundaries, `CLAUDE.md` import path, optional MCP exposure, and the retrieval-deferral
  boundary.

### `decisions/`

- [Decision: Adopt A Repo-Local Markdown Memory Layer](decisions/2026-04-13_repo_local_memory_layer.md)
  Adopt a Markdown-first repo-local memory tree and keep retrieval/database infrastructure out of
  scope.
- [Decision: Issue 791 narrow benchmark-set claim](decisions/2026-04-20_issue_791_narrow_benchmark_claim.md)
  Paper framing is "strong policy on a broad scenario matrix", not OOD generalization; avoid
  transfer/unseen language and skip the held-out suite.

### `experiments/`

- [Experiment: Agent Memory Round-Trip Dry Run](experiments/2026-04-13_agent_memory_roundtrip_dry_run.md)
  Example session log showing how a new memory entry should be indexed and replayed in a new
  session.
- [Issue 791: distribution alignment dominates PPO lift](experiments/2026-04-20_issue_791_distribution_alignment_dominates.md)
  Wave-5/6 attribution: eval-aligned training explains ~97% of the 0.586→0.929 PPO lift;
  curriculum/capacity/foresight add only a small foresight-gated residual.

### `failures/`

- [Failure Patterns: Memory Misuse And Staleness](failures/2026-04-13_memory_misuse_and_staleness.md)
  Known failure modes for this memory layer: stale summaries, duplicated notes, and mixing issue
  logs into stable memory.
- [Camera-ready benchmark halts on ORCA when rvo2 missing](failures/2026-04-20_benchmark_orca_rvo2_missing.md)
  `uv sync --extra orca` is required before submitting the camera-ready benchmark; fallback
  policy does not cover the rvo2 import check.
- [WandB rejects runs when job_type exceeds 64 characters](failures/2026-04-20_wandb_job_type_64_char_limit.md)
  Keep `tracking.wandb.job_type` ≤ 64 chars; long labels belong in tags instead.

### `benchmarks/`

- [Benchmark Memory Scope](benchmarks/2026-04-13_benchmark_memory_scope.md)
  What benchmark-facing evidence belongs in memory, what must stay in `docs/context/`, and why
  fallback/degraded outcomes cannot be stored as success claims.

## Maintenance

- `CLAUDE.md` imports this file so Claude Code can see the index at session start.
- Other agents should treat this directory as optional but preferred startup context for durable
  repo facts.
- If the tree grows large, keep this index concise and move detail into additional topic files.
