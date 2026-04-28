# Benchmark Memory Scope

Date: 2026-04-13

## What Belongs In Memory

- stable benchmark workflow rules
- recurring interpretation caveats
- known failure modes that matter across multiple benchmark tasks
- links to canonical proof notes or release docs

## What Should Stay Elsewhere

- full execution logs and issue-specific investigation details: `docs/context/`
- generated artifacts and raw outputs: `output/`
- transient benchmark experiments that are not yet stable enough to reuse as memory

## Fail-Closed Reminder

Memory notes must not present fallback, degraded, or partial-failure benchmark outcomes as success.
When a benchmark result matters, memory should summarize the reusable lesson and link to the
canonical evidence, such as `docs/context/issue_691_benchmark_fallback_policy.md` or a benchmark
issue note.
