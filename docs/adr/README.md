# Architecture Decision Records (ADR)

This directory holds the lightweight record process for major design decisions that are
intended to outlive a single issue or PR.

## Why ADRs here

- Capture long-lived architecture, contract, or dependency choices with clear rationale.
- Provide a durable pointer before behavior changes become difficult to reverse.
- Separate durable decisions from issue-level execution notes and ad-hoc run notes.

## ADR scope

Create an ADR for changes that affect:

- public interfaces, contracts, or semantics that can invalidate downstream integrations
- repository-wide simulation/benchmark architecture
- dependency or runtime strategy with broad impact
- evidence, provenance, or promotion workflow boundaries with non-trivial external meaning

Prefer a context note for:

- short-lived implementation logs
- benchmark executions, reruns, and diagnostics
- one-off bugfixes or local tuning decisions

## How this differs from other notes

- `docs/context/` notes are issue execution, validation, and handoff surfaces. They may
  explain tradeoffs, but they are not the canonical home for a durable architecture decision
  unless the decision remains issue-scoped.
- `memory/` entries are compact cross-session facts and reusable knowledge for future agents.
- ADRs in this directory are for durable architectural and process decisions; each ADR
  should be short, source-backed, and cross-referenced from issues, PRs, or related notes.

## Current index

- [ADR Template](./template.md)
- No durable ADRs have been recorded yet.

## Creating a new ADR

- Copy `template.md` and rename with a short, numeric filename (for example:
  `0001-simulator-backend.md`).
- Keep one decision per file.
- Include at least:
  - context and problem statement
  - decision rationale
  - alternatives considered
  - concrete impacts
  - what evidence or repository changes would invalidate the decision
  - references to source docs, tests, PRs, or specs
