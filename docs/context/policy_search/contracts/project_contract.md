# Project Contract

## Goal

Create a reproducible local-policy search workflow that improves the current
ORCA versus PPO benchmark tradeoff without relying on long local training runs.

## In Scope

- candidate registry and stage-gated evaluation funnel,
- reusable local runners and reporting tools,
- non-training planner candidates that can be evaluated immediately,
- local smoke and narrow validation only,
- markdown reasoning, validation, and report artifacts under this directory.

## Out Of Scope

- long horizon training or evaluation on the laptop,
- paper-grade claims from smoke or stress-only evidence,
- hidden results that are not captured in markdown or structured JSON.

## Canonical Inputs

- `docs/context/policy_search/2026-04-29_policy_search.md`
- `docs/context/policy_search/2026-04-29_broad_policy_search.md`
- `configs/policy_search/funnel.yaml`
- `docs/context/policy_search/candidate_registry.yaml`
- `configs/policy_search/promotion_gates.yaml`

## Execution Rule

Promote only from consistent evidence. A candidate must pass nominal sanity
before any stress or full-matrix comparison is considered meaningful.