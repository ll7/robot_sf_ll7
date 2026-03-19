# Planning Convention

Use this file when the work is large enough that an agent should externalize its plan before or
during implementation.

## When To Write A Plan

Write a plan when the task:

- spans multiple subsystems,
- changes benchmark semantics, training contracts, or planner provenance,
- adds documentation intended to guide future agent work,
- or is likely to require follow-up issues.

Skip formal planning only for narrow, obviously local edits.

## Plan Template

Keep plans short and operational:

```md
# Goal
- One or two sentences on the desired outcome.

# Boundaries
- What is in scope.
- What is explicitly out of scope.

# Evidence
- Files, issues, docs, configs, or upstream sources that define the contract.

# Steps
- Ordered implementation steps.

# Validation
- Commands to run.

# Risks / Follow-ups
- Remaining uncertainty, deferred scope, or issue candidates.
```

## Required Behaviors

- Restate the issue or user goal in repository terms, not generic assistant language.
- Separate observed evidence from assumptions.
- Prefer canonical scripts and committed configs over ad-hoc commands.
- When benchmark or planner claims are involved, include the exact docs/configs that anchor the claim.
- If scope expands, create a follow-up issue instead of silently broadening the implementation.

## Review Expectations

A good plan makes it easy for a reviewer to answer:

- what changed,
- why that scope is correct,
- how it was validated,
- and what risk remains.
