---
name: implementation-verification
description: "Verify current branch implementations against origin/main by mapping each claimed feature to code, docs, tests, and executable proof; use before PR handoff or after substantial changes when passing tests alone is not enough."
---

# Implementation Verification

## Overview

Use this skill when the task is to prove that the current branch actually implements the intended
features compared to `origin/main`, not merely that the repository test suite passes.

Good triggers:
- "verify this branch against main"
- "check whether every feature works as designed"
- "audit implementation proof before PR"
- "compare current implementation to issue/PR claims"

This skill complements `pr-ready-check`. Run readiness gates, but also inspect whether each changed
behavior has direct evidence.

## Read First

- `AGENTS.md`
- `docs/code_review.md`
- `docs/dev_guide.md`
- `.specify/memory/constitution.md`
- `.agent/PLANS.md` when the branch changes benchmark, training, or public workflow semantics

## Workflow

1. Establish the comparison base
   - Default to `origin/main`.
   - Run `git status --short --branch`.
   - Inspect `git diff --stat origin/main...HEAD` and `git diff --name-only origin/main...HEAD`.

2. Extract implementation claims
   - Read the issue, PR body, changelog entries, new docs, and test names.
   - Convert claims into a short checklist of expected features or behavior changes.
   - Separate intended behavior from incidental refactors.

3. Map claims to evidence
   - For each claim, identify the exact code path, docs/config surface, and tests or commands that
     should prove it.
   - Prefer repo-native entry points under `scripts/dev/`, committed configs, and targeted tests.
   - If a feature has no direct proof path, mark it as a gap before running broad tests.

4. Verify behavior
   - Run the smallest targeted check for each feature first.
   - Then run the appropriate broader gate, usually:
     `BASE_REF=origin/main scripts/dev/pr_ready_check.sh`
   - For benchmark/planner changes, require benchmark-safe evidence and do not count fallback or
     degraded execution as success unless that fallback is the explicit subject.

5. Report an evidence matrix
   - Include one row per claimed feature:
     `Claim | Evidence surface | Validation command/artifact | Result | Residual risk`
   - Distinguish observed evidence from inference.
   - State any features that were changed but not proven, and create or recommend follow-up issues
     when proof is out of scope.

Example:

| Claim | Evidence surface | Validation command/artifact | Result | Residual risk |
| --- | --- | --- | --- | --- |
| Legacy entrypoint fails closed | `scripts/training_ppo.py`, contract test | `uv run pytest tests/training/test_train_expert_ppo_contract.py` | Observed pass | External docs may still mention old command |

## Guardrails

- Do not treat "tests pass" as sufficient unless the tests directly exercise the claimed behavior.
- Do not broaden implementation scope while verifying; open follow-up issues for unrelated gaps.
- Do not revert or rewrite user changes discovered during the diff review.
- If `origin/main` is stale or unavailable, state the alternate base and why it was used.
