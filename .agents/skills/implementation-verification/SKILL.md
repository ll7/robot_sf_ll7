---
name: implementation-verification
description: Verify branch changes against origin/main with claim-based evidence, not only test status.
category: validation
kind: atomic
phase: verification
requires_write: false
requires_slurm: false
requires_benchmark_artifacts: false
delegates_to: []
output_schema: skill_run_summary.v1
---

# Implementation Verification

## Purpose

Prove that branch changes implement the stated claims versus `origin/main`, with evidence per behavior
rather than only a global test pass signal.

## Workflow

1. Set baseline:
   - `git status --short --branch`
   - `git diff --stat origin/main...HEAD`
   - `git diff --name-only origin/main...HEAD`
2. Extract claims from issue/PR body, docs, and changed code/docs.
3. Map each claim to a concrete evidence surface:
   - scripts, configs, tests, CLI commands, or artifacts.
4. Validate claim by claim:
   - run minimal targeted checks first,
   - then run `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` when broader validation is required.
5. Include benchmark safety checks when relevant:
   - explicitly classify fallback/degraded execution as a limitation, never as success.
6. Record residual gaps where no direct proof path exists.

## Guardrails

- Do not treat a passing test suite as sufficient if it does not exercise the claim.
- Do not edit unrelated code while proving current claims.
- If `origin/main` is unavailable, switch to an explicit documented base and note why.
- Treat delegated worker artifacts as leads until locally checked. `status=0` only proves the wrapper
  exited, not that the model returned useful findings.
- For read-only delegates, require a structured final answer in `RESULT.md`; otherwise keep the
  worker output at low confidence and inspect local evidence directly.
- When worker metadata says `changed_files: []`, still inspect `status.txt` or `git status --porcelain`
  for untracked files before using the run as implementation evidence.

## Output

- Evidence matrix (`claim`, `proof`, `command/artifact`, `result`, `residual risk`).
- Items proven vs unresolved.
- Recommended follow-up tickets for unproven claims.
## When to use

Use this skill for the scope named in its frontmatter description and registry metadata.
