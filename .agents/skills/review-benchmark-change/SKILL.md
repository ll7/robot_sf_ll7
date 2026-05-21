---
name: review-benchmark-change
description: "Review benchmark-sensitive code or docs changes for semantic regressions, normalization drift, reproducibility gaps, and provenance overclaim."
---

# Review Benchmark Change

## When to use

Use this skill when reviewing PRs, patches, or docs that can change benchmark semantics,
interpretation, or reproducibility.

## Read First

- `docs/code_review.md`
- `docs/benchmark_spec.md`
- `docs/dev/observation_contract.md`
- `docs/benchmark_planner_family_coverage.md`
- `docs/context/issue_691_benchmark_fallback_policy.md`

## Workflow

1. Confirm the claimed benchmark contract and changed contract surface.
2. Validate observation/normalization contract compatibility.
3. Check seed/scenario policy and artifact reproducibility entries.
4. Verify provenance and readiness labels match implementation status.
5. Assess whether changes are intentional or regressions and document severity.

## Review Checklist

- evaluation semantics unchanged or intentionally documented,
- observation normalization and dtype/bounds still match the stated contract,
- scenario distributions and seed policy remain explicit,
- reproducibility path still exists,
- upstream provenance and benchmark-readiness labels remain accurate.

## Proof and Guardrails

- Apply fail-closed logic if any changed path lacks reproducible evidence.
- Do not accept fallback-degraded execution as a benchmark success.
- Keep any claim in sync with `benchmark` contract source files.

## Output

- findings should prioritize:
  1. incorrect benchmark semantics,
  2. broken learned-policy contract or normalization,
  3. provenance overclaim,
  4. missing reproducibility hooks,
  5. missing tests/docs for changed public behavior.
