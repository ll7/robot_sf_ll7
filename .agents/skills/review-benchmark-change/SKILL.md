# review-benchmark-change

Use this skill when reviewing a PR, patch, or doc change that could alter benchmark outcomes or
their interpretation.

## Read First

- `code_review.md`
- `docs/benchmark_spec.md`
- `docs/dev/observation_contract.md`
- `docs/benchmark_planner_family_coverage.md`

## Review Checklist

- evaluation semantics unchanged or intentionally documented,
- observation normalization and dtype/bounds still match the stated contract,
- scenario distributions and seed policy remain explicit,
- reproducibility path still exists,
- upstream provenance and benchmark-readiness labels remain accurate.

## Output Expectations

Review findings should prioritize:

1. incorrect benchmark semantics,
2. broken learned-policy contract or normalization,
3. provenance overclaim,
4. missing reproducibility hooks,
5. missing tests/docs for changed public behavior.
