# Issue #7030: Agent Figure Interpretation Evaluation

This fixture-only evaluator measures whether a workflow preserves quantitative figure and interpretation details; it does not make a scientific or benchmark claim.

## Reproduce the current diagnostic

From the repository root:

```bash
uv run pytest -q tests/benchmark/test_agent_figure_interpretation_eval.py
uv run python scripts/analysis/run_agent_figure_interpretation_eval.py \
  --manifest tests/fixtures/agent_figure_interpretation_eval/v1/manifest.json \
  --pretty
```

The manifest pins every packet, source fixture, and reference fixture by SHA-256. The evaluator
fails closed on digest, identity, path, schema, and claim-boundary drift. Its output is explicitly
`evaluation_artifacts_only`; generated output must not be treated as benchmark, paper, or
dissertation evidence.

## Report contract

The report retains case-level scores and critical-error flags, then adds an aggregate summary with:

- pass/fail counts for each scoring dimension;
- exact packet identifiers for every critical failure kind;
- reviewer coverage, disagreement, and adjudication status;
- paired baseline versus packet-constrained workflow status when both variants are present.

Missing reviewer records or workflow variants are reported as `not_available` or `partial`. The
summary never infers an improvement or authorizes workflow adoption. Independent reference review,
adjudication, and any external model execution remain separate follow-up work under #7030.

## Current local baseline

On the 2026-08-17 current-main fixture corpus, the focused suite passed 50 tests and the replay
contained eight cases: one clean case and seven deliberate failure cases. All seven existing
critical mutation classes were detected. Reviewer accounting and paired workflow comparison were
`not_available` because the committed corpus contains no independently reviewed workflow runs.

This is diagnostic implementation evidence only. It does not establish model quality, visualization
quality on unseen inputs, a packet-constrained improvement, or any result-admission decision.
