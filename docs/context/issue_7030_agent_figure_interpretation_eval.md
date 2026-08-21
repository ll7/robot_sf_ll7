# Issue #7030: Agent Figure Interpretation Evaluation

This fixture-only evaluator measures whether a workflow preserves quantitative figure and interpretation details; it does not make a scientific or benchmark claim.

## Reproduce the current diagnostic

From the repository root:

```bash
uv run pytest -q tests/benchmark/test_agent_figure_interpretation_eval.py
uv run python scripts/analysis/run_agent_figure_interpretation_eval.py \
  --manifest tests/fixtures/result_interpretation_packet/agent_figure_interpretation_eval_manifest.json \
  --pretty
```

The bounded replay harness is provider-free and uses the same evaluator owner:

```bash
uv run python scripts/analysis/run_agent_figure_interpretation_eval.py \
  --manifest tests/fixtures/result_interpretation_packet/agent_figure_interpretation_eval_manifest.json --list
```

`--list` reports the verified source fixture IDs, mutation IDs, and the
expected deterministic scientific-error detector for each mutation. A
single candidate envelope is replayed with `--candidate`; `--fixture-id` and
`--mutation-id` may be supplied as an additional identity check:

```bash
uv run python scripts/analysis/run_agent_figure_interpretation_eval.py \
  --manifest tests/fixtures/result_interpretation_packet/agent_figure_interpretation_eval_manifest.json \
  --candidate candidate.json \
  --fixture-id ch7_visualization_causal_abstention_fixture \
  --mutation-id causal_overclaim
```

The candidate envelope is versioned and must contain workflow identity and
revision, figure specification and caption, interpretation, limitations,
confidence, unresolved questions, mutation identity plus expected detector,
per-dimension findings, explicit unavailable/not-applicable lists, and
manifest-bound source/packet/reference/candidate/figure/caption/review digests.
Unavailable or not-applicable evidence uses an explicit status and null
digest. It must not include a reference answer. Dimension findings may use
`requires_semantic_review` when deterministic replay cannot establish a
meaning-dependent judgment; this status is not a pass or benchmark result.
A replay-all candidate file is a JSON array of these envelopes:

```bash
uv run python scripts/analysis/run_agent_figure_interpretation_eval.py \
  --manifest tests/fixtures/result_interpretation_packet/agent_figure_interpretation_eval_manifest.json \
  --candidate candidates.json --replay-all
```

Replay-all fails closed unless it covers every verified fixture/mutation pair
exactly once. `--list` reports deterministic operators for every required
critical-error mutation, including analysis-unit mismatch, reversed effect
direction/desirability, native and adapter row merging without disclosure, and
inconsistent multiplicity language. `digest_omission` and `stale_post_review_bytes` are explicit
manifest-validation mutations; they fail closed before scoring. Exit code 1
means the candidate did not reproduce the expected detector; exit code 2 means
the manifest, envelope, identity, digest, or claim-boundary contract failed.
These commands remain diagnostic fixture replays only and do not call
providers or create benchmark evidence.

Replay reports bind `code_sha256` to the evaluator module, `config_sha256` to
the digest-pinned manifest configuration, and `fixture_sha256` to the ordered
manifest fixture records. A candidate `review_sha256`, when available, binds
the deterministic review fields and fails closed if post-review bytes drift.

The manifest pins one existing canonical packet and its source-binding digest by SHA-256. The
evaluator calls `load_result_interpretation_packet` before creating ephemeral mutation projections;
it carries no second packet or reference-fixture registry. It fails closed on digest, identity,
path, schema, and claim-boundary drift. Its output is explicitly
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

On the current-main fixture corpus, the focused suite passes 82 tests and the replay inventory
contains one canonical source-backed packet plus eleven deterministic mutation cases: one clean
case and eleven deliberate failure cases. All eleven critical mutation classes are detected.
Reviewer accounting and paired workflow comparison are `not_available` because the committed
corpus contains no independently reviewed workflow runs.

This is diagnostic implementation evidence only. It does not establish model quality, visualization
quality on unseen inputs, a packet-constrained improvement, or any result-admission decision.
