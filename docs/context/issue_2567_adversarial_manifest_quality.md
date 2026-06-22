# Issue #2567 Adversarial Manifest Quality Metrics 2026-06-07

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2567>

Status: current workflow/quality-metric smoke evidence. This note does not claim adversarial
coverage, planner weakness, leaderboard movement, or paper-facing benchmark evidence.

## Result

Issue #2567 adds a compact `adversarial_manifest_quality_summary.v1` path for
`adversarial_scenario_manifest.v1` batches. The summary is a generator-quality and pre-benchmark
gate: it can show whether generated manifests are valid, non-degenerate, novel, meaningfully
perturbed from a reference manifest, whether generated candidates pass the additive
`naturalistic_vru_prior.v1` plausibility screen, and whether an optional planner smoke summary
contains aggregate or per-episode failure/near-miss yield signals.

## What Was Built

- `robot_sf/adversarial/manifest_quality.py`
  - Loads manifest YAML files or directories.
  - Computes validity, invalid, and degeneracy rates from manifest validation status.
  - Computes novelty and duplicate rates from normalized control hashes.
  - Computes perturbation distance from an optional reference manifest using continuous controls
    only; `scenario_seed` remains part of duplicate hashing but is not treated as a geometric
    perturbation dimension.
  - Summarizes additive naturalistic-prior metadata as available/pass/fail/missing counts and
    violation counts. Legacy manifests without prior metadata are `missing`, not failed.
  - Supports `--naturalistic-status passed|violated|missing|all` filtering so plausible hard cases
    and intentionally unrealistic stress cases can be inspected separately.
  - Reads optional planner-smoke summaries. When episode JSONL rows are available, it computes
    failure and near-miss yields from rows. When only aggregate metrics are durable, it derives
    failure yield from aggregate success counts and leaves near-miss yield unavailable unless
    aggregate near-miss counts exist.
- `scripts/tools/summarize_adversarial_manifest_quality.py`
  - CLI wrapper around the metric path with manifest inputs, optional `--reference-manifest`,
    optional `--smoke-summary-json`, and optional `--output-json`.
- `tests/adversarial/test_adversarial_manifest_quality.py`
  - Focused fixtures for validity/novelty, perturbation distance, episode-row yields,
    aggregate-smoke yields, and CLI JSON output.

## Smoke Evidence

Durable compact evidence:
[evidence/issue_2567_adversarial_manifest_quality/summary.json](evidence/issue_2567_adversarial_manifest_quality/summary.json)

The smoke regenerated four bounded manifests from the #2524 generator, used candidate 0 as a
reference, summarized candidates 1-3, and read the tracked #2562 smoke summary for aggregate planner
yield. Observed quality signals:

- manifest count: 3 summarized candidates, all `valid`;
- validity rate: `1.0`;
- invalid rate: `0.0`;
- degeneracy rate: `0.0`;
- novelty rate: `1.0`;
- duplicate rate: `0.0`;
- perturbation distance from reference: min `0.585564`, mean `1.499197`, max `2.060612`;
- aggregate planner yields from #2562 smoke:
  - `goal`: failure yield `0.0` from 2 episodes;
  - `social_force`: failure yield `1.0` from 2 adapter-mode episodes;
  - near-miss yield unavailable because the durable aggregate summary has no near-miss metric.

Raw generated manifests and the CLI output under `output/adversarial/issue2567_quality_cli_smoke/`
remain disposable worktree-local artifacts.

## Claim Boundary

These metrics are quality and gating signals for generated candidate batches. They do not prove
adversarial coverage, planner weakness, benchmark ranking, or paper-facing results. The #2562
`social_force` signal is adapter-mode diagnostic evidence only. Any #2568 RL/diffusion expansion
should use these metrics to reject invalid, duplicate, degenerate, or low-yield batches before
larger execution. Naturalistic-prior pass/fail is an authored local plausibility screen, not a
real-world calibration claim; prior-fail candidates should be reported as stress-only or
intentionally unrealistic rather than mixed into plausible-hard-case summaries.

## Validation

Focused validation:

```bash
uv run pytest tests/adversarial/test_adversarial_manifest_quality.py \
  tests/adversarial/test_adversarial_scenario_manifest.py -q
uv run ruff check robot_sf/adversarial/manifest_quality.py \
  robot_sf/adversarial/__init__.py \
  scripts/tools/summarize_adversarial_manifest_quality.py \
  tests/adversarial/test_adversarial_manifest_quality.py
uv run ruff format --check robot_sf/adversarial/manifest_quality.py \
  robot_sf/adversarial/__init__.py \
  scripts/tools/summarize_adversarial_manifest_quality.py \
  tests/adversarial/test_adversarial_manifest_quality.py
uv run python scripts/tools/generate_adversarial_scenario_manifests.py \
  --search-space configs/adversarial/crossing_ttc_space.yaml \
  --scenario-template configs/scenarios/templates/crossing_ttc.yaml \
  --count 4 --seed 42 \
  --output-dir output/adversarial/issue2567_quality_cli_smoke/manifests
uv run python scripts/tools/summarize_adversarial_manifest_quality.py \
  output/adversarial/issue2567_quality_cli_smoke/manifests \
  --reference-manifest output/adversarial/issue2567_quality_cli_smoke/manifests/candidate_0000.yaml \
  --smoke-summary-json docs/context/evidence/issue_2562_adversarial_manifest_smoke/summary.json \
  --output-json output/adversarial/issue2567_quality_cli_smoke/quality_summary.json
```

Observed result: manifest-quality and existing manifest tests passed with `51 passed`; Ruff check
and format check passed; the CLI smoke produced `adversarial_manifest_quality_summary.v1` with
aggregate planner yields from the tracked #2562 summary.
