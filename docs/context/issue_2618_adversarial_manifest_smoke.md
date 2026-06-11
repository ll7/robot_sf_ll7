# Issue #2618 Adversarial Manifest Planner Smoke 2026-06-11

Issue: [#2618](https://github.com/ll7/robot_sf_ll7/issues/2618)
Status: diagnostic result; not benchmark evidence.

## Claim Boundary

This note records the first Issue #2618 smoke using generated
`adversarial_scenario_manifest.v1` candidates against one planner pair. It does not claim
adversarial coverage, planner ranking, leaderboard movement, or paper-facing benchmark evidence.
The `social_force` row is adapter-mode and remains a diagnostic stress signal only.

Tracked compact evidence:
[summary.json](evidence/issue_2618_adversarial_manifest_smoke/summary.json).
Raw generated manifests, route overrides, materialized matrix files, episode JSONL rows, and local
smoke summaries remain ignored under `output/`.

## Required Output

```yaml
adversarial_manifest_smoke:
  generator_version: RandomCandidateSampler
  seed_family: crossing_ttc_space.yaml:random_seed_43
  generated_count: 4
  valid_count: 4
  invalid_count: 0
  degenerate_count: 0
  planner_pair: [goal, social_force]
  executed_count: 2
  collision_signal: true
  near_miss_signal: unavailable
  low_progress_signal: true
  novelty_or_perturbation_distance:
    unique_hash_count: 4
    duplicate_hash_count: 0
    novelty_rate: 1.0
    perturbation_reference: null
  claim_boundary: diagnostic_only
```

Decision outcome: `generator_produces_useful_valid_cases`.

Follow-up recommendation: `continue`. The generated cases are valid and non-degenerate, and the
two executed cases expose collision and low-progress signal. A larger campaign should still add
certification, explicit near-miss metrics, and native/comparable planner rows before making
benchmark-strength claims.

## Manifest Quality

| Metric | Value |
| --- | ---: |
| Generated manifests | 4 |
| Valid manifests | 4 |
| Invalid manifests | 0 |
| Degenerate manifests | 0 |
| Parse failures | 0 |
| Unique normalized hashes | 4 |
| Duplicate normalized hashes | 0 |
| Novelty rate | 1.0 |

No perturbation reference manifest was supplied, so perturbation distance is not available for this
smoke. Novelty is measured only by normalized control hashes within the generated batch.

## Planner Smoke

| Planner | Mode | Episodes | Success sum | Collision sum | Time-to-goal norm mean | Signal |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `goal` | native | 2 | 1.0 | 1.0 | 0.8333 | mixed: one success and one collision |
| `social_force` | adapter | 2 | 0.0 | 2.0 | 1.0 | collision and low-progress diagnostic signal |

`near_misses` was not present in the aggregate smoke summary, so the near-miss signal is
`unavailable` rather than negative. Both planners completed in available execution modes, but only
`goal` is native; `social_force` is adapter-mode and should not be compared as native benchmark
evidence.

## Validation

Smoke command:

```bash
TF_CPP_MIN_LOG_LEVEL=2 LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/tools/run_adversarial_manifest_smoke.py \
  --search-space configs/adversarial/crossing_ttc_space.yaml \
  --scenario-template configs/scenarios/templates/crossing_ttc.yaml \
  --count 4 --seed 43 --max-valid 2 \
  --output-dir output/adversarial/issue2618_manifest_smoke \
  --summary-json output/adversarial/issue2618_manifest_smoke/summary.json \
  --planner goal --planner social_force \
  --horizon 60 --dt 0.1 --workers 1
```

Quality summary command:

```bash
uv run python scripts/tools/summarize_adversarial_manifest_quality.py \
  output/adversarial/issue2618_manifest_smoke/manifests \
  --smoke-summary-json output/adversarial/issue2618_manifest_smoke/summary.json \
  --output-json output/adversarial/issue2618_manifest_smoke/quality_summary.json
```

Both commands completed successfully. The smoke emitted the existing SVG invalid-polygon warning for
`uni_campus_big.svg` and force-metric warnings from missing pedestrian force data; neither warning
blocked the smoke or the compact metric summary.
