# Trained SAC Autoresearch Follow-Up

Date: 2026-05-01

## Contract

Goal: test whether a locally trained learned policy can beat the best
deterministic policy-search incumbent on success rate.

Incumbent to beat:

- `hybrid_rule_v3_fast_progress`
- nominal_sanity success: `0.3333`
- nominal_sanity collision: `0.0000`
- stress_slice success: `0.2917`
- stress_slice collision: `0.0000`

Training command launched in tmux:

```bash
uv run python scripts/training/train_sac_sb3.py \
  --config configs/training/sac/autoresearch_nominal_sanity_high_success.yaml \
  --log-level INFO
```

Training surface: `configs/policy_search/nominal_sanity_matrix.yaml`.

Evaluation command:

```bash
LOGURU_LEVEL=INFO uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate sac_autoresearch_nominal_sanity_v1 \
  --stage nominal_sanity \
  --output-dir output/ai/autoresearch/trained_high_success/manual_nominal_sanity \
  --workers 1
```

## Result

The trained SAC policy did not beat the incumbent.

| Candidate | Stage | Success | Collision | Decision |
|---|---|---:|---:|---|
| `sac_autoresearch_nominal_sanity_v1` | nominal_sanity | 0.1667 | 0.3889 | revise |

Report:
`docs/context/policy_search/reports/2026-05-01_sac_autoresearch_nominal_sanity_v1_nominal_sanity.md`

The experiment was rejected and its candidate/training configs were reverted.
The generated checkpoint remains only as disposable local output under
`output/ai/autoresearch/trained_high_success/`.

## Interpretation

Direct 300k-step SAC training on the nominal_sanity surface learned enough to
solve the trivial nominal sanity episodes, but failed all classic and
Francis-style interaction episodes and introduced seven static collisions.
This is worse than the rule-based incumbent on both success and safety.

Current best local policy remains `hybrid_rule_v3_fast_progress`.

## Next Direction

More short constant-tweak or short SAC runs are unlikely to close the gap to the
nominal_sanity gate. The remaining high-value path is to change the route/local
minima behavior directly, or run a longer training campaign with a curriculum
and periodic policy-search evaluation after the evaluation harness is extended
to consume the policy-search seed manifest.
