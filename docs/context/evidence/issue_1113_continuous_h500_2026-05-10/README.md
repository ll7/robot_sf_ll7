# Issue 1113 Continuous Candidate H500 Evidence

Date: 2026-05-10

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1113>

This bundle stores the compact, reviewable evidence from the local full `full_matrix_h500`
validation run for `hybrid_rule_v3_fast_progress_static_escape_continuous`.

## Contents

- `continuous_candidate_summary.json`: policy-search summary for the continuous candidate.
- `promotion_decision.{json,md}`: evaluation against
  `configs/policy_search/promotion_gates.yaml`.
- `comparison.{json,md}`: comparison against
  `scenario_adaptive_hybrid_orca_v2_collision_guard` and `scenario_adaptive_hybrid_orca_v1`.

## Source Command

```bash
rtk timeout 7200s env LOGURU_LEVEL=WARNING DISPLAY= MPLBACKEND=Agg \
  SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 \
  uv run python scripts/validation/run_policy_search_candidate.py \
  --candidate hybrid_rule_v3_fast_progress_static_escape_continuous \
  --stage full_matrix_h500 \
  --allow-expensive-stage \
  --workers 2 \
  --output-dir output/policy_search/hybrid_rule_v3_fast_progress_static_escape_continuous/full_matrix_h500/issue1113_continuous_h500
```

Raw JSONL records and generated algo YAML files remain ignored under `output/` because they are
reproducible from the tracked candidate config, scenario matrix, seeds, and command above.

