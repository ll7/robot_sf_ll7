# Negative Result Scenario Candidates (Issue #2788)

## Purpose
This directory contains diagnostic-only scenario candidates derived from the `docs/context/negative_result_register.md`. These candidates encode trace-seeded starting points for future adversarial research, grounded in known failed or diagnostic-only results.

## Candidates
- `candidate_nr001_doorway.json`: Derived from NR-001 (issue-2716), targeting `doorway_transfer`.
- `candidate_nr001_t_intersection.json`: Derived from NR-001 (issue-2716), targeting `t_intersection_transfer`.
- `candidate_nr002_observation.json`: Derived from NR-002 (issue-2749), targeting a near-field `classic_bottleneck_medium` pedestrian-route mutation for future observation-noise stress.

## Scope: Diagnostic Only
All candidates in this directory are `not_promoted`. They do NOT constitute benchmark-strength evidence, paper-facing results, or safety claims. They are used for:
1. Encoding trace-level failure modes for generator testing.
2. Providing candidate mutations for heuristic perturbation design.
3. Validating the `generated_scenario_candidate.v1` schema on real-world negative results.

## Validation
To validate these candidates against the schema:
```bash
uv run python scripts/validation/validate_generated_scenario_candidate.py \
  docs/context/evidence/issue_2788_negative_result_scenario_candidates/*.json
```

## Promotion Path
Promotion to `promoted` status requires passing all gates defined in `docs/context/issue_2725_generator_readiness.md`.
