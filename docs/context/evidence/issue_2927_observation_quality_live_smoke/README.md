# Issue #2927 Observation-Quality Live Smoke

## Claim Boundary

Smoke/diagnostic live step-diagnostics evidence only. The report attaches bounded simulator observation-quality metadata to an existing same-seed near-field live replay summary and reports safety effects from that smoke. It is not paper-grade, benchmark-strength planner superiority evidence, or hardware-calibrated sensor realism.

## Source

- Source summary: `docs/context/evidence/issue_3233_near_field_observation_noise/summary.json`
- Source issue: `#3201`
- Source classification: `non_null_behavior_delta`

## Execution Boundary

- Evidence status: `smoke evidence`
- Near-field satisfied: `True`
- Clean closest robot-pedestrian distance: `1.4515750296008105` m
- Fallback/degraded rows: none in the source summary.
- Not-available rows: false-positive actor injection is explicitly excluded.

## Observation Quality

- Schema: `observation_quality.v1`
- Perturbed false-negative rate: `0.5`
- Perturbed false-positive rate: `0.0`
- Perturbed angular noise std: `0.0`
- Perturbed range limit: `None`

## Safety Effects

- False-negative effect: `non_null_behavior_delta_with_false_negative_perturbation`
- False-negative rationale: The perturbed live row removed or occluded pedestrian observations. Behavior/progress fields changed in the same-seed comparison.
- False-positive effect: `not_available_excluded`
- False-positive rationale: The source live smoke did not inject false-positive actors. False-positive safety effects are explicitly excluded rather than treated as successful evidence.

## Caveats

- One scenario, one seed, one live step-diagnostics summary.
- Uses non-calibrated simulator observation perturbations only.
- This report does not claim planner superiority or paper-grade evidence.