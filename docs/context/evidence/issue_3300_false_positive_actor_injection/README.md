# Issue #3300 False-Positive Actor-Injection Smoke

## Claim Boundary

Diagnostic same-seed local smoke for issue #3300. This is executable
planner/environment replay evidence for one scenario, seed, planner, and
five-step horizon. It is not benchmark-strength, hardware-calibrated sensor, or
paper-facing evidence.

## Replay Inputs

- Candidate: `hybrid_rule_v0_minimal`
- Scenario: `issue_3233_near_field_observation_noise`
- Stage: `stress_slice`
- Seed: `3233`
- Horizon: `5`
- Execution path: `scripts/validation/run_policy_search_step_diagnostics.py`
- Comparator: `scripts/benchmark/compare_observation_noise_live_smoke_issue_3201.py`

Raw traces were generated under ignored `output/` paths and are summarized by
the tracked JSON/Markdown files in this directory.

## Commands

```bash
uv run python scripts/validation/run_policy_search_step_diagnostics.py --candidate hybrid_rule_v0_minimal --stage stress_slice --funnel-config configs/policy_search/issue_3201_observation_noise_funnel.yaml --scenario-name issue_3233_near_field_observation_noise --seed 3233 --horizon 5 --output-dir output/benchmarks/issue_3300_false_positive_actor_injection/clean_h5
uv run python scripts/validation/run_policy_search_step_diagnostics.py --candidate hybrid_rule_v0_minimal --stage stress_slice --funnel-config configs/policy_search/issue_3201_observation_noise_funnel.yaml --scenario-name issue_3233_near_field_observation_noise --seed 3233 --horizon 5 --false-positive-actor-count 1 --false-positive-offset-x-m 0.75 --false-positive-offset-y-m 0.0 --observation-perturbation-seed 3300 --output-dir output/benchmarks/issue_3300_false_positive_actor_injection/false_positive_h5
uv run python scripts/validation/run_policy_search_step_diagnostics.py --candidate hybrid_rule_v0_minimal --stage stress_slice --funnel-config configs/policy_search/issue_3201_observation_noise_funnel.yaml --scenario-name issue_3233_near_field_observation_noise --seed 3233 --horizon 5 --missed-detection-probability 1.0 --observation-perturbation-seed 3300 --output-dir output/benchmarks/issue_3300_false_positive_actor_injection/missed_detection_h5
```

## Results

- False-positive replay: `non_null_behavior_delta`; the injected observed-only
  actor produced 5 false-positive actor observations, changed the selected
  command sequence to repeated stop commands, and had no pedestrian, obstacle,
  or robot collision flags in the 5-step smoke.
- Missed-detection replay: `non_null_behavior_delta`; the false-negative slice
  produced 4 missed actor observations, changed the command sequence, and hit
  1 pedestrian collision flag by step 3.
- Near-field target: the clean trace closest robot-pedestrian distance was
  `1.7872738570026998` m, satisfying the <= `2.0` m target used by the
  comparator.

## Durable Summaries

- `false_positive_vs_clean_summary.json`
- `false_positive_vs_clean.md`
- `missed_detection_vs_clean_summary.json`
- `missed_detection_vs_clean.md`
