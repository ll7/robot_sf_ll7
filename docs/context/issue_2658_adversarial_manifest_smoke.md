# Issue #2658 Adversarial Manifest Smoke

Issue: [#2658](https://github.com/ll7/robot_sf_ll7/issues/2658)
Status: diagnostic smoke evidence only; not benchmark or safety evidence.

## Claim Boundary

This note records a bounded execution of validated `adversarial_scenario_manifest.v1` candidates
through the planner smoke runner. It proves that accepted manifests can pass deterministic
validation, materialize into route-overridden scenarios, and execute a small `goal`/`social_force`
planner pair while rejected or degenerate manifests remain filtered before runner execution.

It does not establish adversarial coverage, planner weakness, leaderboard movement, or
paper-facing benchmark evidence. The `social_force` row is adapter-mode and remains a diagnostic
stress signal only.

Tracked compact evidence:
[summary.json](evidence/issue_2658_adversarial_manifest_smoke/summary.json).

Raw generated manifests, route overrides, materialized matrix files, planner episode JSONL rows,
and local smoke summaries remain ignored under `output/adversarial/issue2658_manifest_smoke/`.

## Result

```yaml
adversarial_manifest_smoke:
  evidence_classification: adversarial_smoke
  result_classification: smoke_passed
  manifest_schema_version: adversarial_scenario_manifest.v1
  validator_ref: robot_sf.adversarial.scenario_manifest.validate_manifest_payload
  seed_family: crossing_ttc_space.yaml:random_seed_44
  generated_count: 4
  valid_count: 4
  invalid_count: 0
  degenerate_count: 0
  planner_pair: [goal, social_force]
  executed_count: 2
  horizon: 60
  dt: 0.1
  claim_boundary: diagnostic_only
```

Both planners completed two episodes. In this smoke, both selected cases ended in collision for both
`goal` and `social_force`; that is a diagnostic signal only, not a planner comparison claim.

## Validation

```bash
uv run pytest tests/tools/test_run_adversarial_manifest_smoke.py tests/adversarial/test_adversarial_scenario_manifest.py -q
uv run ruff check scripts/tools/run_adversarial_manifest_smoke.py tests/tools/test_run_adversarial_manifest_smoke.py
uv run ruff format --check scripts/tools/run_adversarial_manifest_smoke.py tests/tools/test_run_adversarial_manifest_smoke.py
TF_CPP_MIN_LOG_LEVEL=2 LOGURU_LEVEL=WARNING PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  uv run python scripts/tools/run_adversarial_manifest_smoke.py \
  --search-space configs/adversarial/crossing_ttc_space.yaml \
  --scenario-template configs/scenarios/templates/crossing_ttc.yaml \
  --count 4 --seed 44 --max-valid 2 \
  --output-dir output/adversarial/issue2658_manifest_smoke \
  --summary-json output/adversarial/issue2658_manifest_smoke/summary.json \
  --planner goal --planner social_force \
  --horizon 60 --dt 0.1 --workers 1
```

Observed result: targeted tests passed with 49 tests; Ruff check and format check passed; the
bounded smoke command exited successfully and wrote the compact summary promoted above.
