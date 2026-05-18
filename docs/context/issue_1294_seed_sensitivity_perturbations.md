# Issue #1294 Seed-Sensitivity Perturbations

## Scope

Issue #1294 extends the adversarial seed-sensitivity explorer from Issue #1271 with an opt-in
timing/speed perturbation grid around one fixed adversarial candidate.

The implementation adds `SeedSensitivityPerturbation` to
`robot_sf.adversarial.seed_sensitivity`. Each perturbation explicitly records:

- `pedestrian_speed_delta_mps`
- `pedestrian_delay_delta_s`
- `spawn_time_delta_s`
- optional `label`

`run_seed_sensitivity` still defaults to one no-op perturbation, preserving the original
seed-only behavior. When perturbations are provided, replay order is deterministic:
`seed x perturbation`.

## Evidence Boundary

Perturbation summaries remain development evidence for adversarial analysis, not durable benchmark
claims. Generated summaries under `output/` are disposable unless promoted through the artifact
policy with provenance, checksums, and a durable storage pointer.

Certification still runs per perturbed candidate. Rejected perturbations are recorded as
`fail_closed_exclusion` and do not count as persisted failures.

## Validation Evidence

- Red first:
  `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_records_timing_speed_perturbation_grid tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_rejects_unbounded_perturbations -q`
  failed because `SeedSensitivityPerturbation` was not implemented.
- Focused proof:
  `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_classifies_stable_and_brittle_failures tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_records_fail_closed_rejected_perturbations tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_records_timing_speed_perturbation_grid tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_rejects_unbounded_perturbations -q`
  passed with `4 passed`.
- Adjacent adversarial proof:
  `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run pytest tests/adversarial/test_adversarial_search.py -q`
  passed with `42 passed`.
- API smoke:
  built `SearchConfig` from `configs/scenarios/templates/crossing_ttc.yaml` and
  `configs/adversarial/crossing_ttc_space.yaml`, then ran `run_seed_sensitivity` over seed `[100]`
  with one no-op perturbation and one speed/delay/spawn perturbation using an injected evaluator.
  Observed `no_failure`, two replay rows, and summary path
  `output/adversarial/issue1294_seed_sensitivity_perturbation_smoke/seed_sensitivity_summary.json`.
- Lint/type proof:
  `uv run ruff check robot_sf/adversarial/seed_sensitivity.py robot_sf/adversarial/__init__.py tests/adversarial/test_adversarial_search.py`
  passed.
  `uvx ty check robot_sf/adversarial/seed_sensitivity.py --exit-zero` passed.
- Post-merge proof after refreshing against `origin/issue-1271-seed-sensitivity-explorer`:
  `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run --active pytest tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_classifies_stable_and_brittle_failures tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_records_fail_closed_rejected_perturbations tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_records_timing_speed_perturbation_grid tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_rejects_unbounded_perturbations tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_rejects_empty_perturbation_iterables tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_records_fail_closed_evaluator_rejections tests/adversarial/test_adversarial_search.py::test_seed_sensitivity_rejects_non_integral_replay_seeds -q`
  passed with `7 passed`.
  `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run --active pytest tests/adversarial/test_adversarial_search.py -q`
  passed with `45 passed`.
  `uv run --active ruff check robot_sf/adversarial/seed_sensitivity.py tests/adversarial/test_adversarial_search.py`
  and
  `uv run --active ruff format --check robot_sf/adversarial/seed_sensitivity.py tests/adversarial/test_adversarial_search.py`
  passed.
  `BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh` passed.
- Post-retarget proof after Issue #1271 merged and this branch refreshed against `origin/main`:
  the same seven focused seed-sensitivity tests passed with `7 passed`, and
  `DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run --active pytest tests/adversarial/test_adversarial_search.py -q`
  passed with `45 passed`.

## Remaining Notes

- The perturbation surface is deliberately bounded in code to small deltas:
  speed deltas up to `1.0 m/s` in either direction and timing deltas up to `5.0 s` in either
  direction, while keeping resulting speed positive and resulting timing values non-negative.
- This does not add distributed sweeps or promote any generated perturbation output as benchmark
  evidence.
