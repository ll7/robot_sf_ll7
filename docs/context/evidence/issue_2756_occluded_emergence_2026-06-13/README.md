# Issue 2756 Occluded Emergence Fixture

This evidence note records the deterministic occluded pedestrian emergence fixture added for
issue #2756.

## Claim Boundary

Smoke/diagnostic only. The fixture proves that the repository has a durable trace with separated
ground-truth and observed pedestrian state for an occluded-emergence pattern. It does not establish
real-world occlusion representativeness, paper-facing benchmark coverage, or learned-prediction
quality.

## Durable Inputs

- Summary:
  `docs/context/evidence/issue_2756_occluded_emergence_2026-06-13/summary.json`
- Trace fixture:
  `tests/fixtures/analysis_workbench/simulation_trace_export_v1/occluded_emergence_episode_0000.json`
- Source metadata:
  `tests/fixtures/analysis_workbench/simulation_trace_export_v1/sources/issue_2756_occluded_emergence_fixture_111_ep0000.meta.json`

## Fixture Properties

- The pedestrian is present in ground truth for all 16 frames.
- `observed_pedestrians` is empty through step 4, then first visible at step 5.
- Per-frame metadata records `occlusion_status`, `first_visible`, `time_to_conflict_s`,
  `stop_feasible`, and `yield_feasible`.
- The fixture has nonzero pedestrian motion and enough frames for 0.5s and 1.0s forecast metrics.

## Validation

Commands run during implementation:

```bash
uv run pytest tests/benchmark/test_occluded_emergence_fixture.py -q
uv run pytest tests/benchmark/test_observation_perturbation.py tests/benchmark/test_pedestrian_forecast.py tests/benchmark/test_pedestrian_forecast_cv_eval.py tests/benchmark/test_occluded_emergence_fixture.py -q
uv run ruff check scripts/benchmark/run_cv_forecast_eval.py tests/benchmark/test_occluded_emergence_fixture.py
uv run ruff format --check scripts/benchmark/run_cv_forecast_eval.py tests/benchmark/test_occluded_emergence_fixture.py
uv run python scripts/benchmark/run_cv_forecast_eval.py --output-dir <ignored scratch directory>
git diff --check
```

The evaluator smoke reported
`occluded_emergence/deterministic_occluded_emergence: evaluated (11 samples)`. The committed fixture
and tests are the durable artifacts.
