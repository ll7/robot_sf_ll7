# Issue 869 Adversarial Runner

Issue: https://github.com/ll7/robot_sf_ll7/issues/869

## Goal

Issue 869 adds a programmable adversarial scenario search runner under
`robot_sf/adversarial/`. The implementation boundary is the Python API, not a CLI, so search
programs, tests, notebooks, and future optimizer adapters can call the same core directly.

## Current V1 Boundary

- `SearchConfig.from_files(...)` loads a scenario template and search-space YAML.
- `run_adversarial_search(...)` runs a bounded random-search loop by default.
- Candidate generation is deterministic for a fixed seed.
- Search-space validation rejects malformed candidates before policy evaluation.
- `require_certification=True` fails closed when `scenario_cert.v1` is not available.
- The default evaluator delegates to the existing benchmark `run_batch` path.
- Tests can inject evaluator, certifier, and sampler callables to cover orchestration without
  spawning a subprocess.

## Artifact Contract

Each valid candidate gets a bundle directory under the configured `output_dir`:

- `scenario.yaml` for replay through the normal scenario loader,
- `route_overrides.yaml` with generated robot route waypoints,
- `episode_records.jsonl` from the benchmark runner when evaluation runs,
- `trajectory.csv` as a replay index until the benchmark runner exposes dense per-step export,
- `failure_attribution.json`,
- top-level `manifest.json` summarizing all candidates and the best scored bundle.

Generated bundles under `output/` are development stress-test artifacts. They are not durable
benchmark evidence until separately certified, promoted, and published through the repository's
normal artifact policy.

## Validation Path

Focused tests:

```bash
uv run pytest tests/adversarial/test_adversarial_search.py
```

Smoke proof against an existing policy path:

```bash
uv run python - <<'PY'
from pathlib import Path
from robot_sf.adversarial.config import SearchConfig
from robot_sf.adversarial.search import run_adversarial_search

config = SearchConfig.from_files(
    policy="goal",
    scenario_template=Path("configs/scenarios/templates/crossing_ttc.yaml"),
    search_space=Path("configs/adversarial/crossing_ttc_space.yaml"),
    objective="worst_case_snqi",
    output_dir=Path("output/adversarial/issue_869_smoke/goal_crossing"),
    budget=1,
    seed=123,
    horizon=20,
    require_certification=False,
)
result = run_adversarial_search(config)
print(result.best_bundle_path, result.num_failed_evaluations)
PY
```

PR handoff should still run:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

Observed on 2026-05-01 after syncing with `origin/main`:

- `uv run pytest tests/adversarial/test_adversarial_search.py` passed (`8 passed`).
- The smoke search above produced
  `output/adversarial/issue_869_smoke/goal_crossing/candidate_0000` with
  `num_failed_evaluations=0` and `num_invalid_candidates=0`.
- A direct replay of the emitted `candidate_0000/scenario.yaml` through `run_batch(...)` wrote one
  episode to `episode_records_replay.jsonl` with zero failures.
- `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` passed after the coverage-focused tests were
  added (`2998 passed, 19 skipped, 3 warnings`). Changed-file coverage was above the hard 80%
  floor; `bundle.py`, `config.py`, `io.py`, and `search.py` remained below the aspirational 100%
  goal as warning-only items.

Observed on 2026-04-30: the focused tests and smoke path passed. The full readiness script passed
Ruff, then failed in the broad parallel pytest phase due unrelated performance/worker issues
(`tests/examples/test_examples_run.py` timeouts, `tests/test_ci_performance.py` runtime budget,
`tests/research/test_performance_smoke.py` runtime budget, and one xdist worker crash in
`tests/test_force_field_figure.py`). No adversarial tests failed in that run.

## Deferred Follow-Ups

- Add a real `scenario_cert.v1` adapter once the certification package lands.
- Replace the `trajectory.csv` replay index with dense per-step trajectory export when the
  benchmark runner exposes robot/pedestrian trajectories.
- Add optimizer adapters such as CMA-ES or Bayesian optimization only after the scenario semantics
  and bundle contract are stable.
