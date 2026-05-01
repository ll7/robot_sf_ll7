# Issue 868 Scenario Certification

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/868>

## Goal

Add a first `scenario_cert.v1` surface that classifies scenarios before they are used for benchmark
claims. The certificate must fail closed for malformed geometry and infeasible route contracts,
while still representing stress-only and hard-but-solvable cases without flattening everything into
pass/fail.

## Scope Decision

The v1 implementation is intentionally conservative:

- it uses the existing scenario loader so map registry resolution and route overrides match
  benchmark/training behavior,
- it uses the classic A* global planner with obstacle inflation and no fallback to prove an
  inflated path exists,
- it treats scenario-difficulty and planner-residual analysis as optional diagnostic evidence only,
- it excludes only dynamic cases that are clearly blocked by static single pedestrians on the
  inflated robot route corridor.

Out of scope for this issue: adversarial scenario generation, headline benchmark promotion, CARLA
export, and treating campaign outcomes as proof of scenario validity.

## Public Surfaces

- Library API: `robot_sf.scenario_certification`
- CLI: `scripts/tools/certify_scenarios.py`
- JSON schema: `robot_sf/benchmark/schemas/scenario_cert.v1.json`
- User docs: `docs/scenario_certification.md`

## Validation Plan

Targeted tests cover the label taxonomy and schema validation:

```bash
PYTEST_ADDOPTS=--no-cov uv run pytest tests/scenario_certification/test_scenario_certification_v1.py -q
```

Smoke generation should run on the atomic invalid fixture set without requiring durable artifacts:

```bash
uv run python scripts/tools/certify_scenarios.py \
  configs/scenarios/sets/atomic_navigation_validation_fixtures_v1.yaml \
  --output output/scenario_cert/issue868_atomic_validation.json
```

Before handoff, run the repository readiness gate after syncing with `origin/main`:

```bash
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

## Known Limits

- `scenario_cert.v1` is not a complete dynamic oracle. Moving pedestrians are hardness evidence
  unless the scenario is obviously overconstrained by a static pedestrian blocking the route.
- Bicycle kinodynamic checks use authored-route turn geometry. Differential and holonomic robots
  are considered turn-feasible because current settings allow rotation in place.
- Worktree-local outputs under `output/` remain disposable; the durable contract is the code,
  schema, tests, and docs.
