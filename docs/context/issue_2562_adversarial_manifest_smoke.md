# Issue #2562 Adversarial Manifest Planner Smoke (2026-06-07)

Status: current diagnostic planner-smoke evidence only. This note does not claim adversarial
coverage, planner weakness, leaderboard movement, or paper-facing benchmark evidence.

Related surfaces:

- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/2562
- Predecessor manifest generator: [issue_2524_adversarial_manifests.md](issue_2524_adversarial_manifests.md)
- Adversarial generation roadmap: [issue_2468_adversarial_generation_roadmap.md](issue_2468_adversarial_generation_roadmap.md)
- Search space: `configs/adversarial/crossing_ttc_space.yaml`
- Scenario template: `configs/scenarios/templates/crossing_ttc.yaml`
- Smoke runner: `scripts/tools/run_adversarial_manifest_smoke.py`
- Evidence: `docs/context/evidence/issue_2562_adversarial_manifest_smoke/summary.json`

## Result

Issue #2562 adds the first planner-facing smoke for `adversarial_scenario_manifest.v1`. The runner
generates validator-backed manifests, materializes valid candidates through route-overrides, runs a
small planner pair, and writes a compact JSON summary with generation counts, selected candidate
hashes, replay commands, benchmark availability, and episode-level metric aggregates.

The first implementation attempt tried to materialize generated controls as new
`single_pedestrians`, but `classic_cross_trap` does not expose single-pedestrian runtime slots. That
path failed closed with:

```text
ValueError('single_pedestrians overrides provided but map has no single pedestrians')
```

The final smoke follows the existing adversarial search bridge instead: each selected manifest
writes a route-overrides file for the generated route candidate and adds `peds_speed_mult` plus
manifest metadata to the materialized scenario.

## Evidence

Command:

```bash
TF_CPP_MIN_LOG_LEVEL=2 LOGURU_LEVEL=WARNING uv run python scripts/tools/run_adversarial_manifest_smoke.py \
  --search-space configs/adversarial/crossing_ttc_space.yaml \
  --scenario-template configs/scenarios/templates/crossing_ttc.yaml \
  --count 4 --seed 42 --max-valid 2 \
  --output-dir output/adversarial/issue2562_manifest_smoke \
  --summary-json output/adversarial/issue2562_manifest_smoke/summary.json \
  --planner goal --planner social_force \
  --horizon 60 --dt 0.1 --workers 1
```

Observed compact result:

- Generation: 4 candidates, 4 valid, 0 invalid, 0 degenerate.
- Selected candidates: indexes 0 and 1, hashes `89dabea8ec09870f` and `f7739ece2dae18cb`.
- `goal`: 2/2 episodes written, native/available, success mean 1.0, total collisions 0.
- `social_force`: 2/2 episodes written, adapter/available, success mean 0.0, total collisions 2.
- Result classification: `smoke_passed`.

The tracked JSON summary is the durable evidence. Raw generated manifests, materialized matrix,
route overrides, and episode JSONL rows remain worktree-local under `output/` and are not required
for future runs.

## Claim Boundary

This is a diagnostic smoke. It proves the manifest generator can feed a route-materialized planner
execution path for one bounded seed/config. It does not prove adversarial coverage, rank planners,
certify the generated cases, or establish that the observed `social_force` collisions are a general
planner weakness.

The `social_force` rows are adapter-mode, so they remain a caveated diagnostic signal rather than a
native planner benchmark result.

## Validation

Focused validation on this branch:

```bash
uv run pytest tests/adversarial/test_adversarial_search.py -k 'manifest_materializ' tests/tools/test_run_adversarial_manifest_smoke.py
uv run pytest tests/tools/test_run_adversarial_manifest_smoke.py
uv run ruff check robot_sf/adversarial/materialize.py robot_sf/adversarial/__init__.py \
  scripts/tools/run_adversarial_manifest_smoke.py \
  tests/adversarial/test_adversarial_search.py \
  tests/tools/test_run_adversarial_manifest_smoke.py
```

Observed result: the manifest materialization slice passed with 4 selected tests, the smoke-runner
tests passed with 2 tests, and Ruff check passed.

## Follow-Up Direction

The next useful step is a certification or replay-determinism gate for selected manifest candidates
before treating any generated case as more than development stress input. If the adapter-mode
`social_force` collision signal looks interesting, rerun with a native/comparable planner pair or a
larger certified candidate packet before making any planner interpretation.
