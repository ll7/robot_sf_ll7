# Issue #5792 demo validation record

Status: `diagnostic-only` UX/reproducibility evidence; not benchmark, safety, or paper evidence.

This compact record is the durable artifact pointer for PR #5807. The generated JSONL, viewer, and
PNG remain worktree-local under `output/` and are intentionally not copied into git. They can be
recovered from the tracked scenario, code, seed, and command below.

## Reproduction

- Source repair commit: `240963388` (`fix(ux): make demo artifacts deterministic`)
- Scenario: `configs/scenarios/single/quickstart_demo.yaml`
- Scenario name: `quickstart_demo_crossing_basic`
- Planner: built-in `random`
- Seed: `270`
- Command: `uv run robot-sf demo --output-root output/demo/latest --seed 270`
- Artifact root: `output/demo/latest/`
- Artifact set: `episode.jsonl`, `summary.json`, `metrics.json`, `viewer/index.html`,
  `viewer/scene.json`, and `thumbnail.png`

## Observed CPU smoke

Two fresh process invocations on 2026-07-15 completed in `21.04–21.07 s` wall time on this
validation host, each producing 122 JSONL records for a 120-step episode. The two
`episode.jsonl` files and the two `viewer/scene.json` files were byte-for-byte equal.

This timing is a local UX smoke range, not a performance benchmark or cross-host guarantee.

## Validation commands

```bash
uv run ruff check robot_sf/cli.py scripts/demo/quickstart_demo.py tests/test_quickstart_demo.py
uv run ruff format --check robot_sf/cli.py scripts/demo/quickstart_demo.py tests/test_quickstart_demo.py
scripts/dev/run_focused_tests.sh tests/test_quickstart_demo.py -q
```

The focused suite passed 5 tests, including full JSONL/viewer determinism, repository-root-
independent CLI dispatch, an empty scenario manifest, and a failed required-runtime check.

## Claim boundary

The record supports only that the demo path ran on this host for the named scenario and seed and
that its generated recording/viewer inputs were stable across this two-run smoke. It does not
support planner ranking, collision safety, benchmark performance, deployment readiness, or
generalized timing claims.
