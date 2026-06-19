# Issue #3146 Forecast Replay Fixture Suite

Issue: <https://github.com/ll7/robot_sf_ll7/issues/3146>

## Claim Boundary

Evidence status: diagnostic smoke only.

This issue adds a scenario-diverse forecast replay fixture suite for the live
forecast replay gate. The suite exercises the full forecast variant matrix
(`none`, `cv`, `semantic`, `interaction_aware`, `risk_filtered`) across existing
trace fixtures from crossing, route-conflict/corridor, dense bottleneck, and
signalized-crossing families. It does not claim planner superiority, safety
improvement, benchmark-strengthening evidence, or paper-grade evidence.

## Contract

The suite manifest is
`configs/benchmarks/issue_3146_forecast_replay_fixture_suite.yaml`. It names the
trace path, scenario family, expected seed, expected scenario id, and expected
row classification for each fixture. The validator is
`scripts/validation/validate_forecast_replay_fixture_suite.py`.

The diagnostic replay policy now records variant-specific brake distances:
`cv=3.0m`, `semantic=2.5m`, `interaction_aware=2.0m`, and
`risk_filtered=1.5m`. The suite requires at least one native row with multiple
non-`none` closed-loop metric signatures so the full matrix is not merely a
uniform stop/no-stop check.

Rows are fail-closed:

- `native` rows are usable diagnostic smoke rows.
- `degraded` rows remain visible but are not success evidence.
- `blocked` rows fail the suite unless they are explicitly expected by a future
  manifest revision.

## Validation

Canonical command:

```bash
uv run python scripts/validation/validate_forecast_replay_fixture_suite.py \
  --output-dir docs/context/evidence/issue_3146_forecast_replay_fixture_suite \
  --generated-at-utc 2026-06-19T00:00:00Z
```

Focused test:

```bash
uv run pytest tests/validation/test_validate_forecast_replay_fixture_suite.py
```

The tracked evidence summary lives under
`docs/context/evidence/issue_3146_forecast_replay_fixture_suite/`.
