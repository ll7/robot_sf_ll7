# Issue #3146/#3164 Forecast Replay Fixture Suite

Issues:

- <https://github.com/ll7/robot_sf_ll7/issues/3146>
- <https://github.com/ll7/robot_sf_ll7/issues/3164>

## Claim Boundary

Evidence status: frozen-policy diagnostic only.

Issue #3146 added a scenario-diverse forecast replay fixture suite for the live
forecast replay gate. Issue #3164 revises that suite into a frozen-policy
comparison: forecast variant changes, while replay control/braking parameters
stay fixed. The suite exercises the full forecast variant matrix (`none`, `cv`,
`semantic`, `interaction_aware`, `risk_filtered`) across existing trace fixtures
from crossing, route-conflict/corridor, dense bottleneck, and signalized-crossing
families. It does not claim planner superiority, safety improvement,
benchmark-strengthening evidence, or paper-grade evidence.

## Contract

The suite manifest is
`configs/benchmarks/issue_3146_forecast_replay_fixture_suite.yaml`. It names the
trace path, scenario family, expected seed, expected scenario id, and expected
row classification for each fixture. The validator is
`scripts/validation/validate_forecast_replay_fixture_suite.py`.

The diagnostic replay policy now records a frozen scalar replay brake distance
(`3.0m`) shared by every non-`none` variant. Variant-specific
`forecast_risk_distance_m` values remain visible as forecast-model parameters,
but they do not change replay braking. The #3164 tracked evidence shows three
native rows and one degraded row; under frozen replay braking, native non-`none`
variants collapse to one closed-loop signature. This is a useful negative
diagnostic result: the earlier full-matrix separation was policy-threshold
confounded and should not be treated as forecast-variant superiority.

Rows are fail-closed:

- `native` rows are usable diagnostic smoke rows.
- `degraded` rows remain visible but are not success evidence.
- `blocked` rows fail the suite unless they are explicitly expected by a future
  manifest revision.

## Validation

Canonical command:

```bash
uv run python scripts/validation/validate_forecast_replay_fixture_suite.py \
  --output-dir docs/context/evidence/issue_3164_frozen_forecast_policy \
  --generated-at-utc 2026-06-20T07:05:00Z
```

Focused test:

```bash
uv run pytest tests/validation/test_validate_forecast_replay_fixture_suite.py
```

The current #3164 tracked evidence summary lives under
`docs/context/evidence/issue_3164_frozen_forecast_policy/`. The previous #3146
diagnostic evidence remains under
`docs/context/evidence/issue_3146_forecast_replay_fixture_suite/` for historical
comparison only.
