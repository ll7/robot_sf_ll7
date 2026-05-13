# Issue 1110 CARLA oracle replay parity adapter

Date: 2026-05-12

## Scope implemented so far

This branch adds the first CARLA-free metric parity adapter for comparing Robot-SF trajectory/episode metrics with CARLA oracle replay metrics.

Implemented files:

- `robot_sf/carla_bridge/parity.py`
- `robot_sf/carla_bridge/__init__.py`
- `scripts/carla_bridge/compare_oracle_replay_metrics.py`
- `tests/carla_bridge/test_parity.py`
- `tests/carla_bridge/test_parity_cli.py`

## Contract

`compare_oracle_replay_metrics(...)` emits `comparison_schema=carla_oracle_replay_parity_v1`.

Default metrics considered:

- `success`
- `collision`
- `ttc_min_s`
- `min_distance_m`
- `comfort`
- `jerk`
- `curvature`
- `intervention_rate`
- `snqi`

Metric behavior:

- Numeric fields are marked `comparable` and report `delta = carla_value - robot_sf_value`.
- Boolean fields are marked `match` or `mismatch`.
- Missing Robot-SF or CARLA fields are marked `unavailable` with explicit reasons.
- Non-finite or non-numeric/non-boolean fields are marked `unavailable` instead of coerced.
- CARLA records whose `mode` or `status` is `fallback`, `degraded`, `not_available`,
  `not-available`, or `failed` make the whole report unavailable.

## Claim boundary

This adapter is not transfer proof by itself. It is a conservative report format for fixture or live replay outputs. A meaningful CARLA transfer claim still requires #1111 live CARLA oracle replay evidence on a CARLA-capable host.

## Remaining work

- Load real Robot-SF and CARLA replay artifact files once the live CARLA output shape is fixed.
- Add timestamp/coordinate-frame alignment metadata if required by #1111 outputs.
- Record first live comparison evidence in a successor context note.

## Validation status

Validation has been run for this implementation slice:

- `UV_NO_CONFIG=1 python -m pytest tests/carla_bridge/test_parity.py tests/carla_bridge/test_parity_cli.py -q`
- `UV_NO_CONFIG=1 BASE_REF=origin/main scripts/dev/pr_ready_check.sh`

These checks cover numeric deltas, boolean parity, missing-metric handling, degraded-mode
rejection, CLI JSON report writing, and the branch-wide readiness gate for the main-based PR.

## Additional foundation: parity CLI

Implemented artifacts:

- `scripts/carla_bridge/compare_oracle_replay_metrics.py`
- `tests/carla_bridge/test_parity_cli.py`

Usage sketch:

```bash
uv run python scripts/carla_bridge/compare_oracle_replay_metrics.py \
  --robot-sf output/carla/robot_sf_metrics.json \
  --carla output/carla/carla_metrics.json \
  --output output/carla/parity_report.json
```

The CLI expects each input to be a JSON object and writes a sorted, pretty-printed `carla_oracle_replay_parity_v1` report.
