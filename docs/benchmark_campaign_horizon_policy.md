# Historical campaign horizon compatibility

Fixed campaign and planner horizons refuse shorter authored scenario budgets by
default. The 0.0.8+ protocol uses a hash-pinned authored schedule, or a fixed
horizon that every scenario admits.

Corrected D-050 preserves main's historical behavior. `horizon: 600` capped the
runner loop; it never extended a shorter authored simulator limit. Historical
configs use `legacy_runner_cap`: the simulator retains its authored limit, the
runner cap is 600, and the effective budget is `min(authored, 600)`. A timeout
at an authored limit below 600 remains `terminated`, exactly as on main.

Production admission uses the exact source-byte SHA-256 registry in
`robot_sf/benchmark/camera_ready/_historical_horizons.py`. Each entry records
its true protocol version and `legacy_runner_cap`. Source filenames are only
documentation: changing any byte removes admission. Historical YAML and its
manifest pins remain unchanged. There is no test-only injection and no alias
for the former policy name. The 0.0.8-cycle `runtime_smoke_v0_4` is excluded.

Scenario and planner preparation independently enforce the historical version
fence. Input `metadata.campaign_horizon` and `metadata.scenario_horizon` are
reserved and refused, including keys planted through matrix overrides. A
passed runner horizon must match its admitted bound. Row provenance records
`policy`, `authored_max_episode_steps`, `runner_horizon`, and
`applied_max_episode_steps` (the effective minimum). New annotations are excluded
from historical identity; all previously emitted stable row fields retain main's
values, including `scenario_params.run_horizon: 600`.

For 0.0.8+, full-release acceptance independently pins the authored schedule in
the manifest and compares every row horizon, provenance horizon, run_horizon,
and effective budget with its scenario. Legacy-policy rows are refused.

Issue #9748's v1 config declared 600 but its simulator budgets were already
500/500/400/400 for doorway medium, group crossing medium, perpendicular
traffic and crowd navigation. These equal the v2 schedule and 0.0.8 authored
budgets: no tuning mismatch exists. Original v1 inputs, log and frozen v4
parameters remain unchanged; a future v2 run has its own input closure and log.
