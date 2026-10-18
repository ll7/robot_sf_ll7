# Historical campaign horizon compatibility

Simulator-budget binding and shorter-budget refusal apply only to campaign or
planner horizons on configs that explicitly declare `protocol_version: 0.0.8`
or a later protocol. Admitted 0.0.8 release inputs use the same declaration.
Other fixed-horizon configs, including those without a protocol version, retain
main `93ba0d75` behavior: the horizon caps the runner loop only. Their authored
simulator limits and complete scenario payloads are untouched, preserving
`config_hash`, `episode_id`, and authored-limit `terminated` labels. No shorter
authored-budget refusal or horizon annotation is added to those inputs.

The 0.0.8+ protocol uses a hash-pinned authored schedule, or a fixed horizon
that every scenario admits.

Older and unversioned `scenario_horizons` configs retain main's scheduled
simulator limits and four-field schedule metadata (`source`,
`recommended_horizon_steps`, `status`, `bucket`). Only declared 0.0.8+ schedules
add `sha256` and `authored_max_episode_steps`; those reserved fields identify
the authored-budget contract to the runner and resume identity. Historical
scheduled rows keep their original episode IDs, timeout labels and row fields,
including the absence of an automatically added `scenario_params.run_horizon`.

D-064 preserves main's historical behavior. `horizon: 600` capped the
runner loop; it never extended a shorter authored simulator limit. Historical
configs use `legacy_runner_cap`: the simulator retains its authored limit, the
runner cap is 600, and the effective budget is `min(authored, 600)`. A timeout
at an authored limit below 600 remains `terminated`, exactly as on main.

Production admission uses the exact source-byte SHA-256 registry in
`robot_sf/benchmark/camera_ready/_historical_horizons.py`. Each entry records
its true protocol version and `legacy_runner_cap`. Source filenames are only
documentation: changing any byte removes admission. Historical YAML and its
manifest pins remain unchanged. There is no test-only injection and no alias
for the former policy name. The 0.0.8-cycle `runtime_smoke_v0_4` is excluded from this historical registry;
its undeclared fixed horizon follows the general runner-cap compatibility path.

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

The sealed three-width doorway input explicitly declares `protocol_version: 0.0.8`,
removes its fixed horizon, and pins
`configs/benchmarks/horizon_schedules/three_width_doorway_release_0_0_8_authored_v1.yaml`.
All three scenarios retain their authored 400-step budgets. Config/schedule admission
checks do not execute an evaluation seed. The v0.2 width-slice template is owned by
stacked PR #10039; it must independently pin this same three-scenario schedule.
