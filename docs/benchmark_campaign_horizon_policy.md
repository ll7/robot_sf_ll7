# Historical campaign horizon compatibility

Fixed campaign and planner horizons refuse shorter authored scenario budgets by
default. Current 0.0.8 and later protocols must use the authored schedule or a
fixed horizon that every scenario admits.

A historical protocol may explicitly reproduce its original extension behavior:

```yaml
protocol_version: 0.0.7
horizon_policy: legacy_fixed_extends_authored
horizon: 600
```

The policy is accepted only for an explicit `protocol_version` from 0.0.2 through
0.0.7. Unknown policies, missing/malformed versions, and 0.0.8+ versions fail
closed. Omitting the policy retains the default refusal even for old protocols.
Schedule mode and the legacy policy cannot be combined.

Do not rewrite a frozen historical campaign or its manifest pins. Its fixture
or reproduction caller can supply the override after reading the immutable input:

```python
from dataclasses import replace

cfg = replace(
    load_campaign_config(frozen_path),
    protocol_version="0.0.7",
    horizon_policy="legacy_fixed_extends_authored",
)
scenarios = _load_campaign_scenarios(cfg)
```

Scenario preparation and planner preparation independently enforce the version
fence. The episode row records `metadata.scenario_horizon` with `policy`,
`authored_max_episode_steps`, and `applied_max_episode_steps`. The applied value
is the actual simulator budget. Historical episode identity remains based on
the authored input and requested `run_horizon`; the new accounting annotation
does not rename historical rows.

The immutable release/stress fixture overrides are explicit in
`tests/benchmark/conftest.py` and are enabled only by fixtures that request
`historical_horizon_policy`. Source YAML and manifest checksums remain active.
They authorize no held-out execution during development.

Issue #9748 uses a new v2 development config and pinned authored schedule for
future execution. The original v1 H600 tuning inputs, log, and frozen v4
parameters remain unchanged. That H600 log is not tuning evidence for v2.
