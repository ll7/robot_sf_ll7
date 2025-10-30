# Data Model - Architectural decoupling and consistency overhaul

## Entities

### Simulator Facade
- Purpose: Stable contract consumed by envs, abstracts underlying physics engines.
- Responsibilities: create/reset/step/get_state; expose deterministic seed propagation.
- Relationships: Selected via Unified Config; wraps concrete simulator implementation.

### Sensor
- Purpose: Produce typed observation slices with name and shape.
- Responsibilities: init(config), read(state) -> observation, close().
- Relationships: Registered with Fusion Pipeline; lifecycle managed by env/fusion.

### Fusion Pipeline
- Purpose: Compose multiple sensor outputs into a single observation per step.
- Responsibilities: register(sensors), compose(observations) -> fused_observation; validate shapes.
- Relationships: Consumes Sensor outputs; used by env to produce observations.

### Unified Config
- Purpose: Single source of truth for env, simulator, sensors, fusion options.
- Responsibilities: Validation (unknown/conflicts), defaults, resolved-config dump.
- Relationships: Drives factory selection of Simulator implementation and Fusion/Sensors.

## Validation Rules
- Unknown config keys are rejected in strict mode.
- Mutually exclusive options (e.g., image and non-image fusion stacks) are conflicts.
- Sensor output schemas must match declared shapes; fusion validates at registration.

## State Transitions (high-level)
- Env lifecycle: created -> reset -> (step)* -> close.
- Simulator: initialized -> reset -> step [t] -> ...
- Sensors: initialized -> read [t]* -> close.
