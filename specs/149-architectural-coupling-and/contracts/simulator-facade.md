# Contract: Simulator Facade (internal component)

## Purpose
Provide a stable interface between environments and underlying simulator implementations.

## Interface (conceptual)
- create(config) -> simulator
- reset(seed?) -> None
- step(n=1) -> None
- get_state() -> SimState (peds_state, groups, optional debug fields)

## Invariants
- Deterministic under fixed seed.
- No direct dependency on specific physics engine types in env code.

## Errors
- Missing required resources -> fatal with remediation guidance.
- Invalid config key -> validation error listing allowed keys.
