# Contract: Sensor Registry

Purpose: Provide a centralized, scriptable registry for declaring and retrieving sensor implementations used by environments and planners.

Interface (conceptual):
- register(name: str, factory: Callable[[Config], Sensor]) -> None
- get(name: str) -> SensorFactory
- list() -> dict[str, SensorFactory]

Invariants:
- Names are unique (case-sensitive); re-register raises.
- Factories are pure and deterministic w.r.t. provided config.

Errors:
- Unknown sensor name -> KeyError with known names suggestion.
- Factory construction error -> propagated with context.
