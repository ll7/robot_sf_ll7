# Contract: Unified Config

Purpose: A single configuration surface for creating environments, selecting backends, and wiring sensors.

Schema (conceptual):
- backend: str  # e.g., "fast-pysf"
- robot: { radius: float, speed: float, policy: str }
- pedestrians: { count: int, interactions: { obstacle_forces: bool, robot_forces: bool } }
- sensors: [ { name: str, params: dict } ]
- map: { path: str, seed: int }

Validation:
- Required: backend, map.path
- Defaults applied for optional fields; semantic checks (e.g., radius>0)

Access:
- Programmatic only (factory functions consume this dataclass/obj)
- CLI flags should be adapters, not new config sources
