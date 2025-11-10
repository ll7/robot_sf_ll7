# Contract: State Schema (runtime)

Purpose: Define the minimal runtime state exchanged between simulators and views/consumers.

Entities:
- SimState := tuple[np.ndarray, Groups]
  - ndarray shape: (N, 6) -> [x, y, vx, vy, ax, ay] per ped (float64)
  - Groups: list[list[int]] or empty list if unused

Evolution:
- step(): mutates internal sim, but get_state() returns a new view (no aliasing contract)

Notes:
- Episode metadata (seed, scenario_id) is not part of runtime state; tracked separately in benchmark APIs
