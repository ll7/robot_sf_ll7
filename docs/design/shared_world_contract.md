# Shared-World Contract: Architecture Decision (Issue #9344)

Status: accepted for the diagnostic adapter slice. No training, fleet, or
benchmark claim follows from this note.

## Current behavior (verified on `origin/main`, not assumed)

- `init_simulators` derives the simulator count from spawn capacity
  (`ceil(requested / num_start_pos)`) and spreads robots across them.
- `MultiRobotEnv.step` distributes one action vector per simulator and calls
  `step_once` per simulator: robots in different simulators never interact.
  When the request fits spawn capacity, all robots already share one
  simulator (one clock, one pedestrian state).
- When any agent terminates, `MultiRobotEnv.step` calls `sim.reset_state()`,
  which resets the shared pedestrian state (`_reset_social_force_state`) and
  respawns finished robots. In a shared simulator this perturbs every other
  agent's world.
- `init_simulators` dropped robots on exact multiples: the last simulator
  received `max(1, requested % capacity)`, so requesting 2 robots on a
  2-start map yielded 1. Fixed by `split_robot_counts` (exact remainder).

## Chosen alternative

A versioned adapter (`robot_sf/gym_env/shared_world.py`,
`shared_world_contract.v1`) next to — not inside — `MultiRobotEnv`:

- Call sequence per step: snapshot → collect full packet → validate packet
  (complete, duplicate-free, known IDs, finite) → parse → exactly one
  `sim.step_once` → per-robot state update → statuses.
- One `world_id`, stable `agent_{index:03d}` IDs, explicit robot count.
- Over-capacity requests are rejected; worlds are never duplicated and starts
  never overlapped to satisfy a request.
- Finished agents remain under a zero hold action; the episode ends under an
  explicit any/all completion policy. Only `reset()` resets the world.
- Invalid packets raise before any mutation; the world is bit-identical after
  a rejected packet.

## Compatibility boundary

- Single-robot paths (`make_robot_env`, `RobotEnv`, existing configs) are
  untouched. `init_simulators` keeps its signature; only the remainder slice
  changed, and total counts are preserved for every previously correct input.
- No PettingZoo or external multi-agent API dependency.

## Unresolved assumptions

- Direct robot-robot force coupling is absent from the simulator; interaction
  flows through the shared pedestrian state (verified: robot motion changes
  pedestrian trajectories with repulsion enabled). Do not claim coordinated
  behavior from shared membership alone.
- Hold semantics for finished agents is zero command (the robot may still
  roll); a timeout/hold policy with active braking is a later contract.
- Cardinality pairs `(capacity, requested)` in the tests admit the full
  request; over-capacity stays a rejection, never a partial world.
