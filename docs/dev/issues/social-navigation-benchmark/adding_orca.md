Here’s a concrete, low‑risk plan to add an ORCA/RVO baseline, followed by a decision with clear arguments so we can move forward confidently.

## plan: how we’d use ORCA (or derivatives)

- Library choice (ranked)
  1) rvo2 (Python bindings to RVO2/ORCA)
     - Pros: Mature, fast, widely used, Apache‑2.0 license.
     - Cons: Native wheel/build risk on some CI runners; obstacle import is extra work.
  2) Pure‑Python ORCA variants (pyorca/others)
     - Pros: Easier install; no native build.
     - Cons: Slower; less feature‑complete; correctness/perf risk.
  3) In‑house “ORCA‑lite”
     - Pros: Full control; tuned to our env.
     - Cons: Time‑consuming; correctness risk; scope creep.

- Baseline adapter design
  - New module: `robot_sf/baselines/orca.py`
  - Interface: matches existing planners (`step(obs) -> action`), wired via `get_baseline('ORCA', config)`.
  - One‑way coupling: Use ORCA only to select the robot’s avoidance velocity; pedestrians stay under our sim physics (FastPysf). This avoids double‑modelling.
  - Action mapping:
    - Extract robot/ped positions and velocities from env observation/state.
    - Build a tiny ORCA simulator per step (or maintain across steps) with:
      - timeStep=env.dt, radius=robot radius, maxSpeed=robot max speed.
      - neighborDist, maxNeighbors, timeHorizon/timeHorizonObst from config.
    - Add robot as an agent (preferred velocity toward goal).
    - Add pedestrians as agents with their current velocity as preferred velocity (or predicted short‑horizon goal direction).
    - ORCA step → read robot’s new velocity → convert to our action space (e.g., target v, ω or Δv with clamp).
  - Obstacles (phase 2):
    - MVP: ignore static obstacles in ORCA (leave obstacle handling to env dynamics/collision checks).
    - Later: map walls/segments to ORCA obstacles via `addObstacle` for tighter turns in clutter.

- Parameters and config
  - Add an ORCA config dataclass (or extend unified config):
    - neighbor_dist, max_neighbors, time_horizon, time_horizon_obst, radius, max_speed, goal_gain, stop_dist.
  - Defaults derived from env config (robot radius/max speed) and standard ORCA heuristics.

- Determinism and performance
  - Deterministic: ORCA is deterministic given inputs (no RNG).
  - Performance: rvo2 is fast; per‑step agent count = 1 robot + N pedestrians. If needed, maintain a persistent simulator and update agents in place to avoid re‑adding.

- Edge cases and fallbacks
  - If ORCA returns a near‑zero or NaN velocity: fall back to straight‑to‑goal (clamped) or prior velocity.
  - If neighbor count is huge: cap to `max_neighbors` and warn via logging (no prints).

- Dependency management
  - Optional dependency: try‑import rvo2; if missing, raise clear error with install hint.
  - Tests that require rvo2 are skipped when not installed (marker).

- Tests (minimal and meaningful)
  - Unit: instantiate ORCA baseline; one step with a few pedestrians → returns finite action.
  - Smoke: run a short episode in a corridor with pedestrians; ensure no exceptions; collisions <= Random baseline.
  - Determinism: fixed observation → same action across runs.
  - Type check and lint clean.

- Docs and wiring
  - Docs: `docs/dev/issues/orca-baseline/design.md` (overview, params, limitations) and `todo.md` for tracking.
  - CLI: expose via existing baseline selection (`--algo ORCA`); add example in docs.
  - Figure/demo (optional): a short clip or metric table comparing SF vs ORCA vs PPO on one scenario.

- Rollout phases
  1) MVP: Robot avoids pedestrians (no ORCA obstacles), no map ingestion.
  2) Obstacles: Import walls to ORCA for better turns.
  3) Polishing: Param sweep for neighborDist/timeHorizon; add tiny sensitivity note.

- Success criteria
  - ORCA baseline runs end‑to‑end on the core suite with no crashes.
  - On crowded scenarios, collision rate lower than Random and comparable or better than SF in at least one scenario cluster.
  - Deterministic outputs across seeds (modulo stochastic env elements).
  - Tests green; optional dependency handling verified in CI.

## decision: include ORCA now or defer?

- Option A — Include ORCA now (via rvo2, MVP without obstacles)
  - Pros:
    - Strong, recognized analytical baseline; improves benchmark credibility.
    - Likely to yield distinct trade‑offs vs SF/PPO (e.g., fewer close calls, possibly longer paths).
    - Small scoped MVP (robot–ped only) can land in ~0.5–1.5 days.
  - Cons:
    - Native wheel risk on some CI/macOS runners; we’ll need optional dependency + skip strategy.
    - Minor schedule risk: we’ll re‑run baseline batches once ORCA lands.

- Option B — Defer ORCA until after initial research runs (recommended)
  - Pros:
    - Zero schedule risk for the immediate research kickoff.
    - We already meet discriminative and reproducibility goals with SF/PPO/Random.
    - Avoid potential friction with native wheels in CI right before a batch run.
  - Cons:
    - Fewer baselines in first report; we’ll miss a classic point of comparison initially.
    - A second round of episodes once ORCA is added.

Recommendation: Defer for this sprint, prepare the adapter skeleton, and schedule ORCA MVP immediately after the first batch/validation. Rationale: We’re ready to begin research now; ORCA adds value but is not blocking for discriminative validation and SNQI analysis. Deferring reduces delivery risk and re‑run costs; we can still highlight ORCA as planned in the benchmark roadmap.
