# Safety revision after the independent FIX review

**Historical sampled-position guard.** The [continuous-interval revision](../revision3/README.md) supersedes this implementation and retains these benchmark traces as its comparator.

Development diagnostic only, seeds 1001–1003. This supersedes the original contributor-only evidence for the current implementation. It does not admit release evidence or provide a general collision-safety guarantee.

## Contract and implementation

Retain sequence-objective beam admission. For `surface_v2`, share an independent 0.35 m surface-margin veto between constant-action and sequence scoring. It checks the learned forecast and constant-velocity paths of observed actors against the physical robot rollout, including measured motion/braking when bound. If the observed population exceeds `predictive_max_agents`, the forecast cannot represent every actor and authorizes no motion. The beam excludes infinite-cost candidates/expansions; infeasible selection commands braking. The legacy center-distance mode retains its historical semantics.

No radii, configuration thresholds, predictor, forecast horizon or goal weights are retuned. The 0.8 m term remains a soft objective; it was not a pre-existing hard pedestrian constraint. The new hard constraint uses `predictive_safe_distance` (0.35 m) exactly once in surface units.

The conservative population refusal is intentional: the released crowding scenario has 24 observed actors against a 16-agent model contract. It loses progress there while retaining its timeout outcome. The predictor still loads and runs; no substitute predictor or degraded row is promoted as success. This limitation applies to physical-clearance configurations whose observed population exceeds the configured model capacity.

## Action-level diagnosis

The reviewed crowding seed-1002 minimum occurs while the robot moves at 0.4 m/s near actor slot 16, outside the represented first sixteen actors. In seed 1003 the robot is moving immediately before its minimum. [Scored-trace observations](reviewed_crowding_minima.csv) preserve the velocities and requested commands. Episode minima alone did not resolve this mechanism.

A hard base-margin veto plus observed constant-velocity checks removed commands from already sub-margin states, but two crowding terminal gaps were still newly below margin. Adding a speed-dependent hard buffer still did not resolve those terminal gaps and lost one reviewed completion. Those attempts are preserved privately as rejected diagnostics. The final capacity refusal addresses the incomplete prediction contract, preserves the three reviewed completions, and keeps every observed minimum and terminal gap above 0.35 m. It avoids scenario identifiers, seed-specific rules and tuned buffers.

## Test proof

The original clear-progress regression is retained and strengthened from positive physical separation to the configured safety margin. On main `53721be3248600e08641a9816d094b605e3c93e4`: **1 failed, 1 passed**, the intended zero-speed assertion. On the guarded fix it passes.

New tests assert stop under sub-margin goal pressure, protection of an omitted close actor, compliant slow progress, and refusal of an over-capacity population even before actors approach. On reviewed head `2a6afc5e2a67668013c24d3a7fa6d875d7bdd654`: **4 failed** for the intended moving-command/clearance assertions. On the final fix: focused module **40 passed**. No learned model download or simulation is needed for these tests.

```sh
scripts/dev/run_worktree_shared_venv.sh -- uv run python -m pytest tests/planner/test_socnav_prediction_module.py -q
```

Test value: these protect margin-constrained command selection under high goal pressure and incomplete forecasts; old-head failures demonstrate sensitivity; expectations use returned commands and independent body geometry, not scorer-call assertions; deterministic bounded searches cost roughly 0.03 s each. The previous clear/touching cases miss close but separated bodies and omitted actors. No production test seam was added.

## Affected cells

[All eighteen paired cells](affected.csv), canonical runner, 400 steps at 0.1 s, same 0.0.8 inputs and robot resets. Main 0/18 successes; reviewed 3/18; guarded 3/18, with precisely the same three completions retained. Zero contacts in every revision. All guarded episode minima are at least 0.517 m. Zero newly sub-margin minima or terminal gaps compared with main; no already sub-margin crowding minimum worsens.

Crowding minimum surface clearance, main → reviewed → guarded:

| seed | main m | reviewed m | guarded m | guarded terminal m |
| --- | ---: | ---: | ---: | ---: |
| 1001 | 0.136 | 0.348 | 1.352 | 2.815 |
| 1002 | 0.138 | 0.121 | 1.184 | 1.415 |
| 1003 | 0.247 | 0.118 | 1.806 | 2.582 |

The guarded crowding robot remains at its reset pose with zero requested speed throughout. These are capacity-limited holds, not successful navigation. Fifteen overall timeouts remain; their global feasibility is not asserted.

## Empty-world gate

The final runtime `df7541979` completed 102/102 expected actor-free rows with complete traces on development seeds 1001–1002. [All paired rows](empty_world.csv), [summary](empty_summary.json).

| suite | success main → guarded | collisions main → guarded | new failures |
| --- | ---: | ---: | ---: |
| main (96 episodes) | 94 → 94 | 0 → 0 | 0 |
| width (6 episodes) | 6 → 6 | 0 → 0 | 0 |

The two main timeouts are the same original narrow-doorway physical refusals. Changed commands and trajectories are classified in the complete paired table; none introduces a new failure.

## Review and limits

The original local audit-service failure is unconfirmed as a main defect: independent review passed its entire module on main with parallel workers. It is kept separate from the planner finding. Independent review at the new head remains required; this lane does not merge or poll CI.

A predicted/current margin guard cannot guarantee that pedestrians will never move toward a stationary robot or that forecasts have zero error. The reported margin criterion is verified for the requested development samples, not held-out scenarios. Overflow refusal can increase timeouts in other crowded physical-clearance scenarios; this is an explicit safety/progress tradeoff for review.
