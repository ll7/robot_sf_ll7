# Goal and scope

Address #10407 with the smallest per-scenario alternative pedestrian model for
predictor development. Reuse existing runtime code; leave defaults unchanged.
Deliver matched selectable scenario packets, executable compatibility/activation
proof, a short indexed note, and a PR. No planner, campaign, release, merge, or
performance/realism claim. Profile: coordinated implementation; research evidence
is limited to implementation integrity.

# Survey (before implementation)

Surveyed current checkout bytes at base
`a52d2cb65a1f1d9850568a9b857fb53a71307405`, initially clean, repository
`ll7/robot_sf_ll7`. Sources: issues #10407 and #10402, root AGENTS.md,
`docs/code_review.md`, and the files below. The retrieval broker returned
INDETERMINATE because its graph adapter lacked an isolated binary; direct `rg`
and source reads supplied the survey. No graph result is authority.

- `robot_sf/ped_npc/ped_behavior.py`: crowded-zone goal resampling, route
  following/respawn, and single scripted pedestrians (including holds/triggers).
  These manage goals and actor lifecycle over the shared force integrator;
  they are not independent ORCA pedestrian solvers.
- `ped_population.py`, `ped_archetypes.py`, `ped_grouping.py`, and
  `ped_robot_force.py`: spawn/speed heterogeneity, groups, and robot responses.
  Adversarial/residual forces are also present. These vary the SFM population or
  forces rather than supply a different crowd solver.
- `fast-pysf/pysocialforce/config.py` and `robot_sf/sim/simulator.py`:
  PySocialForce desired/social/obstacle/group forces are configured at construction.
  SimulationSettings exposes desired-speed, kernel, integration, obstacle-law,
  robot-response, and pedestrian-model choices. Default model is
  `social_force_default`.
- `robot_sf/sim/pedestrian_model_variants.py` and `Simulator._step_pedestrians`:
  existing selectors include total-force heading, TTC additive repulsion,
  anisotropic field of view, alignment torque, and Zanlungo collision prediction.
  Zanlungo replaces the pedestrian-pedestrian SocialForce contribution using
  earliest-approach common-time projection; other force components are retained.
  This is a stronger equation mismatch than adding TTC to the default force.
- ORCA exists for robot planning (`robot_sf/planner/socnav_orca.py`); the inspected
  pedestrian controllers and supported-model registry contain no ORCA crowd
  selector. No new solver dependency is warranted for this low-priority task.
- Replay exists for benchmark trace inspection and generated-scenario state
  initialization (`robot_sf/benchmark/scenario_generation/replay_adapter.py`);
  it is not a supported continuous trajectory-replay pedestrian-model selector.
- Existing #4973 research packets and
  `tests/sim/test_zanlungo_collision_prediction_model.py` already cover pure force
  math and isolated integration. Their research YAML is not a matched, runnable
  default/alternative scenario pair for #10407; its old seed is not used here.
- `robot_sf/training/scenario_loader.py` already accepts per-scenario
  `simulation_config.pedestrian_model` and nested Zanlungo parameters. Reuse this
  public surface instead of inventing an alias or duplicate dynamics module.

# Steps and decisions

1. Add matched development scenarios selecting default and existing Zanlungo
   dynamics. Pin inputs and use only dev seed 1001 (permitted band 1001–1200).
2. Test real scenario loading and real simulator stepping: identical initial
   populations; finite, deterministic, distinct alternative trajectories;
   omitted-selector default equals explicit default and direct legacy PedState
   stepping for the same forces.
3. Document selection, retained shared SFM terms, and limits. This does not prove
   predictor gains, physical realism, or full independence from SFM equations.
4. Run focused tests, lint/format, and required readiness checks. Inspect the
   final diff and GitHub PR head; never self-approve or merge.

# Validation route and test value

- New scenario test protects the actual shipped pair against selector loss and
  mismatched map/config/seeds. Existing tests construct mappings in memory and
  do not load this pair. No production test seam.
- Runtime test protects model activation and deterministic divergence against
  accidental default dispatch. Existing fake-state tests miss full construction
  and repeated stepping from the shipped scenario. No production test seam.
- Default equivalence test protects legacy dispatch against silent default
  changes. Compare to direct PySocialForce stepping as an independent reference;
  existing tests assert selectors/one-step formulas rather than this path.
  Demonstrate sensitivity with a temporary changed dispatch, then restore it.
- Commands: shared-venv `uv run pytest tests/sim/test_predictor_pedestrian_models.py
  -q`, scoped Ruff check/format, then `BASE_REF=origin/main
  scripts/dev/pr_ready_check.sh`. Preserve exact outcomes and any blockers.
  Held-out 111–140 and sealed 50036–59019 are forbidden. Existing broad tests
  run with a lane-local copy of the repository's test seed harness tightened to
  reject simulation seeds outside 1001–1200, including child interpreters. This
  harness is outside the checkout and does not change repository seed policy.
  A pre-existing fixture outside that band blocks the broad check rather than
  authorizing a different seed. No benchmark execution is authorized.

# Implementation evidence

- Reused the existing Zanlungo implementation with zero runtime code changes.
  The scenario pair loads through the public loader and the factory constructs
  real simulators with identical four-pedestrian starting populations.
- Focused suite: 31 passed (new pair/default-equivalence/runtime tests plus
  existing Zanlungo and TTC tests), under the stricter lane seed guard.
- Test sensitivity: temporarily scaling default dispatched forces by 0.5 made
  the default-equivalence assertion fail (16/28 state elements differed).
  Temporarily omitting Zanlungo from variant dispatch made the trajectory
  difference exactly zero and failed the activation assertion. Both mutations
  were restored; neither is delivered. These are mutation controls, not a claim
  of a pre-existing runtime bug: base runtime already supports this model.
- Scoped Ruff lint and format checks passed. Final broad readiness outcomes are
  recorded in the external task report and PR validation section.

# Recovery and handoff

Task-owned worktree and branch are isolated. Keep plan and code in Git, raw
validation logs in the lane directory outside the checkout, and final report.md
there with terminal state and PR head SHA. A reviewer can revert the scenario,
test, and note additions together. Record evidence, owner, next action, and
revival conditions in report.md. Downstream predictor evaluation stays in the
epic; no evidence admission or registry update is claimed.
