# Issue 604 Pred2Nav Assessment Note

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#604` Pred2Nav feasibility spike
- `robot_sf_ll7#593` predictive planner v2 / staged benchmark comparison
- `robot_sf_ll7#592` hybrid graph + obstacle-context predictive model idea
- `robot_sf_ll7#603` planner-family coverage matrix

## Goal

Assess Pred2Nav as an external predictive-MPC planner-family anchor for `robot_sf_ll7`.

This issue is an external-family feasibility and overlap assessment. It is **not** an implementation
task.

## Canonical source anchors

- upstream repo: <https://github.com/sriyash421/Pred2Nav>
- local checkout: `output/repos/Pred2Nav`
- paper family:
  - *From Crowd Prediction Models to Robot Navigation in Crowds*
  - *Winding Through: Crowd Navigation via Topological Invariance*
- key source files:
  - `output/repos/Pred2Nav/crowd_nav/policy/vecMPC/controller.py`
  - `output/repos/Pred2Nav/crowd_nav/policy/vecMPC/predictors/cv.py`
  - `output/repos/Pred2Nav/evaluation_mpc.py`
  - `output/repos/Pred2Nav/requirements.txt`

## License status

Current judgment: `license unclear / missing`

Observed evidence:

- the checkout contains no `LICENSE` or `COPYING` file,
- the README names upstream inspirations and dependencies but does not state a reuse license,
- the repository is therefore not safe to vendor or wrap as reusable external code.

Immediate consequence:

- direct code import remains blocked,
- this repository can only be used as a reference source unless upstream licensing is clarified.

## What the upstream method actually is

Pred2Nav is a prediction-aware crowd-navigation stack built on a legacy CrowdSim lineage.

The `vecMPC` controller path combines:

- explicit action rollout generation,
- predictor-pluggable future crowd trajectories,
- cost terms for:
  - goal progress,
  - obstacle/crowd proximity,
  - winding or topological invariance,
  - predictor-specific penalties.

The default `cv.py` predictor is a constant-velocity baseline, but the controller is structured so
other predictors can be swapped in.

Action semantics from `controller.py`:

- the policy returns `ActionXY`,
- rollouts are built as 2D velocity-like guidance trajectories,
- the stack is crowd-simulation-centric and not natively expressed as Robot SF `unicycle_vw`.

## Runtime and stack assessment

Observed runtime burden from the checked-out source:

- Python `3.6` recommended in the README,
- legacy `gym`,
- mixed `torch` and `tensorflow` requirements,
- dependency on `Python-RVO2`,
- CrowdSim-specific environment contract.

This is materially weaker than the current Robot SF stack for near-term integration:

- Python `>=3.11`,
- Gymnasium-first,
- explicit benchmark artifact and provenance workflow,
- native predictive planners already integrated.

## Overlap with current internal planners

### Relative to `prediction_planner`

Strong overlap:

- short-horizon pedestrian prediction,
- explicit rollout scoring,
- progress-versus-risk tradeoff,
- benchmark-facing crowd-state evaluation.

Pred2Nav-specific additions worth noting:

- predictor-pluggable rollout scoring architecture,
- explicit winding / topological cost terms,
- clearer decomposition between prediction module and rollout evaluator.

### Relative to `predictive_mppi`

Strong overlap:

- action-sequence reasoning under predicted pedestrian motion,
- explicit cost-based optimization,
- navigation under robot dynamics rather than pure reactive velocity selection.

Pred2Nav-specific difference:

- uses a legacy CrowdSim + `ActionXY` contract,
- is structurally closer to predictive vector-MPC than to native unicycle-sequence optimization.

## Required inputs, outputs, and kinematics

Required inputs from the checked-out code:

- self-state position, velocity, radius,
- human states position, velocity, radius,
- trajectory history for prediction,
- goal position,
- environment timestep and rollout horizon parameters.

Action semantics:

- output action is `ActionXY`,
- therefore holonomic / velocity-vector oriented at the controller boundary.

Kinematics judgment:

- this is not a direct drop-in for Robot SF unicycle execution,
- adaptation would require an explicit projection or a deeper contract rewrite,
- benchmark-faithful reuse would need that mismatch documented rather than hidden.

## Integration-shape judgment

Current recommendation for integration shape:

- `reference only`

Reason:

- license is unclear,
- runtime is legacy,
- action contract is holonomic,
- the repo is more valuable as a concept source than as a direct external benchmark entry.

## Reusable concepts worth extracting

Pred2Nav remains useful as a design source for native planner work.

Most reusable concepts:

- predictor-pluggable rollout scoring,
- winding / topological cost terms around crowd interactions,
- cleaner separation between crowd prediction and local control selection.

Those ideas map naturally onto existing native tracks:

- `robot_sf_ll7#593` predictive planner v2,
- `robot_sf_ll7#592` hybrid guidance / obstacle-context ideas.

## Recommendation

Recommendation: `do not pursue now`

Meaning:

- do not vendor or wrap the external code until licensing is clarified upstream,
- do not spend benchmark-integration effort on this legacy runtime today,
- treat Pred2Nav as a reference source for native predictive-planner improvements instead.
