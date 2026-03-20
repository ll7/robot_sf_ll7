# External Repository Overview

This folder contains ignored working copies of external repositories inspected for benchmark and planner inspiration. These directories are not part of the tracked `robot_sf` codebase.

Do not copy code from these repositories into `robot_sf` blindly.

- Any future reuse should start from the upstream remote URL, not the local ignored clone.
- Preserve attribution and comply with the upstream license before importing any code or assets.
- Missing or unclear licenses block direct code import until the upstream project is clarified.
- Restrictive licenses can still allow design inspiration, but not necessarily code reuse.

## Benchmark Takeaways

- `Pred2Nav` is the strongest planner inspiration for the current benchmark direction because it combines short-horizon rollout generation, trajectory prediction, and explicit cost shaping.
- `CrowdNav` and `gym-collision-avoidance` are useful reference points for ORCA/RVO baselines and benchmark lineage, but they are better treated as comparison context or lightweight adapter inspiration than as code sources.
- `go-mpc` should not be vendored into this repository because it is GPL-3.0 and depends on a FORCESPro-generated solver.
- `CrowdNav_DSRNN` and `SoNIC-Social-Nav` are mainly useful as alternate learning-based baseline references, not as the primary source for a new classical planner.

## Research Intake Shortlist (2026-03-19)

The current planner-zoo research intake adds a few high-value external candidates. These entries are
research targets, not approved imports. Keep the same guardrails:

- upstream provenance must be verified from a canonical remote URL
- license terms must remain compatible with the intended integration shape
- source-harness or model-only inference must be validated before any benchmark claim

Shortlist summary:

- `CrowdNav-SB3`
  - recommendation: promising learned-policy candidate
  - caveat: the intake provided a non-canonical search URL, so the exact upstream repo still needs
    re-verification before it can be treated as a concrete integration target
- `PySocialForce`
  - recommendation: strong classical baseline candidate
  - caveat: preserve the force-model implementation and add an explicit unicycle adapter rather than
    silently treating desired velocity as executable control
- `SocNavGym`
  - recommendation: `prototype only`
  - caveat: GPL-3.0 blocks direct vendoring, so any use should stay wrapper-only or external
    dependency based
- `SDA`
  - recommendation: `assessment only`
  - caveat: Habitat coupling makes direct 2D benchmark reuse high-risk
- `RVO2-python`
  - recommendation: strong subtree or wrapper candidate for a clean ORCA-family baseline
- `PythonRobotics` DWA
  - recommendation: `inspiration only`
  - caveat: useful for native ports or ideas, not as a provenance-preserving benchmark import

Second-pass ranking highlights:

- best immediate production candidate:
  - `Python-RVO2`
- best Gymnasium-native breadth anchor:
  - `Social-Navigation-PyEnvs`
- best learned-policy breadth candidate:
  - `CrowdNav_HEIGHT`
- most likely dead end despite strong reported results:
  - `SoNIC-Social-Nav`

Current `Social-Navigation-PyEnvs` follow-on judgment after source-harness and ORCA prototype work:

- usable next:
  - non-trainable `socialforce`, `sfm_*`, and `hsfm_*` policies
- blocked for now:
  - learned `cadrl`, `lstm_rl`, and `sarl` from this repo
- blocker:
  - the checked-out repo does not bundle the expected `social_gym/robot_models/...` assets
    (`policy.config`, `env.config`, and learned weight files), so learned-policy reuse would not be
    source-faithful yet

Second-pass execution order:

1. `Python-RVO2`
   - prove upstream example, then add an explicit `velocity_vector -> unicycle_vw` projection
2. `Social-Navigation-PyEnvs`
   - run source harness first, then wrap selected planner modules rather than the whole simulator
3. `CrowdNav_HEIGHT`
   - validate checkpoint-backed inference in a side environment before any main-stack adapter work

Additional guardrails from the second-pass intake:

- `CrowdNav_HEIGHT`, `CrowdNav`, and related learned-policy repos should stay in frozen side
  environments until source-harness parity is demonstrated
- ROS-heavy planners such as `LT_DWA` and MPC repos are method-credible, but they are bridge-first
  candidates, not low-friction in-process wrappers
- `Pred2Nav` remains blocked by unclear license status

## Repository Inventory

### Pred2Nav

- Remote: <https://github.com/sriyash421/Pred2Nav>
- License summary: no license file was found in the upstream repository at review time
  (2026-03-19); treat the repository as reference-only until the upstream license is confirmed.
- Recommendation: `candidate to study`

This repository focuses on crowd navigation with predictive MPC variants and topology-aware cost terms. It is the closest match to the current `robot_sf` benchmark gap because it couples short-horizon robot action rollouts with interchangeable pedestrian trajectory predictors.

Relevant upstream files:

- Controller: [controller.py](https://github.com/sriyash421/Pred2Nav/blob/main/crowd_nav/policy/vecMPC/controller.py)
- Constant-velocity predictor: [cv.py](https://github.com/sriyash421/Pred2Nav/blob/main/crowd_nav/policy/vecMPC/predictors/cv.py)

Planner relevance to `robot_sf`:

- Strong source of ideas for predictor-pluggable local planning.
- Useful reference for rollout generation and explicit multi-term scoring.
- Useful inspiration for winding or topology-sensitive passing costs.

Constraints and risks:

- No visible local license file; do not import code directly.
- The stack is old and built around a different simulator layout.
- Parts of the controller are ballbot-specific, so the dynamics should be reinterpreted rather than copied.

### CrowdNav

- Remote: <https://github.com/vita-epfl/CrowdNav>
- License summary: MIT
- Recommendation: `assessment target for external reproduction`

This repository is a classic crowd-navigation benchmark codebase centered on RL policies, with a small ORCA baseline implementation used as a comparison policy.

Relevant upstream file:

- ORCA baseline: [orca.py](https://github.com/vita-epfl/CrowdNav/blob/master/crowd_sim/envs/policy/orca.py)

Planner relevance to `robot_sf`:

- Historical attention-based family anchor for external reproduction assessment.
- Useful benchmark lineage for historical comparisons.
- Useful reference for how source policies are evaluated inside a CrowdNav-style harness.

Constraints and limits:

- Most of the repository is focused on RL policy training rather than reusable planner abstractions.
- Public pretrained weights are not obviously bundled in the upstream repository.
- Treat CrowdNav as the family anchor, not the first runnable spike target.

### CrowdNav_DSRNN

- Remote: <https://github.com/Shuijing725/CrowdNav_DSRNN>
- License summary: MIT
- Recommendation: `family context only`

This repository extends the CrowdNav family toward graph-structured recurrent RL policies for robot crowd navigation.

Planner relevance to `robot_sf`:

- Useful as an alternate learning-based baseline family.
- Useful for observing policy-factory structure and benchmark conventions across related repos.

Constraints and limits:

- Useful family context, but not the first reproduction target.
- Most of the reusable value is at the benchmark-comparison level, not at the first-spike level.

### SoNIC-Social-Nav

- Remote: <https://github.com/tasl-lab/SoNIC-Social-Nav>
- License summary: MIT at the repo root; bundled subcomponents also include their own licenses, including Apache-2.0 for `Python-RVO2`.
- Recommendation: `assessment target for external reproduction`

This repository combines learning-based crowd navigation policies with simpler non-learning baselines such as social force and ORCA.

Planner relevance to `robot_sf`:

- Practical pretrained/test-only spike candidate for the broader CrowdNav-family roadmap.
- Useful for understanding how a modern crowd-navigation stack ships bundled checkpoints and test-only evaluation code.

Constraints and limits:

- Training is not fully released and the workflow is Docker/NVIDIA-heavy.
- Any reuse would need per-subcomponent license review, not just the root license.
- Treat SoNIC as the first practical inference spike candidate, not as drop-in benchmark support.

### go-mpc

- Remote: <https://github.com/tud-amr/go-mpc>
- License summary: GPL-3.0
- Recommendation: `do not vendor`

This repository implements a hybrid approach where a learned policy recommends subgoals and an MPC layer executes local collision-aware planning.

Relevant upstream file:

- MPC policy: [MPCPolicy.py](https://github.com/tud-amr/go-mpc/blob/main/mpc_rl_collision_avoidance/policies/MPCPolicy.py)

Planner relevance to `robot_sf`:

- Useful as design inspiration for decomposing long-horizon guidance and local control.
- Useful conceptually for hybrid planner portfolios and guidance-plus-reactive control structures.

Constraints and blockers:

- GPL-3.0 makes direct vendoring undesirable for this repository.
- The implementation depends on a FORCESPro-generated solver and a heavy legacy stack.
- Treat this repo as design inspiration only, not as a code source.

### gym-collision-avoidance

- Remote: <https://github.com/mit-acl/gym-collision-avoidance>
- License summary: MIT
- Recommendation: `prototype only`

This repository is a mature collision-avoidance benchmark environment with historical SA-CADRL, GA3C-CADRL, PPO-CADRL, and RVO baselines plus bundled learned-policy checkpoints.

Relevant upstream files:

- Example entrypoint: [example.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/experiments/src/example.py)
- CADRL policy: [CADRLPolicy.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/envs/policies/CADRLPolicy.py)
- GA3C-CADRL policy: [GA3CCADRLPolicy.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/envs/policies/GA3CCADRLPolicy.py)
- RVO policy: [RVOPolicy.py](https://github.com/mit-acl/gym-collision-avoidance/blob/master/gym_collision_avoidance/envs/policies/RVOPolicy.py)

Planner relevance to `robot_sf`:

- Best historical external reference for the SA-CADRL / GA3C-CADRL family.
- Strong candidate for a fail-fast source-harness reproduction spike because the repo includes runnable entrypoints and bundled learned checkpoints.
- Lower value as an ORCA import target because `Python-RVO2` already provides a cleaner upstream-backed ORCA path.

Constraints and limits:

- The stack is still built around legacy `gym` and TensorFlow-era learned policies.
- Observation packing and normalization are strongly source-specific.
- Treat it as `prototype only`: source-harness reproduction first, then a wrapper only if parity is demonstrated.

## Recommended Benchmark Directions

Near-term candidates for `robot_sf`:

- Reimplement a lightweight `Pred2Nav`-inspired predictive MPC baseline in native `robot_sf` style instead of copying external code.
- Consider a minimal ORCA or RVO comparison baseline only if it adds real benchmark breadth beyond the existing SocNav-style ORCA adapter.

Planner ideas worth translating into native `robot_sf` code:

- Predictor-pluggable local planner interfaces.
- Short-horizon action rollout generation with explicit scoring.
- Cost decomposition into goal progress, obstacle interaction, and passing-preference terms.
- Hybrid long-horizon guidance with a local reactive or optimization layer.

Avoid:

- Importing GPL solver-backed `go-mpc` code.
- Copying code from `Pred2Nav` before its license is confirmed upstream.
- Vendoring entire external repositories into tracked `robot_sf` code.
