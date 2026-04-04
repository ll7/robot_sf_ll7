# Issue 771 DR-MPC + SICNav MPC Assessment Note

Date: 2026-04-03
Related issues:
* `robot_sf_ll7#771` DR-MPC + SICNav MPC feasibility spike
* `robot_sf_ll7#760` benchmark method recommendations
* `robot_sf_ll7#599` Go-MPC assessment note

## Goal

Assess whether the MPC-based social navigation candidates DR-MPC and SICNav are worth pursuing in `robot_sf_ll7` and, if so, identify the most promising next benchmark-safe integration boundary.

## Sources / upstream anchors

### DR-MPC

* paper: RA-L 2025, [arXiv:2410.10646](https://arxiv.org/abs/2410.10646)
* repo: https://github.com/James-R-Han/DR-MPC
* visible flow: `scripts/online_continuous_task.py` + `scripts/configs` + `environment/HA_and_PT`
* dependencies:
  + `conda env create -f environment.yml`
  + `Python-RVO2` (CMake build)
  + `pysteam`

  + PyTorch
* license: no explicit license file in repo (risk for downstream reuse)

### SICNav

* paper: T-RO 2024, [arXiv:2310.10982](https://arxiv.org/abs/2310.10982)
* project page: http://sepehr.fyi/projects/sicnav/
* repo: https://github.com/sepsamavi/safe-interactive-crowdnav (MIT)
* checkpoints: available
  + e.g. `sicnav_diffusion/JMID/MID/checkpoints/jrdb_bev_0_25_multi_class_epoch16.pt`
* dependencies:
  + `crowd_sim_plus` simulation
  + `Python-RVO2`
  + `CasADi` + `IPOPT` (or `acados` for real-time path)
  + Stable-Baselines3

## Verification of Done Condition 1

* DR-MPC has a usable upstream repo with runnable training and evaluation scripts. No published pretrained model artifact is obvious in repo root, but the training pipeline is present.
* SICNav has a usable upstream repo and explicit checkpoint artifacts in `sicnav_diffusion/JMID/MID/checkpoints`.

## Interface mapping (Done Condition 2)

### DR-MPC control/state interface

* robot action output: 2D actions (`[v, w]` unicycle command, plus residual correction via DR-MPC policy)
* interaction pattern: residual action added to MPC base command (MPC produces nominal `v,w`; net command applies weighted sum via learned `alpha`/`beta` and OOD-safety module)
* robot state required: pose `(x, y, theta)`, linear/angular velocity `(v, w)`, goal position, path-tracking reference
* human state required: set of `N` neighbors with positions, velocities, radii, history (lookback window, per `env.lookback=6`)
* internal numerics: path tracking MPC solver in `environment/path_tracking`, human avoidance in `environment/human_avoidance` (ORCA-like), residual RL policy in `scripts`.

### SICNav control/state interface

* robot action output: 2D velocity/curvature command; overall API is an MPC `select_action` call in `sicnav` / `sicnav_diffusion` policy classes.
* lower-level model: bilevel MPC with human model solved as ORCA constraints; action horizon updating each step.
* robot state required: pose, heading, speed, goal/waypoint, collision geometry, environment map.
* human state required: per-agent position/velocity/radius, predicted future trajectories via integrated ORCA-based human dynamics.
* returns: safe, interactive multi-human motion decisions with explicit safety buffers.

## Runtime and integration constraints

* DR-MPC:
  + medium-weight Python/torch stack + RVO2 + pysteam; no heavy solver locking.
  + no explicit license makes direct bundling unclear; if accepted, wrap as adapter with a clear `SOURCE` provenance and potential vendor divergence.

* SICNav:
  + heavier solver stack: `CasADi` + `IPOPT` or `acados`, has a working checkpoint pipeline and a `simple_test.py` runner.
  + MIT license – cleaner for Robot SF integration.

## Recommendation (Done Condition 3)

* DR-MPC: `assessment only`
  + Strengths: directly a residual MPC concept that fits Robot SF goal (path tracking + crowd reasoning / constraints)
  + Weaknesses: no prepackaged license file, reliance on non-robot-SF sim stack, training pipeline is necessary for benchmark use.

* SICNav: `prototype only` (for path to `assessment only`)
  + Strengths: MIT-licensed, code + checkpoints + canonical paper, most mature operational pipeline in upstream.
  + Weaknesses: solver dependency stack is heavy (CasADi/IPOPT/Acados), may require simulated state mapping and C++ net install barrier in CI.

* Overall family level: keep MPC approaches as a core benchmark candidate but in the `conceptually adjacent` track until a thin wrapper exists.

## Suggested smallest benchmark-safe boundary (Done Condition 4)

1. Add external anchor rows to `docs/benchmark_planner_family_coverage.md` for `DR-MPC` and `SICNav` as `conceptually adjacent only`.
2. Implement two experimental adapters:
   - `algo=dr_mpc` ; minimal wrapper in `robot_sf/planner` that provides: robot + human state -> unicycle `[v,w]` (via environment state translation), using an external process call to DR-MPC Python (e.g. `scripts/online_continuous_task.py` wrapped as policy inference)
   - `algo=sicnav` ; same mapping, using upstream `simple_test.py` style API with `sicnav_policy.select_action()` .
3. Start with zero-copy injection: generator in `scripts/validation` or `output/repos` that converts Robot SF benchmark scenario into upstream `crowd_sim_plus` format and a stable interface for action back-projection to `unicycle_vw`.
4. Guard with `allow_testing_algorithms: true` and `include_in_paper=False` until performance and reproducibility are verified.

## Notes

* This assessment excludes direct training from scratch in the local repo (out-of-scope by request).
* Both methods are still “assessment-level” from Robot SF benchmark governance; “fallback execution” is not treated as a success condition.
* In next pass, generate a dedicated issue (`#771-drmpscnav-implementation`) for wrapper prototype and CI smoke-run scripts.

## Result Verification & Current Status

### What was done

1. Assessed DR-MPC and SICNav upstream assets for repository availability, license, and checkpoint status.
2. Mapped control and state interfaces for both methods to Robot SF categories:
   - robot pose/velocity/goal, human state adjacency, and MPC command semantics.
3. Classified family status:
   - DR-MPC: `assessment only` .
   - SICNav: `prototype only` .
4. Suggested minimal benchmark-safe integration boundary and guidance toward wrapper prototypes.
5. Updated repo docs:
   - `docs/context/issue_771_drmpscnav_assessment.md`

   - `docs/benchmark_planner_family_coverage.md` (added DR-MPC and SICNav rows).

### Verification checklist

* `gh issue view 771 --json ...` confirmed issue exists and matches done criteria.
* Branch created: `codex/771-assess-mpc-based-social-navigation-candidates`.
* Lint gate passed: `uv run ruff check .`
* Git commit and push performed.
* Draft PR created: https://github.com/ll7/robot_sf_ll7/pull/774.
* Issue label update: `benchmark`,  `priority: medium`.

### What is now working

* Robot SF docs now include a clear MPC arcade evaluation path for issue #771.
* Team can use the document as a direct source of truth for:
  + DR-MPC and SICNav viability, 
  + integration interface mapping, 
  + classification decision and next-step boundary.
* Benchmarks and reviewers can now identify the expected implementation ladder (external anchor → wrapper prototype → full integration).
