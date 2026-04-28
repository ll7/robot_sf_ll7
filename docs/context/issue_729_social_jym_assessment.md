# Issue 729 Social-Jym Assessment

Date: 2026-04-08
Related issues:
- `robot_sf_ll7#729` research: evaluate social-jym for Robot SF fit
- `robot_sf_ll7#792` research: reproduce social-jym source harness before integration
- `robot_sf_ll7#629` Planner zoo research prompt
- `robot_sf_ll7#601` CrowdNav family feasibility note

## Goal

Evaluate whether `TommasoVandermeer/social-jym` is a reusable integration candidate, a comparison
reference, or out of scope for Robot SF benchmark and reproducibility requirements.

Canonical upstream repository:
`https://github.com/TommasoVandermeer/social-jym`

Upstream revision inspected:
`212ea7759f614ff646f39462c4f51dca67d01ed0`

## Upstream Shape

`social-jym` is a JAX-first social-navigation environment and RL training stack. Its README frames
the project as a crowded mobile-robot training environment with human motion models, RL policies,
and vectorized JAX execution.

Relevant source surfaces at the inspected revision:

- `socialjym/envs/base_env.py`: base environment, scenario registry, human-motion-model registry,
  kinematic-model registry, and shared state update logic.
- `socialjym/envs/socialnav.py`: state-aware `SocialNav` environment for policies that consume
  human and robot state.
- `socialjym/envs/lasernav.py`: LiDAR-style `LaserNav` environment.
- `socialjym/policies/`: CADRL, SARL, SARL*, SARL-PPO, DWA, MPPI, DRA-MPPI, JESSI, DIR-SAFE, and
  end-to-end policy implementations.
- `JHSFM`, `JSFM`, and `JORCA` submodules: JAX human-motion-model dependencies for HSFM, SFM, and
  ORCA.

The upstream README and `BaseEnv` expose these overlapping benchmark ideas:

- human models: ORCA, SFM, HSFM,
- scenarios: circular crossing, parallel traffic, perpendicular traffic, robot crowding, delayed
  circular crossing, circular crossing with static obstacles, crowd navigation, corner traffic,
  door crossing, crowd chasing, and hybrid scenario,
- kinematics: holonomic and unicycle/differential-drive-style actions,
- policies: CADRL, SARL, SARL*, SARL-PPO, and additional local-planning or safety-oriented policy
  families.

License signal: the root repository contains an Apache-2.0 license file. This is a positive
provenance signal for studying or adapting ideas, but submodule provenance must still be checked
before any vendoring or wrapper work.

## Reproducibility Check

Commands attempted from this Robot SF worktree:

```bash
git ls-remote https://github.com/TommasoVandermeer/social-jym.git HEAD
git clone https://github.com/TommasoVandermeer/social-jym.git output/repos/social-jym
git -C output/repos/social-jym rev-parse HEAD
git -C output/repos/social-jym submodule status
git -C output/repos/social-jym submodule update --init --recursive
PYTHONPATH=output/repos/social-jym uv run python -c "import socialjym; print('import_ok')"
PYTHONPATH=output/repos/social-jym uv run python -c \
  "from socialjym.envs.socialnav import SocialNav; print('socialnav_import_ok')"
PYTHONPATH=output/repos/social-jym uv run --with jax --with jax_tqdm --with dm-haiku --with optax --with scipy --with pandas \
  python -c "from socialjym.envs.socialnav import SocialNav; print('socialnav_import_ok')"
```

Observed results:

- `git ls-remote` resolved upstream `HEAD` to `212ea7759f614ff646f39462c4f51dca67d01ed0`.
- The root repository cloned at that revision.
- Root package namespace import succeeded from the cloned checkout when it was placed on
  `PYTHONPATH`: `import_ok`.
- `SocialNav` import failed in the current Robot SF environment because JAX is not installed:
  `ModuleNotFoundError: No module named 'jax'`.
- After injecting the declared Python dependencies ephemerally with `uv run --with ...`, `SocialNav`
  import progressed further but failed on the missing human-motion-model submodule package:
  `ModuleNotFoundError: No module named 'jhsfm'`.
- The documented submodule path failed closed because `.gitmodules` uses SSH URLs such as
  `git@github.com:TommasoVandermeer/JHSFM.git`; this environment has no GitHub SSH key available
  for that path, so `git submodule update --init --recursive` aborted with
  `Permission denied (publickey)`.
- No root `requirements.txt`, `pyproject.toml`, or environment file was found beyond `setup.py`.

Interpretation:

- The upstream repository is reachable and the root package can be inspected locally.
- A meaningful environment import is not currently reproducible from the documented workflow in
  this Robot SF runtime because the submodule transport requires SSH access and the environment does
  not already contain the required JAX/submodule packages.
- This is a fail-closed reproduction result, not benchmark evidence.

## Robot SF Compatibility

Direct environment reuse is not a good fit today.

Robot SF environment construction is centered on `robot_sf.gym_env.environment_factory` and
Gymnasium-compatible reset/step signatures, with structured simulation configs and benchmark
artifact metadata. `social-jym` uses a functional JAX interface where `step` receives explicit
`state`, `info`, `action`, reset keys, and optional auto-reset controls. That is useful for high
throughput JAX rollouts, but it is not a drop-in Gymnasium environment for this repository's
factory path.

Planner-wrapper reuse is also adapter-heavy.

The Robot SF planner protocol expects `step(obs) -> {"vx", "vy"}` or `{"v", "omega"}` and explicit
state reset/configure hooks. `social-jym` policies generally operate inside source-harness state,
info, reward, network parameter, and JAX PRNG-key contracts. Some action semantics overlap with
Robot SF because upstream supports holonomic and unicycle actions, but the observation construction,
normalization, scenario bookkeeping, and source simulator state are not directly supplied by Robot
SF benchmark observations.

Scenario-level comparison is the best near-term use.

The scenario taxonomy and human-model list overlap with Robot SF's benchmark interests. They are
useful as a reference for future scenario framing, ablation design, or related-systems discussion.
However, they should not be presented as current in-repo benchmark support unless a future
source-harness reproduction and wrapper proof both pass.

## Benchmark And Provenance Risks

- Dependency risk: JAX, Haiku, Optax, JAX tqdm, pandas, SciPy, and three submodule packages create a
  separate runtime surface from the current Robot SF stack.
- Submodule transport risk: documented setup uses SSH submodule URLs, which failed in this
  environment before any source-harness run could be attempted.
- Adapter risk: source observations and policy state are not Robot SF planner observations.
- Benchmark-faithfulness risk: running a Robot SF fallback or local heuristic in place of missing
  `social-jym` components would violate `docs/context/issue_691_benchmark_fallback_policy.md`.
- Provenance risk: the root repo is Apache-2.0, but each submodule and any model artifacts should be
  checked before reuse or redistribution.
- Claim risk: `social-jym` contains CADRL/SARL-family and local-planner-like policies, but Robot SF
  must not cite this as implemented CADRL/SARL/SFM/ORCA family coverage without source-harness and
  wrapper parity proof.

## Recommendation

Recommendation: `follow-up research only`

`social-jym` is relevant as a comparison reference and possible future source-harness candidate, but
it is not a scoped integration issue yet. The next useful task is tracked separately in
`robot_sf_ll7#792` as a narrow source-harness reproduction spike that:

1. checks whether the JHSFM, JSFM, and JORCA submodules are reachable via HTTPS or another
   reproducible transport,
2. installs the root package and submodules in an isolated side environment,
3. imports and resets one minimal `SocialNav` or `LaserNav` episode,
4. runs one source-harness policy step without Robot SF adaptation,
5. records source-harness outputs and exact blockers.

Only after that should Robot SF consider a wrapper issue. The most plausible wrapper target would be
one policy/scenario path with native source-harness proof, not a broad `social-jym` family
integration.

Not recommended now:

- direct dependency addition to `pyproject.toml`,
- benchmark runner integration,
- local fallback behavior,
- claims that Robot SF currently supports `social-jym` policy families.

## Coverage Matrix Status

Add `social-jym` as `conceptually adjacent only` in
`docs/benchmark_planner_family_coverage.md`.

Safe claim after this assessment:

- Robot SF has assessed `social-jym` as a relevant JAX social-navigation reference with overlapping
  scenarios, human models, and policy families.

Unsafe claim after this assessment:

- Robot SF has a runnable `social-jym` benchmark baseline or integration path.
