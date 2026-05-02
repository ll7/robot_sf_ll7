# Issue #792 Social-Jym Source-Harness Reproduction

Date: 2026-05-02
Related issues:
- <https://github.com/ll7/robot_sf_ll7/issues/792>
- <https://github.com/ll7/robot_sf_ll7/issues/729>

## Goal

Reproduce `TommasoVandermeer/social-jym` through its own source harness in an isolated side
environment before proposing any Robot SF wrapper or benchmark integration.

This is source-harness evidence only. It does not add `social-jym`, JAX, or its submodules to the
Robot SF runtime, and it does not count as Robot SF benchmark support.

## Source Revisions

Root repository:

- `https://github.com/TommasoVandermeer/social-jym.git`
- revision: `212ea7759f614ff646f39462c4f51dca67d01ed0`

Submodules resolved through HTTPS at the SHAs recorded by the root checkout:

- `JHSFM`: `acf2946aebf117c9426d7d914c29c003bd825250`
- `JSFM`: `13950d9ec003076394e705b6e0ae2a6b4d15f2b0`
- `JORCA`: `59ec886757ca9cf7dceb3ea7b8ec0b37f1dbe7b2`

The root `.gitmodules` still records SSH URLs, but the same repositories were reachable over HTTPS
for this reproduction.

## Isolated Environment

Side checkout and environment:

- source checkout: `output/repos/social-jym`
- virtual environment: `output/benchmarks/external/social_jym_issue792/.venv`
- summary artifacts:
  - `output/benchmarks/external/social_jym_issue792/source_harness_summary.json`
  - `output/benchmarks/external/social_jym_issue792/source_policy_step_summary.json`

Selected package versions:

- Python `3.12.8`
- `jax==0.10.0`
- `jaxlib==0.10.0`
- `dm-haiku==0.0.16`
- `optax==0.2.8`
- `numpy==2.4.4`
- `scipy==1.17.1`
- `pandas==3.0.2`
- `matplotlib==3.10.9`
- editable local packages: `socialjym==0.0.1`, `jhsfm==0.0.1`, `jsfm==0.0.1`,
  `jorca==0.0.1`

JAX reported that a GPU may be present but CUDA-enabled `jaxlib` was not installed, so the proof ran
on CPU.

## Commands Run

Resolve the upstream root revision:

```bash
git ls-remote https://github.com/TommasoVandermeer/social-jym.git HEAD
```

Clone the source checkout:

```bash
git clone https://github.com/TommasoVandermeer/social-jym.git output/repos/social-jym
git -C output/repos/social-jym checkout 212ea7759f614ff646f39462c4f51dca67d01ed0
```

Verify HTTPS submodule reachability:

```bash
git ls-remote https://github.com/TommasoVandermeer/JHSFM.git HEAD
git ls-remote https://github.com/TommasoVandermeer/JSFM.git HEAD
git ls-remote https://github.com/TommasoVandermeer/JORCA.git HEAD
```

Use HTTPS URLs in the ignored side checkout and hydrate submodules:

```bash
git -C output/repos/social-jym config submodule.JHSFM.url https://github.com/TommasoVandermeer/JHSFM.git
git -C output/repos/social-jym config submodule.JSFM.url https://github.com/TommasoVandermeer/JSFM.git
git -C output/repos/social-jym config submodule.JORCA.url https://github.com/TommasoVandermeer/JORCA.git
git -C output/repos/social-jym submodule update --init --recursive
```

Create the isolated side environment and install only into that environment:

```bash
uv venv output/benchmarks/external/social_jym_issue792/.venv
uv pip install \
  --python output/benchmarks/external/social_jym_issue792/.venv/bin/python \
  -e output/repos/social-jym/JHSFM \
  -e output/repos/social-jym/JSFM \
  -e output/repos/social-jym/JORCA \
  -e output/repos/social-jym
```

## Source-Harness Result

The source harness successfully imported the root package and submodules, constructed a minimal
`SocialNav` environment, reset one episode, and stepped the environment once with a zero holonomic
action.

Observed summary:

- environment: `SocialNav`
- scenario: `circular_crossing`
- human policy: `hsfm`
- observation shape: `[2, 6]`
- state shape: `[2, 6]`
- reward after zero-action step: `0.0`
- time after step: `0.25`
- step after step: `1`
- outcome after step: `{"nothing": true, "success": false, "failure": false, "timeout": false}`

The source-harness policy proof also initialized the upstream `SARL` policy with random source-model
parameters, produced one action, and stepped the same `SocialNav` environment once.

Observed policy-step summary:

- policy: `SARL`
- policy parameters: randomly initialized source model
- action: `[0.3826833963394165, 0.9238794445991516]`
- observation shape: `[2, 6]`
- reward: `0.0`
- time after step: `0.25`
- step after step: `1`
- outcome after step: `{"nothing": true, "success": false, "failure": false, "timeout": false}`

## Interpretation

Issue #729's SSH-submodule blocker is no longer a hard blocker for this pinned revision if the side
checkout rewrites submodule URLs to HTTPS. A minimal `social-jym` source-harness reset and policy
step is reproducible in an isolated environment on this machine.

This does not make `social-jym` integration-ready. The proof is intentionally narrow:

- it does not run a trained policy checkpoint;
- it does not run a full episode or upstream benchmark script;
- it does not adapt `social-jym` observations/actions to Robot SF;
- it does not add JAX dependencies to Robot SF;
- it does not provide benchmark-facing parity evidence.

## Recommendation

Recommendation: `wrapper-spike justified, benchmark integration not yet justified`

Follow-up issue: <https://github.com/ll7/robot_sf_ll7/issues/905>

The next issue should be a narrow wrapper feasibility spike for one source-harness-proven path:

- environment: `SocialNav`
- scenario family: start with `circular_crossing`
- human policy: `hsfm`
- robot action contract: holonomic source action mapped explicitly to Robot SF planner semantics
- policy target: source `SARL` only after deciding whether randomly initialized, trained, or
  heuristic/source-policy parameters are acceptable for the comparison question

Benchmark-facing support still requires a separate wrapper smoke test, action/observation parity
note, and fail-closed readiness status. Fallback or degraded execution must remain non-success under
`docs/context/issue_691_benchmark_fallback_policy.md`.
