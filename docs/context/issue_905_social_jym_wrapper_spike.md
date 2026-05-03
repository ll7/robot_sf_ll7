# Issue #905 Social-Jym Wrapper Spike

Date: 2026-05-02
Related issues:
- <https://github.com/ll7/robot_sf_ll7/issues/905>
- <https://github.com/ll7/robot_sf_ll7/issues/792>

## Goal

Test whether the source-harness-proven `social-jym` path can drive one Robot SF step through a
thin wrapper without adding JAX or `social-jym` to the main Robot SF runtime.

This is a wrapper feasibility spike only. It is not trained-policy evidence, source-policy parity
evidence, or Robot SF benchmark support.

## Probe Added

New optional probe:

- `scripts/tools/probe_social_jym_wrapper.py`
- focused tests: `tests/tools/test_probe_social_jym_wrapper.py`

The probe is dependency-isolated. It lazily imports `social-jym`, JAX, and upstream submodules only
when the caller provides them through an external or ephemeral environment.

The implementation:

1. Creates a Robot SF environment through `make_robot_env` with
   `ObservationMode.SOCNAV_STRUCT`.
2. Maps one Robot SF SocNav observation into the upstream SARL observation shape
   `(max_humans + 1, 6)`.
3. Provides the minimum source-style info needed by upstream SARL:
   `robot_goal` and `time`.
4. Uses upstream `socialjym.policies.sarl.SARL` with randomly initialized source-model parameters
   and `epsilon=1.0`.
5. Projects the source holonomic `(vx, vy)` action into a heading-safe Robot SF `(v, omega)`
   command.
6. Uses `PlannerActionAdapter` to step the real Robot SF environment once.

## Commands Run

Focused unit tests:

```bash
uv run pytest tests/tools/test_probe_social_jym_wrapper.py -q
```

Real optional probe with ephemeral dependencies and the ignored #792 source checkout:

```bash
uv run \
  --with-editable output/repos/social-jym/JHSFM \
  --with-editable output/repos/social-jym/JSFM \
  --with-editable output/repos/social-jym/JORCA \
  --with-editable output/repos/social-jym \
  python scripts/tools/probe_social_jym_wrapper.py \
  --repo-root output/repos/social-jym \
  --seed 7 \
  --max-steps 1 \
  --output-json output/benchmarks/external/social_jym_issue905/wrapper_probe.json \
  --output-markdown output/benchmarks/external/social_jym_issue905/wrapper_probe.md
```

The first real probe attempt failed closed with:

- `KeyError: 'time'`

That showed upstream SARL traces the reward path during JAX compilation even when random action
selection is forced with `epsilon=1.0`. The wrapper now provides source-style `info["time"]` in
addition to `info["robot_goal"]`.

## Result

The second real probe succeeded.

Recorded local artifacts:

- `output/benchmarks/external/social_jym_issue905/wrapper_probe.json`
- `output/benchmarks/external/social_jym_issue905/wrapper_probe.md`

Observed summary:

- verdict: `wrapper prototype viable`
- source environment: `socialjym.envs.socialnav.SocialNav`
- source policy: `socialjym.policies.sarl.SARL`
- source scenario: `circular_crossing`
- source humans policy: `hsfm`
- projection policy: `heading_safe_holonomic_xy_to_unicycle_vw`
- Robot SF steps executed: `1`
- latest source action xy: `[-0.10953568667173386, 0.26444247364997864]`
- latest Robot SF command vw: `[0.0, 2.0]`
- latest Robot SF action: `[0.0, 1.0]`
- latest heading error: `1.74375185840817`
- Robot SF observation keys: `goal`, `map`, `pedestrians`, `robot`, `sim`

## Contract Mismatches

The wrapper is feasible but still adapter-heavy:

- `social-jym` SARL consumes source-shaped absolute-state observations plus a source `info`
  dictionary; Robot SF provides structured nested observations with additional map/sim fields.
- The source SARL action is holonomic `(vx, vy)`, while the default Robot SF environment step uses
  action-space-specific commands via `PlannerActionAdapter`.
- The current projection is heading-safe and conservative, but it is not source-policy parity.
- The policy parameters are randomly initialized; no trained upstream checkpoint or model artifact
  lineage has been validated.
- The probe uses one active human slot for the narrow smoke path, so multi-human masking/parity is
  still unproven.
- JAX runs through ephemeral dependencies; the main Robot SF runtime still does not depend on
  `social-jym`.

## Recommendation

Recommendation: `narrow parity issue justified, benchmark integration not justified`

Follow-up issue: <https://github.com/ll7/robot_sf_ll7/issues/907>

The next step should be a parity-focused follow-up, not a benchmark runner integration:

- define exact action projection semantics for holonomic source actions into Robot SF commands;
- compare wrapper-built SARL inputs against source-harness SARL inputs for matched simple states;
- decide whether a randomly initialized source policy is sufficient for adapter parity, or whether
  a trained upstream checkpoint/artifact must be reproduced first;
- keep any benchmark-facing availability status non-success until parity and trained-policy
  provenance are proven.

Under `docs/context/issue_691_benchmark_fallback_policy.md`, this result remains exploratory
adapter evidence only. It must not be reported as benchmark success.
