# Issue #907 Social-Jym SARL Wrapper Parity

Date: 2026-05-02
Related issues:
- <https://github.com/ll7/robot_sf_ll7/issues/907>
- <https://github.com/ll7/robot_sf_ll7/issues/905>
- <https://github.com/ll7/robot_sf_ll7/issues/792>

## Goal

Check whether the thin `social-jym` SARL wrapper from #905 builds source-faithful SARL inputs for a
controlled state, and quantify the semantic loss from projecting source holonomic actions into
Robot SF commands.

This is controlled adapter parity evidence only. It is not trained-policy provenance or benchmark
support.

## Probe Update

`scripts/tools/probe_social_jym_wrapper.py` now supports:

```bash
--mode parity
```

The parity mode:

1. Builds a controlled one-robot/one-human Robot SF SocNav observation.
2. Builds the matched source-shaped SARL observation and `info` dictionary.
3. Compares wrapper-built observation, `robot_goal`, and `time` against the matched source inputs.
4. Uses upstream `socialjym.policies.sarl.SARL.batch_compute_vnet_input` to compare the final
   reparameterized SARL VNet inputs.
5. Reports projection-loss cases for source holonomic actions mapped into Robot SF `(v, omega)`.

## Commands Run

Focused tests:

```bash
uv run pytest tests/tools/test_probe_social_jym_wrapper.py -q
```

Real optional parity probe with ephemeral dependencies:

```bash
uv run \
  --with-editable output/repos/social-jym/JHSFM \
  --with-editable output/repos/social-jym/JSFM \
  --with-editable output/repos/social-jym/JORCA \
  --with-editable output/repos/social-jym \
  python scripts/tools/probe_social_jym_wrapper.py \
  --mode parity \
  --repo-root output/repos/social-jym \
  --output-json output/benchmarks/external/social_jym_issue907/parity_probe.json \
  --output-markdown output/benchmarks/external/social_jym_issue907/parity_probe.md
```

## Result

The controlled input parity probe passed.

Recorded local artifacts:

- `output/benchmarks/external/social_jym_issue907/parity_probe.json`
- `output/benchmarks/external/social_jym_issue907/parity_probe.md`

Observed input parity:

- observation max absolute error: `0.0`
- robot-goal max absolute error: `0.0`
- source time absolute error: `0.0`
- upstream SARL VNet input max absolute error: `0.0`

This means the wrapper-built controlled observation and `info` dictionary match the source-shaped
inputs for the tested one-human holonomic state, including the upstream SARL reparameterized VNet
input.

## Projection Findings

The holonomic-to-unicycle projection remains lossy:

- forward source action `[1.0, 0.0]`: no speed loss;
- lateral source action `[0.0, 1.0]`: almost all instant linear speed is lost, command saturates
  angular velocity;
- reverse source action `[-1.0, 0.0]`: all instant linear speed is lost, command saturates negative
  angular velocity;
- diagonal source action `[1.0, 1.0]`: about half the instant speed is lost and angular velocity
  saturates.

The current projection is appropriate as a heading-safe smoke-path bridge, but it is not equivalent
to source holonomic dynamics.

## Recommendation

Recommendation: `input parity proven for one controlled state, benchmark integration still blocked`

Follow-up issue: <https://github.com/ll7/robot_sf_ll7/issues/909>

Randomly initialized SARL is sufficient for validating wrapper input construction and action-path
plumbing. It is not sufficient for benchmark-facing policy-quality claims.

Any later benchmark-facing `social-jym` issue still needs:

- trained upstream SARL checkpoint or explicit decision to use a source heuristic/random-policy
  baseline;
- multi-human and scenario-variant parity checks;
- projection-policy acceptance criteria or a holonomic Robot SF comparison surface;
- fail-closed readiness metadata under `docs/context/issue_691_benchmark_fallback_policy.md`.
