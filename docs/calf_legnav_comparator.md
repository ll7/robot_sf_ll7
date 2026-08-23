# CALF/LegNav-inspired comparator diagnostic

This page documents a narrowly scoped, one-seed Robot SF smoke for asking whether
the existing policy-search trace can separate an ideal observation contract from a
perception-limited contract. It is diagnostic-only evidence. It does not reproduce
the external CALF policy, the LegNav simulator, a calibrated leg sensor, TurtleBot 4
hardware, or real-world deployment.

## Source and method boundary

The source paper, [Learning Social Robot Navigation By Sensing Human Legs](https://arxiv.org/abs/2607.27922),
describes CALF (Convolutional Attention for Leg Features), a leg-sensing policy evaluated
in the LegNav lightweight 2-D simulator and reported with a TurtleBot 4 deployment. The
Robot SF comparator records that method card for provenance, but imports none of its
checkpoint, sensor calibration, gait model, simulator, embodiment, or training recipe.
Temporal features, source action bounds, and Social Force/HSFM calibration are also
recorded as unavailable rather than inferred from the local PPO adapter.

The local question is smaller: with the same Robot SF PPO candidate, scenario, and seed,
what changes when the policy receives the fixture's ideal state versus a fixed-shape,
perception-limited observation? A result cannot support a CALF ranking, sensor-realism
claim, safety claim, universal planner claim, or zero-shot transfer claim.

## Reproducible fixture smoke

From the repository root, run:

```bash
uv run python scripts/benchmark/run_calf_legnav_comparator_issue_7318.py \
  --config configs/benchmarks/issue_7318_calf_legnav_comparator_smoke.yaml \
  --output-dir /tmp/issue-7318-calf-smoke
```

The config freezes candidate `ppo_issue791_best_v1`, scenario
`issue_2756_occluded_emergence`, seed `111`, horizon `12`, and a `1.5 m` personal-space
threshold. The perfect-perception condition disables only the fixture's first-visible
mask; the sensor-limited condition retains that mask and adds bounded position noise.
The runner executes both conditions through the existing policy-search step diagnostics,
then validates `summary.json` against
`robot_sf/benchmark/schemas/calf_legnav_comparator.v1.json`.
The YAML itself is checked against
`robot_sf/benchmark/schemas/calf_legnav_comparator_config.v1.json` before either
condition starts, including a strict finite-JSON check for numeric values.

Use `--dry-run` to inspect the two generated commands without executing the policy.
Generated traces and summaries belong under ignored `output/` or a disposable temporary
directory; they are not durable benchmark evidence by themselves.

## Metric mapping

The report keeps local and proxy semantics explicit:

| Report field | Local source | Mapping |
| --- | --- | --- |
| Success and collision rates | trace outcome and collision flags | exact local |
| Minimum human distance and personal-space compliance | simulator ground-truth distance | qualified proxy |
| Angular jerk | second differences of commanded angular action | qualified proxy |
| Action smoothness | successive two-channel action difference | exact local |
| Timeout rate | trace termination fields | exact local |

For each executed action, the distance metrics use the conservative minimum of
the available pre-step and post-step ground-truth distances. This keeps the
shared state between adjacent rows from being counted twice while preserving
within-step clearance violations. Outcome flags must be JSON booleans; malformed
flags are reported as unavailable rather than coerced into results. Trace rows
must also have contiguous step identities, a complete fixed horizon or explicit
terminal `done_info`, recognized execution-mode provenance, non-negative integer
observed-actor counts, and finite non-negative distance values. Violations block
the condition or materialize a schema-valid blocked handoff.

One paired episode has no uncertainty estimate. Missing observations, runner errors,
fallback/degraded execution, or an unrecognized observation contract produce `blocked`
or `unavailable` fields rather than fabricated zeros.
An unavailable or mislabelled observation contract also blocks that condition's metric
values and paired deltas, so a failed contrast cannot be read as a valid comparison.

## Evidence and next proof

The fixture smoke is useful for validating provenance, observation separation, and metric
plumbing. It is not a research result about CALF or LegNav. A stronger research result
requires a preregistered multi-seed, multi-scenario comparison with explicit sensor and
embodiment assumptions, followed by review of whether a real manifest and source-policy
artifacts are actually available. Until then, keep this comparator in the diagnostic
lane and do not promote its output to a benchmark or paper-facing claim.
