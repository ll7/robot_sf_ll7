# SNQI Output JSON Schema (Draft)

This document formalizes the structure of the JSON outputs produced by the SNQI optimization and recomputation tools.

Status: Draft (stability: minor changes possible). A jsonschema validator test will enforce these contracts for CI.

## Top-level structure
- summary: object (human-readable summary; non-stable formatting)
- results: object
- _metadata: object

## results object
Required keys (optimize and recompute share most fields):
- recommended_weights: object<string, number> — normalized per-weight in [0, 1]
- recommended_score: number — mean episodic SNQI at recommended weights
- sensitivity: object — method-dependent sensitivity metrics
- strategies: object — per-strategy results (recompute only)
- bootstrap: object — present when bootstrap enabled

### results.bootstrap.recommended_score
- samples: integer > 0 — number of bootstrap resamples
- mean_mean: number — mean of the resampled means
- std_mean: number — standard deviation of the resampled means
- ci: [lower: number, upper: number] — two-sided confidence interval bounds
- confidence_level: number in (0,1) — e.g., 0.95

## _metadata object
- schema_version: string (semver)
- generated_at: string (ISO8601)
- git_commit: string | null
- seed: integer | null
- provenance:
  - invocation: string — the full CLI command used
  - tool: string — e.g., robot_sf_bench snqi optimize|recompute
- timings: object — phase timings in seconds (optimize, recompute, io, bootstrap, etc.)

## Notes
- All numeric fields are finite numbers (no NaN/Inf); writer validates prior to dump.
- Weight keys are stable and drawn from WEIGHT_NAMES in code.
- Additional fields may be added in a backwards-compatible way (additive only).

## Future
- Consider publishing a machine-readable JSON Schema under docs/snqi-weight-tools/schema.json and validating via tests.
