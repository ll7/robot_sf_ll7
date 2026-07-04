# Issue #3287 Cross-Benchmark Limitations Template

Plain-language summary: this template records why a Robot SF versus external social-navigation
benchmark comparison is bounded evidence, not a direct simulator-equivalence claim.

## Scenario Mapping Quality

- Robot SF scenario IDs:
- External benchmark scenario IDs:
- Mapping method:
- Known geometry, route, pedestrian-script, and sensor differences:
- Mapping quality label: exact / approximate / diagnostic-only / unavailable

## Metric Denominator Differences

- Metrics compared:
- Robot SF denominator definition:
- External-suite denominator definition:
- Normalization or aggregation differences:
- Metrics that must stay suite-specific:

## Observation-Space Differences

- Robot SF observation contract:
- External-suite observation contract:
- Sensor range, visibility, latency, and noise differences:
- Wrapper or adapter behavior:

## Action-Space Differences

- Robot SF action contract:
- External-suite action contract:
- Projection, clipping, fallback, or unsupported action modes:
- Policy outputs that are not comparable without an adapter:

## Dynamics And Pedestrian-Model Differences

- Robot and pedestrian update rates:
- Collision geometry:
- Pedestrian model and interaction assumptions:
- Dynamics gaps that may explain performance differences:

## Unsupported Direct-Equivalence Claims

The comparison must not claim that scores are directly equivalent across suites unless the relevant
scenario, metric, observation, action, dynamics, and asset mappings have been validated. Approximate
mappings should be labeled as approximate and interpreted as diagnostic sim-to-sim evidence.

## Valid Bounded Comparison Statements

- The same named policy was evaluated under the recorded suite-specific wrappers.
- Metrics were reported with suite-specific denominators preserved.
- Observed differences are compatible with the listed simulator, wrapper, and asset limitations.
- The result is suitable for follow-up selection or diagnosis, not paper-facing equivalence without
  additional validated evidence.
