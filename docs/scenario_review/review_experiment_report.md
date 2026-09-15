# Recorded experiment comparison

The `srev25-experiment-report` component creates a small, offline JSON and HTML
comparison from recorded experiment results. It does not start a simulator,
require an executor, or turn a fixture into benchmark evidence.

## Run the fixture

From the repository root:

```bash
uv run python -m robot_sf.analysis_workbench.review_experiment_report \
  --input tests/fixtures/scenario_review/review_experiment_report/request.json \
  --config tests/fixtures/scenario_review/review_experiment_report/config.json \
  --output output/scenario_review/srev-25-smoke
```

The output directory must not already exist. The command writes:

- `experiment-comparison.json`, a versioned `experiment-comparison.v1` report;
- `experiment-comparison.html`, a dependency-free static view of the same comparison.

Each result envelope includes SHA-256 digests for both artifacts and provenance
for the recorded source. The fixture is intentionally labelled `source_kind:
fixture` and `execution_mode: recorded_results_only`.

## Interpretation rules

Results are grouped by `shared_parent_id`. Branches in one group are a single
`dependent_family`, not independent samples. Each condition retains its role,
outcome, measurement values and units, fidelity status, and activation status.
The report keeps `survived`, `falsified`, `inconclusive`, and `contradictory`
outcomes in the inventory and negative-findings ledger without ranking them or
converting them into a scientific claim.

An effect is `interpretable` only when both conditions have verified fidelity
and activation and both values are present with matching units. A missing or
failed control blocks effect interpretation for its whole family. Missing
treatment measurements or prerequisites block only that treatment's effect;
the recorded outcome is still retained. A blocked effect is not silently
treated as zero.

The component returns `complete` only after both artifacts are written.
Unsupported requested capabilities and unsupported source versions return
`unavailable`; malformed input, invalid configuration, output collisions, and
write failures return `failed`. These non-complete statuses carry no output
artifacts. The component descriptor is available from
`component_descriptor()` and declares the supported request and result
versions.

## Evidence boundary

This is a diagnostic review surface. Source integrity and artifact digests do
not establish benchmark admission, causal effect, planner superiority,
physical validity, safety, population prevalence, or paper-facing evidence.
Use the source's own provenance and the repository's applicable evidence gates
before making any stronger claim.
