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
for the recorded source. A source identity is a closed contract: it must carry
`source_kind`, `source_commit`, `execution_mode`, `readiness_status`, and
`availability_status`. `source_kind` is `fixture` or `recorded_results`;
`execution_mode` is one of `recorded_results_only`, `native`, `adapter`, or
`mixed`; and `source_commit` must be a 40-hex commit identifier. The fixture is
intentionally labelled `source_kind: fixture`, `execution_mode: recorded_results_only`,
`readiness_status: verified`, and `availability_status: available`.

When a request-level `SourceRef` declares `sha256` or `source_commit`, the
component compares those values with the bytes it actually reads and the
source identity in the source document. A mismatch fails closed. The source
must be a regular file no larger than 8 MiB (8,388,608 bytes); the size check
happens before source contents are read. Source, request, and optional config
JSON reject duplicate object names. Retained API and HTML text rejects NUL and
other C0/C1 control characters, as well as malformed Unicode surrogates.

The HTML artifact is a human-readable summary. The JSON artifact is authoritative
for complete per-condition measurement values, units, expected directions, source
identity, and provenance; consumers that need the full report contract must read
`experiment-comparison.json`.

## Interpretation rules

Results are grouped by `shared_parent_id`. Branches in one group are a single
`dependent_family_by_shared_parent_id`, not independent samples. Each condition retains its role,
outcome, measurement values and units, fidelity status, and activation status.
The report keeps `survived`, `falsified`, `inconclusive`, and `contradictory`
outcomes in the inventory and negative-findings ledger without ranking them or
converting them into a scientific claim. Summary counts carry a `count_units`
map: family counts are dependent-family records, outcome/condition counts are
condition records including controls and treatments, and negative-finding counts
are non-survived condition records. The report publishes no independent sample
count because no independence contract is supplied or validated.

An effect is `interpretable` only when the source has
`readiness_status=verified` and `availability_status=available`, and both
conditions have verified fidelity and activation and both values are present
with matching units. A missing or failed control blocks effect interpretation
for its whole family. Missing treatment measurements or prerequisites block only
that treatment's effect; the recorded outcome is still retained. A blocked
effect is not silently treated as zero.

Source `missing`, `fallback`, `degraded`, or `unavailable` readiness remains
visible for diagnostic inspection, but produces a `partial` component result,
marks the comparison `diagnostic_tainted`, and blocks every effect. The source's
declared execution mode is preserved separately from the component's
`recorded_results_only` execution mode. Unsupported readiness or availability
values and unrecognized source-identity keys fail closed.

The component returns `complete` only after both artifacts are written and the
source readiness/availability contract permits interpretation. A diagnostic
`partial` report may carry both artifacts, but it is not a verified comparison.
Unsupported requested capabilities and unsupported source versions return
`unavailable`; malformed input, invalid configuration, output collisions, and
write failures return `failed`. These non-complete statuses carry no output
artifacts except for the diagnostic `partial` case described above. The
component descriptor is available from
`component_descriptor()` and declares the supported request and result
versions.

## Evidence boundary

This is a diagnostic review surface. Source integrity and artifact digests do
not establish benchmark admission, causal effect, planner superiority,
physical validity, safety, population prevalence, or paper-facing evidence.
Use the source's own provenance and the repository's applicable evidence gates
before making any stronger claim.
