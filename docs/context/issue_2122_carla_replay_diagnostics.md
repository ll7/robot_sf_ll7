# Issue #2122 CARLA Replay Diagnostics 2026-06-02

Issue: [#2122](https://github.com/ll7/robot_sf_ll7/issues/2122)

Related:

- [Issue #1169 CARLA live replay](issue_1169_carla_live_replay.md)
- [Issue #1467 CARLA replay metrics](issue_1467_carla_replay_metrics.md)
- [Debug visualization](../debug_visualization.md)
- [Artifact evidence vocabulary](artifact_evidence_vocabulary.md)

## Outcome

Issue #2122 adds a CARLA-free diagnostics workflow for Robot-SF trace or episode JSON and CARLA
live-replay summary JSON. The workflow classifies what can be compared and what remains degraded,
not available, or unsupported without requiring a live CARLA runtime.

The user-facing command is:

```bash
uv run robot-sf-carla-replay-diagnostics \
  --robot-sf path/to/robot_sf_trace_or_episode.json \
  --carla path/to/carla_live_replay_summary.json \
  --output-dir output/carla_replay_diagnostics/<run-id>
```

It writes:

- `carla_replay_diagnostics.json`
- `carla_replay_diagnostics.md`
- `carla_capability_matrix.csv`
- `unsupported_semantics.csv`

## Schema And Status Boundary

The report schema is `carla-replay-diagnostics.v1`, exposed through
`load_carla_replay_diagnostics_schema` and the CARLA bridge schema catalog.

Rows use only these statuses:

- `available`: both inputs expose the field or capability needed for a bounded comparison.
- `degraded`: the CARLA summary mode/status or coordinate alignment prevents native comparison.
- `not_available`: the field is absent from one or both inputs.
- `unsupported`: the semantic surface is outside the current replay contract.

Required CARLA summary fields such as schema version, replay status, map metadata, and actor summary
become explicit `not_available` rows when omitted. Missing fields are not converted to zeros or
treated as successful replay evidence.

## Interpretation Boundary

The diagnostics workflow is not simulator-equivalence evidence by itself. It is a report-preparation
and artifact-pack helper for identifying which state, geometry, timing, actor, and metric surfaces
are comparable under the current CARLA replay contract.

In particular:

- fallback, degraded, failed, adapted, or not-available CARLA summaries remain caveats or
  exclusions;
- unsupported semantics such as sensor/perception replay and broad simulator equivalence remain
  explicit unsupported rows;
- local `output/` diagnostics are staging artifacts until promoted through a durable artifact pack
  or copied into a small tracked evidence bundle with provenance.

## Validation

CARLA-free targeted tests:

```bash
uv run pytest tests/carla_bridge/test_diagnostics.py \
  tests/carla_bridge/test_t0_export.py::test_schema_catalog_lists_all_carla_bridge_contracts \
  tests/carla_bridge/test_t0_export.py::test_schema_catalog_validates_against_schema \
  tests/carla_bridge/test_t0_export_cli.py::test_export_t0_cli_is_packaged_as_project_script -q
```

Result: `8 passed`.

Nearby CARLA bridge regression suite:

```bash
uv run pytest tests/carla_bridge -q
```

Result: `103 passed`.

No live CARLA runtime, long replay campaign, or benchmark run is required for this diagnostics
workflow.
