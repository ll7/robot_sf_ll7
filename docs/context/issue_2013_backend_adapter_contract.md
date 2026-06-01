# Issue #2013 Backend Adapter Contract

Date: 2026-06-01

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/2013>
- <https://github.com/ll7/robot_sf_ll7/issues/1646> — analysis-workbench epic
- <https://github.com/ll7/robot_sf_ll7/issues/1491> — CARLA native/aligned parity

## Scope

This note defines the minimum required contract for any alternate simulator backend
(e.g., MuJoCo, Webots, Gazebo, Isaac Sim, Habitat, or expanded CARLA) that replays
a Robot SF scenario or command trace. It is not a request to implement any backend;
it is the handoff contract for future backend spikes.

## Required Adapter Fields

Every backend adapter must declare:

| Field | Type | Description |
|---|---|---|
| `backend_name` | string | Unique identifier for the simulator backend |
| `scenario_input` | schema reference | Robot SF scenario or manifest, or a declared mapping |
| `command_input` | schema reference | Command/action trace or native plan input |
| `observation_output` | schema reference | Frame-level observation structure emitted by the backend |
| `trace_output` | schema reference | Output artifact schema, named as `simulation_trace_export.v1` or a clearly named derived diagnostic variant |
| `supported_metrics` | list of strings | Metric IDs the backend natively supports for benchmark comparison |
| `unsupported_semantics` | list of strings | Simulator semantics that cannot be reproduced and trigger fail-closed behavior |
| `claim_boundary` | string | Short statement of what evidence the backend output may or may not support |

## Relationship To Existing Trace And Review Artifacts

- An adapter must emit `simulation_trace_export.v1`-compatible output, or a clearly
  named derived variant (e.g., `backend_<name>_trace.v1`) when native fields cannot
  be mapped.
- Trace review reports under `docs/context/` or analysis-workbench fixtures must
  identify the emitting backend by `backend_name`.
- The adapter's `trace_output` must reference the source `simulation_trace_export.v1`
  schema and document every field that differs.

## Fail-Closed Behavior

The adapter must fail closed (non-zero exit, clear error message, no partial output
treated as valid) for:

- **Unsupported semantics**: any scenario or command element the backend cannot
  reproduce.
- **Fallback/degraded execution**: substitution of a different simulation behavior
  without explicit documentation and mode classification.
- **Missing assets**: required map, model, or fixture files.
- **Non-native replay modes**: adapted, aligned-with-warning, or degraded replay
  that does not match the native scenario contract.

## Status Classification

Every trace or replay run must report status fields using the vocabulary in
`docs/context/issue_691_benchmark_fallback_policy.md`:

- `execution_mode`: `native`, `adapter`, `mixed`, or `unknown`.
- `readiness_status`: `native`, `adapter`, `fallback`, or `degraded`.
- `availability_status`: `available`, `partial-failure`, `failed`, or `not_available`.

Fallback, degraded, failed, partial-failure, and not-available runs are diagnostic evidence only.
They are not benchmark success evidence and must be labeled as such in any report, context note,
or PR.

## Claim Boundary

Alternate backend output is **diagnostic** unless separately proven through
a dedicated benchmark campaign with explicit eligibility, comparison metrics, and
reproducibility constraints. The following are never benchmark-success evidence:

- fallback execution
- degraded execution
- adapted execution that changes scenario semantics
- output from backends with unsupported semantics that fall back silently
- replay modes with non-native execution, fallback/degraded readiness, or non-success availability

Any PR, report, or context note using alternate backend output must state whether
each run was native, adapter-backed, fallback, degraded, failed, or not available. If the backend
does not satisfy the native contract and report `available`, the output is a limitation, not success
evidence.

## Validation

```bash
# This is a docs-only contract. Validation is by inspection and link check.
test -f docs/context/issue_691_benchmark_fallback_policy.md
test -f docs/context/issue_1689_simulation_trace_export_schema.md
test -f docs/context/issue_1169_carla_live_replay.md
test -f docs/debug_visualization.md
```

## References

- `docs/context/issue_691_benchmark_fallback_policy.md` — Canonical fail-closed
  benchmark vocabulary and execution mode semantics.
- `docs/context/issue_1689_simulation_trace_export_schema.md` — The
  `simulation_trace_export.v1` schema that backends must consume or reference.
- `docs/context/issue_1169_carla_live_replay.md` — CARLA live replay boundaries
  that informed the fail-closed contract.
- `docs/debug_visualization.md` — Debug timeline export that accepts
  `simulation_trace_export.v1` artifacts.
