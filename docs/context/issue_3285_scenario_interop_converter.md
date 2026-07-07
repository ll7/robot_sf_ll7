# Issue #3285 Dry-Run Scenario Interop Converter

Plain-language summary: Robot SF can emit deterministic local handoff artifacts for
SocNavBench and HuNavSim scenario conversion, but it still cannot claim runnable external
benchmark export until the external assets and adapters are staged.

Status: local, asset-free implementation slice (`smoke evidence`). This is not benchmark
evidence and does not claim cross-benchmark validity.

Related:
- GitHub issue: https://github.com/ll7/robot_sf_ll7/issues/3285
- Conceptual mapping seed: [`issue_2928_socnavbench_hunavsim_metric_correspondence.md`](issue_2928_socnavbench_hunavsim_metric_correspondence.md)
- Source-side contract: `robot_sf/benchmark/scenario_contract.py`,
  `robot_sf/benchmark/schema/scenarios.schema.json`
- External-asset prerequisites block external run: #1456, #1498, #2414, #1134

## Slice Delivers

The local converter emits deterministic, schema-validated intermediate representation (IR)
Robot SF scenario-matrix entries, explicit unsupported-field reports, fail-closed target
compatibility reports, export manifests, target-shaped previews, and local-only prerequisite
reports. It does not download assets and does not emit runnable SocNavBench or HuNavSim files.

- Module: `robot_sf/benchmark/scenario_interop.py`
- CLI: `scripts/tools/convert_scenario_interop.py`
- Tests: `tests/benchmark/test_scenario_interop.py`,
  `tests/benchmark/test_scenario_interop_target_compatibility.py`
- IR schema: `robot_sf/benchmark/schemas/scenario_interop_ir.v1.json`
- Target handoff schemas:
  `robot_sf/benchmark/schemas/scenario_interop_target_compatibility.v1.json`,
  `robot_sf/benchmark/schemas/scenario_interop_target_export_manifest.v1.json`,
  `robot_sf/benchmark/schemas/scenario_interop_target_export_preview.v1.json`,
  `robot_sf/benchmark/schemas/scenario_interop_target_prerequisite_report.v1.json`

## Mapping Contract

| IR section | Source fields | Notes |
| --- | --- | --- |
| `provenance` | `id`, `scenario_id`, `name`, source file, source fields | Stable source trace. |
| `geometry` | `obstacle`, `map_file` | Coarse target-neutral environment type. |
| `environment` | `density`, `flow`, `groups`, `speed_var`, `goal_topology`, `robot_context` | Target-neutral environment semantics. |
| `agents` | `single_pedestrians[]` | Start/goal points, preferred speed, role, role target, wait points. |
| `timing` | `repeats`, `seeds` | Preserves repeat count and explicit seed set when present. |
| `unsupported_fields` | everything else | Every unmapped top-level source key is reported with a reason. |

## Dry-Run Commands

```bash
# Print IR plus per-scenario validity summary.
uv run python scripts/tools/convert_scenario_interop.py \
  --matrix configs/baselines/example_matrix.yaml

# Write one <scenario_id>.ir.json per scenario.
uv run python scripts/tools/convert_scenario_interop.py \
  --matrix configs/baselines/example_matrix.yaml \
  --out-dir output/interop_ir

# Write fail-closed target export manifests.
uv run python scripts/tools/convert_scenario_interop.py \
  --matrix configs/baselines/example_matrix.yaml \
  --target socnavbench \
  --target-out-dir output/issue_3285_target_export_manifest_smoke

# Write fail-closed target export previews.
uv run python scripts/tools/convert_scenario_interop.py \
  --matrix configs/baselines/example_matrix.yaml \
  --target socnavbench \
  --target-preview-out-dir output/issue_3285_target_export_preview_smoke

# Write local-only prerequisite reports for runnable export readiness.
uv run python scripts/tools/convert_scenario_interop.py \
  --matrix configs/baselines/example_matrix.yaml \
  --target socnavbench \
  --target-prerequisite-out-dir output/issue_3285_target_prerequisite_smoke
```

The CLI prints IR to stdout only when no output directory is requested. It always writes a
`{"dry_run_summary": [...]}` block to stderr and exits non-zero when scenario IR fails schema
validation.

## Claim Boundary

- Dry-run IR validation, target compatibility, export manifest, export preview, and target
  prerequisite reports make no claim about simulator equivalence, planner transferability,
  benchmark-score parity, runnable external export, or paper/dissertation evidence.
- Real runnable SocNavBench/HuNavSim scenario asset export remains blocked on staged external
  assets/adapters (#1456, #1498, #2414, #1134).

## Closure Audit Integration 2026-07-07

Merged evidence through PR #4718 satisfies the original asset-free acceptance criteria:

| Acceptance criterion | Evidence |
| --- | --- |
| Converter output is schema-validated for deterministic fixture scenarios. | PR #3735 added the IR converter, `scenario_interop_ir.v1.json`, deterministic serialization tests, and schema validation. |
| Unsupported fields are reported explicitly instead of silently dropped. | PR #3735 records unmapped top-level source fields in `unsupported_fields`; PR #4562 and PR #4711 propagate blockers into target compatibility and preview artifacts. |
| Smoke dry-run command is documented. | PR #3735 documented the IR CLI; PR #4673 and PR #4711 extended the note with manifest and preview dry-run commands. |
| External asset prerequisites required for full run are linked. | The issue thread and this note keep #1456, #1498, #2414, and #1134 as full-run blockers. |

The current integration slice adds a schema-validated target prerequisite report so the remaining
external blockers become one machine-readable local preflight artifact. It does not change the
closure boundary: runnable SocNavBench/HuNavSim export remains intentionally blocked until external
assets and adapters are staged. The next empirical action is an asset-backed adapter smoke after a
prerequisite issue provides staged target inputs.
