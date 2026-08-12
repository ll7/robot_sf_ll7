# Provenance-first case workbench

The workbench turns an existing episode JSONL into a reviewable proposal package:

```text
original traces -> normalized campaign-result-store.v2 -> metrics/events
                 -> role-local Pareto proposal -> author admission overlay
                 -> Rerun review and reduced Matplotlib publication figure
```

The first command is deterministic and does not run the simulator:

```bash
robot_sf_bench analyze-cases \
  --config configs/analysis/case_workbench.v1.yaml \
  --result-store <episodes.jsonl-or-store> \
  --output <package> \
  --check-determinism
```

The command is intentionally source-gate aware. Without a receipt from the
reviewed source-restoration step it still writes an audit package, but emits
`publication/UNAVAILABLE.json` and does not create a publication figure. Once
the exact source package has been restored and its digest verified, pass the
receipt explicitly:

```bash
robot_sf_bench analyze-cases \
  --config configs/analysis/case_workbench.v1.yaml \
  --result-store <episodes.jsonl-or-store> \
  --output <package> \
  --source-gate-receipt <source-integrity-gate.json>
```

The receipt is accepted only when its approval ID and source digest occur in the
repository-controlled `configs/analysis/source_gate_registry.v1.json`. That
registry is intentionally empty in the tooling PR; restoring and approving the
RobotSF #6792/#6814 package is a separate evidence gate and must populate it in
the evidence work, not in this code merge.

`analysis_trace: all` is opt-in and additive to the legacy trace booleans. It
records an explicit `t=0` state and, when the simulator supplies the required
identity/geometry/control fields, stable actor IDs/radii, world-frame units,
controls, typed events, and provenance. Missing or positional-only identities
remain typed `unavailable` and cannot enter an evidence portfolio. The capture
path is data-only and must not alter actions or outcomes. Raw traces remain
outside Git; packages retain manifests and checksums.

The profile does not invent planner implementation commits, actor registries,
or missing control dimensions. Lightweight/non-map episodes and planners that
do not expose those receipts therefore carry an explicit unavailable coverage
record until their adapter supplies the fields; they are never silently treated
as analysis-ready.

The dedicated `configs/benchmarks/analysis_ready_full_campaign.yaml` overlay
is the only canonical opt-in. Camera-ready, smoke, and historical configs remain
unchanged until trace-storage overhead has been measured.

For an individual legacy run, pass the same profile with
`robot_sf_bench run ... --telemetry-config configs/benchmarks/analysis_ready_full_campaign.yaml`.

Eligibility fails closed for fallback/degraded rows, missing artifact hashes,
incomplete analysis telemetry, and incompatible comparison starts. Historical
v1 traces are adapted with typed `unavailable` fields rather than inferred
values. Machine selection is never author admission: edit the digest-bound
`admission_overlay.json` with a decision and rationale after review.
The `apply_admission_overlay` API verifies the proposal digest and keeps the
original machine portfolio beside the admitted/rejected/replaced result.

The canonical admission command verifies the package manifest, checksums, source
gate, and overlay before refreshing the package receipts:

```bash
robot_sf_bench admit-cases \
  --package <package> \
  --overlay <package>/admission_overlay.json
```

The default publication renderer accepts only an admitted package whose source
gate passed. Before admission, use the package's audit dossier and interactive
viewer for review, or request an explicitly diagnostic-only preview through the
private API flag; such a preview is not evidence.

The full audit dossier and interactive viewer are diagnostic artifacts. The
publication renderer is intentionally reduced:

```bash
uv run python scripts/analysis/render_case_publication.py \
  --package <package> --output <figure.preview.pdf>

uv run --with 'rerun-sdk==0.34.1' python scripts/tools/trace_viewer.py \
  --package <package> --case-id <case-id> --spawn
```

Package traces use the recorded robot and actor radii. Legacy bundles without
those fields show surface-clearance tracks as unavailable instead of applying a
default geometry assumption.

Doorway seed 113/114 remains `shared_prefix=false`; the package must not claim
a first divergence, causal pivot, or planner superiority. Source restoration
for RobotSF issues [#6792](https://github.com/ll7/robot_sf_ll7/issues/6792) and
[#6814](https://github.com/ll7/robot_sf_ll7/issues/6814) is a separate admission
gate before Chapter 7 integration through
[diss #698](https://github.com/ll7/diss/issues/698).
