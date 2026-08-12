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

`analysis_trace: all` is opt-in and additive to the legacy trace booleans. It
records an explicit `t=0` state, stable actor IDs/radii, world-frame units,
controls, typed events, and provenance. The capture path is data-only and must
not alter actions or outcomes. Raw traces remain outside Git; packages retain
manifests and checksums.

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

The full audit dossier and interactive viewer are diagnostic artifacts. The
publication renderer is intentionally reduced:

```bash
uv run python scripts/analysis/render_case_publication.py \
  --package <package> --output <figure.preview.pdf>

uv run --with 'rerun-sdk==0.34.1' python scripts/tools/trace_viewer.py \
  --package <package> --case-id <case-id> --spawn
```

Doorway seed 113/114 remains `shared_prefix=false`; the package must not claim
a first divergence, causal pivot, or planner superiority. Source restoration
for RobotSF issues [#6792](https://github.com/ll7/robot_sf_ll7/issues/6792) and
[#6814](https://github.com/ll7/robot_sf_ll7/issues/6814) is a separate admission
gate before Chapter 7 integration through
[diss #698](https://github.com/ll7/diss/issues/698).
