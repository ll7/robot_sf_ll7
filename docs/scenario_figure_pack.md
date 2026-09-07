# Scenario analysis figure packs

Turn the case workbench's selected recorded episodes into readable, individually
exported scenario views. This is an offline presentation tool, not a simulator,
new relevance ranking, or evidence-admission mechanism.

**Claim boundary:** exact recorded episodes only. Selection does not estimate
failure prevalence, planner superiority, causal mechanisms, or deployment safety.
**Evidence status:** tooling with synthetic-fixture validation; a generated pack
is either `diagnostic` or derived from an already `admitted` source package.
Neither a successful export nor a checksum promotes the underlying evidence.
Missing geometry and telemetry remain explicit; native campaign validation is a
separate proof obligation.

## Ownership and scope

The existing [case workbench](case_workbench.md) owns discovery, role-local
selection, the complete proposal ledger, source trust and author admission.
`robot_sf/benchmark/figures/scenario_pack.py` owns only bounded composition and
transport packaging. It delegates trace series, events and verified map loading
to `case_publication_figure.py`, and uses the existing publication style,
provenance and export helpers. It does not supersede the reduced comparison
renderer, interactive viewer, episode replay bridge or scenario thumbnails.

Every case gets separate figures; the exporter never invents a comparison pair,
normalizes episode duration, interpolates a snapshot, or infers a causal pivot.
A missing actor frame breaks that actor's path. Missing applied controls are not
replaced by estimated velocity. A clearance sample with any missing body radius
is unavailable rather than a minimum over only the conveniently known actors.
Canonical benchmark metrics and collision labels are not rewritten.

## Generate a pack

First produce the existing case-workbench package from a completed, immutable
run. The denominator in a pack is the **workbench portfolio**, not the benchmark
population; ineligible candidates and discovery decisions remain in the source
workbench ledger.

```bash
robot_sf_bench analyze-cases \
  --config configs/analysis/case_workbench.v1.yaml \
  --result-store output/run/episodes.jsonl \
  --output output/run-cases \
  --check-determinism
```

For review before author admission, explicitly choose the diagnostic profile:

```bash
uv run python -m robot_sf.benchmark.figures.scenario_pack \
  --package output/run-cases \
  --config configs/analysis/scenario_figure_pack_diagnostic.v1.json \
  --output output/run-figures-diagnostic
```

For an already source-verified, author-admitted workbench package, use
`configs/analysis/scenario_figure_pack.v1.json` instead. The default profile calls
both existing source-integrity and publication-admission checks. Diagnostic mode
skips **only admission**, never source checksums or trace validation, and every
figure visibly says `DIAGNOSTIC ONLY - not author admitted`. This feature does not
populate the source-gate registry or perform admission on the operator's behalf.

Pass repeated `--case-id` arguments to restrict the existing portfolio. Its order
is preserved regardless of argument order. Every excluded case is recorded as
`not_requested` or `presentation_budget`; no case disappears silently. For a
smaller export, copy the JSON profile and choose a subset of `views` or `formats`.
Unknown fields, invalid settings and duplicate requests are refused.

## Views and presentation

| View | What it displays | Important boundary |
| --- | --- | --- |
| `trajectory` | Full robot and stable-identity pedestrian paths, start/end markers, recorded body footprints at the selected frame | Missing actor frames remain gaps; map loading requires matching bytes. |
| `snapshot` | Recorded state nearest the first event, otherwise minimum fully observed clearance, otherwise first frame | Reports the actual frame time and selection rule; never interpolated or called a causal pivot. |
| `clearance` | Minimum robot-pedestrian disc-surface separation from recorded radii | Missing radii invalidate that sample; zero is not a recomputed benchmark collision label. |
| `speed` | Recorded applied linear command series | Missing commands remain unavailable, not zero or estimated motion. |
| `turn` | Recorded applied turn-rate series | Same absolute recorded times; no invented controls or dual axes. |

The `single` (3.4-inch) and `double` (7-inch) profiles use separate one-axis views,
readable typography, units, non-color marker/line distinctions, and restrained
legends. Trajectory and snapshot views share trace-derived world limits that
also contain recorded body footprints. Separate cases retain their own scales:
these are not automatically comparable panels. Map absence is printed explicitly;
no map, perception cone, obstacle clearance or control dimension is fabricated.

## Portable output

`README.md` links to every exported view. PDF and SVG preserve vector paths;
PNG is exported at 300 dpi. Each view has a provenance JSON companion and an
escaped caption fragment. `manifest.json` inventories every artifact by relative
path, byte count and SHA-256, and records source/config/producer digests, source
trace identifiers, the Matplotlib version, selection/omission receipts and panel
availability. Case IDs are hashed into path-safe filenames; raw traces and local
absolute map/source paths are not copied into the output.

Reproduction requires the exact source package (identified by its complete
inventory digest), its referenced map bytes where applicable, the generated
`config.json`, the recorded requested case IDs, and the same producer/dependency
versions. The source package is not embedded. No cross-platform byte-identical
PDF/SVG claim is made: fonts, backends and export metadata can differ. Review
rendered pages and retain the final artifact byte hashes.

Outputs must be new directories outside the source package. A sibling lock
coordinates writers; exclusive directory creation prevents replacing another
writer's output. Publication finishes only when the manifest exists **and**
`.INCOMPLETE` is absent. The source inventory is checked before and after render.
An interrupted publish retains an incomplete target, never a false success
receipt. Do not remove an output or lock owned by a running process. Once its
owner is confirmed stopped, preserve the incomplete directory for diagnosis and
retry with a new output name.

The defaults limit cases, frames, stable actors and total recorded world points
(`max_points=250000` across selected cases). Exceeding a budget refuses export;
the tool does not silently downsample, hide actors or change selection scores.
Raw campaign traces and generated packs belong outside Git. Track only the small
configuration, required durable artifact references and accepted receipts.

## Validation and review

```bash
uv run pytest tests/benchmark/test_scenario_figure_pack.py -q
uv run ruff check robot_sf/benchmark/figures/scenario_pack.py \
  robot_sf/benchmark/figures/export.py tests/benchmark/test_scenario_figure_pack.py
uv run ruff format --check robot_sf/benchmark/figures/scenario_pack.py \
  robot_sf/benchmark/figures/export.py tests/benchmark/test_scenario_figure_pack.py
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

The tests cover ordering, omission accounting, invalid traces, actor gaps,
partial radius coverage, missing controls, event/frame timing, budget refusals,
layout, path safety, gate delegation, tampering, interrupted exports and concurrent
output appearance. The shared exporter regression intentionally leaves a different
pyplot figure current and proves that the supplied figure is the one saved.

The implementation was exercised with synthetic traces in an isolated dependency
harness, with PDF/SVG/PNG export and downstream byte-pinned review staging. That
is component/render proof, **not** a native repository full-suite result or an
approved benchmark-source render. Before merge, run the native commands above,
render one approved package with its exact maps, inspect all views at intended
print size, and obtain domain-aware review. Do not turn a tooling pass into a new
paper-facing claim.
