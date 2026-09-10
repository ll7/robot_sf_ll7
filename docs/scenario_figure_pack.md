# Scenario analysis figure packs

Turn the case workbench's selected recorded episodes into readable, individually
exported scenario views. This is an offline presentation tool, not a simulator,
new relevance ranking, or evidence-admission mechanism.

**Claim boundary:** exact recorded episodes only. Selection does not estimate
failure prevalence, planner superiority, causal mechanisms, or deployment safety.
**Evidence status:** every generated pack and view sidecar is
`diagnostic-only`; `source_admission_status` separately records whether the
input package was `admitted` or `not_admitted`. A generated pack may be derived
from an admitted source package, but neither a successful export nor a checksum
promotes the underlying evidence.
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
or expected pedestrian identity is unavailable rather than a minimum over only
the conveniently known actors. Series receipts use `available`,
`partly_unavailable`, or `unavailable` and include the missing-sample count and
reason.
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

For review before author admission, explicitly choose the diagnostic pack
configuration:

```bash
uv run python -m robot_sf.benchmark.figures.scenario_pack \
  --package output/run-cases \
  --config configs/analysis/scenario_figure_pack_diagnostic.v1.json \
  --output output/run-figures-diagnostic
```

For an already source-verified, author-admitted workbench package, use
`configs/analysis/scenario_figure_pack.v1.json` instead. The admitted mode calls
both existing source-integrity and publication-admission checks and requires
canonical trace hash and complete coverage validation. Diagnostic mode skips
**only admission**, never source checksums or structural trace validation; when
canonical trace provenance is incomplete it records `structural-only` rather
than emitting an unverified source hash. Every figure visibly says
`DIAGNOSTIC ONLY - not author admitted`. This feature does not populate the
source-gate registry or perform admission on the operator's behalf.

Pass repeated `--case-id` arguments to restrict the existing portfolio. Its order
is preserved regardless of argument order. Every excluded case is recorded as
`not_requested` or `presentation_budget`; no case disappears silently. For a
smaller export, copy the JSON pack configuration and choose a subset of `views`
or `formats`. Unknown fields, invalid settings and duplicate requests are
refused.

## Render at the final printed size

Use `--figure-profile` when a downstream document or venue owns exact dimensions
and typography. A figure profile is a strict
`robot-sf-figure-profile.v1` JSON object with a stable ID, width in inches,
height ratio, requested font families, language, point sizes, line/marker sizes,
and raster DPI. For example, a dissertation-owned profile can be supplied from a
separate checkout:

```bash
uv run python -m robot_sf.benchmark.figures.scenario_pack \
  --package output/run-cases \
  --config configs/analysis/scenario_figure_pack_diagnostic.v1.json \
  --figure-profile ../diss/configs/figures/dissertation-full-width.v1.json \
  --output output/run-figures-dissertation
```

The renderer creates every figure at the profile's exact natural width instead
of relying on LaTeX or another consumer to downscale a generic 7-inch export.
When the option is omitted, the legacy `single` (3.4 by 6.2 inches) or `double`
(7.0 by 5.6 inches) scenario-view dimensions remain in effect, so existing
callers do not silently change layout.

The normalized profile is copied as `figure_profile.json`, hashed independently
from the pack configuration, and summarized in `manifest.json`. The summary
includes exact target width and height, requested font family list, and the
family Matplotlib actually resolved. It deliberately records no machine-local
font path. The profile is presentation intent, not source admission or scientific
authority.

## Views and presentation

| View | What it displays | Important boundary |
| --- | --- | --- |
| `trajectory` | Full robot and stable-identity pedestrian paths, start/end markers, recorded body footprints at the selected frame | Missing actor frames remain gaps; selected-frame footprint receipts identify missing expected actors; map loading requires matching bytes. |
| `snapshot` | Recorded state nearest the first event, otherwise minimum fully observed clearance, otherwise first frame | Reports the actual frame time and selection rule; missing expected actors make the footprint status partly unavailable; never interpolated or called a causal pivot. |
| `clearance` | Minimum robot-pedestrian disc-surface separation from recorded radii | Missing radii or any expected actor invalidate that sample; zero is not a recomputed benchmark collision label. |
| `speed` | Recorded applied linear command series | Missing commands remain unavailable, not zero or estimated motion; partial coverage is labeled. |
| `turn` | Recorded applied turn-rate series | Same absolute recorded times; no invented controls or dual axes; partial coverage is labeled. |

All modes use separate one-axis views, units from the shared publication-label
owner, non-color marker/line distinctions, and restrained legends. Trajectory
and snapshot views share trace-derived world limits that also contain recorded
body footprints. Separate cases retain their own scales: these are not
automatically comparable panels. Map absence is printed explicitly; no map,
perception cone, obstacle clearance or control dimension is fabricated.

## Portable output

`README.md` links to every exported view. PDF and SVG preserve vector paths;
PNG uses the figure profile's requested DPI. Each view has a provenance JSON
companion and an escaped caption fragment. `manifest.json` inventories every
artifact by relative path, byte count and SHA-256, and records
source/config/profile/producer digests, source trace identifiers, the Matplotlib
version, selection/omission receipts and panel availability. Admitted mode
requires canonical trace hash and complete coverage validation; diagnostic mode
labels incomplete trace provenance `structural-only` and does not emit it as a
verified source hash. Case IDs are hashed into path-safe filenames; raw traces
and local absolute map/source/font paths are not copied into the output.

Reproduction requires the exact source package (identified by its complete
inventory digest), its referenced map bytes where applicable, the generated
`config.json`, the generated `figure_profile.json`, the recorded requested case
IDs, and the same producer/dependency versions. The source package is not
embedded. No cross-platform byte-identical PDF/SVG claim is made: fonts,
backends and export metadata can differ. Review rendered pages at intended print
size and retain the final artifact byte hashes.

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
uv run pytest tests/benchmark/test_scenario_figure_pack.py \
  tests/benchmark/test_figure_profile.py -q
uv run ruff check robot_sf/benchmark/figures/scenario_pack.py \
  robot_sf/benchmark/figures/profile.py robot_sf/benchmark/figures/style.py \
  robot_sf/benchmark/figures/export.py \
  tests/benchmark/test_scenario_figure_pack.py tests/benchmark/test_figure_profile.py
uv run ruff format --check robot_sf/benchmark/figures/scenario_pack.py \
  robot_sf/benchmark/figures/profile.py robot_sf/benchmark/figures/style.py \
  robot_sf/benchmark/figures/export.py \
  tests/benchmark/test_scenario_figure_pack.py tests/benchmark/test_figure_profile.py
BASE_REF=origin/main scripts/dev/pr_ready_check.sh
```

The tests cover ordering, omission accounting, invalid traces, actor gaps and
complete expected actor sets, partial radius/control coverage with explicit
statuses, event/frame timing, budget refusals, final-size profiles, stable profile
digests, centralized telemetry labels, path safety, gate delegation, tampering,
interrupted exports and concurrent output appearance. The shared exporter
regression intentionally leaves a different pyplot figure current and proves
that the supplied figure is the one saved.

The implementation was exercised with synthetic traces in an isolated dependency
harness, with PDF/SVG/PNG export and downstream byte-pinned review staging. That
is component/render proof, **not** a native repository full-suite result or an
approved benchmark-source render. Before merge, run the native commands above,
render one approved package with its exact maps and intended figure profile,
inspect all views at intended print size, and obtain domain-aware review. Do not
turn a tooling pass into a new paper-facing claim.
