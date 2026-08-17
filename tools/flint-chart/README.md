# Flint chart foundation

This directory defines the Robot-SF side of the Flint chart adoption contract
from Issue #6720. The foundation is deliberately analysis-only: it creates
deterministic candidate surface JSON and an atlas manifest, but it does not
render with Flint, replace the canonical Matplotlib/PGF/TikZ renderer, promote
an artifact, admit evidence, or edit a dissertation claim.

## Inputs and outputs

`surface-input-schema.v1.json` describes a surface input containing:

- a durable pinned source context (`release` or `replay`), full source commit,
  release/bundle identity, artifact-catalog hash, and input hashes;
- an explicit planner-by-scenario-family display population;
- canonical and candidate cells with value, denominator, exposure definition,
  exclusions, uncertainty, capability track, and evidence status; and
- the renderer policy that keeps Matplotlib/PGF/TikZ canonical and exact ties
  free of catalog-order ranks.

`scripts/tools/build_flint_chart_surface.py` compares the canonical and
candidate cells before producing `surface-schema.v1.json`. Any missing,
duplicate, dropped, reordered, provenance-invalid, or parity-drifting cell
fails closed. The output is marked `render_status: not_run`.

`scripts/tools/build_flint_chart_atlas_manifest.py` accepts one or more built
surface files and produces `atlas-manifest-schema.v1.json`. Release and replay
surfaces remain separate entries, including when they share a surface id.
Every input surface must already report complete population coverage and passed
canonical parity.

Example commands:

```bash
uv run python scripts/tools/build_flint_chart_surface.py \
  --input tests/fixtures/flint_chart/figure_7_6_release_input.json \
  --output output/flint/figure_7_6_release.surface.json

uv run python scripts/tools/build_flint_chart_atlas_manifest.py \
  --surface output/flint/figure_7_6_release.surface.json \
  --surface output/flint/figure_7_6_replay.surface.json \
  --output output/flint/atlas.manifest.json
```

The fixture inputs are synthetic/public-safe contract fixtures. They are not a
pinned Flint release or replay bundle and cannot support a dissertation,
benchmark, ranking, or scientific claim. Downstream promotion additionally
requires a real durable input bundle, deterministic Flint export, provenance
sidecars, Robot-SF figure QA, print-scale and accessibility checks, and an
explicit per-surface admission decision.
