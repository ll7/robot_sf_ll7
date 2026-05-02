# Issue 328: Real-World Map Parent Tracker

## Current Role

Issue #328 is the parent tracker for real-world benchmark map coverage. It should not own direct
map implementation. Concrete map assets, configs, tests, and validation evidence belong in child
issues with one reviewable map family per PR.

## Child Issue State

- #547, urban crossing: closed. This child owns the handcrafted urban crossing map contract;
  current repository surfaces include `maps/svg_maps/classic_urban_crossing.svg` and
  `configs/scenarios/archetypes/classic_urban_crossing.yaml`.
- #548, bottleneck: closed. This child owns the handcrafted bottleneck map contract; current
  repository surfaces include `maps/svg_maps/classic_realworld_bottleneck.svg`,
  `configs/scenarios/archetypes/classic_realworld_bottleneck.yaml`, and
  `tests/maps/test_realworld_bottleneck_map.py`.
- #549, station/platform: open. Draft PR #733 added the initial station/platform map, and draft
  PR #913 adds an exploratory station-platform scenario pack on top of that map.
- #334, SocNavBench import: separate staged-import track. It should not be folded into #328 child
  map PRs because external asset provenance, licensing, geometry conversion, and batch selection
  need their own acceptance gate.

## Shared Child-PR Contract

Each concrete map child PR should include:

- one bounded map family or explicit import batch,
- parser-loadable SVG assets under `maps/svg_maps/`,
- config-first scenario or benchmark wiring under `configs/`,
- route, zone, and label validation through the normal parser path,
- a benchmark smoke or equivalent scenario-loading proof,
- a short benchmark-coverage rationale in the child issue, PR, or context note.

## Parent Close-Out Boundary

Do not close #328 until the selected child work is merged or deliberately deferred with explicit
reasons. The final close-out comment should link the merged child PRs, summarize the accepted map
coverage, and list any deferred map families or SocNavBench import work.

## Validation For This Note

```bash
gh issue view 547 --json state,title
gh issue view 548 --json state,title
gh issue view 549 --json state,title,comments
gh issue view 334 --json state,title,comments
test -f maps/svg_maps/classic_urban_crossing.svg
test -f maps/svg_maps/classic_realworld_bottleneck.svg
uv run pytest tests/test_ai_prompt_surfaces.py -q
git diff --check
```
