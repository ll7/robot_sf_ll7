# Issue 435: Map Coverage Flow

## Current Role

Issue #435 is the parent tracker for map coverage and map-quality sequencing. It should stay a
coordination surface rather than accumulating direct SVG, config, or parser changes. Concrete map
work belongs in one child issue and one PR per map family or validation repair.

## Flow State

- #328, real-world benchmark maps: open parent tracker. Current parent note:
  `docs/context/issue_328_real_world_map_parent.md`.
- #334, SocNavBench map import: open staged-import track. It remains separate from handcrafted maps
  because upstream asset licensing, batch selection, geometry conversion, and provenance need their
  own gate.
- #351, Inkscape template/plugin: closed after the template-only MVP path was selected and
  implemented.
- #352, committed map formatting: closed.
- #365, OSM obstacle holes: closed.
- #380, campus lake shape: open with draft PR #889.
- #388, self-intersecting OSM obstacles: closed with PR #514; follow-up #515 owns full compound
  obstacle semantics.

## Sequencing Rule

Use this parent to answer "what map work is next," not "how should this specific SVG be edited."
The next concrete child should be selected only when it has:

- a bounded map family or repair target,
- explicit file/config/test entry points,
- parser-facing validation criteria,
- benchmark relevance or quality rationale,
- and a validation command that can run locally without depending on untracked `output/` artifacts.

## Close-Out Boundary

Do not close #435 until the open child trackers are either merged, closed, or explicitly deferred.
The final close-out comment should link accepted child PRs, rejected/deferred map work, and the
benchmark evidence that shows the selected map coverage is usable.

## Validation For This Note

```bash
gh issue view 328 --json state,title
gh issue view 334 --json state,title
gh issue view 351 --json state,title,comments
gh issue view 352 --json state,title
gh issue view 365 --json state,title
gh issue view 380 --json state,title,comments
gh issue view 388 --json state,title,comments
uv run pytest tests/test_ai_prompt_surfaces.py -q
git diff --check
```
