# Issue #1646 child — run annotation model (`run_annotation.v1`)

**Status:** schema/data deliverable for the analysis-workbench epic #1646. **Respects the epic's
queue note:** browser/Three.js UI stays blocked; the sanctioned next step is trace/schema/report
evidence — which this provides. No UI.

## What this is

`robot_sf/analysis/run_annotation.py` adds the annotation model the epic names: a way for humans and
agents to attach comments to important moments in a run, anchored to durable evidence (frame ranges,
event IDs, entities, a source artifact) and keeping **observation separate from hypothesis**. It
complements the existing renderer-neutral timeline (`simulation_timeline.v1`), which annotations
anchor to.

## Model (`run_annotation.v1`)

- `RunAnnotation` — `annotation_id`, `author` + `author_kind` (`human`/`agent`), `frame_start` ≤
  `frame_end`, a **canonical** `label` (`success` / `near_miss` / `collision_precursor` / `deadlock`
  / `social_force_artifact` / `planner_issue` / `policy_uncertainty`), a `statement_kind`
  (`observation`/`hypothesis`), `text`, `source_artifact` (provenance), and optional `event_ids` /
  `entity_ids`. Fails closed on contract violations.
- `AnnotationSet` — a validated collection over one source artifact (unique ids, consistent
  provenance) with `observations()` / `hypotheses()` accessors and an observation/hypothesis split in
  `to_dict()`.

## Scope boundary

Pure schema/data — no UI, no runs. Anchors to the existing `simulation_timeline.v1` artifact. The
inspector, agent-discussion workflow, and review-report generation remain separate (de-emphasized)
children.

## Tests

`tests/analysis/test_run_annotation.py` (11 tests): valid serialization with anchors, fail-closed
validation of every contract field, observation/hypothesis separability, duplicate-id rejection, and
source-artifact consistency.

## Related

- Epic: #1646. Renderer-neutral timeline: `scripts/analysis/export_trace_timeline_issue_1646.py`
  (`simulation_timeline.v1`).
