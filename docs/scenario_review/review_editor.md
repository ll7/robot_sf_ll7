# SREV-17 offline review editor

The SREV-17 editor extends the SREV-16 panel model with three annotation
speeds, source-bound spatial references, and undoable storyboard edits. It is
an offline diagnostic component: it does not start a simulation, serve a
remote application, fetch media, or write the original scene/video/metric
sources.

## Launch

The fixture route is deterministic and needs only the repository environment:

```bash
uv run python -m robot_sf.render.review_editor \
  --input tests/fixtures/scenario_review/review_editor/request.json \
  --output output/scenario_review/srev-17-smoke \
  --base .
```

The output directory must be new. A complete run emits a
`review-editor.v1.json` model, a self-contained HTML shell that mounts the
dependency-free editor, the `components/review_editor/review_editor.js` module,
and a capability report. Durable browser saves require an injected BA-03/BA-05
callback; without one, edits remain explicitly transient.
Generated output belongs under ignored `output/`, not Git. The model is also
available without files through `build_editor_model(request, base=...)`.

## Annotation contract

`make_one_click_annotation` maps Normal/Suspicious/Bug/Unsure to the BA-03
classifications `normal`, `interesting_valid`, `planner_defect`, and `unclear`.
`make_quick_annotation` records a classification, tags, and a timestamp or
interval without requiring a cause, confidence, or measured evidence.
`make_full_annotation` keeps observed behaviour, measured evidence, hypothesis,
confidence, actors, references, and notes in their separate fields. A
`ReviewRecord(scope="full_episode", author_kind="human")` is the explicit
coverage receipt; triage, interval, detector, and agent rows do not create
human full-episode coverage.

Spatial references carry source execution/time and either verified world
geometry or image coordinates with source media/frame/crop identity. A changed
source hash is retained as `stale`; the editor never silently rebinds the
point. `snap_reference` accepts recorded actor, goal, waypoint, map, event,
and metric targets. Distance overlays require two recorded world points and
declared units; browser pixels are never treated as metres.

## Persistence and recovery

`AuditStoreAdapter` delegates to BA-03's canonical `AuditStore` and therefore
uses the durable NDJSON journal, SQLite projection recovery, operation IDs, and
caller-captured expected revisions. The `PersistenceAdapter` protocol is the
explicit seam for BA-05's future service; BA-05 is not merged here and SREV-17
does not invent a second store. `ReviewEditorSession` reports `pending`,
`saved`, or `error` autosave state. A stale revision preserves the local record
and includes the remote record and selection revision in `SaveConflict`; it is
never a silent overwrite. Storyboard edits use an auditable BA-03
`ActionRecord` carrying the versioned storyboard, with the same CAS contract.
The headless session serializes selection changes with the adapter call, so a
blocked synchronous write cannot commit a record after another thread changes
the captured selection.
Browser annotation/storyboard saves use injected atomic transactions and never
`localStorage`. A transaction must advertise `atomic: true`, return an
uncommitted proposal from `prepare(record, token)`, and implement a durable
compare-and-swap `commit(proposal, token)` using the immutable operation,
expected-revision, selection, source-identity, source-revision, and context
token. Direct callbacks are rejected: a delayed or buggy callback cannot gain a
durable write capability from the controller. Reload accepts only the explicit
`review-storyboard-edit.v1` schema, matching record/action/source identities and
revisions, a nonnegative non-tombstone record revision, and source-bounded
intervals.

`StoryboardEditor` validates source intervals, preserves captions and order,
supports undo/redo, and writes only to a caller-declared export destination.
Saving and loading use canonical JSON so a round trip is byte-stable.

The result is implementation/diagnostic evidence only. It is not benchmark
evidence, a causal diagnosis, or an automatic bug grade.
