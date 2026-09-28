# BA-06 fixture-backed audit workbench slice

This page documents the small, offline UI slice for Benchmark Auditor (BA-06)
issue #9489. It is diagnostic UI-wiring evidence only: the fixture facade is
not the BA-05 service, does not rank a campaign, and does not replace the
BA-03 audit store or the scenario-review (SREV-17) editor.

## Try the fixture artifact

Generate a disposable artifact directory with repository-local JavaScript
assets:

```bash
uv run python -c 'from pathlib import Path; from robot_sf.render.audit_workbench import write_fixture_workbench; write_fixture_workbench(Path("output/audit-workbench-fixture"))'
```

Serve the disposable directory on loopback, then open
`http://127.0.0.1:8765/audit-workbench.v1.html` in a local browser:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory output/audit-workbench-fixture
```

Stop the server with Ctrl-C when finished. Direct `file://` opening blocks
the page's local JavaScript module imports in Chrome; loopback serving keeps
all assets local and does not contact an external service. The fixture
demonstrates this bounded flow:

1. `Queue Next` selects a normal-control episode and shows its cursor,
   selection reasons, metric units, and declared media presentation timestamps (PTS).
2. The existing SREV-16 scene, video, and metric panels provide playback,
   step and seek controls. Their local simulation-time cursor stays aligned
   with the SREV-17 annotation editor, including during playback; moving it
   does not advance the queue selection revision on every frame.
3. The existing SREV-17 editor creates a quick annotation. Its atomic save
   transaction sends the record through the injected facade and exposes
   pending/saved/error autosave state.
4. `Persist finding` updates the visible coverage receipt and the next queue
   identity, then `Reopen` restores the saved annotation/finding state.
5. The second packet has no recording. The scene/metric context stays
   reviewable while the media pane reports `recording_not_present` explicitly.

The page uses only local loopback requests and marks `native: false` and
`evidence_status: diagnostic_only` in the generated model. Uneven declared
media PTS values are copied from the fixture; the UI never derives timestamps
from frame rate or duration.

## Integration boundary

The current helper is a fixture-backed adapter while BA-05 is being finalized.
The canonical SREV-15 `review_workbench` request/launch path now mounts this
fixture in its existing `panels` extension slot and copies the shell plus the
SREV-16/SREV-17 browser modules into the requested output directory. The
extension remains explicitly `diagnostic_fixture` with `native: false` and
`evidence_status: diagnostic_only`; it is not a second renderer entry point.
The remaining integration seam is to replace `FixtureAuditService` with the
accepted BA-05 facade. The shared cursor is currently local UI state;
durable BA-05 context updates at action boundaries still need integration and
proof. Native retained episodes, materialization status,
Codex/MCP activity, related-case search, and BA-03 durable receipts are
explicit follow-ups; this slice does not claim those capabilities.
