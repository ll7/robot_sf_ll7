# Audit episode materialization

`robot_sf.analysis_workbench.audit_materialize` is the bounded source-first
slice of BA-05 (Benchmark Auditor lazy materialization). It provides diagnostic
artifacts for one already-selected episode. Its generic materializer does not
start a simulator or rerun a planner. A separate, explicitly trusted native
adapter may regenerate one admitted state-free `simple_policy` input for a
derived diagnostic view; neither route creates benchmark evidence.

## Resolution order

`materialize_episode` follows one fail-closed order:

1. Verify an explicitly declared original recording under `source_root`. The
   declaration must include a relative URI, SHA-256 digest, and supported
   format: `video/mp4` or `simulation_trace_export.v1`. MP4 must decode at
   least one video frame with the local bounded FFmpeg decoder; an opaque
   byte sequence or a format label alone is never enough. If the decoder is
   unavailable, original video admission fails closed. The source is
   read and rechecked through protected no-follow descriptors, and the
   normalized URI/digest receipt identifies the bytes verified at admission;
   bytes are not copied or relabeled. The returned source URI remains mutable,
   so consumers requiring immutability must revalidate it or use an immutable
   source store.
2. If the original is unavailable, validate and render an explicitly retained
   `simulation_trace_export.v1` payload through the canonical SREV-09 scene
   renderer ([`review_scene.md`](review_scene.md)). A retained native
   `analysis-trace.v1` from the canonical runner is digest- and
   selection-checked, including its own episode ID and seed. Older native
   traces without that binding fail closed. If the selected row claims an
   execution ID, the trace must carry the same ID. The one exception is a
   BA-01 scanner-added default `execution_id = episode_id`, marked in the
   scanned row as `_audit_scan_identity_defaults`; it is not a historical
   execution claim. An explicit execution ID remains binding even when it
   equals the episode ID. Inline `retained_trace` and
   digest-verified `retained_trace_source` paths are supported; a native trace
   file or a runner-record JSON wrapper around it misdeclared as an original
   recording is not accepted as original. Conflicting planner aliases on the
   selected episode also fail closed. JSON analysis artifacts, including
   nested, malformed, or alternatively encoded wrappers, do not satisfy the
   MP4 decoder boundary and cannot be certified as original video.
   Explicit selected identity claims must likewise match a field in the trace
   (or its artifact digest), otherwise the projection is unavailable. BA-01
   marks inherited campaign ID and campaign-file `source_digest` defaults in
   the same scanner-owned metadata. A standalone runner trace may omit those
   wrapper identities, but a trace that declares them must match; an explicit
   source-row claim is never treated as a scanner default. The campaign-file
   digest is not the native trace's `artifact_sha256`, which is checked
   independently. Campaign identity aliases (`campaign_id`, `study_id`, and
   `campaign`, including `config.campaign`) are resolved consistently before
   a selected row is projected. A conflicting identity declared by a retained
   trace makes materialization unavailable even when that trace has a valid
   self-digest. The direct `materialize_episode` helper ignores
   `_audit_scan_identity_defaults` inside its input row; only `AuditService`
   may pass scanner-derived defaults through the separate trusted argument.
   Untrusted campaign sources must enter through `scan_campaign` and
   `AuditService`.
   The verified trace is then projected only to recorded time, robot pose, and
   pedestrian positions for the replay figure renderer. This path does not
   invent missing planner commands, stable actor IDs, or a video stream.
3. If no trace is retained, validate and render explicit retained replay
   states through the existing replay figure renderer.
4. If no retained state is available, return `unavailable`. Rows carrying
   exact-input execution information return
   `exact_input_execution_deferred`; the generic materializer still never starts
   a runner or substitutes another policy.
5. `AuditService.materialize_selected` may explicitly hand that deferred row to
   the launcher-owned native adapter. This is a separate diagnostic execution,
   not a continuation of the generic materializer. It is admitted only when
   the launcher has supplied a digest-verified native source bundle with a
   closed `simple_policy` `runner_input`, and the selected row has no recording,
   retained trace, replay states, checkpoint, or stateful planner input. The
   source bundle—not row fields—supplies the executable inputs.

Convenience fields such as `video_path` are not source declarations. A path is
used only when it appears in `recording`, `original_recording`,
`original_trace`, `retained_trace_source`, or `retained_trace_recording`.
Serialized episode identities are admitted from their canonical top-level and
nested `source`/`source_identity` envelopes. Conflicting aliases are rejected;
an identity declared only by an original recording is not treated as a binding
to the selected row.

## Result boundary

Every result has `diagnostic_only: true` in `to_dict()` and exposes an explicit
classification:

- `historical_original`: a declared source was digest-verified at admission.
  Fidelity is `verified`; no renderer is invoked. The receipt is a mutable
  source-path reference, not an immutable byte handle.
- `derived_render`: retained state was rendered. Fidelity is
  `unverifiable`; provenance records the retained-state digest, renderer, and
  `simulation_advanced: false`. A native analysis-trace projection reports
  `native_retained_trace_projected` in diagnostics. Its `source_digest` and
  `native_trace_digest` identify the canonical trace payload, while
  `retained_state_digest` identifies the projected states; path-backed traces
  also report `retained_source_file_digest` for the verified file bytes. Its
trajectory image is a derived view, never the historical recording. An
exact-input regeneration is also `derived_render`, with
`simulation_executed: true` and a fresh execution ID linked to the historical
episode/execution ID. Its manifest records the admitted source commit, closed
runner-input/config identity, initial-state digest, environment identity,
bounded timeout, source-admission receipt, and a fidelity classification of
`verified`, `diverged`, or `unverifiable`. Only the newly generated trace is
rendered. Historical metrics, trace steps, and telemetry are never copied into
the artifact; missing historical telemetry therefore remains `unverifiable`,
not fabricated.
- `unavailable`: source proof or retained state was insufficient. The reason
  and bounded diagnostics explain the next missing proof.

Derived output is published under `output_root` with a deterministic cache key
that includes episode identity, source digest, renderer version, and render
configuration. Existing output directories are never overwritten. Artifact
URIs in a materialization result are relative to that result's output
directory. Trace rendering uses private staging, then copies each staged regular
file from a retained no-follow descriptor into a fresh inode, fsyncs it, and
publishes it with Linux libc `renameat2(RENAME_NOREPLACE)` into the reserved
directory; replay uses the same publication model. This avoids both renderer
hardlink aliases and overwrite races. Source and renderer staging is created
below the retained output-root descriptor on the same filesystem, never by
selecting `/tmp` or a mutable output-root parent pathname. Root/parent/output
replacement is checked before manifest publication, and the manifest is
re-opened no-follow and digest-verified after the writer returns, so a visible
rename or symlink cannot redirect renderer bytes or the published manifest
outside the admitted `output_root`. If the Linux no-replace primitive is
unavailable, publication fails closed rather than using an overwrite-prone
fallback. If publication fails, or a renderer returns no non-empty regular
artifact, partial files remain in the admitted directory for diagnosis but no
manifest is published; retry with a fresh output directory.

## Deliberate limits

`AuditService.materialize_selected` now binds this adapter to the authenticated
session's selected episode and source revision. It requires the configured
source root and caller-specified output root to be permitted by the session
policy, then hands the materializer a descriptor-bound output-root capability.
The capability retains the selected policy-root identity and creates a missing
output root only through no-follow descriptor-relative operations. A root
replacement between the policy check and materialization is therefore denied
or fails closed; the service never creates renderer output through the
replacement pathname.
The same handoff retains an identity-checked descriptor for the configured
source root, so a source-root replacement cannot redirect original or retained
trace reads after policy admission. If cancellation commits after the final
service check and adapter admission, the adapter may finish its diagnostic
render; the outer operation is still recorded as `cancelled`, returns no
successful value, keeps the one-unit charge, and does not promote the
diagnostic artifacts to evidence. The kill switch prevents later agent
operations, while already-started diagnostic files remain subject to normal
operator cleanup.
The service records an operation before adapter work and charges one
conservative compute unit. That unit is an admission/accounting charge, not
measured CPU time. An interrupted operation stays inflight and is not
automatically re-rendered; replay returns the durable receipt without a media
value. The adapter uses the source/context snapshot admitted before the
charge; a later source-revision mutation is detected on replay rather than
silently treated as a new request.
The `materialize_selected` MCP tool invokes this same service method; its
request cannot supply a different session identity or bearer token.

This module does not own Codex integration. Exact-input regeneration is a
separate diagnostic-only path, admitted only for a launcher-owned source bundle
whose closed runner input is the supported stateless `simple_policy` form. It
rejects checkpoint/stateful/model paths, retained state or recordings, source
or identity mismatches, and ambiguous native bindings before execution. The
single canonical child is bounded by the adapter timeout, observes cancellation,
and performs a final source-integrity check. A duplicate service operation ID
returns its durable receipt without starting another child. The focused native
smoke removes the trace from a selected row, starts one real canonical episode,
and renders that generated trace; separate tests cover divergence,
unverifiable historical telemetry, cancellation, mutation, and idempotent
replay. These are diagnostic checks, not benchmark or scientific evidence.

Focused validation:

```bash
DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
  scripts/dev/run_worktree_shared_venv.sh -- \
  uv run pytest tests/analysis_workbench/test_audit_materialize.py \
  tests/analysis_workbench/test_audit_materialize_native.py \
  tests/analysis_workbench/test_audit_native_diagnostic.py \
  tests/analysis_workbench/test_audit_native_service.py -q
```
