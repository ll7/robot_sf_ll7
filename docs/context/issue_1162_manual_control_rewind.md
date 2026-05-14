# Issue #1162 Manual-Control Active-Attempt Rewind

## Decision

The first active-attempt rewind slice uses deterministic replay-to-step, not opaque simulator
checkpoint/restore.

This keeps the contract narrow enough for the current manual-control foundation:

- JSONL remains append-only and preserves the original user behavior.
- A rewind is represented as an explicit `event="rewind"` record with `ManualRewindMetadata`.
- The live runner can restore a bounded prefix by replaying records through `to_step_idx`.
- Behavior-cloning export excludes training samples from the discarded pre-rewind suffix.

## Rejected alternatives

Simulator checkpoint/restore is deferred because the current stack does not expose a single stable
state object covering simulator internals, robot state, pedestrian state, RNG state, metrics
accumulators, and session-controller state. Adding that as a first slice would risk silently
claiming exact restore semantics without a proven simulator-level invariant.

Destructive log editing is rejected because manual-control provenance should remain append-only.
Users may rewind, but the stream must still show what happened before the rewind.

## Current proof boundary

This implementation proves the replay/export semantics on tiny deterministic attempts. It does not
yet claim pixel-perfect or physics-state rewind in a live Pygame runner. A later runner integration
should wire `plan_replay_to_step_rewind` into the live environment and add a headless smoke that
demonstrates simulator state restoration against a concrete tiny scenario.

Repeated rewind planning from an append-only replay stream is currently fail-closed. Once a replay
already contains `ManualRewindMetadata`, callers must derive the active timeline first or wait for a
follow-up slice that adds explicit repeated-rewind planning semantics.

## Validation path

Targeted validation should include:

- `tests/test_manual_control_replay.py` for replay-to-step planning and invalidated sample indexes,
- `tests/test_manual_control_recording.py` for explicit rewind schema round-tripping,
- `tests/test_manual_control_export.py` for BC export exclusion across rewind boundaries.
