# Issue 1154: Web-game manual-control data collection path

Date: 2026-05-12
Issue: #1154
Parent: #1151

## Decision

A browser/web-game data collection path is worth keeping as a follow-up direction, but it should
not be implemented before the local Pygame manual-control recorder has a stable session and attempt
schema.

The web path should start as a small compatibility prototype rather than a separate product stack.
Its first success condition is schema parity with the local recorder, not polished gameplay or broad
public deployment.

## Rationale

A web game could collect more human demonstrations than a local Pygame tool, but it introduces
product and data-governance risks that are not present in local debugging:

- consent and retention rules for external user trajectories,
- hosted scenario and asset versioning,
- data-quality checks for incomplete, noisy, or adversarial inputs,
- browser input latency and frame-rate variability,
- compatibility with the local manual-control behavior-cloning export path.

Those risks are manageable only after #1151 defines the local recording contract: scenario ID, seed,
attempt ID, input mapping version, view mode, user input events, mapped actions, simulator states,
training-sample markers, terminal reason, and baseline-to-beat metadata.

## Recommended sequence

1. Finish the #1151 local Pygame MVP recorder first.
2. Freeze a minimal local session/attempt schema version.
3. Define a tiny hosted scenario subset with deterministic assets and seeds.
4. Write consent, privacy, retention, and deletion rules before collecting external data.
5. Build a browser prototype that exports the same session/attempt manifest fields or a lossless
   converter into the local schema.
6. Add data-quality checks before accepting trajectories for behavior-cloning experiments.

## Minimal hosted scenario subset

The first web prototype should use a deliberately small set:

- one or two short scenarios,
- fixed seeds,
- short horizons,
- no private model checkpoints in the client,
- explicit scenario version metadata,
- lightweight assets that can be reproduced or regenerated from tracked definitions.

The goal is to validate data collection and schema compatibility, not benchmark breadth.

## Consent and retention requirements

Before collecting any external user data, the implementation must define:

- what input and trajectory data is collected,
- whether any browser/device metadata is collected,
- where data is stored,
- how long data is retained,
- how a user can request deletion when identifiable data is present,
- whether data may be used for behavior-cloning training,
- whether data may be published as an anonymized aggregate or artifact.

Until this policy exists, web collection should remain local-only or synthetic-test-only.

## Schema compatibility requirements

The web path must preserve or convert into the local #1151 schema fields:

- `schema_version`,
- `scenario_id`,
- `scenario_version`,
- `seed`,
- `attempt_id`,
- `retry_index`,
- `input_mapping_version`,
- `view_mode`,
- input events,
- mapped actions,
- simulator timestamps/step indexes,
- dynamic state observations,
- `training_sample` markers,
- terminal status and failure reason,
- optional baseline-to-beat metadata.

If the browser implementation cannot produce a field exactly, it should record an explicit
`unavailable_reason` rather than silently changing semantics.

## Data-quality checks

A web trajectory should be flagged or rejected when:

- it is incomplete or terminates before minimum usable duration,
- it contains no meaningful control input,
- client timing is too irregular for action/state alignment,
- scenario version metadata is missing,
- input mapping metadata is missing,
- terminal status is missing,
- the attempt is dominated by pause/countdown/non-training frames.

These checks should run before a trajectory is used for behavior-cloning training.

## Acceptance mapping for #1154

- A written feasibility decision exists: this note.
- The web schema maps to the local manual-control schema: required fields are listed above.
- Consent/privacy/retention requirements are documented: see the consent section above.
- A future implementation can start from a narrow prototype: recommended sequence and hosted subset
  are defined above.

## Open dependency

This issue remains implementation-blocked on #1151's local recorder schema. Once #1151 lands, the
next web task should be a narrow prototype/export converter, not broad web deployment.
