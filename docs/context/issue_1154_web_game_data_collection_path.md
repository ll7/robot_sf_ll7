# Issue #1154 Web-Game Data Collection Path

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1154>

## Goal

Record the feasibility decision for browser/web-game manual-control data collection before any
public deployment or external trajectory upload path is implemented.

This note intentionally does not implement a browser runner. It defines the gate that must be
satisfied after the local manual-control recorder, replay, and export workflow stabilizes.

## Current decision

Web-game data collection is a distant-future path, not part of the local manual-control MVP.

The first viable browser deliverable should be a local/offline compatibility prototype or converter
that emits the same session and attempt fields as the local manual-control recorder. Public or
semi-public collection must remain disabled until the repository has explicit consent, privacy,
retention, deletion/export, schema-parity, and hosted-scenario versioning requirements.

## Dependencies

The web path depends on the local manual-control data contract landing first:

- `#1151`: local Pygame manual-control runner and versioned recording schema.
- `#1153`: schema-validating loader, completed-attempt replay, and behavior-cloning export.

Until those are merged, browser work should stay design-only. A browser-only schema would make
future behavior-cloning and benchmark comparison harder because records could not be filtered or
replayed using the local tooling.

## Required gate before external collection

External upload remains disabled unless all of these are documented and reviewable:

- Consent language that says what is recorded and why.
- Privacy policy covering user identifiers, IP/session metadata, and optional demographics.
- Retention and deletion/export rules for raw trajectories and derived datasets.
- Deterministic hosted scenario subset with map, scenario, seed, and asset-version identifiers.
- Schema mapping from browser records to the local manual-control session/attempt records.
- Data-quality checks for incomplete attempts, noisy/adversarial input, duplicate sessions, and
  impossible action streams.
- Baseline comparison semantics that match the local runner: success first, then time-to-goal,
  then SNQI and safety diagnostics.

## Minimal browser record contract

Each browser record stream must either use the local schema directly or provide a lossless converter
that produces the local fields:

- Session identity: schema version, recorder version, scenario id, map id, seed, attempt id, and
  user/session pseudonym.
- Mode identity: control mode, view mode, input mapping version, and action-space family.
- Event stream: ordered input events, simulator steps, pause/resume, retry, timeout, and terminal
  events.
- Training samples: explicit `training_sample` marker, observation, mapped action, reward/metric
  snapshot, and source provenance.
- Provenance: browser app version, scenario asset version, local-schema target version, timestamp,
  and validation status.

Any field without a local equivalent must be documented as browser-only metadata and excluded from
planner or behavior-cloning claims unless a later issue defines its semantics.

## Narrow first prototype

If the gate is accepted after `#1151` and `#1153`, the first implementation should be one of these
small compatibility paths:

1. An offline browser export fixture that writes one deterministic attempt in the local schema.
2. A converter that transforms a deliberately tiny browser JSON fixture into local
   manual-control JSONL records.

The prototype should not deploy a hosted service, upload raw trajectories, or introduce a
browser-specific training dataset format.

## Validation path

For the design gate:

- Review this note against the fields provided by the merged local manual-control schema.
- Verify every browser field has a local schema equivalent or explicit converter mapping.
- Verify hosted scenarios are deterministic and versioned before external collection is enabled.

For a later prototype:

- Add schema round-trip tests between browser-exported attempts and local manual-control records.
- Add data-quality tests for malformed records, incomplete attempts, duplicate sessions, and
  adversarial action streams.

## Current conclusion

Close `#1154` as a design/feasibility decision when this note lands. Reopen or create a new
implementation issue only after the local manual-control schema, replay/export path, and data-policy
gate are all stable enough to support a browser compatibility prototype.
