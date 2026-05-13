# Issue 1154 web-game manual-control data collection path

Date: 2026-05-12
Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1154>

## Goal

Record the feasibility decision for browser/web-game manual-control data collection before any
public deployment or external trajectory upload path is implemented.

This note intentionally does not implement a browser runner. It defines the gate that must be
satisfied after the local manual-control recorder, replay, and export workflow stabilizes.

## Current decision

Web-game data collection is a follow-up direction, not part of the local manual-control MVP.

The first viable browser deliverable should be a local/offline compatibility prototype or converter
that emits the same session and attempt fields as the local manual-control recorder. Public or
semi-public collection must remain disabled until the repository has explicit consent, privacy,
retention, deletion/export, schema-parity, hosted-scenario versioning, and validation rules.

## Rationale

A web game could collect more human demonstrations than a local Pygame tool, but it introduces
product and data-governance risks that are not present in local debugging:

- consent and retention rules for external user trajectories,
- hosted scenario and asset versioning,
- data-quality checks for incomplete, noisy, duplicate, or adversarial inputs,
- browser input latency and frame-rate variability,
- compatibility with the local manual-control replay and behavior-cloning export path.

Those risks are manageable only after the local recording contract is stable enough to pin browser
records to the same schema, replay path, and benchmark-comparison semantics.

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
- Browser timing rules for action/state alignment when frame rate or input latency varies.
- Baseline comparison semantics that match the local runner: success first, then time-to-goal,
  then SNQI and safety diagnostics.

## Initial hosted scenario subset

The first browser-compatible subset should be intentionally small and deterministic. It should use
tracked SVG assets only, and every collected attempt must include the `scenario_id`, `map_id`,
`seed`, and `asset_version_id` below:

| Scenario id | Map id | Seed | Purpose |
| --- | --- | --- | --- |
| `web_manual_debug_06_seed_1154001` | `maps/svg_maps/debug_06.svg` | `1154001` | Basic local-runner parity and smoke validation. |
| `web_manual_classic_crossing_seed_1154002` | `maps/svg_maps/classic_crossing.svg` | `1154002` | Pedestrian-crossing timing and safety comparison. |
| `web_manual_classic_doorway_seed_1154003` | `maps/svg_maps/classic_doorway.svg` | `1154003` | Narrow-passage control and retry behavior. |

The asset-versioning plan for this subset is:

- Define one manifest id, `web_manual_control_v0`, before any hosted collection starts.
- Record the repository commit, browser app build id, local manual-control schema version, and
  SHA-256 for each SVG map asset in that manifest.
- Treat any SVG edit, scenario seed change, input mapping change, or browser simulator change as a
  new manifest id rather than mutating `web_manual_control_v0`.
- Store the manifest path or durable artifact reference in every exported session so attempts from
  different hosted scenario versions cannot be mixed silently.

## Minimal browser record contract

Each browser record stream must either use the local schema directly or provide a lossless converter
that produces the local fields:

- Session identity: `schema_version`, recorder version, `scenario_id`, `map_id` or
  `scenario_version`, `seed`, `attempt_id`, `retry_index`, and user/session pseudonym.
- Mode identity: control mode, `view_mode`, `input_mapping_version`, and action-space family.
- Event stream: ordered input events, mapped actions, simulator timestamps or step indexes,
  dynamic-state observations, pause/resume, retry, timeout, and terminal events.
- Training samples: explicit `training_sample` marker, observation, mapped action, reward or
  metric snapshot, optional baseline-to-beat metadata, and source provenance.
- Provenance: browser app version, scenario asset version, local-schema target version,
  validation status, and export timestamp.

Any field without a local equivalent must be documented as browser-only metadata and excluded from
planner or behavior-cloning claims unless a later issue defines its semantics. If a browser build
cannot supply a local field exactly, it should record an explicit `unavailable_reason` rather than
silently changing semantics.

## Data-quality expectations

A browser trajectory should be flagged or rejected when:

- it is incomplete or terminates before minimum usable duration,
- it contains no meaningful control input,
- client timing is too irregular for action/state alignment,
- scenario or asset-version metadata is missing,
- input-mapping metadata is missing,
- terminal status is missing,
- the attempt is dominated by pause, countdown, or non-training frames,
- the session is duplicated or the action stream is impossible for the declared control mode.

These checks should run before a trajectory is used for behavior-cloning training.

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
- Verify the hosted scenario subset uses the exact map ids, seeds, and asset-version manifest rules
  documented above before external collection is enabled.

For a later prototype:

- Add schema round-trip tests between browser-exported attempts and local manual-control records.
- Add data-quality tests for malformed records, incomplete attempts, duplicate sessions, and
  adversarial action streams.
- Verify benchmark comparisons still use success first, then time-to-goal, then SNQI and safety
  diagnostics.

## Current conclusion

Close `#1154` as a design or feasibility decision when this note lands. Reopen or create a new
implementation issue only after the local manual-control schema, replay/export path, and data-policy
gate are all stable enough to support a browser compatibility prototype.
