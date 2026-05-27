# Issue #1571 Adversarial Smoke Packet Sharpening (2026-05-27)

Date: 2026-05-27

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1571> (this analysis note)
- <https://github.com/ll7/robot_sf_ll7/issues/1501> (one-family smoke)
- <https://github.com/ll7/robot_sf_ll7/issues/1500> (frozen manifest)
- <https://github.com/ll7/robot_sf_ll7/issues/1488> (umbrella)
- <https://github.com/ll7/robot_sf_ll7/issues/1457> (guided route-search smoke anchor)
- <https://github.com/ll7/robot_sf_ll7/issues/1433> (crossing/TTC search design)
- <https://github.com/ll7/robot_sf_ll7/issues/691> (fail-closed fallback policy)
- <https://github.com/ll7/robot_sf_ll7/issues/1542> (claim-evidence map)

## Goal

Provide the durable background-research summary requested by #1571 so #1501 can run as an
agent-ready one-family smoke without broadening the campaign or weakening fail-closed evidence
rules.

This note is a durable handoff artifact. It does **not** execute the adversarial campaign, create
new benchmark evidence, or promote local `output/` artifacts into git.

## Background-Research Synthesis

| Mechanism | Source issue | Evidence tier | Config | Seeds | Artifacts | Metrics | Verdict | Caveats |
|---|---:|---|---|---|---|---|---|---|
| Crossing/TTC parametric adversarial search | #1433, #1500 | design + manifest launch packet | `configs/adversarial/crossing_ttc_space.yaml` and `configs/adversarial/issue_1500_adversarial_comparison_manifest.v1.yaml` | global seed `42`; candidate `scenario_seed` range `[100, 999]` | `adversarial-search-manifest.v1`, compact failure archive | row counts, objective traces, replay determinism | **Run first in #1501** | `generated_cases_are_benchmark_evidence: false`; no cross-engine claim yet |
| Guided route search | #1457, #1500 | bounded route-generation smoke + manifest | `configs/adversarial_routes/default.yaml` | seed `123` | route override artifact, compact smoke summary | valid-trial ratio, composite objective components | **Do not run in the crossing/TTC smoke** | `guided_route_search` is `not_available` for `crossing_ttc`; keep it as explicit exclusion row |
| Fail-closed benchmark status policy | #691, #1500 evidence | policy + classification contract | n/a | n/a | compact row-classification table | readiness/availability status | **Mandatory accounting rule** | `fallback`, `degraded`, `simulation_error`, and `not_available` never count as success evidence |
| Manuscript / claim boundary | #1542 | evidence map | manifest note + future execution notes | n/a | durable notes only | none at this stage | **Still not claim-ready** | #1501 remains a smoke/execution gate, not paper-facing evidence |

## Packet Decision For #1501

### 1. What #1501 should run first

Run the `crossing_ttc` family only.

- Available engines for this family: `random`, `optuna_tpe`
- Planner rows remain the frozen manifest rows:
  - `classic_global_theta_star` (`native`)
  - `orca` (`adapter`)
- Preferred command surface for the available search engines:
  `scripts/tools/compare_adversarial_samplers.py`

`generate_adversarial_routes.py` remains the command surface for the separate
`classic_head_on_corridor` route-search lane, but that is **not** the first smoke for #1501.

### 2. How to handle `guided_route_search`

For a one-family `crossing_ttc` smoke, `guided_route_search` must be recorded as an explicit
`not_available` design-exclusion row, not executed and not silently dropped.

Reason: the frozen manifest marks `guided_route_search` as `not_available` for `crossing_ttc`
because the route-override search paradigm does not map to the parametric CandidateSpec template.

This preserves the original #1501 intent ("all engines are accounted for") without pretending the
route-search surface is runnable on the crossing/TTC family.

### 3. Budget normalization rule

Do **not** invent a new cross-paradigm normalization rule inside #1501.

Use the frozen manifest budget for the one-family smoke:

- `random`: `32`
- `optuna_tpe`: `32`
- `guided_route_search`: `not run` / `not_available`

`guided_route_search` trial-count normalization remains relevant to the
`classic_head_on_corridor` lane, but that belongs to the later two-family comparison issue rather
than this one-family smoke.

### 4. Row-status accounting

Issue #1501 must count the following campaign row types separately:

- `valid_behavioral_failure`
- `success`
- `invalid_candidate`
- `simulation_error`
- `fallback`
- `degraded`
- `not_available`

In addition, if the smoke cannot start because a required dependency, planner mapping, or replay
surface is missing, the execution packet should classify the attempted slice as **blocked** in the
note/summary rather than collapsing it into a success-shaped result.

Per #691 and the manifest row-classification contract:

- `fallback` and `degraded` are caveats, not success evidence.
- `simulation_error` is an explicit exclusion, not a discovered stress failure.
- `not_available` is a design exclusion, not an implicit zero.

## Required Artifacts Before #1501 Is Complete

Issue #1501 should be considered complete only when it leaves a durable, reproducible packet
containing:

1. A concise durable note or issue follow-up that records commit SHA, config paths, commands, seed
   schedule, output root, and the design-exclusion treatment for `guided_route_search`.
2. `adversarial-search-manifest.v1` outputs for each **available** engine x planner row actually
   run in `crossing_ttc`.
3. One curated `adversarial_failure_archive.v1` for the smoke output set, built only from archive-
   eligible failures.
4. Explicit per-row counts for all seven row types above, with `counts_as_success_evidence=false`
   for `fallback`, `degraded`, `simulation_error`, and `not_available`.
5. Replay determinism results for the manifest checks frozen in #1500:
   `manifest_materialization`, `seed_determinism`, and `search_trajectory`.
6. A compact tracked evidence copy under `docs/context/evidence/` only for small reviewable
   summaries such as row-count tables, determinism summaries, or checksums.

The following are **not** required for #1501:

- `stress_uncertainty_coverage.v1` synthesis (deferred to #1503)
- the `classic_head_on_corridor` family
- raw `output/` bundle promotion into git
- paper-facing interpretation

## Verdict On #1501 Tightness

Issue #1501 is **tight enough with this note/comment-level clarification**. It does not need a scope
expansion or promotion into #1502.

The only sharpening needed is interpretive:

- keep `#1488` as the umbrella,
- keep `#1501` as the first executable child,
- treat the one-family smoke as `crossing_ttc`,
- account for `guided_route_search` as an explicit `not_available` row instead of trying to force a
  route-search execution into the parametric family.

## Recommended Next Prompt

Use the next execution prompt to run #1501 as a bounded `crossing_ttc` smoke:

```text
Implement issue #1501 as the first executable child of #1488 using the frozen contract in
configs/adversarial/issue_1500_adversarial_comparison_manifest.v1.yaml and the clarifications in
docs/context/issue_1571_adversarial_smoke_packet_sharpening.md.

Scope:
- Run only the crossing_ttc family.
- Run only the available engines for this family: random and optuna_tpe.
- Preserve the two frozen planner rows from the manifest.
- Record guided_route_search as an explicit not_available design-exclusion row for crossing_ttc.
- Keep fallback, degraded, simulation_error, and not_available rows separate from success evidence.
- Do not broaden into classic_head_on_corridor, #1502, or #1503.

Required outputs:
- adversarial-search-manifest.v1 for each executed slice
- one adversarial_failure_archive.v1 for the smoke outputs
- compact durable row-count + determinism summary under docs/context/evidence/
- durable note/update stating commands, config paths, seeds, output root, and blockers/caveats

Starting command surface:
- Use `scripts/tools/compare_adversarial_samplers.py` with
  `--scenario-template configs/scenarios/templates/crossing_ttc.yaml`,
  `--search-space configs/adversarial/crossing_ttc_space.yaml`,
  `--sampler random`, and `--sampler optuna`.
- Resolve the frozen planner-row policy/algo mapping from local code and the manifest before
  execution; if `classic_global_theta_star` or `orca` cannot map cleanly to the current CLI, fail
  closed and report the exact blocker instead of substituting a different planner.

Required checks:
- manifest_materialization
- seed_determinism
- search_trajectory

If a planner-row alias or execution surface is ambiguous, resolve it from local code/docs first and
record the mapping explicitly. If it cannot be resolved cleanly, fail closed and report the slice as
blocked rather than silently substituting another contract.
```

## Artifact / Provenance Note

This analysis-only issue does not need a new generated evidence bundle. The durable artifact is this
tracked note plus the existing frozen-manifest evidence under
`docs/context/evidence/issue_1500_adversarial_manifest/`.
