# Issue #2943 Fast-Results Milestone / Claim Map v0

**Status**: current routing surface as of 2026-06-16. Proposal-maintained.
**Issue**: [#2943](https://github.com/ll7/robot_sf_ll7/issues/2943)
**Branch**: `issue-2943-fast-results-claim-map-v0`

## Purpose

This note is a compact, durable handoff surface for routing the fastest paper-relevant evidence
work after the June 2026 research cycle. It defines which claims are moveable now, which are
gated on named preconditions, and which must not be promoted beyond diagnostic evidence under
current tooling.

This is a synthesis/context-doc note. It does not create benchmark runs, change experiment
configs, or produce paper-facing evidence.

## Scope

In scope:

- Claim map covering fast-results targets with evidence tier, required issues, blocked dependencies,
  and do-not-claim boundaries.
- Priority queue with `p0_now`, `p1_after_gate`, and `parked_blocked` entries.
- Alignment with #2910, #2911, #2612, #2937, #2941, #2923.

Out of scope:

- Benchmark campaign execution or result generation.
- Paper draft, manuscript, or dissertation text.
- Changes to planner configs, training pipelines, or experiment scripts.
- Claims beyond what cited durable context notes support.

---

## Claim Map

Evidence tiers used below:

| Tier | Meaning |
|---|---|
| `schema` | A source contract (schema, validator, fixture) exists; no claim that output is paper-ready. |
| `diagnostic` | Mechanism/trace understanding only; not benchmark-candidate evidence. |
| `candidate` | Executable evidence exists but a named requirement (comparator, seed tier, scenario tier, provenance check) is missing. |
| `paper_ready` | Command/config/commit, durable artifact, metrics/schema mode, and explicit fallback/degraded exclusions are all recorded. |
| `blocked` | A named dependency or evidence gate prevents the claim from moving at all. |
| `do_not_claim` | Current evidence is insufficient; promoting this claim is explicitly prohibited until the gate is satisfied. |

Fallback/degraded note: execution in `fallback`, `degraded`, `adapter`, or `not_available` mode
is never acceptable as a successful benchmark outcome per
[issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md). Any row with
those modes must be labeled a caveat or exclusion, not claim-grade success.

### Claim Map Table

| ID | Target table / surface | Required issue(s) | Evidence tier | Blocked dependency | Do-not-claim boundary |
|---|---|---|---|---|---|
| `cm-v0.benchmark_release.odd_coverage` | ODD/hazard-scenario coverage matrix for v0.1 release | #2911 (open/ready) | `schema` -> `candidate` | ODD schema `odd_hazard_coverage.v1` not yet defined | Do not claim gap coverage is closed until schema is checked against scenario family list |
| `cm-v0.benchmark_release.seed_table_semantics` | Release 0.0.2 secondary-report table semantics clarified | #2612 (open/ready) | `diagnostic` | `seed_episode_rows collision=0.0` conflicts with authoritative `campaign_table.csv`; must be resolved before any secondary-table paper claim | Do not cite secondary table in paper until collision-field conflict in #2612 is resolved |
| `cm-v0.benchmark_release.row_claim_enforcement` | `benchmark_row_claim.v1` row-level enforcement on leaderboard sidecars | #2912 (closed; implemented) | `schema` | None; schema and validator are live | Do not add `fallback`/`degraded` planner rows with `row_status: successful_evidence` |
| `cm-v0.prediction.denominator_health` | Horizon x timestep denominator-coverage audit | #2937 (closed; implemented) | `diagnostic` | None; 164/180 cells (91.1%) evaluated; `corridor_interaction` fixtures remain `trace_too_short` | Do not promote to forecast benchmark evidence; this audit does not prove navigation benefit or safety improvement |
| `cm-v0.prediction.native_replay` | Forecast-variant closed-loop replay path is native (not fallback) | #2941 (closed; implemented) | `diagnostic` | Minimal brake-heuristic replay policy only; not a production planner | Do not claim cv improves safety/success/runtime; do not remove caveat that replay uses a simple forecast-brake heuristic |
| `cm-v0.mechanism.trace_schema` | `mechanism_trace.v1` source contract for local-navigation intervention rows | #2923 (closed; implemented) | `schema` | Additional producers needed for `prediction_risk_gating`, `orca_residuals`, `signal_state_logic`, `amv_actuation_constraints` | Do not use `mechanism_trace.v1` rows for benchmark or paper-facing claims until durable trace inputs and producer integrations exist |
| `cm-v0.benchmark_release.suite_freeze` | Nominal/stress/adversarial/AMV suite freeze for v0.1 release | #2910 epic (open) | `blocked` | Requires ODD coverage (#2911) and row-claim matrix frozen before suite can be marked frozen | Do not claim suite is paper-ready until freeze contract from #2910 is satisfied |
| `cm-v0.prediction.full_planner_integration` | Forecast variant integrated into a real planner consuming `ProbabilisticPredictor` | #2960 (this PR; smoke implemented) | `smoke` | Single deterministic local-planner fixture only; no full-episode benchmark or scenario-matrix evidence yet | Do not claim forecast variant is benchmark-capable or beneficial; smoke proves planner consumption mechanics only |

---

## Priority Queue

### p0_now -- No blocking gates; work can start or land immediately

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| Resolve `seed_episode_rows collision=0.0` conflict | #2612 | ready | Artifact: updated release 0.0.2 table-semantics context note and authoritative collision-field decision | Clarify core table semantics and confirm which field is authoritative | Missing until #2612 closes |
| Define `odd_hazard_coverage.v1` schema and gap matrix | #2911 | ready | Artifact: `odd_hazard_coverage.v1` schema plus JSON/Markdown gap matrix | Schema definition checked against cyclist, signalized crossing, occlusion, stairs, dense crowd, narrow-lane conflict, AMV actuation, and sensor-latency scenario families | Missing until #2911 closes |
| Wire first additional `mechanism_trace.v1` producer | #2976 | ready | Artifact: schema-valid `mechanism_trace.v1` producer payload from durable input or tracked fixture | Passing contract tests and at least one durable trace input for `prediction_risk_gating` or `orca_residuals` | Missing until #2976 closes |

### p1_after_gate -- Gated on a named p0 precondition

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| ODD/AMV suite freeze for v0.1 | #2910 | blocked | Artifact: suite-freeze release note and frozen scenario-family list | p0: #2911 schema and gap matrix accepted | Missing until #2911 closes |
| Secondary-table paper promotion | #2612 | blocked | Artifact: paper-promotion note or release-table addendum | p0: #2612 collision-field conflict resolved | Missing until #2612 closes |
| Forecast variant benchmark campaign | #2966 | blocked | Artifact: planner-consumed forecast slice report | p0: production planner integration and durable episode manifest accepted | Missing until #2966 unblocks and lands |
| `mechanism_trace.v1` mechanism reports | #2976 | blocked | Artifact: mechanism report over durable producer payloads | p0: durable trace inputs and at least `prediction_risk_gating` emitter live | Missing until #2976 closes |

### parked_blocked -- Explicitly blocked; do not route until gate is named and unblocked

| Item | Owner issue | Status | Next command or artifact | Evidence gate | Durable evidence |
|---|---|---|---|---|---|
| CARLA replay transfer evidence | #2158 | do-not-claim | Artifact: CARLA transfer eligibility note | CARLA parity proven and `sensor_perception_replay` available with native/aligned fixture semantics | Not available under current diagnostics; see #2158 and #2276 |
| AMV calibrated-actuation paper claim | #2230 | do-not-claim | Artifact: calibrated-actuation provenance note | Runtime/provenance fields for yaw, angular acceleration, latency, and update rate are hardware-calibrated or accepted proxy-source fields | Not available; see #2230 and #2259 |
| SocNavBench planner rows | #2397 | do-not-claim | Artifact: SocNavBench unavailable-row note | SocNavBench control pipeline available natively or explicitly accepted as unavailable-only | Not available; rows must stay `not_available` / `accepted_unavailable_only` |
| Full benchmark claim matrix for #2910 release | #2910 | blocked | Artifact: v0.1 benchmark claim matrix | #2911, #2612, and suite-freeze prerequisites all landed | Missing until p0 prerequisites land |
| Forecast variant safety/success claims | #2966 | do-not-claim | Artifact: forecast safety/success benchmark report | Planner-consumed forecast benchmark run with comparator, scenario matrix, and durable manifest | Not available; #2941 remains mechanism-trace quality only |

---

## Provenance and Validation

This note is a tracked synthesis document. It does not create or depend on local `output/` files.

### Referenced durable context notes

- [issue_2937_horizon_denominator_health.md](issue_2937_horizon_denominator_health.md) -- denominator-health evidence (closed)
- [issue_2941_native_forecast_replay.md](issue_2941_native_forecast_replay.md) -- native forecast-variant replay evidence (closed)
- [issue_2923_mechanism_trace_schema.md](issue_2923_mechanism_trace_schema.md) -- mechanism trace v1 schema (closed)
- [issue_2912_benchmark_row_claim.md](issue_2912_benchmark_row_claim.md) -- benchmark row claim v1 enforcement
- [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md) -- fail-closed fallback policy
- [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md) -- artifact evidence vocabulary
- [issue_2153_research_v1_evidence_map.md](issue_2153_research_v1_evidence_map.md) -- research-v1 claim gate (predecessor map pattern)

### Referenced open issues (live state at 2026-06-16)

- [#2910](https://github.com/ll7/robot_sf_ll7/issues/2910) -- benchmark v0.1 release epic (open)
- [#2911](https://github.com/ll7/robot_sf_ll7/issues/2911) -- ODD hazard scenario coverage matrix (open/ready)
- [#2612](https://github.com/ll7/robot_sf_ll7/issues/2612) -- release 0.0.2 secondary-report ambiguity (open/ready)

### Validation commands

```bash
# Check that linked context-note paths resolve
grep -RiIn "issue_2943_fast_results_claim_map_v0" docs/context/
# Check executable priority-queue fields
uv run python scripts/dev/check_fast_results_claim_map.py --json
# Proof-consistency diff
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
# No trailing whitespace
git diff --check
```

---

## Follow-Up Boundary

This v0 map is intentionally narrow. Extend it when:

1. Issue #2911 lands and `odd_hazard_coverage.v1` is defined -- add a `candidate` ODD coverage row.
2. Issue #2612 is resolved -- move secondary-table claim from `diagnostic` to `candidate`.
3. A full-episode benchmark or scenario-matrix run exercises the planner-consumed `forecast_variant` path -- consider moving `cm-v0.prediction.full_planner_integration` from `smoke` to `candidate`.
4. The #2910 suite freeze happens -- update `cm-v0.benchmark_release.suite_freeze` from `blocked` to `candidate`.

Do not promote any `blocked` or `do_not_claim` row without named evidence that satisfies the
stated gate condition. Numeric uncertainty note: all tier assignments above reflect current
evidence state; any claim tier upgrade below 95% confidence of satisfying the gate requires an
explicit caveat naming the open condition.
