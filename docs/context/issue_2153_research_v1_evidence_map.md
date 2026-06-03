# Issue #2153 Research-v1 Evidence Map And Claim Gate

Issue: [#2153](https://github.com/ll7/robot_sf_ll7/issues/2153)
Status: current research-v1 planning surface as of 2026-06-03.

This note defines compact research-v1 claim IDs for AMV-oriented social-navigation work and links
each claim to the evidence that would be required, the evidence that is explicitly excluded, and the
next issue that can move the claim. It is a claim gate, not a claim upgrade: diagnostic, fallback,
degraded, unavailable, blocked, and proposal-only results remain limited to the status named below.

## Claim Gate

Use these statuses when updating the map:

| Status | Meaning |
| --- | --- |
| `blocked` | A named dependency or evidence gate prevents the claim from moving. |
| `diagnostic` | The evidence is useful for mechanism, trace, or workflow understanding only. |
| `proposal` | The issue or protocol is planned but not yet executed on durable inputs. |
| `candidate` | The claim has executable evidence, but still lacks a required comparator, seed tier, scenario tier, schema field, or artifact provenance check. |
| `paper_ready` | The claim names command/config/commit, durable artifact or manifest, metrics/schema mode, fallback/degraded exclusions, and limitations. |

Do not mark a claim `candidate` or `paper_ready` when the supporting row is `fallback`,
`degraded`, `failed`, `not_available`, or `accepted_unavailable_only`. Those rows may explain a
blocker or diagnostic mechanism, but they are not successful benchmark evidence under
[issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md).

## Claim Map

| ID | Claim boundary | Required evidence | Current status | Current evidence and caveat | Next movement |
| --- | --- | --- | --- | --- | --- |
| `research-v1.amv.model_behavior` | AMV-aware social-force or interaction-model behavior improves a named mechanism without breaking existing planner contracts. | Configured model term, unit/integration tests, diagnostics showing the term activates on intended scenarios, and no paper-facing performance language until calibrated evidence exists. | `proposal` | Issue [#2154](https://github.com/ll7/robot_sf_ll7/issues/2154) is ready but high risk; existing manuscript map treats synthetic actuation smoke as non-paper-facing. | Implement #2154 with targeted tests and a compact diagnostic report. |
| `research-v1.amv.benchmark_matrix` | A frozen AMV social-navigation matrix can support reproducible planner comparison work. | Tracked config/protocol note, scenario families, planner rows, seed policy, artifact policy, fail-closed row semantics, and config validation. | `candidate` | Issue [#2155](https://github.com/ll7/robot_sf_ll7/issues/2155) freezes the matrix contract with 23 scenarios in 11 families, 12 planner rows in 4 families, paper_eval_s5 seed policy, and fail-closed fallback handling. No benchmark run is implied by this status. `docs/context/issue_2155_research_v1_ammv_matrix.md` is the contract note. | Move to `diagnostic` or retain `candidate` after a named campaign execution with artifact provenance. |
| `research-v1.amv.scenario_criticality` | Hazard and ODD coverage can identify critical AMV social-navigation scenarios. | Tracked coverage report, scenario taxonomy, ODD/hazard mapping, row status counts, and explicit missing-coverage list. | `proposal` | Issue [#2156](https://github.com/ll7/robot_sf_ll7/issues/2156) is the report lane. Existing seed and perturbation notes remain diagnostic unless a stronger matrix supports them. | Generate the #2156 coverage report after the selected matrix surface is clear. |
| `research-v1.amv.transfer_boundary` | CARLA or alternate-simulator replay can bound where Robot SF AMV evidence transfers. | Native/aligned fixture or replay semantics diagnostics, adapter status, durable trace/replay artifacts, and fail-closed exclusions. | `blocked` | [issue_2014_simulator_backend_matrix.md](issue_2014_simulator_backend_matrix.md) recommends trace/report work before full integration. CARLA parity remains blocked by replay/actor-spawn gaps in the manuscript map. | Use #2158 for a diagnostic pack; do not use it as parity evidence until the CARLA gates close. |
| `research-v1.amv.calibrated_actuation` | AMV actuation profiles are calibrated enough for paper-facing AMV claims. | Direct hardware or accepted proxy-source manifest, field-level source boundaries, runtime metrics, checksums or durable pointers, and unavailable-field handling. | `blocked` | [issue_2001_amv_actuation_proxy_source_analysis.md](issue_2001_amv_actuation_proxy_source_analysis.md) supports longitudinal proxy values only. [issue_2011_amv_actuation_sensitivity_sweep.md](issue_2011_amv_actuation_sensitivity_sweep.md) produced `accepted_unavailable_only` rows, not benchmark evidence. | Resolve the missing runtime/provenance fields before claim movement; keep yaw, angular acceleration, latency, and update rate synthetic or unavailable until sourced. |
| `research-v1.amv.failure_case_review` | Representative failure-case traces can explain mechanisms worth testing next. | Selected cases, trace-viewer pack, qualitative tags, row statuses, and no aggregation beyond observed cases. | `proposal` | Issue [#2159](https://github.com/ll7/robot_sf_ll7/issues/2159) is qualitative and speculative until cases are selected. | Build the #2159 review pack after claim IDs and matrix/report lanes provide selection criteria. |

## Downstream Update Rule

When a research-v1 issue closes, update the matching row above with:

- the issue or PR number;
- command/config paths and commit when the result is executable evidence;
- durable tracked evidence path, manifest, or external pointer;
- row status and whether fallback/degraded/unavailable rows were excluded;
- status change, or the reason the status did not change.

If the work only adds diagnostics, keep the status `diagnostic` and name the smallest next proof step.
If the proof fails, keep or move the status to `blocked` with the exact blocker.

## Validation

This note is a tracked synthesis artifact. It does not create or depend on local `output/` files.
Minimum validation for updates:

```bash
grep -Rin "research-v1\\|evidence map\\|AMV" docs/context docs/README.md
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
git diff --check
```

## Artifact Decision

Artifact class: tracked context note. No generated benchmark, replay, training, or local `output/`
artifact was created for this map. Diagnostic-only and proposal-only evidence is intentionally not
promoted to benchmark or paper-facing status here.
