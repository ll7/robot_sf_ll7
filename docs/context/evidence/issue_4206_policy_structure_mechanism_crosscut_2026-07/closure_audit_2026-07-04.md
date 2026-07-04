# Issue #4206 Closure Audit

Plain-language summary: issue #4206 is not ready to close. The repository now has the CPU-only
mechanism-cross-cut builder, blocked evidence packet, schema instrumentation contract, trace-capable
h600 rerun pre-registration, and runnable trace-capable h600 config. The remaining acceptance
criterion is empirical: run the trace-capable h600 campaign and feed trace-verified mechanism
labels back into the builder. That action is outside this PR because it requires private queue
execution, not another tracked checker.

## Audit Scope

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/4206>
- Audit date: 2026-07-04
- Claim boundary: closure audit and integration status only.
- Out of scope: no benchmark campaign run, no Slurm/GPU submission, no paper or dissertation claim
  edits, no causal claim promotion.

## Acceptance Evidence

| Acceptance criterion | Delivered evidence | Status |
| --- | --- | --- |
| Add an explicit analysis config declaring #4206 inputs, taxonomy source, structural classes, accepted mechanism-confidence classes, and forbidden geometry fallback. | PR #4219 added `configs/analysis/issue_4206_policy_structure_mechanism_crosscut.yaml`; PR #4341 tightened the declared sidecar consumption and predates-trace-capture block. | Delivered for CPU-side contract. |
| Add a deterministic CPU-only builder that loads h600 rows, optional declared sidecars, job-13175 continuity packet, and produces compact evidence outputs. | PR #4219 added `scripts/validation/build_issue_4206_policy_structure_mechanism_crosscut.py`; PR #4293 wired input provenance into blocked outputs; PR #4312 split missing-input and missing-label blockers; PR #4305 and PR #4341 added native/declared sidecar consumption. | Delivered for CPU-side contract. |
| Fail closed when h600 rows lack trace-verified mechanism labels; do not substitute scenario geometry buckets. | PR #4219 introduced the fail-closed path; PR #4312 distinguished artifact retrieval from missing labels; PR #4319 refreshed the packet against retrieved 13268/13273 rows and retained `blocked_missing_trace_verified_mechanism_labels`; PR #4341 made `not_derivable` sidecars a precise predates-trace-capture blocker. | Delivered. Current packet is intentionally blocked. |
| Produce mechanism-level rank tables and the two F-C4(ii) probes only when trace-verified labels exist. | The builder and CSV outputs exist, but the current retrieved h600 rows have 2304/2304 rows missing trace-verified mechanism labels, so `f_c4ii_probe_*` remain empty by design. | Blocked on new trace-capable h600 campaign outputs. |
| Produce geometry-vs-mechanism agreement/disagreement comparison without using geometry as a mechanism-label substitute. | PR #4219 introduced `geometry_vs_mechanism_agreement.csv`; PR #4319 and PR #4341 preserved geometry rejection while refreshing against retrieved inputs. | Delivered for blocked packet; final comparison still waits on trace-verified labels. |
| Evidence directory contains claim-boundary README, compact tables, provenance metadata, and checksums. | PR #4319 refreshed `docs/context/evidence/issue_4206_policy_structure_mechanism_crosscut_2026-07/` with `README.md`, `claim_boundary.md`, `metadata.json`, `missing_instrumentation.json`, report/table CSV/JSON files, and `SHA256SUMS`. This PR adds the closure audit report and updates `SHA256SUMS`. | Delivered for current blocked state. |
| If trace-level mechanism labels are unavailable, report exactly what instrumentation is missing and stop before F-C4(ii) conclusions. | `missing_instrumentation.json` reports `blocked_missing_trace_verified_mechanism_labels`, required fields, row count, geometry rejection, and the follow-up skeleton. PR #4350 adds the trace-capable h600 rerun pre-registration and fail-closed validator for the required fields. | Delivered as a precise blocker. |

## Linked PR Integration

| PR | Merge commit | Integration role |
| --- | --- | --- |
| #4219 | `92311713` | Initial mechanism-level policy-structure cross-cut builder, config, tests, and evidence packet shape. |
| #4255 | `0bfc6434` | Canonical episode evidence schema fields that #4206 consumes. |
| #4278 | `afac5d97` | Retained h600 sidecar backfill for mechanism/exposure inputs. |
| #4293 | `04d84e44` | Input provenance in blocked mechanism-cross-cut packet. |
| #4301 | `b18d08b8` | Native mechanism/exposure blocks in the episode writer for future trace-capable rows. |
| #4305 | `889677c9` | #4195/#4206 builders consume native mechanism/exposure fields and declared sidecars. |
| #4312 | `26d49859` | Distinct fail-closed blockers for missing retrieved artifacts vs missing trace labels. |
| #4319 | `6dca725f` | Refreshed blocked evidence packet against retrieved h600 artifacts 13268/13273. |
| #4338 | `48fee80c` | Consolidated builder-consumption state surface for #4242 dependencies. |
| #4341 | `bdb5b0f3` | Declared sidecar consumption plus precise predates-trace-capture block. |
| #4350 | `64605a57` | Trace-capable h600 rerun pre-registration and fail-closed validator. |
| #4373 | `755e4ae1` | Schema instrumentation terminality note for #4242 dependency. |
| #4402 | `dae10d11` | Aligns h600 mechanism packet builder schema. |
| #4414 | `81992aaa` | Hardens h600 mechanism-label validation delegation. |
| #4423 | `63fd36aa` | Documents trace-capable h600 runtime context. |
| #4428 | `ea3583db` | Fixes trace-capture flag plumbing in the camera-ready path. |
| #4449 | `9b260169` | Guards `guarded_ppo` availability in the trace-capable h600 config. |

## Current Blocker

The issue cannot be closed from tracked-code changes alone. The existing h600 rows predate
trace-capable episode export, so all 2304 loaded rows remain missing the fields required for
trace-verified mechanism conclusions:

- `mechanism_schema_version`
- `mechanism_label`
- `mechanism_confidence`
- `mechanism_evidence_mode`
- `mechanism_evidence_uri`
- a trace-verified evidence mode such as `paired_trace`, `deterministic_replay`, `direct_probe`, or
  `root_cause`

The next empirical action is to run the trace-capable h600 rerun declared by
`configs/benchmarks/issue_4206_trace_capable_h600_rerun_preregistration.yaml` and the runnable
config `configs/benchmarks/paper_experiment_matrix_v1_h600_trace_capable_rerun.yaml`, then rerun
the mechanism-cross-cut builder against those outputs. This audit does not authorize or perform that
campaign.
