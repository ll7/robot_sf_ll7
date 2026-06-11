# Issue #2602 Why-First Report Usefulness

Issue: [#2602](https://github.com/ll7/robot_sf_ll7/issues/2602)
Status: reporting-evaluation synthesis; not new benchmark evidence.

## Purpose

This note evaluates whether the why-first reports generated in
[issue_2522_why_first_diagnostics.md](issue_2522_why_first_diagnostics.md) made follow-up
decisions clearer than reading the source context notes alone. It compares one topology diagnostic
and one AMV diagnostic, using only tracked compact evidence and generated reports that already exist
under `docs/context/evidence/issue_2522_why_first_diagnostics/`.

## Selected Diagnostics

| Diagnostic class | Source evidence | Why-first report | Source decision before report | Report decision |
| --- | --- | --- | --- | --- |
| Topology near-parity corrective diagnostics | [Issue #2530 note](issue_2530_topology_near_parity_corrective_smoke.md), [Issue #2530 summary](evidence/issue_2530_topology_near_parity_corrective_smoke_2026-06-07/summary.json), and [Issue #2518 gate evidence](issue_2518_topology_near_parity_gate.md) | [Topology why-first report](evidence/issue_2522_why_first_diagnostics/topology_near_parity_why_first_report.md) | `revise`: the selector signal activated, but corrective behavior did not pass. | `revise`: topology signal exists, but route progress and terminal outcome remain insufficient. |
| AMV actuation timeout diagnostics | [Issue #2443 trace review](issue_2443_amv_trace_review.md), [Issue #2440 timeout closure](issue_2440_amv_timeout_closure.md), and compact summaries linked from the report | [AMV why-first report](evidence/issue_2522_why_first_diagnostics/amv_actuation_why_first_report.md) | `revise`: command clipping improved, but route progress and success did not. | `revise`: feasibility improved, but task completion remains blocked by route-progress or completion geometry. |

## Usefulness Comparison

| Criterion | Topology result | AMV result | Cross-diagnostic judgment |
| --- | --- | --- | --- |
| Decision clarity | Improved. The report puts `revise` next to the exact reason: non-primary selection and command influence were real but not corrective. | Improved. The report separates command-feasibility improvement from route-progress failure. | Positive. Both reports make the continue/revise/stop state easier to recover quickly. |
| Claim-boundary clarity | Improved. The report states failed/non-success rows must not count as benchmark success and limits strength to compact evidence. | Improved. The report prevents actuation-feasibility evidence from becoming a planner-improvement claim. | Positive. The generated claim-boundary section is useful for avoiding overclaiming. |
| Missing-proof clarity | Improved. Missing proof is a paired reuse-penalty diagnostic that moves route progress or terminal behavior. | Improved. Missing proof is route-progress geometry or task-completion evidence, not another broad scorer. | Positive. Both reports expose the next proof gap without requiring a full context-note reread. |
| Next-action clarity | Improved. The report points to the #2563 primary-route reuse-penalty hypothesis before benchmark claims. | Improved. The report routes work away from actuation scoring as the immediate completion blocker. | Positive. Next actions are concrete enough to reduce duplicate exploratory work. |
| Report structure gap | No blocking structure gap found, but dissertation-facing fields from #2545 would make reader takeaways and allowed wording easier to reuse. | No blocking structure gap found, with the same optional dissertation-facing gap. | Continue current format for diagnostics; link #2545 only as an optional reuse/readability improvement. |

## Decision

```yaml
why_first_report_usefulness:
  topology_source: docs/context/evidence/issue_2522_why_first_diagnostics/topology_near_parity_why_first_report.md
  amv_source: docs/context/evidence/issue_2522_why_first_diagnostics/amv_actuation_why_first_report.md
  durable_inputs_available: true
  decision_clarity_delta: improved
  claim_boundary_clarity_delta: improved
  missing_proof_clarity_delta: improved
  next_action_clarity_delta: improved
  recommendation: continue
  claim_boundary: reporting_evaluation_only
```

Recommendation: continue using why-first reports for diagnostic classes where compact inputs already
name mechanism activation, comparator, trace evidence, alternative explanations, and a
continue/revise/stop decision. Do not treat the reports as stronger evidence than their source
diagnostics. Do not require a report for every diagnostic note; use it when the next-action or
claim-boundary state is likely to be reused by later issues, PRs, or dissertation-facing summaries.

## Claim Boundary

This is a reporting-usefulness evaluation only. It does not add a benchmark run, metric proof,
trace export, paper-facing claim, or new source evidence for the topology or AMV mechanisms. The
underlying topology and AMV conclusions remain `revise`.

## Validation

```bash
uv run python scripts/tools/generate_why_first_report.py --input docs/context/evidence/issue_2522_why_first_diagnostics/topology_near_parity_input.json --output /tmp/topology_near_parity_why_first_report.gen.md
diff -u /tmp/topology_near_parity_why_first_report.gen.md docs/context/evidence/issue_2522_why_first_diagnostics/topology_near_parity_why_first_report.md
uv run python scripts/tools/generate_why_first_report.py --input docs/context/evidence/issue_2522_why_first_diagnostics/amv_actuation_input.json --output /tmp/amv_actuation_why_first_report.gen.md
diff -u /tmp/amv_actuation_why_first_report.gen.md docs/context/evidence/issue_2522_why_first_diagnostics/amv_actuation_why_first_report.md
uv run pytest tests/tools/test_generate_why_first_report.py -q
uv run python -m json.tool docs/context/evidence/issue_2602_why_first_report_usefulness/summary.json
BASE_REF=origin/main scripts/dev/check_docs_proof_consistency_diff.sh
```
