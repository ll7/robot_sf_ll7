# Issue #4163 Closure Audit

This audit maps issue #4163 acceptance criteria to merged evidence and keeps the
remaining brute-force reference comparison blocked instead of closing the issue.

Issue: https://github.com/ll7/robot_sf_ll7/issues/4163

Audit date: 2026-07-04

Evidence status: diagnostic-only closure audit. The rare-event harness exists, but
the issue is not complete because the required cheap-family brute-force reference
comparison has not been produced.

## Acceptance Map

| Criterion | Status | Evidence | Notes |
| --- | --- | --- | --- |
| Add a planner-agnostic rare-event estimation harness over scenario parameters. | Delivered | PR #4188 (`c0332f8cceb9fe519f6bd19d8d3b15553626a33e`) added `robot_sf/benchmark/rare_event_sampling.py`, `scripts/benchmark/run_rare_event_estimation_issue_4163.py`, and `configs/benchmarks/rare_event/issue_4163_crossing_smoke.yaml`. | Diagnostic harness only; no full benchmark campaign claim. |
| Record sampled scenario rows with base probability, proposal probability, likelihood ratio, and sampled parameter hash. | Delivered | PR #4188 added `SampledScenarioRow` payloads and tests in `tests/benchmark/test_rare_event_sampling.py`. | The runner writes raw JSONL under ignored `output/`; tracked summaries preserve compact provenance. |
| Report estimated failure probability with confidence interval, standard error, effective sample size, and naive Monte Carlo comparison. | Delivered | PR #4188 added estimator fields in `docs/context/evidence/issue_4163_rare_event_smoke/summary.json`; PR #4233 (`bb674d86833fc8409f359adf93111e603fa27bb2`) added per-arm diagnostic estimates for static constriction in `docs/context/evidence/issue_4163_static_constriction_rare_event_smoke/summary.json`. | These are diagnostic smoke estimates, not benchmark-strength failure-rate evidence. |
| Apply the harness to one real failing family. | Delivered diagnostically | PR #4233 added `configs/benchmarks/rare_event/issue_4163_static_constriction_smoke.yaml` and static-constriction diagnostic evidence. | It binds the harness to the static-constriction family but remains a smoke run. |
| Validate estimates against a larger brute-force reference on one cheap family. | Blocked | PR #4453 (`82afa1658aecf29376695363b72586fd089b9997`) added explicit `empirical_reference` blocker contract in both issue #4163 smoke configs and fail-closed tests for incomplete available references. | Required evidence is still missing: cheap-family naive baseline, accelerated sample, and larger brute-force reference on the same family. |

## Closure Decision

Do not close #4163 from the currently merged evidence. The implementation has
delivered the harness, likelihood-ratio accounting, diagnostic estimates, and
static-constriction smoke application, but the original acceptance criterion for
brute-force validation remains blocked by missing empirical reference results.

Smallest remaining empirical slice:

1. Select one cheap family already covered by the smoke runner.
2. Run a CPU-only naive baseline, accelerated sample, and larger brute-force
   reference on the same family.
3. Record the reference evidence path under `empirical_reference.status:
   available`.
4. Keep the result diagnostic unless the run satisfies the benchmark evidence
   contract.

Out of scope for this audit: no full benchmark campaign, no Slurm or GPU
submission, no paper or dissertation claim edits, and no transient queue-routing
state in tracked files.
