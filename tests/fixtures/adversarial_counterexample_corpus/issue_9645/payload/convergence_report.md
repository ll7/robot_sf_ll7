# Falsification search convergence report

This diagnostic report summarizes persisted search attempts and their best-so-far objective by evaluation budget.

- Claim scope: `diagnostic_only_finite_search_budget`.
- Comparison input SHA-256: `7c8e7d0f6c74cef38bc776d24d19737176eb9eaebcac99ff1a91faeda302e02d`.
- Archived manifest path map: `docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/pilot_report_manifest_path_map.v1.json`; SHA-256 `692db3b44f12b1fd4d0d7ff506b8924e4009710b5062e71c30840ce74e4a8b28`; bound comparison SHA-256 `7c8e7d0f6c74cef38bc776d24d19737176eb9eaebcac99ff1a91faeda302e02d`.
- Source revision status: `unknown`.
- Exact source revision: `unknown`.

## Per-run accounting

| Objective | Method | Seed | Budget | Search runtime (s) | Best observed ≤B | Best eligible ≤B | Trace evidence: eligible / ineligible (≤B) | Criticality: critical / not-critical / unknown (≤B) | Collision/intrusion tier: critical / not-critical / unknown (≤B) | Known critical (all rows) | Eligible critical ≤B | First eligible critical eval | Over-budget rows | Valid / invalid / failed / scoreless / missing ≤B / all expected | Duplicates ≤B | Execution modes | Availability | Run input status | Artifact |
|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `constraints_first_lexicographic_v1` | Random | 1101 | 16 | Not recorded | 0 | Not recorded | 0 / 16 | 0 / 0 / 16 | 0 / 0 / 16 | 0 | 0 | None recorded | 0 | 16 / 0 / 0 / 0 / 0 / 0 | 0/16 | native: 16 | available: 16 | input checks passed | available |
| `constraints_first_lexicographic_v1` | TPE | 1101 | 16 | Not recorded | 0 | Not recorded | 0 / 16 | 0 / 0 / 16 | 0 / 0 / 16 | 0 | 0 | None recorded | 0 | 16 / 0 / 0 / 0 / 0 / 0 | 0/16 | native: 16 | available: 16 | input checks passed | available |
| `constraints_first_lexicographic_v1` | Random | 2202 | 16 | Not recorded | 0 | Not recorded | 0 / 16 | 0 / 0 / 16 | 0 / 0 / 16 | 0 | 0 | None recorded | 0 | 16 / 0 / 0 / 0 / 0 / 0 | 0/16 | native: 16 | available: 16 | input checks passed | available |
| `constraints_first_lexicographic_v1` | TPE | 2202 | 16 | Not recorded | 0 | Not recorded | 0 / 16 | 0 / 0 / 16 | 0 / 0 / 16 | 0 | 0 | None recorded | 0 | 16 / 0 / 0 / 0 / 0 / 0 | 0/16 | native: 16 | available: 16 | input checks passed | available |

Budget-limited summaries use only the first B comparison-indexed candidate rows. Extra rows remain in JSON audit history and cannot alter best-so-far values or paired deltas.

Trace evidence counts use the report's byte-verified artifact eligibility check; the producer's `analysis_eligibility` receipt is retained separately in JSON. Criticality columns report known critical, known non-critical, and unknown candidate counts separately. Collision/severe-intrusion tier counts require explicit evidence for both components to report `not_critical`; a missing, malformed, or contradictory component remains `unknown`. Execution eligibility does not establish safety-tier completeness.

Eligible best/critical summaries require a scored objective, `execution_mode=native`, `readiness_status=native`, `availability_status=available`, a parseable episode-record artifact, an effective-scenario hash, and an explicit `analysis_eligibility.eligible=true` receipt; contradictory or incomplete evidence stays ineligible. Per-evaluation reason codes identify failed checks.

Valid candidates within B are derived as `candidate rows within B - invalid - failed`; scoreless valid evaluations remain in that count. Missing and over-budget attempts remain explicit.

## Matched Random vs TPE

| Objective | Budget | Matched eligible seeds | Excluded matched seeds | TPE − Random median | Observed delta range | Inference |
|---|---:|---:|---:|---:|---:|---|
| `constraints_first_lexicographic_v1` | 16 | 0 | 2 | Not recorded | Not available | Not performed; descriptive only |

Run input status reports per-run index/manifest/config checks only; it does not assert that a Random/TPE pair exists or qualifies. Pairs are decided separately and require index/manifest agreement, matching normalized scenario/search/planner configuration, no over-budget rows, complete budgeted evaluations, and an analysis-eligible score from both methods. Missing, unmatched, ambiguous, and counterpart-ineligible cases remain in the JSON with reason codes.

## Figures

![Best-so-far convergence by evaluation budget](convergence_constraints_first_lexicographic_v1.png)

## Interpretation limits

- A finite search budget that finds no counterexample does not establish that none exists.
- Seed/run aggregates are descriptive; observed min/max ranges are not confidence intervals.
- No inferential test is performed, and small pilot seed counts do not support broad claims.
- Search-level runtime is reported only when a finite nonnegative runtime_seconds field is recorded in the search manifest or its summary; comparison-row fields are ignored.
- The current runner's legacy num_valid_candidates summary omits evaluator failures; this report derives valid as candidate rows minus invalid minus failed and retains the legacy field for audit.
- A recorded successful outcome or zero historical v1 score does not establish a negative collision/severe-intrusion tier when either component is absent or contradictory; the report preserves that tier as unknown.
- Fixture tests verify report accounting; finite-budget reports do not establish planner safety, search-space coverage, or absence of counterexamples.

## Provenance

Comparison artifact: `docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/pilot_comparison.json`.

- `random:1101:16:constraints_first_lexicographic_v1:row1`: declared `output/issue9645-pilot/results/constraints_first_lexicographic_v1/budget_0016/seed_1101/random/manifest.json`; archived `docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_manifests/random_seed_1101.json`; SHA-256 `fad937010131d1d12ed97260966383f9c19a41cceefe099f2d76a2c2cce17c13`; config SHA-256 `f174f22672aa05d222f7ce6bb20ecf64a042fb73dca9449dbaf7c44dc3e4ec38`; status `available`.
  Mapping provenance: `docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/pilot_report_manifest_path_map.v1.json`; path-map SHA-256 `692db3b44f12b1fd4d0d7ff506b8924e4009710b5062e71c30840ce74e4a8b28`; archived SHA-256 `fad937010131d1d12ed97260966383f9c19a41cceefe099f2d76a2c2cce17c13`.
- `optuna:1101:16:constraints_first_lexicographic_v1:row2`: declared `output/issue9645-pilot/results/constraints_first_lexicographic_v1/budget_0016/seed_1101/optuna/manifest.json`; archived `docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_manifests/tpe_seed_1101.json`; SHA-256 `00244a3d158079ef36de6e301fa6348b50d879a4a6f48492389776b8b411e9a0`; config SHA-256 `a3e0978a698fa7d3696c101203b39a15702447906249e46f77d80ca2c7b9e6ea`; status `available`.
  Mapping provenance: `docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/pilot_report_manifest_path_map.v1.json`; path-map SHA-256 `692db3b44f12b1fd4d0d7ff506b8924e4009710b5062e71c30840ce74e4a8b28`; archived SHA-256 `00244a3d158079ef36de6e301fa6348b50d879a4a6f48492389776b8b411e9a0`.
- `random:2202:16:constraints_first_lexicographic_v1:row3`: declared `output/issue9645-pilot/results/constraints_first_lexicographic_v1/budget_0016/seed_2202/random/manifest.json`; archived `docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_manifests/random_seed_2202.json`; SHA-256 `d15846cae0423334604b39efa65a15ec8842622edb992b397bf29f973a71fd0c`; config SHA-256 `84de14ba637674502970acd646e915dcc27b85ce454a897beb540af6e8e44162`; status `available`.
  Mapping provenance: `docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/pilot_report_manifest_path_map.v1.json`; path-map SHA-256 `692db3b44f12b1fd4d0d7ff506b8924e4009710b5062e71c30840ce74e4a8b28`; archived SHA-256 `d15846cae0423334604b39efa65a15ec8842622edb992b397bf29f973a71fd0c`.
- `optuna:2202:16:constraints_first_lexicographic_v1:row4`: declared `output/issue9645-pilot/results/constraints_first_lexicographic_v1/budget_0016/seed_2202/optuna/manifest.json`; archived `docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/source_manifests/tpe_seed_2202.json`; SHA-256 `1ace494078f174c1e9cd454ca97c60e2408343e192bc2c3d4f1408128917b00b`; config SHA-256 `7d91ed1a1a7c319bcba735813152b19ab1b202d018b7b4322fbe9e2261663e75`; status `available`.
  Mapping provenance: `docs/context/evidence/issue_9645_bounded_falsification_2026-09-24/payload/pilot_report_manifest_path_map.v1.json`; path-map SHA-256 `692db3b44f12b1fd4d0d7ff506b8924e4009710b5062e71c30840ce74e4a8b28`; archived SHA-256 `1ace494078f174c1e9cd454ca97c60e2408343e192bc2c3d4f1408128917b00b`.
