# Hosted CI value audit — 2026-10-12

Repository: `ll7/robot_sf_ll7`; clean inventory base `33a4735900153e742828227cc2171436ea09cd90`. [Summary](summary.json) and [individual job observations](observations.csv) preserve the measured sample.

44 registered workflows, 5,571 runs, 11,391 job observations. Current checkout source exists for 32 workflows. Up to 200 runs **per workflow**, latest attempt only; sparse workflows use all available runs. REST collected 4,840 run histories; after primary REST quota exhaustion, GraphQL check-suite `LATEST` records completed the remaining 731. All use `gh api` with `GH_TOKEN` obtained from `gh auth token` only in process.

Runtime medians are execution minutes, not billed minutes. Workflow runtime spans first executing job to last completion and excludes initial queue time; job runtime uses started/completed timestamps. Failure rate excludes cancelled, skipped and unfinished outcomes. See per-workflow sample dates in JSON: this is a run-count window, not a fixed-date comparison. Historical names are listed even when no current job has that name. Runs still in progress at collection are reported without implying success.

Classification: **a** preserves real/plausible defect detection; **b** is bookkeeping or non-validation orchestration; **c** is exact duplicate validation. A zero failure rate cannot establish that a check has never caught a defect. Removed overhead has identical preserved checks; no unique functional, security, rights, seed or integrity check is deleted.

## Bounded changes and cost estimates

- **Scripts Catalog:** duplicate `validate_catalog()` and README render comparisons already execute in `tests/dev/test_scripts_catalog.py` through fast-feedback. Retain the `scripts-catalog` check name as a PR/push pass-through; run the standalone audit weekly/manual. Historical median 209.5 s; assuming a 10 s pass-through, expected saving **3.33 runner-minutes per applicable PR run** (scripts paths only), plus main-push savings. Weekly audit costs roughly 3.49 runner-minutes. No observed standalone failures in the 200-run sample.
- **Docs checks:** retain changed-path integrity and full docs-link scan, sharing checkout/Python/PyYAML. Preserve `docs-link-integrity` as a dependent alias that fails when the shared check fails. Historical separate full-scan job median 24 s and scan-step median 4 s; with a conservative 10 s alias, expected saving **0.17 runner-minutes per applicable docs PR run**: `(24 - 4 - 10) / 60`. Full scan still catches broken links in untouched docs. The observed changed-path failures reject unregistered evidence; that checker is retained.

These estimates include cancelled/skip exclusions and apply to one workflow execution, not an entire PR lifetime. Rounded per-job GitHub billing can erase the docs saving; no dollar saving is asserted.

## Retained bookkeeping and residual work

- Coverage percentage gates and the advisory new-tests-on-base job remain candidates in #10070. Changed-coverage proof is coupled to merge admission. This lane does not silently replace that policy with a passing check. Keep the current names and guards until a separate bounded reviewed change addresses the proof consumer. Test-value auditing also has current policy in #9986.
- Queue reports/scanners already run on schedules. Runner recovery and exact-check reconciliation preserve documented infrastructure repairs. Publication/delivery jobs have no PR admission cost and are outside CI-gate cleanup. GitHub-managed workflows require platform settings, which this task prohibits. Retired workflow records have no current source to edit.
- Linked #10070 PRs #10148, #10150 and #10152 are already closed/merged; fast-by-default admission is complete and is not rebuilt. Open #10196 is duration-upload repair, while #9903 carries old action pins; neither implements these two consolidations. They are left independent.
- #10315 config-hash tests, frozen artifacts, branch protection, required-check configuration and repo settings are untouched. Both preserved check names can be simplified by the author in a later settings decision; no such setting change is necessary for these PRs.

## Every registered workflow

| Workflow / source | Runs | Median min | Failure rate | Class |
| --- | ---: | ---: | --- | :---: |
| RobotSF actionable-change monitor<br>`.github/workflows/actionable-change-monitor.yml` | 200 | 3.80 | 0.0% (0/200) | b |
| Actionlint<br>`.github/workflows/actionlint.yml` | 200 | 0.22 | 0.0% (0/184) | a |
| Apply issue 8134 canonical lint correction (historical/platform source unavailable)<br>`.github/workflows/apply-8134-lint-v2.yml` | 2 | 0.29 | 50.0% (1/2) | b |
| Apply issue 8134 lint correction (historical/platform source unavailable)<br>`.github/workflows/apply-8134-lint.yml` | 3 | 0.27 | 100.0% (3/3) | b |
| Apply issue 8134 patch (historical/platform source unavailable)<br>`.github/workflows/apply-8134.yml` | 4 | 0.23 | 75.0% (3/4) | b |
| Apply issue 8677 projection repair (historical/platform source unavailable)<br>`.github/workflows/apply-issue-8677-projection-repair.yml` | 1 | 0.30 | 100.0% (1/1) | b |
| Asset Rights Inventory<br>`.github/workflows/asset-rights-inventory.yml` | 200 | 0.72 | 0.0% (0/132) | a |
| Benchmark Docker Repro Smoke<br>`.github/workflows/benchmark-docker-repro-smoke.yml` | 184 | 3.58 | 1.2% (2/166) | a |
| Blocker staleness report<br>`.github/workflows/blocker-staleness.yml` | 14 | 3.14 | 0.0% (0/14) | b |
| CI runner queue watchdog<br>`.github/workflows/ci-runner-watchdog.yml` | 194 | 2.27 | 3.2% (6/188) | b |
| CI<br>`.github/workflows/ci.yml` | 200 | 49.17 | 56.1% (60/107) | a |
| CodeQL<br>`.github/workflows/codeql.yml` | 200 | 3.95 | 0.0% (0/145) | a |
| Dependency Review<br>`.github/workflows/dependency-review.yml` | 200 | 0.27 | 6.5% (11/169) | a |
| Docs & Evidence Integrity<br>`.github/workflows/docs-evidence-integrity.yml` | 200 | 0.33 | 2.1% (3/146) | a |
| Docs Link Integrity<br>`.github/workflows/docs-link-integrity.yml` | 200 | 0.40 | 0.0% (0/122) | a |
| Evidence-registry ratchet<br>`.github/workflows/evidence-registry-ratchet.yml` | 200 | 3.20 | 11.4% (12/105) | a |
| Figure-render recurrence guard<br>`.github/workflows/figure-render-recurrence.yml` | 200 | 3.98 | 0.9% (1/116) | a |
| Main CI incident reconciliation<br>`.github/workflows/main-ci-incident-reconcile.yml` | 150 | 2.95 | 0.0% (0/149) | b |
| Merge Queue Gate<br>`.github/workflows/merge-queue-gate.yml` | 200 | — | — | a |
| Model License Preflight<br>`.github/workflows/model-license-preflight.yml` | 37 | 0.56 | 10.7% (3/28) | a |
| Mutation testing<br>`.github/workflows/mutation-testing.yml` | 12 | 3.52 | 41.7% (5/12) | a |
| Packaging Extras Matrix<br>`.github/workflows/packaging-extras.yml` | 200 | 7.27 | 0.6% (1/168) | a |
| Nightly Performance<br>`.github/workflows/perf-nightly.yml` | 200 | 3.52 | 49.6% (65/131) | a |
| PR body contracts (historical/platform source unavailable)<br>`.github/workflows/pr-body-contracts.yml` | 200 | 2.54 | 0.0% (0/110) | b |
| PR Contract Check<br>`.github/workflows/pr-contract-check.yml` | 200 | 3.14 | 60.2% (65/108) | a |
| PR Promoted Planner Smoke<br>`.github/workflows/pr-promoted-planner-smoke.yml` | 200 | 4.98 | 0.0% (0/150) | a |
| Publish docs site<br>`.github/workflows/publish-docs-site.yml` | 200 | 5.37 | 0.0% (0/146) | a |
| Release issue 8134 atomic claim (historical/platform source unavailable)<br>`.github/workflows/release-8134-claim.yml` | 1 | 0.17 | 0.0% (0/1) | b |
| Release Functional Badge Verification<br>`.github/workflows/release-functional-badge.yml` | 7 | 1.38 | 28.6% (2/7) | a |
| Release Legal Input Coverage<br>`.github/workflows/release-legal-input-coverage.yml` | 200 | 0.53 | 0.0% (0/172) | a |
| Route external review bots<br>`.github/workflows/review-bot-routing.yml` | 200 | 0.08 | 0.0% (0/152) | a |
| Scripts Catalog<br>`.github/workflows/scripts-catalog.yml` | 200 | 3.49 | 0.0% (0/142) | c |
| Security Baseline<br>`.github/workflows/security-baseline.yml` | 200 | 3.42 | 0.0% (0/156) | a |
| Build Immutable Software Candidate<br>`.github/workflows/software-candidate.yml` | 5 | 1.68 | 100.0% (5/5) | a |
| Promote Immutable Software Candidate<br>`.github/workflows/software-promotion.yml` | 0 | — | — | b |
| Strip closed state labels<br>`.github/workflows/strip-closed-state-labels.yml` | 200 | 0.23 | 0.0% (0/200) | b |
| Superseded draft scanner<br>`.github/workflows/superseded-draft-scanner.yml` | 65 | 3.30 | 0.0% (0/64) | b |
| Temporary Ruff diagnostics (historical/platform source unavailable)<br>`.github/workflows/tmp-ruff-diagnostics.yml` | 5 | 2.43 | 0.0% (0/5) | b |
| ty advisory ratchet<br>`.github/workflows/ty-advisory-ratchet.yml` | 200 | 3.78 | 15.5% (27/174) | a |
| OpenAI Codex (historical/platform source unavailable)<br>`dynamic/agents/openai-code-agent` | 1 | 8.25 | 0.0% (0/1) | b |
| Copilot code review (historical/platform source unavailable)<br>`dynamic/copilot-pull-request-reviewer/copilot-pull-request-reviewer` | 52 | 2.77 | 3.8% (2/52) | b |
| Copilot coding agent (historical/platform source unavailable)<br>`dynamic/copilot-swe-agent/copilot` | 32 | 15.07 | 9.7% (3/31) | b |
| Dependabot Updates (historical/platform source unavailable)<br>`dynamic/dependabot/dependabot-updates` | 60 | 4.38 | 35.0% (21/60) | b |
| Dependency Graph (historical/platform source unavailable)<br>`dynamic/dependabot/update-graph` | 142 | 0.94 | 0.7% (1/142) | b |

## Every observed job

| Workflow / job | Executions / observations | Median min | Failure rate | Class |
| --- | ---: | ---: | --- | :---: |
| RobotSF actionable-change monitor / `monitor` | 200/200 | 3.80 | 0.0% (0/200) | b |
| Actionlint / `actionlint` | 184/200 | 0.22 | 0.0% (0/184) | a |
| Apply issue 8134 canonical lint correction / `apply-canonical-lint` | 2/2 | 0.29 | 50.0% (1/2) | b |
| Apply issue 8134 lint correction / `apply-lint-correction` | 3/3 | 0.27 | 100.0% (3/3) | b |
| Apply issue 8134 patch / `apply-patch` | 4/4 | 0.23 | 75.0% (3/4) | b |
| Apply issue 8677 projection repair / `apply` | 1/1 | 0.30 | 100.0% (1/1) | b |
| Asset Rights Inventory / `classify tracked assets` | 132/200 | 0.72 | 0.0% (0/132) | a |
| Benchmark Docker Repro Smoke / `docker-repro-smoke` | 166/184 | 3.58 | 1.2% (2/166) | a |
| Blocker staleness report / `blocker-staleness` | 14/14 | 3.14 | 0.0% (0/14) | b |
| CI runner queue watchdog / `recover` | 188/194 | 2.27 | 3.2% (6/188) | b |
| CI / `browser-witnesses` | 106/172 | 3.88 | 0.0% (0/106) | a |
| CI / `changed-coverage-gate` | 32/180 | 4.12 | 9.4% (3/32) | b |
| CI / `ci` | 155/157 | 0.23 | 45.2% (70/155) | a |
| CI / `ci-draft` | 23/23 | 0.07 | 13.0% (3/23) | a |
| CI / `compat-matrix` | 0/48 | — | — | a |
| CI / `compat-matrix (macos-latest, 3.11)` | 119/142 | 8.78 | 0.8% (1/119) | a |
| CI / `compat-matrix (macos-latest, 3.13)` | 119/142 | 8.63 | 0.8% (1/119) | a |
| CI / `compat-matrix (ubuntu-latest, 3.11)` | 108/142 | 9.38 | 0.0% (0/108) | a |
| CI / `compat-matrix (ubuntu-latest, 3.13)` | 107/142 | 9.37 | 0.9% (1/107) | a |
| CI / `coverage-gate` | 1/180 | 7.28 | 0.0% (0/1) | b |
| CI / `determinism-gate` | 106/186 | 0.33 | 0.0% (0/106) | a |
| CI / `dispatch-ownership` | 171/192 | 0.48 | 5.3% (9/171) | a |
| CI / `exact-repeat-model-preflight` | 114/190 | 4.72 | 0.0% (0/114) | a |
| CI / `examples-smoke` | 76/186 | 9.50 | 7.9% (6/76) | a |
| CI / `fast-feedback` | 0/67 | — | — | a |
| CI / `fast-feedback (1)` | 86/119 | 36.31 | 29.1% (25/86) | a |
| CI / `fast-feedback (2)` | 87/119 | 34.12 | 27.6% (24/87) | a |
| CI / `fast-feedback (3)` | 89/119 | 35.17 | 38.2% (34/89) | a |
| CI / `fast-feedback (4)` | 87/119 | 35.30 | 21.8% (19/87) | a |
| CI / `fast-feedback (5)` | 85/119 | 36.43 | 25.9% (22/85) | a |
| CI / `fast-feedback (6)` | 86/119 | 34.72 | 31.4% (27/86) | a |
| CI / `fast-pysf-compat` | 122/190 | 0.23 | 0.0% (0/122) | a |
| CI / `new-tests-fail-on-base` | 100/190 | 4.29 | 0.0% (0/100) | b |
| CI / `notebooks-smoke` | 78/186 | 4.88 | 0.0% (0/78) | a |
| CI / `reproducibility-check` | 97/190 | 3.95 | 0.0% (0/97) | a |
| CI / `reproducibility-check-reconciliation` | 95/186 | 0.25 | 1.1% (1/95) | b |
| CI / `runner-availability` | 115/186 | 0.22 | 0.0% (0/115) | a |
| CI / `scenario-validation` | 119/190 | 3.58 | 1.7% (2/119) | a |
| CI / `self-hosted-admission` | 171/192 | 0.22 | 0.0% (0/171) | a |
| CI / `smoke-artifacts` | 108/186 | 5.35 | 0.0% (0/108) | a |
| CI / `wheel-smoke-install` | 107/186 | 6.40 | 0.0% (0/107) | a |
| CI / `xdist-scratch-isolation` | 86/190 | 5.49 | 0.0% (0/86) | a |
| CodeQL / `Analyze Python` | 145/200 | 3.95 | 0.0% (0/145) | a |
| Dependency Review / `review` | 169/200 | 0.27 | 6.5% (11/169) | a |
| Docs & Evidence Integrity / `docs-evidence-integrity` | 146/200 | 0.33 | 2.1% (3/146) | a |
| Docs Link Integrity / `docs-link-integrity` | 122/200 | 0.40 | 0.0% (0/122) | a |
| Evidence-registry ratchet / `evidence-registry-ratchet` | 105/200 | 3.20 | 11.4% (12/105) | a |
| Figure-render recurrence guard / `figure-render-recurrence` | 116/200 | 3.98 | 0.9% (1/116) | a |
| Main CI incident reconciliation / `reconcile` | 149/150 | 2.95 | 0.0% (0/149) | b |
| Merge Queue Gate / `merge-queue-gate` | 0/200 | — | — | a |
| Model License Preflight / `model-license-preflight` | 28/37 | 0.56 | 10.7% (3/28) | a |
| Mutation testing / `mutation-testing` | 12/12 | 3.52 | 41.7% (5/12) | a |
| Packaging Extras Matrix / `clean wheel [${{ matrix.extra }}]` | 0/32 | — | — | a |
| Packaging Extras Matrix / `clean wheel [all]` | 168/168 | 5.90 | 0.0% (0/168) | a |
| Packaging Extras Matrix / `clean wheel [benchmark]` | 168/168 | 3.56 | 0.6% (1/168) | a |
| Packaging Extras Matrix / `clean wheel [maps]` | 168/168 | 3.73 | 0.0% (0/168) | a |
| Packaging Extras Matrix / `clean wheel [training]` | 168/168 | 4.59 | 0.0% (0/168) | a |
| Packaging Extras Matrix / `clean wheel [viz]` | 168/168 | 3.58 | 0.0% (0/168) | a |
| Nightly Performance / `perf-nightly` | 131/200 | 3.52 | 49.6% (65/131) | a |
| PR body contracts / `pr-body-contracts` | 111/198 | 2.55 | 0.0% (0/111) | b |
| PR Contract Check / `pr-contract-check` | 108/195 | 3.14 | 60.2% (65/108) | a |
| PR Promoted Planner Smoke / `promoted-planner-smoke` | 150/199 | 4.98 | 0.0% (0/150) | a |
| Publish docs site / `build-site` | 146/200 | 5.23 | 0.0% (0/146) | a |
| Publish docs site / `deploy-site` | 25/200 | 0.17 | 0.0% (0/25) | b |
| Release issue 8134 atomic claim / `release-claim` | 1/1 | 0.17 | 0.0% (0/1) | b |
| Release Functional Badge Verification / `verify-badge` | 7/7 | 1.38 | 28.6% (2/7) | a |
| Release Legal Input Coverage / `release-legal-input-coverage` | 172/200 | 0.53 | 0.0% (0/172) | a |
| Route external review bots / `route-coderabbit` | 152/200 | 0.08 | 0.0% (0/152) | a |
| Scripts Catalog / `scripts-catalog` | 142/200 | 3.49 | 0.0% (0/142) | c |
| Security Baseline / `ruff-security-advisory` | 156/200 | 3.42 | 0.0% (0/156) | a |
| Build Immutable Software Candidate / `Build and verify once` | 5/5 | 1.68 | 100.0% (5/5) | a |
| Promote Immutable Software Candidate / no recorded job | 0/0 | — | — | b |
| Strip closed state labels / `strip-state-labels` | 200/200 | 0.23 | 0.0% (0/200) | b |
| Superseded draft scanner / `scan-superseded-drafts` | 64/65 | 3.30 | 0.0% (0/64) | b |
| Temporary Ruff diagnostics / `diagnostics` | 5/5 | 2.43 | 0.0% (0/5) | b |
| ty advisory ratchet / `ty-advisory-ratchet` | 175/199 | 3.78 | 15.4% (27/175) | a |
| OpenAI Codex / `codex` | 1/1 | 8.25 | 0.0% (0/1) | b |
| Copilot code review / `Agent` | 52/52 | 2.21 | 1.9% (1/52) | b |
| Copilot code review / `Autovalidate` | 0/33 | — | — | b |
| Copilot code review / `Cleanup artifacts` | 52/52 | 0.07 | 1.9% (1/52) | b |
| Copilot code review / `CodeQL analysis` | 0/2 | — | — | b |
| Copilot code review / `CodeQL analysis (python)` | 31/31 | 1.47 | 3.2% (1/31) | b |
| Copilot code review / `Prepare` | 47/47 | 0.07 | 0.0% (0/47) | b |
| Copilot code review / `Upload results` | 52/52 | 0.17 | 0.0% (0/52) | b |
| Copilot code review / `prepare` | 5/5 | 0.08 | 0.0% (0/5) | b |
| Copilot coding agent / `copilot` | 31/32 | 15.07 | 9.7% (3/31) | b |
| Dependabot Updates / `Dependabot` | 60/60 | 4.38 | 35.0% (21/60) | b |
| Dependency Graph / `update-uv-graph` | 142/142 | 0.94 | 0.7% (1/142) | b |

## Current job declarations

The observation table expands actual matrix names. These declarations also list jobs with no recorded execution; their runtime/failure rate is unavailable, not zero. Display-name expressions are templates, not a measured job identity.

| Workflow / job ID | Display name | Class |
| --- | --- | :---: |
| RobotSF actionable-change monitor / `monitor` | `monitor` | b |
| Actionlint / `actionlint` | `actionlint` | a |
| Asset Rights Inventory / `inventory` | `classify tracked assets` | a |
| Benchmark Docker Repro Smoke / `docker-repro-smoke` | `docker-repro-smoke` | a |
| Blocker staleness report / `blocker-staleness` | `blocker-staleness` | b |
| CI runner queue watchdog / `recover` | `recover` | b |
| CI / `self-hosted-admission` | `self-hosted-admission` | a |
| CI / `dispatch-ownership` | `dispatch-ownership` | a |
| CI / `runner-availability` | `runner-availability` | a |
| CI / `fast-feedback` | `fast-feedback` | a |
| CI / `browser-witnesses` | `browser-witnesses` | a |
| CI / `changed-coverage-gate` | `changed-coverage-gate` | b |
| CI / `coverage-gate` | `coverage-gate` | b |
| CI / `compat-matrix` | `compat-matrix` | a |
| CI / `fast-pysf-compat` | `fast-pysf-compat` | a |
| CI / `smoke-artifacts` | `smoke-artifacts` | a |
| CI / `scenario-validation` | `scenario-validation` | a |
| CI / `new-tests-fail-on-base` | `new-tests-fail-on-base` | b |
| CI / `reproducibility-check` | `reproducibility-check` | a |
| CI / `reproducibility-check-reconciliation` | `reproducibility-check-reconciliation` | b |
| CI / `xdist-scratch-isolation` | `xdist-scratch-isolation` | a |
| CI / `wheel-smoke-install` | `wheel-smoke-install` | a |
| CI / `examples-smoke` | `examples-smoke` | a |
| CI / `notebooks-smoke` | `notebooks-smoke` | a |
| CI / `determinism-gate` | `determinism-gate` | a |
| CI / `exact-repeat-model-preflight` | `exact-repeat-model-preflight` | a |
| CI / `ci` | `${{ github.event_name == 'pull_request' && github.event.pull_request.draft && 'ci-draft' || 'ci' }}` | a |
| CodeQL / `analyze` | `Analyze Python` | a |
| Dependency Review / `review` | `review` | a |
| Docs & Evidence Integrity / `docs-evidence-integrity` | `docs-evidence-integrity` | a |
| Docs Link Integrity / `docs-link-integrity` | `docs-link-integrity` | a |
| Evidence-registry ratchet / `evidence-registry-ratchet` | `evidence-registry-ratchet` | a |
| Figure-render recurrence guard / `figure-render-recurrence` | `figure-render-recurrence` | a |
| Main CI incident reconciliation / `reconcile` | `reconcile` | b |
| Merge Queue Gate / `merge-queue-gate` | `merge-queue-gate` | a |
| Model License Preflight / `model-license-preflight` | `model-license-preflight` | a |
| Mutation testing / `mutation-testing` | `mutation-testing` | a |
| Packaging Extras Matrix / `clean-wheel-extra` | `clean wheel [${{ matrix.extra }}]` | a |
| Nightly Performance / `perf-nightly` | `perf-nightly` | a |
| PR Contract Check / `pr-contract-check` | `pr-contract-check` | a |
| PR Promoted Planner Smoke / `promoted-planner-smoke` | `promoted-planner-smoke` | a |
| Publish docs site / `build-site` | `build-site` | a |
| Publish docs site / `deploy-site` | `deploy-site` | b |
| Release Functional Badge Verification / `verify-badge` | `verify-badge` | a |
| Release Legal Input Coverage / `release-legal-input-coverage` | `release-legal-input-coverage` | a |
| Route external review bots / `route-coderabbit` | `route-coderabbit` | a |
| Scripts Catalog / `scripts-catalog` | `scripts-catalog` | c |
| Security Baseline / `ruff-security-advisory` | `ruff-security-advisory` | a |
| Build Immutable Software Candidate / `build-candidate` | `Build and verify once` | a |
| Promote Immutable Software Candidate / `verify-candidate` | `Verify immutable candidate` | b |
| Promote Immutable Software Candidate / `testpypi-upload` | `Publish exact bytes to TestPyPI` | b |
| Promote Immutable Software Candidate / `testpypi-cold-install` | `Verify TestPyPI public-index install` | b |
| Promote Immutable Software Candidate / `pypi-upload` | `Publish exact bytes to PyPI` | b |
| Promote Immutable Software Candidate / `pypi-cold-verify` | `Verify PyPI public-index bytes` | b |
| Strip closed state labels / `strip-state-labels` | `strip-state-labels` | b |
| Superseded draft scanner / `scan-superseded-drafts` | `scan-superseded-drafts` | b |
| ty advisory ratchet / `ty-advisory-ratchet` | `ty-advisory-ratchet` | a |

Per-job reasons, skipped/cancelled/incomplete counts and sample freshness are in [summary.json](summary.json). GitHub job provenance is reconstructable from each observation as `https://github.com/ll7/robot_sf_ll7/actions/runs/<run_id>/job/<job_id>`. GraphQL records lack runner labels and failed-step details; this is explicitly recorded as unavailable, rather than guessed. Raw API responses and private validation logs remain outside the public report.
