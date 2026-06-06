---
name: Research / Experiment
about: Propose a research task or experiment with reproducible validation
title: ""
labels: ["research"]
assignees: []
---

## Goal / Problem

**Research question**
<!-- What specific question are we trying to answer? -->

**Hypothesis**
<!-- What do we expect to find or prove? -->

**Evidence grade / claim class**
<!-- Mark observed, inferred, speculative, blocked, diagnostic-only, benchmark, or paper-facing. -->

**Competing explanations**
<!-- List plausible alternatives this work should distinguish, not just the preferred hypothesis. -->
<!-- Use "N/A - blocked: ..." when blocked prerequisites prevent this comparison. -->
1.
2.
3.

**Discriminating evidence**
<!-- What trace, metric, table, diagnostic, or experiment would separate the explanations above? -->
<!-- Use "N/A - blocked: ..." when no discriminating evidence can be gathered yet. -->

**Mechanism signal required**
<!-- Which fields or observable mechanism signals must change for this to move the claim boundary? -->
<!-- Use "N/A - support/docs-only: ..." when this is research-workflow support, not a mechanism probe. -->

**Stop rule**
<!-- When should this mechanism be retired, narrowed, marked blocked, or kept diagnostic-only? -->
<!-- If the result is blocked or diagnostic-only, state why and keep the claim class unchanged. -->

**Scientific motivation**
<!-- Why is this research important? What gap does it fill? -->

## Scope

- Independent variables:
- Dependent variables:
- Control variables:
- In scope:
- Out of scope:

## Added Value Estimation

- Research value:
- Benchmark value:
- Publication value:
- Why now:

## Effort Estimation

- Rough estimate (hours):
- Best estimate (hours):
- Unknowns:

## Complexity Estimation

- Experimental complexity:
- Dependencies:
- Open questions:

## Risk Assessment

- Statistical risk:
- Reproducibility risk:
- Runtime risk:
- Mitigation:

## Affected Files

- `experiments/experiment_name/` - Experiment directory structure.
- `experiments/experiment_name/run_experiment.py` - Main experiment script.
- `experiments/experiment_name/config.py` - Experiment configuration.
- `experiments/experiment_name/analysis.py` - Analysis script.

## Definition of Done

- [ ] Experimental setup is reproducible.
- [ ] Baseline comparison is documented.
- [ ] Competing explanations and discriminating evidence are addressed, or the result is marked blocked/diagnostic-only.
- [ ] Evidence grade or claim class is explicit.
- [ ] The mechanism signal and stop rule are recorded before any claim-strength upgrade.
- [ ] Results are recorded and interpreted.
- [ ] Analysis artifacts can be regenerated from versioned inputs.

## Success Metrics

- The experiment can be rerun from a documented command.
- The results are reproducible across repeated runs with the same seed.
- The analysis output supports the research question.

## Validation / Testing

- [ ] Random seeds are set for reproducible results.
- [ ] Environment configuration is documented.
- [ ] Dependencies are pinned or versioned.
- [ ] The experiment runs from a single canonical command.

## Estimate Discussion

- Why these values were proposed:
- Uncertainty / confidence:
- What evidence would move the estimate:

## Project Metadata

- Priority:
- Effort (h):
- Reviewed:
