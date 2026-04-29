---
name: Issue 791 — paper claim is narrow benchmark-set, not OOD generalization
description: Maintainer chose option (a) for the issue-791 promotion story; no held-out OOD suite, and manuscript text must avoid generalization/transfer/unseen language.
type: project
---

Maintainer decision on 2026-04-20: the issue-791 PPO promotion will be reported as
**a strong policy evaluated on a broad benchmark of social-navigation scenarios**,
not as a generalization or transfer claim.

**Why:** the project priority is to land a publication-grade benchmark result with
a strong candidate and broad scenario coverage. An OOD study adds weeks of design
and execution for a claim the paper does not need to make. Seed variance on the
benchmark matrix + clean bootstrap CIs are sufficient evidence for the narrow
framing.

**How to apply:**

- In manuscript / issue / PR text, **do not** use the words "generalize",
  "transfer", "unseen", or "novel environment". Prefer "benchmark-set performance",
  "on the scenario matrix", "across seeds".
- Do not contrast the old 0.586 plateau against the 0.929 leader as a "fixed
  generalization gap" externally. Internally the attribution (distribution
  alignment dominates over curriculum / capacity / foresight) is still true and
  useful for engineering decisions, but it does not belong in paper framing.
- The OOD suite scaffold at
  `docs/context/issue_791_ood_holdout_suite_design.md` is retained as a deprioritized
  design record, not an active work item.
- Strength of the primary claim rests on: (i) breadth of the camera-ready scenario
  matrix, (ii) seed-variance tightness across 10M replicas (jobs 11872 / 11873),
  (iii) clean bootstrap CIs and SNQI contract compliance from the benchmark rerun
  (job 11871).
