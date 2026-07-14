# Pre-Registration Inference Contract Template

## Why this matters

The S30 interpretation work (issue #4882, packet PR #5515) demonstrated that a preregistration
which declares neither the bootstrap resampling unit nor the decision rule produces **different
branch verdicts** depending on which unspecified methodological choices a later analyst makes:

- Scenario-clustered vs seed-block resampling produced overlapping vs non-overlapping CIs
- CI-overlap vs paired-delta rules produced Boundary vs Separation verdicts

This template section prevents that ambiguity by requiring every comparative preregistration
to freeze its inference contract *before* the campaign executes.

## Template section

Add the following YAML section to your preregistration config (or the equivalent in a Markdown
preregistration). Every field is required for comparative campaigns:

```yaml
inference_contract:
  # 1. Resampling unit: what do you resample during bootstrap?
  #    Describe the hierarchy and WHY this unit matches the inference target.
  resampling_unit:
    method: >-        # e.g. "scenario-clustered hierarchical bootstrap" or
                      # "seed-block (fixed 48-scenario suite)"
    rationale: >-     # e.g. "Treat scenarios as the outer unit because
                      # between-scenario heterogeneity is large and we want
                      # to generalize to unseen scenarios."
    bootstrap_confidence: 0.95  # optional but recommended
    resampling_order: "scenario→seed"  # optional hierarchy description

  # 2. Inference population: is the scenario suite fixed or sampled?
  #    This determines whether CIs reflect only stochastic uncertainty or
  #    also between-scenario variability.
  inference_population:
    type: fixed_suite    # OR: sampled_population
    rationale: >-        # e.g. "These 48 curated scenarios ARE the benchmark.
                      # We quantify seed/stochastic uncertainty only."

  # 3. Estimand: what are you estimating?
  estimand:
    type: paired_delta   # OR: per_arm_interval | ratio | paired_delta_and_per_arm
    description: >-       # e.g. "Paired per-episode delta between treatment and
                      # control, aggregated via hierarchical bootstrap."

  # 4. Decision rule: the exact criterion for the branch verdict.
  decision_rule:
    rule: >-             # e.g. "CI-excludes-zero" or "p<0.05 two-sided"
    threshold: >-        # e.g. "The 95% bootstrap CI for the paired delta
                      # must exclude zero for the intervention to be supported."

  # 5. Primary metrics: what metrics does the decision rule apply to?
  primary_metrics:
    metrics:             # List of metric keys, ordered by importance
      - success_rate
      - collision_rate
    ordered_by_importance: true  # true if first metric is primary tiebreaker

  # 6. Multiplicity handling: how do you control for multiple tests?
  multiplicity_handling:
    strategy: >-         # e.g. "Holm-Bonferroni adjustment across the 3
                      # preregistered contrasts to control FWER at alpha=0.05."
    rationale: >-        # e.g. "Without adjustment, three contrasts at alpha=0.05
                      # would inflate family-wise error to ~14%."
    adjustment_method: holm_bonferroni  # optional: holm_bonferroni, bonferroni,
                                        # hochberg, holmdunn, none_single_metric,
                                        # none_single_contrast
```

## Checklist before campaign submission

Every preregistration must satisfy all six fields above. The shared checker
(`scripts/validation/check_preregistration_inference_contract.py`) validates
the section is present and well-formed:

```bash
python scripts/validation/check_preregistration_inference_contract.py \
  configs/benchmarks/my_preregistration.yaml
```

## Decision rule examples by estimand

- **Paired delta:** "The 95% bootstrap CI for the treatment-control paired delta
  on the primary metric must not include zero."
- **Per-arm interval:** "The leading treatment arm's 95% per-arm CI must not overlap
  the control arm's 95% CI on the primary metric."
- **Ratio:** "The treatment/control effect ratio must exceed 1.5 with 90% lower
  confidence bound."
- **Paired delta + per-arm:** "Both the paired-delta CI excludes zero AND per-arm
  CIs separate on the primary metric."

## Relationship to existing fields

Some preregistrations already encode parts of this contract under other keys:
- `uncertainty.method` / `resampling_order` → maps to `resampling_unit`
- `outcomes.primary_in_order` → maps to `primary_metrics`
- `decision_rule` → maps to `decision_rule`
- `multiplicity_policy` → maps to `multiplicity_handling`

If those already exist, you can still add the `inference_contract` section to
formalize the four checklist items explicitly. The checker validates the
dedicated section, not legacy aliases.