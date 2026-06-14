# Issue #2724 - Adversarial Candidate Scout Register

**Status:** diagnostic-only, not benchmark evidence
**Parent issue:** [#2724](https://github.com/ll7/robot_sf_ll7/issues/2724)
**Negative-result source:** NR-001, `issue-2716-topology-reselection-cross-slice`
**Schema:** `robot_sf/benchmark/schemas/generated_scenario_candidate.v1.json`
**Claim boundary:** single scout candidate; no adversarial diversity or benchmark-strength claim

---

## Purpose

This directory holds a single trace-seeded diagnostic candidate derived from the negative-result
register. NR-001 recorded that progress-gated topology reselection reached `horizon_exhausted` on
all hard non-canonical slices, including `bottleneck_transfer`. This candidate encodes one bounded
successor perturbation so the `generated_scenario_candidate.v1` path can be validated without
claiming broad adversarial diversity, benchmark-strength evidence, or paper-facing evidence.

## Candidate

| Field | Value |
|---|---|
| `candidate_id` | `issue-2724-nr001-bottleneck-route-offset-001` |
| `generator_family` | `heuristic_perturbation` |
| `negative_result_source` | NR-001, `issue-2716-topology-reselection-cross-slice` |
| `source_scenario` | `bottleneck_transfer/classic_bottleneck_medium` |
| `perturbation` | `robot_route_offset` (dx_m=0.25, dy_m=0.0, max_magnitude_m=0.5) |
| `promotion_status` | `not_promoted` |
| `scenario_certified` | `false` |
| `severity_metrics` | null (unevaluated) |
| `diversity_metrics` | null (unevaluated) |

## Validation

```bash
uv run python scripts/validation/validate_generated_scenario_candidate.py \
  docs/context/evidence/issue_2724_adversarial_candidate/candidate.json
uv run pytest tests/docs/test_generated_scenario_candidate_register.py -q
uv run pytest tests/benchmark/test_generated_scenario_candidate_schema.py -q
```

## Claim Boundary

- This is a **scout diagnostic** to verify schema encoding and validation plumbing.
- The source scenario is **not certified** for benchmark claims in this candidate packet.
- The candidate is derived from a `revise`-classified negative-result entry and does not rerun or
  repair the failed topology mechanism.
- Severity and diversity metrics are **unevaluated** (null).
- `promotion_status` is **not_promoted**.
- Do **not** cite this candidate as adversarial diversity evidence, benchmark-strength
  evidence, or paper-facing evidence.
