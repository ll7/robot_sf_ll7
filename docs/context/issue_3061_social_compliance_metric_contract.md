# Issue #3061 Social-Compliance Metric Contract (2026-06-18)

Issue: [#3061](https://github.com/ll7/robot_sf_ll7/issues/3061)

Status date: 2026-06-18

## Goal

Define the first versioned contract for robot-influence and social-compliance diagnostics before
downstream pedestrian-model, fairness, and influence campaigns start inventing incompatible metric
definitions.

## Contract

The machine-readable contract lives at:

```text
configs/benchmarks/social_compliance_metric_contract_v1.yaml
```

It defines `social-compliance-metric-contract.v1` with these metric families:

- `pedestrian_deviation`
- `flow_disruption`
- `comfort_exposure`
- `legibility_progress`
- `distributional_inconvenience`

Each metric names its units, denominator, direction, required signals, aggregation policy,
missing-data behavior, and a concrete fixture expectation. The intended report namespace is
`metrics.social_compliance`, with aggregate tables grouped under `social_compliance`.

## Claim Boundary

These metrics are diagnostic simulation proxies. They may help compare planners or scenario families
inside controlled Robot SF traces, but they do not establish real-world pedestrian comfort,
fairness, legibility, or social validity. SNQI weights and headline score definitions are unchanged.
Any future promotion into benchmark-strength or paper-facing claims needs calibrated evidence,
durable artifacts, and explicit review under the repository evidence vocabulary.

## Validation

Focused contract validation:

```bash
uv run pytest tests/benchmark/test_social_compliance_metric_contract.py
```

The tests verify family coverage, diagnostic-only claim wording, missing-data/support expectations,
and the numeric fixture examples embedded in the contract.

## Follow-Up Surface

The next implementation PR should wire episode emission and aggregation for
`metrics.social_compliance` using this contract. Until then, downstream campaigns should reference
this file as the metric definition source and classify any result as `diagnostic-only`.
