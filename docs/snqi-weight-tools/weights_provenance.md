# SNQI Weights Artifact Provenance

**Purpose**: Document the current Social Navigation Quality Index (SNQI) weight artifact
provenance, validation status, and lifecycle guidance.

## Overview

SNQI uses a weighted composite score to summarize robot navigation performance. This document
records the current weight-file situation and the fail-closed diagnostics that keep unresolved
governance issues visible. The current checks are secondary diagnostic evidence only: they do not
choose canonical weights, change `compute_snqi`, or make SNQI a primary safety ranking.

## Current Artifacts

### v1 Canonical-Labeled Weights (`model/snqi_canonical_weights_v1.json`)

**Current status**: unresolved diagnostic input, not a final canonical decision.

The filename carries a canonical label, but issue #3723 documents that it conflicts with the
`recompute_snqi_weights("canonical")` code default. Treat this file as an existing weight artifact
that must be inventoried explicitly, not as proof that the final canonical SNQI weights are settled.
Any older optimization or validation notes for this file are legacy context until the maintainer
resolves the canonical-weight decision.

## Known Provenance Conflict (Issue #3723)

> **Status:** unresolved, `decision-required`. The "canonical" label is currently attached to more
> than one disagreeing weight set, so which weights produce a given planner ranking depends on the
> path loaded. Choosing a single source of truth is a maintainer decision and is out of scope for the
> diagnostic checks described below.

Conflicting sets exist today:

| Source | Designation | Dominant term | Scale |
| --- | --- | --- | --- |
| `recompute_snqi_weights("canonical")` (code default) | canonical | `w_collisions` (2.0) | raw |
| `model/snqi_canonical_weights_v1.json` | canonical | `w_jerk` (3.0) | raw |
| `configs/benchmarks/snqi_weights_camera_ready_v1.json` | versioned | `w_jerk` (3.0) | raw |
| `configs/benchmarks/snqi_weights_camera_ready_v2.json` | versioned | `w_time` | normalized, sum about 1 |
| `configs/benchmarks/snqi_weights_camera_ready_v3.json` | versioned | `w_near` | normalized, sum about 1 |

The code default (collision-dominant) and `model/snqi_canonical_weights_v1.json` (jerk-dominant)
both claim or imply "canonical" yet can yield different rankings. The raw-vs-normalized scale split
overlaps issue #3699.

## Mixed Normalization Basis (Issue #3699)

> **Status:** unresolved, `decision-required`. SNQI currently mixes raw, unbounded penalty terms
> (`time`, `comfort`) with baseline-normalized penalty terms (`collisions`, `near`, `force_exceed`,
> `jerk`). The diagnostic checks make that visible and fail closed, but they do not choose between
> normalizing the raw terms and documenting a bounded raw-term asymmetry.

## Diagnostics

### Weight-Provenance Inventory

Until a canonical set is designated, the read-only inventory/preflight surfaces conflicts instead
of letting them stay silent. It does not change scoring, re-tune weights, or pick a winner.

```bash
# Human-readable inventory + conflicts; exits non-zero (2) on blocking conflict.
uv run python -m robot_sf.benchmark.snqi.cli inventory

# Machine-readable report; inspection mode never fails.
uv run python -m robot_sf.benchmark.snqi.cli inventory --json --no-fail-on-conflict
```

Programmatic use:

```python
from robot_sf.benchmark.snqi import preflight_snqi_weight_sets

# Raises SNQIWeightProvenanceError on a blocking error-severity conflict.
report = preflight_snqi_weight_sets(strict=True)
```

Implementation: `robot_sf/benchmark/snqi/weights_inventory.py`; guard tests in
`tests/test_snqi_weights_inventory.py`.

### Normalization Inventory

The normalization inventory reports which SNQI terms are raw, which are baseline-normalized, and
which terms are unbounded. It can fail closed while the mixed basis remains unresolved.

```bash
uv run python scripts/benchmark/snqi_normalization_inventory_report.py --fail-on-mixed-scale
```

Implementation: `robot_sf/benchmark/snqi/normalization_inventory.py`; guard tests in
`tests/benchmark/test_snqi_normalization_inventory.py`.

### Combined Governance Preflight

Use this check when a workflow needs one fail-closed gate for the current SNQI governance state. It
combines the weight-provenance inventory (#3723) with the normalization-basis inventory (#3699).
It is secondary diagnostic evidence only: it does not choose canonical weights, change
`compute_snqi`, or make SNQI a primary safety ranking.

```bash
# Exits non-zero while the current governance blockers remain unresolved.
uv run python scripts/validation/check_snqi_governance.py

# Inspection mode keeps the same report but returns success for exploratory use.
uv run python scripts/validation/check_snqi_governance.py --allow-current-blockers --json
```

Implementation: `scripts/validation/check_snqi_governance.py`; guard tests in
`tests/benchmark/test_snqi_governance_preflight.py`.

## Lifecycle Guidance

Future SNQI weight changes should be versioned and should state whether they are diagnostic,
experimental, or canonical. A future canonical decision should update this file, the relevant JSON
artifact metadata, and the fail-closed tests so that only one canonical source is accepted.

Do not rename, remove, or reinterpret existing weight files silently. If a file is deprecated, leave
an explicit migration note that names the replacement, the evidence that supports it, and the
benchmark comparability impact.

## Troubleshooting

- If a workflow fails on issue #3723, inspect the weight-provenance inventory and decide whether the
  workflow needs inspection mode or a maintainer-approved canonical-weight decision.
- If a workflow fails on issue #3699, inspect the normalization inventory and decide whether the
  workflow needs inspection mode or a maintainer-approved normalization remedy.
- Do not use either unresolved diagnostic as proof of a primary safety ranking.
