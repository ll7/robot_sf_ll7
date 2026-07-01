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

## Mixed Normalization Basis (Issues #3699 and #3978)

> **Current status:** versioned and bounded for new diagnostic use. `SNQI-v0` remains the default
> legacy score for historical comparability and still preserves the mixed raw/baseline-normalized
> basis from issue #3699. `SNQI-v1`, introduced by issue #3978, is an opt-in bounded
> baseline-relative diagnostic that normalizes all penalty terms through the same median/p95 clamp.

Do not compare `SNQI-v0` and `SNQI-v1` values numerically: their normalization semantics differ.
`SNQI-v1` does not retune weights, repair the unresolved canonical-weight decision from issue
#3723, implement time-to-collision or closing-speed exposure from issue #3700, or make SNQI a
primary safety ranking. Constraints-first benchmark evidence remains primary.

The compact diagnostic fixture for this transition is tracked at
`docs/context/evidence/issue_3978_snqi_v1_recalibration/`.

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

The normalization inventory reports which SNQI terms are raw, which are baseline-normalized, which
terms are bounded, and which score-version contract is active. By default it reports legacy
`SNQI-v0`; pass `--score-version SNQI-v1` to inspect the bounded diagnostic contract.

```bash
uv run python scripts/benchmark/snqi_normalization_inventory_report.py --fail-on-mixed-scale
uv run python scripts/benchmark/snqi_normalization_inventory_report.py --score-version SNQI-v1
```

Implementation: `robot_sf/benchmark/snqi/normalization_inventory.py`; guard tests in
`tests/benchmark/test_snqi_normalization_inventory.py`.

### Combined Governance Preflight

Use this check when a workflow needs one fail-closed gate for the current SNQI governance state. It
combines weight-provenance inventory (#3723), legacy normalization context (#3699), and the active
`SNQI-v1` bounded diagnostic contract (#3978). It is secondary diagnostic evidence only: it does
not choose canonical weights, make SNQI a primary safety ranking, or treat `SNQI-v1` as numerically
comparable to `SNQI-v0`.

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
- If a workflow depends on historical `SNQI-v0` values, inspect the legacy normalization inventory
  and keep the mixed-basis caveat attached.
- If a workflow depends on active bounded diagnostics, use `SNQI-v1` with complete baseline median
  and p95 coverage for every penalty term.
- Do not use either unresolved diagnostic as proof for a primary safety ranking.
