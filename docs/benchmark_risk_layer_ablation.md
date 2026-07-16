# Risk-layer ablation for planner evaluation

This page defines the **risk-layer ablation axis** introduced in
[issue #5832](https://github.com/ll7/robot_sf_ll7/issues/5832) and implemented in
`robot_sf/benchmark/risk_layer_ablation.py`. It is an *evaluation-config + reporting*
change built on the existing metric stack (collision ledger, near-field exposure,
SNQI weights). It runs no simulation and makes no benchmark claim on its own.

## Motivation

Two 2026-07-15 preprints argue that navigation-safety evidence requires
**heterogeneous hazard fusion**, evaluated layer by layer:

- a powered-wheelchair study fusing slope / static / dynamic / semantic hazards
  into a probabilistic risk map (paired Monte Carlo eval cut collisions from
  >73% to <32%);
- a vehicle-in-the-loop platform showing localization uncertainty dominating
  weather as the cooperative-perception error source.

Both support the claim that a planner can look safe only because the evaluation
ignores a hazard layer. The ablation makes that inversion measurable: *a planner
ranked safe at L0 but unsafe at L2 is exactly the publishable finding class.*

## The three risk layers

| Layer | Hazard dimensions | Active SNQI terms |
| --- | --- | --- |
| `L0_geometry_only` | static obstacles, hard contact geometry | `w_success`, `w_collisions`, `w_force_exceed` |
| `L1_plus_dynamics` | L0 + moving-pedestrian **dynamics** | + `w_near`, `w_comfort` |
| `L2_plus_semantic_risk` | L1 + **semantic / zone** risk | + `w_semantic_risk` |

Layer definitions and their exact weight mappings live in
`RISK_LAYERS` / `RISK_LAYER_WEIGHTS` in `robot_sf/benchmark/risk_layer_ablation.py`
and are mirrored in `configs/benchmark/risk_layer_ablation.yaml`.

- **L0** is the current baseline behaviour: only static obstacles and hard
  contact geometry matter.
- **L1** adds moving-pedestrian dynamics — near-field exposure and motion
  comfort — so planners that "look safe" only because pedestrians are treated
  as static get penalized.
- **L2** adds semantic/zone risk weighting. The semantic-risk term is read from
  `metrics.semantic_risk_exposure`, which scenario families carry when they have
  zone metadata (crossings, doorways, high-density zones). When that metric is
  absent from every record, the layer reports it `unavailable` and equals L1.
  Partial coverage fails closed rather than imputing missing records.

## Metric mapping

Each layer is a fixed-schema SNQI weight configuration. Activating a higher layer
enables additional terms; deactivating keeps a `0.0` weight. Because
`compute_snqi_v0` uses the zero-weight contract, a missing metric under a `0.0`
weight never collapses the score, and a missing semantic metric under a non-zero
L2 weight is reported as unavailable rather than imputed.

## Report artifact

`build_risk_layer_report` consumes episode records that already carry `metrics`
and a planner/group key, then produces:

- per-layer mean SNQI per planner;
- per-planner **rank** under each layer;
- **per-planner rank delta from L0** (`rank_delta_from_l0`; positive = worse than at L0);
- an optional **bootstrap stability** block per layer with rank confidence
  intervals (`ci95_low` / `ci95_high`) and a normalized Spearman stability score.

`format_risk_layer_markdown` renders the per-planner delta table. A planner whose
rank changes between L0 and the richest layer has `rank_changed = true`.

### Bounded first slice

The first slice is an **L0-vs-L1 ablation on one released scenario family** with
the standard seed set, reporting the per-planner delta table with bootstrap CIs.
L2 is wired and documented but only exercised when the scenario family carries
zone metadata.

## Usage

```python
from robot_sf.benchmark.risk_layer_ablation import (
    build_risk_layer_report_from_config,
    format_risk_layer_markdown,
)

report = build_risk_layer_report_from_config(
    records,  # episode dicts with metrics + planner key
    baseline_stats=baseline_stats,
)
print(format_risk_layer_markdown(report))
```

## Acceptance criteria (issue #5832)

- Ablation config producing per-layer metric tables for >=2 planners on >=1
  scenario family, seeds pinned — `configs/benchmark/risk_layer_ablation.yaml`.
- A report artifact showing per-layer rank changes (or their absence) with
  bootstrap CIs — `build_risk_layer_report(bootstrap=...)`.
- A docs page defining the layers and their metric mapping — this file.
- Tests: layer configs load, and metrics differ between layers on a fixture
  where they must — `tests/test_risk_layer_ablation.py`.
