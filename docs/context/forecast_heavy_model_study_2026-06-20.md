# Heavy Forecast-Model Family Study / Preflight - Issue #2845 (2026-06-20)

- **Issue**: [#2845](https://github.com/ll7/robot_sf_ll7/issues/2845) — *prediction: offline
  transformer or diffusion benchmark study*
- **Parent lane**: [#2835](https://github.com/ll7/robot_sf_ll7/issues/2835)
- **Archetype / evidence tier**: `analysis` / `blocked` → `analysis_only` (this slice)
- **Authored**: 2026-06-27 (path name keeps the issue's `2026-06-20` agent-exec-spec label for
  traceability)
- **Canonical owner (machine-readable)**:
  [`robot_sf/research/forecast_heavy_model_inventory.py`](../../robot_sf/research/forecast_heavy_model_inventory.py)
- **CLI**:
  [`scripts/research/check_forecast_heavy_model_inventory.py`](../../scripts/research/check_forecast_heavy_model_inventory.py)

## What this is (and is not)

This is the **assessment-first inventory/preflight slice** of issue #2845. It answers *"could we
evaluate heavier predictors offline, and what would the minimum experiment need?"* — not *"are
heavier predictors better?"*.

**This slice does not**: train any model, run any inference, add any dependency, run any benchmark,
submit any SLURM/GPU job, or make any model-quality or paper/dissertation claim. Per the issue's
non-goals, literature plausibility is **not** treated as repository evidence.

The per-family compute / latency / uncertainty / integration tiers below are **qualitative
literature-derived planning estimates for relative triage only**. They are ordinal ranks
(`low < medium < high < very_high`) relative to the in-repo lightweight learned baseline, not
measurements taken in this repository.

## Candidate model families (planning estimates, not measured)

| Family | Compute | Online latency | Uncertainty quality | Integration burden | Candidate offline use cases |
| --- | --- | --- | --- | --- | --- |
| Transformer (deterministic head) | high | medium | medium | medium | online prediction |
| AgentFormer-like socio-temporal | very_high | high | high | high | online prediction, offline scenario generation |
| CVAE multi-modal | medium | medium | high | medium | online prediction, offline scenario generation |
| Diffusion / score-based | very_high | very_high | high | very_high | offline scenario generation, adversarial stress |

Notes:

- The **lightweight baseline anchor** is the constant-velocity Gaussian / signal-aware ladder in
  `robot_sf/benchmark/pedestrian_forecast.py` and the learned scaffold in
  `robot_sf/planner/predictive_model.py` (#2844). All tiers above are relative to that anchor.
- "Uncertainty quality" is a *potential*, not a guarantee: any probabilistic head must still pass
  the in-repo calibration (`forecast_calibration_report.py`) and conformal-coverage
  (`forecast_conformal_pilot.py`) checks before its distribution is trusted.

## Offline-evaluation entry-point surfaces (already present, fail-closed)

The inventory probes that the surfaces an offline experiment would touch import and expose their
key symbols. On the current checkout all *required* surfaces are present (the preflight passes its
import verdict). They are:

- `robot_sf/benchmark/forecast_metrics.py` — ADE/FDE/miss-rate scoring.
- `robot_sf/benchmark/forecast_calibration_report.py` — calibration / reliability.
- `robot_sf/benchmark/forecast_conformal_pilot.py` — distribution-free coverage.
- `robot_sf/benchmark/forecast_dataset_recorder.py` — bounded durable dataset + manifest.
- `robot_sf/benchmark/forecast_batch.py` — `ForecastBatch` / `ActorForecast` data contracts.
- `robot_sf/benchmark/forecast_baseline_comparison.py` — baseline comparison ladder.
- `robot_sf/benchmark/pedestrian_forecast.py` — lightweight CV/graph comparator.

## Minimum viable offline experiment

**One concrete MVP**: score a single trained heavy model (start with the cheapest useful family —
CVAE) against the lightweight baseline ladder on a bounded, versioned held-out forecast dataset,
recording ADE/FDE, calibration coverage gap, conformal coverage, and wall-clock runtime under a
CPU-only budget. Decision metric: the heavy model must beat the baseline ladder on the
`forecast_baseline_comparison` surface by a margin that justifies its integration burden, with
calibration no worse than the baseline.

This experiment is **BLOCKED** today. The preflight reports these prerequisites:

Local (closeable in-repo, currently absent):

1. `staged_holdout_dataset` — a versioned, split-disjoint held-out forecast dataset + manifest.
2. `heavy_model_adapter` — adapter from a heavy model's output to `ActorForecast`/`ForecastBatch`.
3. `cpu_runtime_budget` — a config-first CPU-only runtime budget for the run.

External (standing decisions, out of this slice's scope):

4. `dependency_decision` — maintainer decision to add the heavy-model dependency/training surface.
5. `trained_checkpoint` — at least one trained checkpoint to evaluate (no GPU campaign here).

## Recommendation (decision boundary)

**continue (assessment) — do NOT start implementation/training now.**

- The offline-evaluation surface is sufficient to *score* heavy models, so the assessment lane is
  worth continuing.
- Do **not** begin heavy-model implementation or training until the three local prerequisites are
  staged and the two external decisions are made. The cheapest first concrete step is staging the
  bounded held-out dataset (prerequisite 1), which is also reusable by the lightweight baseline.
- **Stop condition for the heavy-model lane**: if, once a dataset and a single CVAE checkpoint
  exist, the heavy model does not beat the baseline ladder by a margin that justifies its
  integration burden (and at least matches its calibration), recommend `stop` for online
  prediction and re-scope the heavy families to offline scenario generation / adversarial stress
  only.

## Reproduce

```bash
# Markdown verdict (exit 0 = required surfaces import; minimum experiment still reported BLOCKED)
uv run python scripts/research/check_forecast_heavy_model_inventory.py
# Machine-readable report / static inventory
uv run python scripts/research/check_forecast_heavy_model_inventory.py --json
uv run python scripts/research/check_forecast_heavy_model_inventory.py --list
# Fail-closed revival decision packet
uv run python scripts/research/check_forecast_heavy_model_inventory.py --decision-packet
uv run python scripts/research/check_forecast_heavy_model_inventory.py --decision-packet --json
# Tests
uv run python -m pytest tests/research/test_forecast_heavy_model_inventory.py tests/prediction/test_forecast_heavy_model_decision_packet.py -q
```

## Related

- #2844 (lightweight learned baseline), #2915 (deterministic baseline ladder), #3065 (real-traj
  ingestion + staging contract), #1490 / #2843 (coupling / gating constraints to respect).
