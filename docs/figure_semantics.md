# Versioned figure semantics

Publication figures and tables must not invent display names, units, ordering, or
planner distinctions independently. The packaged
`robot_sf/benchmark/figures/figure_semantics.v1.json` registry is the reviewed
presentation-semantics owner for benchmark identifiers.

This registry does **not** compute metrics, change benchmark schemas, or define a
scientific claim. Metric computation remains with the benchmark metric owners and
specifications. The registry only binds known raw keys and aliases to stable
presentation metadata:

- English and German display labels;
- compact labels for constrained layouts;
- units and number formats;
- display order, direction, and default scale;
- planner labels, order, markers, and line styles.

Colors remain a renderer-style concern in `figures/style.py`. Every planner row
also has a marker and line style so color is never the sole distinction.

## Strict use in new figures

New publication renderers and agent workflows should query the strict API:

```python
from robot_sf.benchmark.figures.semantics import default_registry

registry = default_registry()
y_label = registry.metric_label("surface_clearance", language="en")
planner = registry.planner("ORCA")
```

Unknown identifiers raise `KeyError`. The older
`robot_sf.benchmark.figures.style.metric_label()` helper retains its title-case
fallback by default because existing generic reports may encounter extension
metrics. New publication code should call it with `strict=True` or use the
registry directly.

The command-line interface can validate the packaged registry or inspect one row:

```bash
uv run python -m robot_sf.benchmark.figures.semantics
uv run python -m robot_sf.benchmark.figures.semantics --metric min_clearance
uv run python -m robot_sf.benchmark.figures.semantics --metric surface_clearance \
  --language de --short
uv run python -m robot_sf.benchmark.figures.semantics --planner ORCA
```

## Review-only learning loop

An agent must not remember a new `variable -> display name` mapping privately or
write it into a local cache. For an unmapped identifier, generate a proposal:

```bash
uv run python -m robot_sf.benchmark.figures.semantics \
  --suggest-metric lateral_jerk \
  --unit 'm/s^3' \
  --context robot_sf/example.py:42 \
  --context docs/figure-brief.md
```

The output is explicitly `proposal_only`. Its German labels remain
`REVIEW REQUIRED`, its direction is `context`, and its order is deliberately at
the end. Before editing the registry, a reviewer must confirm:

1. the canonical source key and every alias;
2. English and German long/short labels;
3. the physical unit and number format;
4. whether higher or lower values are preferable in the bounded metric contract;
5. scale and presentation order.

Only an accepted repository diff becomes durable behavior. This makes learned
conventions visible, reviewable, versioned, and reusable by different agents.

## Artifact binding

Scenario figure packs copy the normalized registry as `figure_semantics.json` and
record its SHA-256 in the pack manifest and every view provenance sidecar. A
consumer can therefore distinguish:

- source and trace provenance;
- source admission status;
- final-size figure profile;
- display-semantics version;
- final exported artifact bytes.

A semantics hash proves which labels and presentation metadata were requested. It
does not admit source evidence, validate a metric implementation, or authorize a
paper or dissertation claim.

## Validation

```bash
uv run pytest tests/benchmark/test_figure_semantics.py \
  tests/benchmark/test_figure_profile.py -q
uv run ruff check robot_sf/benchmark/figures/semantics.py \
  robot_sf/benchmark/figures/style.py \
  robot_sf/benchmark/figures/scenario_pack.py \
  tests/benchmark/test_figure_semantics.py
uv run ruff format --check robot_sf/benchmark/figures/semantics.py \
  robot_sf/benchmark/figures/style.py \
  robot_sf/benchmark/figures/scenario_pack.py \
  tests/benchmark/test_figure_semantics.py
```

Focused tests cover alias collision refusal, strict unknown-key behavior,
English/German labels, compatibility fallback, profile-selected language,
non-color planner distinctions, stable normalized hashes, and proposal-only
suggestions that cannot mutate the loaded registry.
