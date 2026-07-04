# Issues #1598 and #1599 Root Compatibility Decisions (2026-05-28)

Date: 2026-05-28

Issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1598>
- <https://github.com/ll7/robot_sf_ll7/issues/1599>

Predecessor: [issue_1583_high_risk_root_boundaries.md](issue_1583_high_risk_root_boundaries.md)

Status: Superseded for `tests/fixtures/scenarios/` by
[root_layout_structured_migration_2026-06-01.md](../root_layout_structured_migration_2026-06-01.md).
Keep this note as provenance for the earlier conservative decision.

## Decision

Keep both `tests/fixtures/scenarios/` and `specs/` at the repository root for now. They are not cleanup
leftovers: current tests, examples, docs, schemas, and runtime comments treat those root paths as
stable contracts. A future move should be a dedicated compatibility migration, not part of the
root-layout cleanup stream.

| Path | Decision | Compatibility burden | Required migration shape | Rollback plan |
| --- | --- | --- | --- | --- |
| `tests/fixtures/scenarios/` | Keep at root. | Medium: OSM examples and tests directly reference `tests/fixtures/scenarios/osm_fixtures/sample_block.pbf`. | If a later fixture move is accepted, keep a compatibility path or update every example/test default in one PR, then run the OSM gates below. | Restore root `tests/fixtures/scenarios/` fixture paths and revert example/test path defaults. |
| `specs/` | Keep at root. | Very high: docs, example manifests, schema tests, runtime comments, and schema loaders deep-link into `specs/...`. | If a later migration is accepted, preserve root `specs/` compatibility links until all docs, tests, schema `$id` references, and runtime path assumptions are updated. | Restore root `specs/` and any contract paths consumed by schema tests or runtime loaders. |

## Reference Inventory

`tests/fixtures/scenarios/` root references are concentrated in OSM fixture consumers:

- `examples/osm_map_editor_demo.py`
- `examples/osm_map_quickstart.py`
- `tests/test_osm_map_builder.py`
- `tests/test_osm_backward_compat.py`
- `tests/test_osm_background_renderer.py`

`specs/` references are broad and mixed-purpose:

- user docs including `docs/README.md`, `docs/dev_guide.md`,
  `docs/imitation_learning_pipeline.md`, and benchmark docs,
- example metadata in `examples/examples_manifest.yaml`,
- schema and visual contract tests under `tests/visuals/` and `tests/maps/`,
- runtime comments and path construction in `robot_sf/benchmark/`,
  `robot_sf/maps/verification/`, `robot_sf/research/`, and `robot_sf/telemetry/`,
- validation and tooling scripts under `scripts/`.

This distribution argues against moving either path without a separate compatibility phase.

## Validation Gates

Reference inventory:

```bash
rg -nF "tests/fixtures/scenarios/" tests examples docs scripts robot_sf
rg -nF "specs/" AGENTS.md docs tests scripts robot_sf examples .github .vscode
```

Targeted gates before any future `tests/fixtures/scenarios/` move:

```bash
uv run pytest tests/test_osm_map_builder.py tests/test_osm_background_renderer.py tests/test_osm_backward_compat.py -q
uv run python examples/osm_map_quickstart.py
```

Targeted gates before any future `specs/` move:

```bash
uv run pytest tests/visuals/test_plot_schema_validation.py \
  tests/visuals/test_video_schema_validation.py \
  tests/visuals/test_performance_schema_validation.py \
  tests/maps/test_map_verifier.py -q
```

The future migration PR should also run the normal PR readiness path after updating all path
contracts.
