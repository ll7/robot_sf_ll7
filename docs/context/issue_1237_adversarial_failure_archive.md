# Issue #1237 Adversarial Failure Archive

## Goal

Issue #1237 asks for a first-class failure archive workflow on top of adversarial search outputs.
The first slice should turn disposable `output/` candidate bundles into a compact, replayable
manifest without copying raw episode JSONL, trajectories, or videos into git.

## Contract

The v1 archive schema is `adversarial_failure_archive.v1`. It ingests one or more
`adversarial-search-manifest.v1` files and writes a compact JSON manifest containing:

- source manifest paths,
- selected failure entries,
- deterministic clusters,
- representative archive IDs,
- replay command pointers back to source candidate bundles.

Selection is conservative: entries must have `failure_attribution.primary_failure` present and not
`success` or `invalid_candidate`. Invalid candidates, `not_evaluated` rows, missing attribution,
and successful episodes are excluded from the archive rather than treated as reusable failures.

Grouping is deterministic by:

- search `config.policy`,
- search `config.scenario_template`,
- `failure_attribution.primary_failure`,
- `failure_attribution.details.termination_reason`.

The v1 representative for each cluster is the member with the smallest normalized perturbation
from the configured search-space midpoint, then the highest objective value, then archive ID. This
is a heuristic minimization proxy only; it does not rerun adversarial search to shrink candidates.

## Implementation Surfaces

- `robot_sf/adversarial/archive.py` implements manifest loading, filtering, grouping,
  representative selection, and JSON writing.
- `scripts/tools/curate_adversarial_failure_archive.py` provides the CLI wrapper.
- `tests/adversarial/test_failure_archive.py` covers deterministic archive output, grouping,
  representative choice, and CLI summary output.

## Validation

Red proof:

```bash
uv run --active pytest tests/adversarial/test_failure_archive.py -q
```

This initially failed during collection because `robot_sf.adversarial.archive` did not exist.

Targeted green proof:

```bash
uv run --active pytest tests/adversarial/test_failure_archive.py -q
```

Result: `3 passed`.

Adjacent regression proof:

```bash
uv run --active pytest tests/adversarial/test_failure_archive.py tests/adversarial/test_adversarial_search.py tests/test_failure_extractor.py -q
```

Result: `40 passed`.

CLI smoke:

```bash
uv run --active python scripts/tools/curate_adversarial_failure_archive.py output/adversarial/issue1237_failure_archive/manifest.json --out output/adversarial/issue1237_failure_archive/archive.json
```

Result summary:
`{"archived_failure_count": 1, "cluster_count": 1, "source_candidate_count": 2, "source_manifest_count": 1}`.

## Follow-Up Boundary

This slice does not promote archived failures to representative benchmark distributions and does
not copy large artifacts into durable docs. Follow-up work can add trajectory-signature clustering,
metric buckets, or search-driven perturbation minimization, but those changes need separate proof
and should keep replay commands and source bundle provenance explicit.
