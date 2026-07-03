# SCAND Socially-Compliant Navigation Demonstrations External Data

Plain-language summary: Robot SF can record the expected local layout and provenance of the SCAND
(Socially CompliAnt Navigation Dataset) demonstrations and validate a cheap **shape contract**
against a locally staged copy, but it does **not** download, redistribute, or benchmark with the
terms-gated data unless you stage it yourself under the upstream terms.

This is a SCAND slice of the #4224 external-data program. It adds a license-safe loader plus
skip-if-absent shape-contract tests on top of the previously merged registry entry (PR #4238). No
dataset bytes, no automated acquisition, no benchmark run, and no paper-facing claim are introduced
here.

Related issues: [#4224](https://github.com/ll7/robot_sf_ll7/issues/4224),
[#1470](https://github.com/ll7/robot_sf_ll7/issues/1470),
[#1496](https://github.com/ll7/robot_sf_ll7/issues/1496)

## What it is

SCAND (Karnan et al. 2022) is 8.7 hours of socially-compliant human-teleoperated navigation
demonstrations, distributed as ROS bag recordings. It targets a real-demonstration alternative for
the oracle-imitation lane (#1470 / #1496).

## Canonical registry identity

The canonical Robot SF registry entry is **`scand-demos`** in
`scripts/tools/manage_external_data.py`. Inspect the authoritative, always-up-to-date contract with:

```bash
uv run python scripts/tools/manage_external_data.py explain scand-demos
```

## Official acquisition

Acquire the dataset directly from the official Texas Data Repository release and keep it under the
upstream terms:

- Official source:
  <https://dataverse.tdl.org/dataset.xhtml?persistentId=doi:10.18738/T8/0PRYRH>

The dataset is large; select files manually through the official DataVerse interface. The public
Robot SF repository does not encode a scriptable download URL and does not commit dataset bytes: the
registry keeps `auto_download_allowed=False`. Keep a local copy of the terms/README alongside the
data.

## Expected layout

By default the registry expects the SCAND root at `output/external_data/scand_demos/`. To share one
staged copy across worktrees, set `ROBOT_SF_EXTERNAL_DATA_ROOT` and place the tree under
`$ROBOT_SF_EXTERNAL_DATA_ROOT/scand_demos/` (the registry `shared_root_subpath`).

```text
$ROBOT_SF_EXTERNAL_DATA_ROOT/scand_demos/
  LICENSE.txt        (or README* / TERMS* — a local copy of the upstream terms)
  <demo>.bag         (one or more ROS bag demonstrations; nested subdirectories are fine)
  <export>.csv       (optional exported trajectory/command tables)
  <metadata>.json    (optional metadata/annotation JSON)
```

Required-path groups (at least one match per group; run `explain` for the exact patterns):

- **recording**: `**/*.bag`, `**/*.csv`, or `**/*.json`.
- **license_or_readme**: `**/README*`, `**/LICENSE*`, or `**/TERMS*`.

These are layout and presence checks only. They do not assert recording-content correctness,
licensing status, benchmark readiness, or any paper-facing evidence.

## Shape contract loader

`robot_sf/data/external/scand.py` is a license-safe loader that only inspects locally staged files.
It never downloads, vendors, or redistributes dataset bytes, and it makes no oracle-imitation claim.
It validates a cheap, structural shape contract: the documented recording + license/readme layout
exists, every staged CSV export is a non-empty rectangular table, every staged JSON export parses,
and no ROS bag is empty. It intentionally asserts no content values.

```python
from robot_sf.data.external import scand

if scand.is_available():
    contract = scand.load_shape_contract()
```

- `is_available(root=None)` returns `False` for an absent or incomplete layout.
- `require_available(root=None)` raises `ScandDataError` with an actionable docs pointer when the
  layout is absent/incomplete.
- `load_shape_contract(root=None)` returns recording counts plus per-file shape metadata (CSV
  row/column counts, JSON top-level type, bag byte sizes) and the resolved license/readme files.

```bash
uv run pytest tests/data/external/test_scand_shape.py -q
```

The shape-contract tests never require the gated bytes: the only real-data path skips when the
dataset is absent, and every other test builds a synthetic layout under a temp directory.

## Claim boundary

External-data registry and shape contract only. A passing `scand-demos` shape contract means only
that a local copy matched the documented recording/license layout and parsed structurally. It does
not mean the data has been downloaded, license-approved for redistribution, ingested, or used to
support a benchmark or paper-facing claim. This asset is an alternative/complement to the blocked SDD
route (#4079), not a resolution of it.

## Citation

> H. Karnan, A. Nair, X. Xiao, et al., "Socially CompliAnt Navigation Dataset (SCAND): A Large-Scale
> Dataset of Demonstrations for Social Navigation," *IEEE Robotics and Automation Letters*, 2022.
