# inD Intersection Drone External Data

Plain-language summary: Robot SF can record the expected local layout and provenance of the inD
dataset (naturalistic road-user trajectories at German intersections), but it does **not** download,
redistribute, or benchmark with the request-gated data unless you request and stage it yourself
under the upstream non-commercial research-use terms.

This is a dataset family in the #4224 external-data program (alongside the Stanford Drone Dataset,
ETH/UCY, and ATC pedestrian tracking). This page, its registry entry, and the skip-if-absent
shape-contract loader are **contract-only**: no dataset bytes, no automated acquisition, no benchmark
run, and no paper-facing claim are introduced here. The shape-contract loader
(`robot_sf/data/external/ind.py`) inspects a locally staged copy and skips cleanly when the data is
absent.

Related issues: [#4290](https://github.com/ll7/robot_sf_ll7/issues/4290),
[#4224](https://github.com/ll7/robot_sf_ll7/issues/4224),
[#3161](https://github.com/ll7/robot_sf_ll7/issues/3161)

## What it is

The inD dataset (Bock et al. 2020) contains naturalistic trajectories of pedestrians, cyclists, and
vehicles recorded by a camera-equipped drone at four German intersections. Each recording is
published as a per-recording group of CSV files (`*_tracks.csv` trajectory samples,
`*_tracksMeta.csv` per-track class/size metadata, `*_recordingMeta.csv` recording-level metadata such
as frame rate, location, and object counts) plus an aerial `*_background.png` reference image, and
upstream also ships OpenDRIVE/Lanelet2 maps.

## Canonical registry identity

The canonical Robot SF registry entry is **`ind-crossings`** in
`scripts/tools/manage_external_data.py`.

Issue #4290 proposed the bare id `ind`. The existing id `ind-crossings` is retained deliberately: it
was registered as canonical in the merged #4224 slice and matches the qualified-id convention of its
sibling assets from that program (`atc-pedestrian`, `eth-ucy`, `crowdbot`, `scand-demos`). The merged
ATC slice (#4289) made the same retain-the-qualified-id decision and explicitly lists `ind-crossings`
as part of that convention, so renaming a single member here would break the convention and churn the
registry, its tests, and the #4224 context note for no functional benefit. The maintainer
implementation plan explicitly permits keeping the existing id with this rationale; a later
cohort-wide rename to bare ids across the whole program remains possible if maintainers prefer it.

Inspect the authoritative, always-up-to-date contract with:

```bash
uv run python scripts/tools/manage_external_data.py explain ind-crossings
```

## Official acquisition (request-based)

inD uses a **request-based, manually reviewed** access model. There is no scriptable direct
download, and Robot SF does not commit dataset bytes: the registry keeps `auto_download_allowed=False`.

- Official source: <https://levelxdata.com/ind-dataset/>

Steps for external users:

1. Open the official leveLXData inD page and submit the access-request form, describing your project
   and intended non-commercial research use. Each request is reviewed manually.
2. After approval, download your own copy under the upstream terms.
3. Keep a local copy of the terms/README alongside the data; you must obtain your own copy — Robot SF
   does not redistribute inD.

The upstream terms state the dataset is **free for non-commercial use only** (academic research,
teaching, or scientific publications), require citing the inD reference paper (see
[Citation](#citation)), prohibit commercial use, and prohibit redistributing the dataset or modified
versions. Derived works that do not allow recovery of the original data (for example trained models)
are permitted under those terms; this registry/docs slice does not itself stage or derive anything.

## Expected layout

By default the registry expects the inD root at `output/external_data/ind_crossings/`. To share one
staged copy across worktrees, set `ROBOT_SF_EXTERNAL_DATA_ROOT` and place the tree under
`$ROBOT_SF_EXTERNAL_DATA_ROOT/ind_crossings/` (the registry `shared_root_subpath`).

```text
$ROBOT_SF_EXTERNAL_DATA_ROOT/ind_crossings/
  README.md   (or LICENSE.txt / TERMS.txt — a local copy of the upstream non-commercial terms)
  00_tracks.csv          (per-recording trajectory samples; one or more recordings)
  00_tracksMeta.csv      (per-track class/size metadata)
  00_recordingMeta.csv   (recording-level metadata: frame rate, location, counts)
  00_background.png       (aerial background image; nested subdirectories are fine)
```

Required-path groups (at least one match per group; run `explain` for the exact patterns):

- **tracks**: `**/*_tracks.csv`.
- **tracks_meta**: `**/*_tracksMeta.csv`.
- **recording_meta**: `**/*_recordingMeta.csv`.
- **background**: `**/*_background.png` (preferred), or any `**/*.png` if staged without the original
  name.
- **license_or_readme**: `**/README*`, `**/LICENSE*`, or `**/TERMS*`.

These are layout and presence checks only. They do not assert trajectory-content correctness,
licensing status, benchmark readiness, or any paper-facing evidence. Staging a subset of recordings
is fine; each staged recording should still carry its tracks/tracksMeta/recordingMeta/background
group so the presence check reflects a real inD-shaped copy rather than a lone stray CSV.

## Validation

Without staged data, the registry check fails closed with `status: missing` / `ok: false`:

```bash
uv run python scripts/tools/manage_external_data.py explain ind-crossings
uv run python scripts/tools/manage_external_data.py check ind-crossings
uv run python scripts/tools/manage_external_data.py --json list
```

With locally staged official data, `check` (or `stage`) validates the required-path groups and can
write a compact provenance manifest. A successful presence check is only local staging evidence; it
is not a full benchmark campaign, a trajectory-loader validation, or a dissertation claim.

## Shape-contract loader

`robot_sf/data/external/ind.py` is a license-safe, skip-if-absent accessor over a locally staged inD
copy. It never downloads or redistributes inD bytes; it only inspects structure:

- `ind.is_available(root=None)` returns whether at least one complete recording group is staged
  (resolving the root via `ROBOT_SF_EXTERNAL_DATA_ROOT` when `root` is `None`). Skip-if-absent tests
  branch on this so external clones without the data skip cleanly.
- `ind.require_available(root=None)` returns the resolved per-recording paths or raises
  `IndDataError` with an actionable pointer back to this page.
- `ind.load_shape_contract(root=None)` validates each recording's `*_tracks.csv`, `*_tracksMeta.csv`,
  and `*_recordingMeta.csv` structurally (non-empty, rectangular, and carrying inD's documented
  header columns) and returns per-recording row/column shape metadata plus resolved relative paths.

The contract is deliberately structural: it confirms the documented per-recording layout and that the
coordinate/id columns in `*_tracks.csv` parse as finite floats. It asserts no scene content (exact
frame counts, coordinates, or class distributions) and makes no benchmark or prediction-comparability
claim. Run the shape-contract tests with:

```bash
uv run pytest tests/data/external/test_ind_shape.py -q
```

Every test except the single real-data probe builds a synthetic layout under `tmp_path`; the probe
skips when inD is not staged.

## Citation

> J. Bock, R. Krajewski, T. Moers, S. Runde, L. Vater, L. Eckstein, "The inD Dataset: A Drone Dataset
> of Naturalistic Road User Trajectories at German Intersections," *2020 IEEE Intelligent Vehicles
> Symposium (IV)*, pp. 1929–1934, 2020, doi:10.1109/IV47402.2020.9304839.

## Boundary and follow-up

The registry metadata and acquisition/layout documentation were added in #4290; the skip-if-absent
shape-contract loader and tests were added for #4224, following the exemplar pattern established for
`socnavbench-s3dis-eth` and `eth-ucy`. No download code, dataset bytes, benchmark consumer, or claim
edit is part of that work. Deferred follow-up (private ops, after local staging exists): acquire →
seed on the hub → fetch on one spoke → registry `check` flips from `missing`, then pin
`expected_tree_sha256`.
