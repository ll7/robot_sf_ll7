# ATC Pedestrian Tracking External Data

Plain-language summary: Robot SF can record the expected local layout and provenance of the ATC
(Osaka shopping-center) pedestrian tracking dataset, but it does **not** download, redistribute, or
benchmark with the license-gated data unless you stage it yourself under the upstream research-use
terms.

This is the third dataset family in the #4224 external-data program (after the Stanford Drone
Dataset and the ETH/UCY trajectory benchmarks). This page and its registry entry are
**registry/docs only**: no dataset bytes, no automated acquisition, no loader, no shape-contract
tests, no benchmark run, and no paper-facing claim are introduced here.

Related issues: [#4289](https://github.com/ll7/robot_sf_ll7/issues/4289),
[#4224](https://github.com/ll7/robot_sf_ll7/issues/4224),
[#2918](https://github.com/ll7/robot_sf_ll7/issues/2918),
[#3161](https://github.com/ll7/robot_sf_ll7/issues/3161),
[#3971](https://github.com/ll7/robot_sf_ll7/issues/3971),
[#3813](https://github.com/ll7/robot_sf_ll7/issues/3813),
[#3950](https://github.com/ll7/robot_sf_ll7/issues/3950)

## What it is

The ATC dataset contains pedestrian position and body-direction trajectories collected with 3-D
range sensors in part of the ATC (Asia and Pacific Trade Center) shopping-and-business center in
Osaka, Japan. The full public release covers 92 collection days (24 Oct 2012 – 29 Nov 2013,
typically Wednesday and Sunday, ~09:40–20:20) distributed as daily CSV files, packaged upstream as
nine 7-Zip archives of 10–11 single-day files each, plus one standalone sample day.

## Canonical registry identity

The canonical Robot SF registry entry is **`atc-pedestrian`** in
`scripts/tools/manage_external_data.py`.

Issue #4289 originally proposed the bare id `atc`. The existing id `atc-pedestrian` is retained
deliberately: it was registered as canonical in the merged #4224 slice (PR #4238) and matches the
qualified-id convention of its sibling assets from that same slice (`eth-ucy-trajectories`,
`ind-crossings`, `crowdbot`, `scand-demos`). Renaming a single member would break that convention
and churn the registry, its tests, and the #4224 context note for no functional benefit. The
maintainer implementation plan explicitly permits keeping the existing id with this rationale; a
later cohort-wide rename remains possible if maintainers prefer bare ids across the whole program.

Inspect the authoritative, always-up-to-date contract with:

```bash
uv run python scripts/tools/manage_external_data.py explain atc-pedestrian
```

## Official acquisition

Acquire the dataset directly from the official ATR project page and keep it under the upstream terms:

- Official source: <https://dil.atr.jp/crest2010_HRI/ATC_dataset/>

The upstream terms state the data is *free to use for research purposes only* and that use must cite
the reference paper (see [Citation](#citation)). The public Robot SF repository does not encode a
scriptable download URL and does not commit dataset bytes: the registry keeps
`auto_download_allowed=False` until a maintainer records stable terms and URL approval. Download the
archives manually, extract the daily CSVs, and keep a local copy of the terms/README alongside the
data.

## Chosen subset policy

The full ATC release is weeks of data. For local staging you do **not** need the entire release. The
registry layout requires only a **small documented subset**: one or more daily trajectory CSV files
(for example the upstream standalone sample day) plus a local terms/README note. Stage more days
only if a later, explicitly scoped benchmark or loader task needs them.

Each daily CSV file is named `atc-YYYYMMDD.csv` and has the columns (no header row upstream):

```text
time [ms, unixtime+ms/1000], person_id, x [mm], y [mm], z/height [mm],
velocity [mm/s], motion_angle [rad], facing_angle [rad]
```

## Expected layout

By default the registry expects the ATC root at `output/external_data/atc_pedestrian/`. To share one
staged copy across worktrees, set `ROBOT_SF_EXTERNAL_DATA_ROOT` and place the tree under
`$ROBOT_SF_EXTERNAL_DATA_ROOT/atc_pedestrian/` (the registry `shared_root_subpath`).

```text
$ROBOT_SF_EXTERNAL_DATA_ROOT/atc_pedestrian/
  README.md   (or LICENSE.txt / TERMS.txt — a local copy of the upstream research-use terms)
  atc-20121024.csv   (one or more daily trajectory CSVs; nested subdirectories are fine)
```

Required-path groups (at least one match per group; run `explain` for the exact patterns):

- **trajectory**: `**/atc-*.csv` (preferred), or any `**/*.csv` if staged without the original name.
- **license_or_readme**: `**/README*`, `**/LICENSE*`, or `**/TERMS*`.

These are layout and presence checks only. They do not assert trajectory-content correctness,
licensing status, benchmark readiness, or any paper-facing evidence.

## Validation

Without staged data, the registry check fails closed with `status: missing` / `ok: false`:

```bash
uv run python scripts/tools/manage_external_data.py explain atc-pedestrian
uv run python scripts/tools/manage_external_data.py check atc-pedestrian
uv run python scripts/tools/manage_external_data.py --json list
```

With locally staged official data, `check` (or `stage`) validates the required-path groups and can
write a compact provenance manifest. A successful presence check is only local staging evidence; it
is not a full benchmark campaign, a trajectory-loader validation, or a dissertation claim.

## Citation

> D. Brščić, T. Kanda, T. Ikeda, T. Miyashita, "Person position and body direction tracking in
> large public spaces using 3-D range sensors," *IEEE Transactions on Human-Machine Systems*,
> Vol. 43, No. 6, pp. 522–534, 2013.

## Boundary and follow-up

This slice adds registry metadata and acquisition/layout documentation only. Deferred follow-up
(after local staging exists): an ATC trajectory loader plus skip-if-absent shape-contract tests,
following the exemplar pattern established for #4279 (`socnavbench-s3dis-eth`). No download code,
dataset bytes, benchmark consumer, or claim edit is part of #4289.
