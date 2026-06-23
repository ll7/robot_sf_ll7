# External Data Setup Assistant

Robot SF does not redistribute licensed or bulky external datasets. Use the local setup assistant
to discover supported asset groups, validate manually staged data, and write compact provenance
manifests without committing raw files.

```bash
uv run python scripts/tools/manage_external_data.py list
uv run python scripts/tools/manage_external_data.py explain sdd
uv run python scripts/tools/manage_external_data.py check sdd
uv run python scripts/tools/manage_external_data.py stage sdd --source /path/to/local/sdd
```

The first supported asset groups are:

- `sdd`: Stanford Drone Dataset annotation text files for the SDD scenario importer.
- `socnavbench-s3dis-eth`: SocNavBench/S3DIS ETH mesh and traversible inputs.
- `socnavbench-control`: SocNavBench `wayptnav_data` control-pipeline assets.
- `amv-calibration`: local-only AMV actuation calibration source provenance for #1585/#1559.

`download` intentionally fails closed for these groups because the repository has not encoded a
license-safe direct download path. Follow the official source instructions printed by `explain`,
stage the data locally, and then run `stage`.

For `amv-calibration`, the helper only records provenance for an already accepted source class:
an official platform/controller specification, a maintainer-accepted platform-class source, or a
command-response trace bundle. It does not turn synthetic AMV actuation diagnostics into calibrated
hardware evidence.

Generated manifests default to `output/external_data/manifests/`, which is local-only by default.
They include source URL, license/access notes, local path, required layout, file count, total size,
an aggregate tree checksum, and sample file hashes. Raw datasets, videos, pickles, meshes, archives,
and private traces remain untracked; repo-local raw paths must be covered by `.gitignore` before a
manifest can be written.

## Where to obtain and where to place each dataset

All of these are **license/agreement-gated** — there is no scriptable direct download. Acquire each
under its own terms, then place it at the path below (relative to the repository root). Run
`explain <asset>` for the authoritative, always-up-to-date contract; this table is the human-readable
summary.

| Asset (`explain` id) | Where to obtain | License / access | Place it at (repo-relative) | Required contents |
| --- | --- | --- | --- | --- |
| `sdd` | <https://cvgl.stanford.edu/projects/uav_data/> | CC BY-NC-SA 3.0 (non-commercial); download per the dataset terms | `output/external_data/sdd/` | the original annotation text files (`**/annotations.txt`, columns `track_id xmin ymin xmax ymax frame lost occluded generated label`) |
| `socnavbench-s3dis-eth` | <https://github.com/CMU-TBD/SocNavBench> install docs → the curated S3DIS/SBPD asset package | **S3DIS** meshes/traversibles have a separate access agreement; not redistributed by Robot SF | `third_party/socnavbench/` | `sd3dis/stanford_building_parser_dataset/mesh/ETH/` (dir) and `sd3dis/stanford_building_parser_dataset/traversibles/ETH/data.pkl` |
| `socnavbench-control` | <https://github.com/CMU-TBD/SocNavBench> (`wayptnav_data`) | external SocNavBench assets, not redistributed | `third_party/socnavbench/` | `wayptnav_data/` (precomputed waypoint/control data) |

After placing the files, validate and write a provenance manifest:

```bash
uv run python scripts/tools/manage_external_data.py check sdd
uv run python scripts/tools/manage_external_data.py stage socnavbench-s3dis-eth \
  --source third_party/socnavbench
```

For SocNavBench/ETH specifically, a successful `check` is what moves the `eth_first` map converter
(#1134/#1498) from `blocked_pending_source_assets` to a real conversion gate.

## Sharing external data across git worktrees

The expected paths above (`output/external_data/<asset>`, `third_party/socnavbench`) resolve
**relative to the current working tree**. Robot SF is frequently developed in
[linked git worktrees](https://git-scm.com/docs/git-worktree), so data staged in one worktree is
**not** visible from another — each worktree has its own `output/` and `third_party/`. Stage the
data **once** at a machine-stable location outside any worktree, then expose it to every worktree.

Both options below target git-ignored paths, so nothing about the raw data is ever committed.

### Recommended today: symlink a shared location into each worktree

```bash
# 1. Keep one physical copy outside the repo (any stable path you control):
mkdir -p ~/robot_sf_external_data
# ...place SDD under ~/robot_sf_external_data/sdd and SocNavBench under ~/robot_sf_external_data/socnavbench...

# 2. In EACH worktree (and the main checkout), create ignored parent dirs and link the expected paths:
mkdir -p output/external_data third_party
ln -sfn ~/robot_sf_external_data/sdd        output/external_data/sdd
ln -sfn ~/robot_sf_external_data/socnavbench third_party/socnavbench
```

`ln -sfn` is safe to re-run. The link lives at a `.gitignore`-covered path, so it is never tracked;
the tool, the SDD importer, and the scenario-prior gate all then read the one shared copy. Re-create
the two links after `git worktree add` (a future enhancement, below, removes this step).

### Validate a shared copy without symlinks

`check`/`stage` accept `--source` with an absolute path, so you can validate a shared copy directly
without linking it into the worktree:

```bash
uv run python scripts/tools/manage_external_data.py stage sdd --source ~/robot_sf_external_data/sdd
```

Note: `--source` covers *validation and manifest writing*. Downstream consumers (the SDD importer,
the scenario-prior gate, SocNavBench map conversion) still read the asset's expected repo-relative
path, so use the symlink above when you need those to see a shared copy.

### Planned: a single shared data root (no per-worktree symlinks)

A follow-up will teach `manage_external_data.py` to honor a `ROBOT_SF_EXTERNAL_DATA_ROOT`
environment variable so every worktree reads from one machine-level root automatically — removing
the per-worktree symlink step entirely. Tracked on top of the staging-consolidation work (#3473).
