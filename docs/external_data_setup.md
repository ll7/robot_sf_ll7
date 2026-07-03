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
- `socnavbench-s3dis-eth`: SocNavBench/S3DIS ETH mesh and traversible inputs; see
  [SocNavBench S3DIS ETH External Data](datasets/socnavbench-s3dis-eth.md).
- `socnavbench-control`: SocNavBench `wayptnav_data` control-pipeline assets.
- `amv-calibration`: local-only AMV actuation calibration source provenance for #1585/#1559.
- `eth-ucy`: ETH BIWI and UCY Crowds-by-Example pedestrian-trajectory
  acquisition/provenance metadata; see
  [ETH/UCY External Trajectory Data](./datasets/eth-ucy.md).

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

For worktree-portable staging, set `ROBOT_SF_EXTERNAL_DATA_ROOT` to one machine-stable directory
outside any checkout. When set, the assistant and downstream SDD/SocNavBench consumers resolve
supported assets there instead of under the current worktree:

```bash
export ROBOT_SF_EXTERNAL_DATA_ROOT="$HOME/robot_sf_external_data"
```

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
| `eth-ucy` | <https://vision.ee.ethz.ch/datsets.html> plus UCY Crowds-by-Example source/paper provenance | research-use terms from upstream or maintainer-approved mirror; not redistributed by Robot SF | `output/external_data/eth-ucy/` or `$ROBOT_SF_EXTERNAL_DATA_ROOT/eth-ucy/` | ETH/Hotel `obsmat.txt`, UCY `.vsp` or trajectory `.txt`, plus local `README*`, `LICENSE*`, or `TERMS*`; see [ETH/UCY External Trajectory Data](./datasets/eth-ucy.md) |

After placing the files, validate and write a provenance manifest:

```bash
uv run python scripts/tools/manage_external_data.py check sdd
uv run python scripts/tools/manage_external_data.py stage socnavbench-s3dis-eth \
  --source third_party/socnavbench
```

For SocNavBench/ETH specifically, a successful `check` is what moves the `eth_first` map converter
(#1134/#1498) from `blocked_pending_source_assets` to a real conversion gate.

## Sharing external data across git worktrees

By default, the expected paths above (`output/external_data/<asset>`, `third_party/socnavbench`)
resolve **relative to the current working tree**. Robot SF is frequently developed in
[linked git worktrees](https://git-scm.com/docs/git-worktree), so data staged in one worktree is not
visible from another unless you opt into a shared root or create links.

### Recommended: one shared data root

Set `ROBOT_SF_EXTERNAL_DATA_ROOT` in your shell, `local.machine.md`, or job launcher environment and
place each dataset under its shared subdirectory:

```bash
export ROBOT_SF_EXTERNAL_DATA_ROOT="$HOME/robot_sf_external_data"
mkdir -p "$ROBOT_SF_EXTERNAL_DATA_ROOT"

# ...place SDD under $ROBOT_SF_EXTERNAL_DATA_ROOT/sdd...
# ...place SocNavBench under $ROBOT_SF_EXTERNAL_DATA_ROOT/socnavbench...

uv run python scripts/tools/manage_external_data.py check sdd
uv run python scripts/tools/manage_external_data.py check socnavbench-control
```

With that variable set, `list`, `explain`, `check`, `stage`, and `download` report the shared-root
path. `stage` defaults to that path when `--source` is omitted. The SDD scenario-prior gate and the
SocNavBench asset helper also consume the same resolved paths, so two worktrees pointed at the same
root see one physical staged copy without per-worktree links.

### Validate a one-off path

`check`/`stage` still accept `--source` with an absolute path when you need to validate a copy that
is not under the shared root:

```bash
uv run python scripts/tools/manage_external_data.py stage sdd --source ~/robot_sf_external_data/sdd
```

`--source` affects only that validation/manifest command; downstream consumers use
`ROBOT_SF_EXTERNAL_DATA_ROOT` or the default repo-relative path.

### Fallback: symlink a shared location into each worktree

If you cannot set an environment variable for a given tool, expose the shared copy through
git-ignored symlinks in each worktree:

```bash
mkdir -p output/external_data third_party
ln -sfn ~/robot_sf_external_data/sdd        output/external_data/sdd
ln -sfn ~/robot_sf_external_data/socnavbench third_party/socnavbench
```

`ln -sfn` is safe to re-run. The links live at `.gitignore`-covered paths, so they are never tracked.
