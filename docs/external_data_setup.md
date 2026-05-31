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
