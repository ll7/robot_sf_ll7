# ETH/UCY External Trajectory Data

Plain-language summary: ETH/UCY is a public pedestrian-trajectory benchmark family, but Robot SF does not redistribute the raw files. This page documents how a contributor can acquire and stage their own local copy so the external-data checker can report either `missing` or locally `ready` without promoting any benchmark claim.

## Scope

This page covers the ETH BIWI Walking Pedestrians data (`eth`, `hotel`) and the UCY Crowds-by-Example data commonly staged as `univ`, `zara01`, and `zara02` for pedestrian-trajectory prediction comparisons.

This documentation does not add a loader, dataset bytes, benchmark scenarios, prediction-comparability results, or paper/dissertation claims. Until a later loader and shape-contract slice is implemented, the registry entry is acquisition and provenance metadata only.

## Sources And Citations

Use the official source first when it is reachable:

- ETH BIWI Walking Pedestrians dataset: <https://vision.ee.ethz.ch/datsets.html>
- UCY Crowds-by-Example project/paper provenance: Lerner et al., "Crowds by Example", Computer Graphics Forum, 2007, DOI `10.1111/j.1467-8659.2007.01089.x`

If an official download endpoint is unavailable, use a maintainer-approved mirror and record that mirror in the local provenance manifest. Common research distributions preserve the five-sequence ETH/UCY layout, but a mirror is not automatically accepted as Robot SF benchmark evidence.

Citation block for downstream notes:

```bibtex
@inproceedings{pellegrini2009you,
  title = {You'll Never Walk Alone: Modeling Social Behavior for Multi-target Tracking},
  author = {Pellegrini, Stefano and Ess, Andreas and Schindler, Konrad and Van Gool, Luc},
  booktitle = {IEEE International Conference on Computer Vision},
  year = {2009}
}

@article{lerner2007crowds,
  title = {Crowds by Example},
  author = {Lerner, Alon and Chrysanthou, Yiorgos and Lischinski, Dani},
  journal = {Computer Graphics Forum},
  volume = {26},
  number = {3},
  pages = {655--664},
  year = {2007},
  doi = {10.1111/j.1467-8659.2007.01089.x}
}
```

## Expected Layout

Set a shared external-data root when you use multiple worktrees:

```bash
export ROBOT_SF_EXTERNAL_DATA_ROOT="$HOME/robot_sf_external_data"
mkdir -p "$ROBOT_SF_EXTERNAL_DATA_ROOT/eth-ucy"
```

Stage the acquired data under:

```text
$ROBOT_SF_EXTERNAL_DATA_ROOT/eth-ucy/
  eth/
    obsmat.txt
    README-or-terms-file
  hotel/
    obsmat.txt
    README-or-terms-file
  univ/
    *.vsp or trajectory *.txt
    README-or-terms-file
  zara01/
    *.vsp or trajectory *.txt
    README-or-terms-file
  zara02/
    *.vsp or trajectory *.txt
    README-or-terms-file
```

The checker accepts equivalent nested layouts as long as it finds at least one trajectory file group and one local README, license, or terms file. Keep source terms with the local copy.

Without `ROBOT_SF_EXTERNAL_DATA_ROOT`, the default repo-local staging path is:

```text
output/external_data/eth-ucy/
```

That path is ignored and must stay local.

## Check And Stage

Expected state before local acquisition is `missing`:

```bash
uv run python scripts/tools/manage_external_data.py --json check eth-ucy
```

After acquiring the data under the expected layout, validate and write a local provenance manifest:

```bash
uv run python scripts/tools/manage_external_data.py stage eth-ucy \
  --source "$ROBOT_SF_EXTERNAL_DATA_ROOT/eth-ucy" \
  --manifest-out output/external_data/manifests/eth-ucy.provenance.json
```

The manifest records source URL, access notes, matched required paths, file counts, and aggregate checksums. Do not commit the raw dataset or the local manifest unless a maintainer explicitly asks for a small reviewable provenance copy.

## Claim Boundary

`eth-ucy` registry `ready` means only that a local copy matched the documented file/layout contract and provenance fields. It does not mean Robot SF has a validated ETH/UCY loader, shape-contract tests, benchmark scenarios, prediction-comparability evidence, or permission to redistribute the data.
