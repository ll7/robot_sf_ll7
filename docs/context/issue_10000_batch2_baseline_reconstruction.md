# Batch 2 release comparison: reconstruction stopped (R1)

The paired schema and metric-semantics checks remain enforced, including collection
of existing sweeps. The requested baseline source is the immutable release commit
`373dbfde4f39667cf9e8732dabe7df5118bdeab1`. Its original schema and metric code differ
from the integrated source, so the comparison continues to refuse acquisition.

The maintainer-selected route was to reconstruct release behaviour with the
harness source's measurement layer. That would describe the baseline as
"release behaviour, measured with the head's metric layer". This route requires
both schemas and the actual executed metric code to match; it permits no change
to the release planner, simulator, maps, configs or models. Frozen release
artifacts, release tags, release configs and `*_frozen.yaml` files are untouched.

The current metric-semantics digest covers these five files, in addition to the
separately digested episode schema:

- `robot_sf/benchmark/metrics.py`
- `robot_sf/benchmark/metric_definitions.py`
- `robot_sf/benchmark/constants.py`
- `robot_sf/benchmark/robot_force_contract.py`
- `robot_sf/benchmark/map_runner/map_runner_episode.py`

The required schema is `robot_sf/benchmark/schemas/episode.schema.v1.json`.
A reconstruction probe installed the existing committed acquisition bundle in a
linked checkout of the release, then installed exactly these six head files and
recorded every original and installed SHA-256 digest. Importing the reconstructed
episode runner exited 1 with:

```text
ModuleNotFoundError: No module named 'robot_sf.benchmark.effective_physics'
```

That module does not exist at the release source. The head episode runner also
imports the absent `robot_sf.common.hybrid_defaults` module and wraps production
`run_map_episode` with `episode_default_policy`. It performs environment and policy
construction and the step loop, including new seed guards. Replacing this whole
file is therefore a runtime overlay, not a measurement-only overlay. Adding its
missing dependencies or extracting a new metric implementation would exceed the
selected route; omitting the file would leave the unchanged metric digest unequal.

R1 is stopped under the maintainer's explicit stop rule. No measurement overlay
has been enabled, no alternate comparator has been invented, and no compatible
paired preflight is claimed. The original mismatch remains a fail-closed refusal.
The private repair report retains the reconstruction manifest, import stderr,
source diff and original guard reproduction. A new independent combined review
must assess this stopped route alongside the other integration repairs before
any later source freeze or canonical acquisition.
