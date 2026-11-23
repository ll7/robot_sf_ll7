# Predictive pipeline training seed roles

The six randomized predictive recipes use `base_collection.master_seed` to draw
unique episode seeds without replacement from the training pool 100000–199999.
The former numeric scenario offsets are removed. The base dataset manifest records
the master seed and the generated manifest path; changing the recipe changes
experiment identity, and historical results do not transfer.

Hardcase collection has its own `hardcase_collection.seed_manifest` and
`master_seed`. The shipped training manifest uses master seed 100002 and retains
the four hardcase families and their per-family episode counts. Proxy selection
and development diagnostics use a separate development manifest. Optimizer RNG
seeds in the active randomized recipes also use the training role.

Both collection commands pass `--training-seeds-only`. The collector rejects any
non-training episode seed before environment construction in that mode. Its
separate small development-collection interface remains available for diagnostics.
No historical benchmark manifest or frozen recipe is rewritten or relabelled.

The integration tests run each actual shipped recipe through the real generator,
CLI parsing, manifest validation and environment construction, then immediately
close the environment and stop before rollout collection or model training. A
separate regression proves a development manifest cannot reach training simulation.
