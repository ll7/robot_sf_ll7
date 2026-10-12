# Released PPO wall probe on five development seeds

Diagnostic only. Refs #9609, #9628 and #10071. CPU inference, no training or
Slurm submission. Source main: `33a4735900`; checkpoint: job 11724 / W&B
`ibo3aqus`. Artifact URLs, digests and sizes are recorded in [probe.json](probe.json).
The source checkpoint uses raw dictionary observations, without VecNormalize.

Reproduce from the repository root:

```bash
OMP_NUM_THREADS=1 LOGURU_LEVEL=ERROR scripts/dev/run_worktree_shared_venv.sh -- \
  python -m scripts.analysis.ppo_wall_probe_dev --out output/ftplan/probe.json
```

The command fixes seeds to 1001–1005 and runs the narrow-doorway scenario through
the canonical benchmark velocity-command adapter. It rejects unavailable PPO or
predictive artifacts and predictive fallback. At contact-minus 20/10/5/1 steps,
it records Gaussian actor mean before clipping, adapter command, native action,
executed speed, critic value, canonical wall geometry, and a same-state static-grid
ablation. Removing the static channel also removes its contribution to the combined
channel. This ablation does not remove other geometry-correlated inputs.

| Dev seed | Static contact step | Raw forward means at the four offsets (m/s) | Strict signature |
| --- | ---: | --- | --- |
| 1001 | 58 | 3.36, 3.01, 2.89, 2.39 | holds |
| 1002 | 61 | 2.93, 1.85, 1.78, 1.96 | does not hold |
| 1003 | 66 | 3.00, 2.11, 1.04, 2.44 | does not hold |
| 1004 | 58 | 3.37, 1.98, 1.26, 2.03 | does not hold |
| 1005 | 84 | 3.12, 2.29, 1.72, 0.85 | does not hold |

The strict criterion was coded before execution: all four raw forward means must
be at least 1.85 m/s, with the final forward occupied cell closer than the first.
Classification uses full precision; rounded 1.85 in the table is not a threshold
comparison. All five have static contact. The sustained-high-speed signature
replicates on **1/5**, so this does not establish a consistent five-seed mechanism.
For seed 1001, the nearest forward static cell center approaches from 4.95 to
1.21 m, while critic value drops from -11.21 to -14.92. The policy partly slows in
the other seeds. A nearest-obstacle observation extension is therefore deferred;
it is not a demonstrated remedy from this result.

Grid distances use actual channel indices and cell centers in the reported frame,
in a forward corridor of robot radius plus one grid cell. They are neither surface
clearance nor an identification of a particular SVG wall. Physical wall clearance
is reported separately. Seed 1005 changes heading and its forward distance is not
monotonic. No broader population, causal attribution, release or dissertation
claim follows from this packet. Two CPU repetitions produced identical measured
rows. The JSON is committed evidence, and model inputs remain independently
recoverable release artifacts; ignored model-cache symlinks are disposable caches.
