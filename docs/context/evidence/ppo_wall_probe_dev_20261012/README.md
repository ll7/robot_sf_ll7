# April issue-791 PPO wall probe on five development seeds

<!-- AI-GENERATED NEEDS-REVIEW -->

Diagnostic only. Refs #9609, #9628 and #10071. CPU inference, no training or
Slurm submission. Source main: `33a4735900`; requested April checkpoint: job 11724 /
W&B `ibo3aqus`, registry ID
`ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417`.
Artifact URLs, digests, sizes and producer identity are in [probe.json](probe.json).
This checkpoint uses raw dictionary observations without VecNormalize. Its
historical training plant allowed 3 m/s and reverse. It is distinct from the
October B1002 checkpoint used by plain PPO in release 0.0.8.

Reproduce from the repository root:

```bash
OMP_NUM_THREADS=1 LOGURU_LEVEL=ERROR scripts/dev/run_worktree_shared_venv.sh -- \
  python -m scripts.analysis.ppo_wall_probe_dev --out output/ftplan/probe.json
```

The command fixes seeds to 1001–1005 and runs the narrow-doorway scenario through
the canonical benchmark adapter. It verifies PPO and predictive release artifact
hashes and rejects predictive fallback. At contact-minus 20/10/5/1 steps, it records
Gaussian means before clipping, absolute adapter targets, native actions, executed
speed, critic value, canonical wall geometry, and a same-state static-grid ablation.
Static ablation also removes the static contribution to the combined channel.
Other geometry-correlated inputs remain.

The current registry declares `velocity_delta` for this April checkpoint. Its
Gaussian forward means are **velocity deltas**, not absolute speeds. The adapter
adds clipped deltas to current physical speed and then limits the absolute target.
A smaller positive delta does not establish slowing. Schema v2 labels the raw and
ablated means explicitly and classifies the absolute adapter target speed.

| Dev seed | Static contact step | Raw forward delta means at four offsets (m/s) | Absolute target speeds (m/s) | Target-speed signature |
| --- | ---: | --- | --- | --- |
| 1001 | 58 | 3.36, 3.01, 2.89, 2.39 | 2, 2, 2, 2 | holds |
| 1002 | 61 | 2.93, 1.85, 1.78, 1.96 | 2, 2, 2, 2 | holds |
| 1003 | 66 | 3.00, 2.11, 1.04, 2.44 | 2, 2, 2, 2 | holds |
| 1004 | 58 | 3.37, 1.98, 1.26, 2.03 | 2, 2, 2, 2 | holds |
| 1005 | 84 | 3.12, 2.29, 1.72, 0.85 | 2, 2, 2, 2 | holds |

All **20 sampled adapter targets are 2 m/s**, and their recorded executed speeds
remain approximately 2 m/s. All five seeds have static contact. The absolute-target
criterion holds on **5/5**: all four pre-contact targets must be at least 1.85 m/s,
with the final forward occupied cell closer than the first. The previous 1/5 result
used a threshold on raw velocity deltas and must not be interpreted as a slowing
or absolute-speed result. The independent review reproduced all measured rows;
this correction changes labels and interpretation, not the recorded trajectory.

For seed 1001, the forward static cell center approaches from 4.95 to 1.21 m while
critic value drops from -11.21 to -14.92. Grid distances use actual channel indices
and cell centers in the reported frame, in a forward corridor of robot radius plus
one grid cell. They are neither surface clearance nor identification of a particular
SVG wall. Physical wall clearance is reported separately. Seed 1005 changes heading,
so its forward distance is not monotonic. This establishes the measured target-speed
signature in these five trajectories; it does not establish the learned cause or a
population-level effect. Optional obstacle-feature implementation remains tracked
on #9609 and is not supplied by this diagnostic repair.

Repeated CPU measurements agree. The JSON is tracked evidence, and model inputs
remain independently recoverable release artifacts; ignored model-cache symlinks
are disposable caches. No benchmark admission, release replacement or dissertation
claim follows from this packet.
