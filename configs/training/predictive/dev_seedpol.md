# Reproducible training-only seed policy

Author ruling: issue #9628, 2026-10-12. The `dev_seedpol_*` configs are seed-policy
successors of train010. The PPO fixed/safety reward weights, plant settings, 5M
training budget, 20 workers, 256-step rollouts, native horizons and optimization
parameters remain unchanged. Selection uses **1031–1060 × 48 scenarios** (1,440
rather than train010's 480 selection episodes); comparison uses **1101–1130 × 48**.
These are development tuning results, with no release or paper claim admission.

`training_seed_range: [100000, 199999]` is inclusive. `training_master_seed: 20261012`
seeds one shuffled permutation, distributed by worker stride. Construction and
all resets consume the worker stream, including single-scenario runs. Workers
never reuse an episode seed; exhaustion fails explicitly. Master-seeded scenario
and domain randomization are reproducible for the same config/worker count.
CUDA/backend variations may still prevent bitwise optimizer reproducibility.
The master is logged to W&B, resolved configs and training manifest notes.

Resolution refuses range overlap with selection, comparison, reserved development
and either sealed band, covering the complete sealed interval. An explicit legacy
reset pool also refuses selection leakage and sealed identities. Legacy scalar
seed configs retain their static loading behavior and remain subject to existing
runtime sealed-seed guards. A range requires fixed selection/comparison seeds,
empty legacy seeds/pool, and randomization enabled. Resume is refused until episode
stream state can be restored, preventing silent layout reuse after restart.

Train with `scripts/training/train_ppo.py --config
configs/training/ppo/dev_seedpol_fixed_5m.yaml` (or its safety sibling).
The new config branch requires independent review before merge. Experimental
execution may use its archived, exact commit; it does not replace a released model.

The forecast and prediction collection configs preserve all train010 scientific
fields, including eight-step prediction, collection lengths, hardcase repetition,
identity supervision and the 220-epoch learning recipe. The canonical collector
accepts `--training-config configs/training/predictive/dev_seedpol_forecast_collection.yaml`
(or the prediction sibling) instead of `--seed-manifest`. Supply the existing
matrix, output and scientific CLI settings from that config. Collection creates
15 fresh episodes per scenario and writes a `.seed-manifest.yaml` sidecar.
For base/hardcase collection in one run, use `--seed-worker-count 2` and distinct
`--seed-worker-index 0` / `1`, so the two datasets use disjoint identities.
`--seed-split selection` and `--seed-split evaluation` collect the fixed held-out
identities for separate datasets; those rows must never enter training data.
Checkpoint selection must use the separate selection dataset, rather than treating
the inherited random `val_split` diagnostic as held-out navigation evidence.

Preserve commit, input and resolved config hashes, master/ranges, per-episode
manifests, hardware, checkpoints and W&B links on the external data hub, with
hash readback; W&B artifacts provide independent preservation. Public issue and
PR comments contain no infrastructure identities or storage paths. Both linked
issues remain open; this bounded seed-policy work does not finish #9628's broader
curriculum and reward programme.
