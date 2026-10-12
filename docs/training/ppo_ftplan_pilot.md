# Unsubmitted released-PPO fine-tuning pilot

Refs #9628, #10071 and #9609. Author approval: 2026-10-12. This is development
tuning preparation. No GPU training or Slurm submission was performed for this
packet. No frozen artifact, reward implementation, alignment config, or seed-policy
PR file is changed.

The machine-readable [protocol](../../configs/training/ppo/ftplan/pilot_protocol_20261012.yaml)
and [training config](../../configs/training/ppo/ftplan/released_warm_weighted_3m_20261012.yaml)
bind the released checkpoint (job 11724, W&B `ibo3aqus`), three initialization seeds
100001/100002/100003, 3M steps per seed, and a factor-two training oversample of
bottleneck, doorway, exiting elevator/room and frontal approach. Sampling factors
are a predeclared pilot choice, not optimized on released results. Training layouts
use the training-only range under each logged master. Initialization changes future
optimizer/minibatch draws; the initial deterministic actor/value weights remain the
same released policy in all three runs.

Use the released observation/action contract. The canonical trainer must reject
any space mismatch. This pilot does not silently remap the released policy to the
alignment lane's new plant or action schema. If a migration is needed, stop and
prepare a separately reviewed conversion before admission. The inherited recipe
also retains its historical reward/curriculum settings: this pilot isolates warm
initialization plus family sampling, and does not claim to implement the reward
lane's improvements. Record the resolved settings alongside every segment.

The five-dev-seed diagnostic found the strict signature on 1/5, with static contact
on all five. Partial replication does not establish a consistent actor mechanism;
therefore the conditional obstacle observation extension is deferred on #9609.

The [Slurm script](../../SLURM/ftplan/released_ppo_pilot.slurm) is staged in 1M-step
segments. **Do not submit before the controlled seed-policy replication reports**
from #10343 are complete with worker receipt audits and independent hub/W&B readback.
The seed-policy implementation (#10324/#10343), warm-start support, family weighting,
exact-head independent review, required CI and the orchestrator's behavior receipt
are also prerequisites. Current-main preparation does not satisfy these future
execution prerequisites. The public script has no site partition/account/QoS and
never invokes `sbatch`; those settings come from the private operations overlay.

After prerequisites pass, the orchestrator binds `FTPLAN_SOURCE_HEAD` to the reviewed
checkout and supplies `FTPLAN_REPLICATION_MANIFEST`, `FTPLAN_SELECTION_RECEIPT`,
`FTPLAN_HUB_ROOT`, and `FTPLAN_SEGMENT` (1–3), using the repository max-time submission
wrapper. Create `output/slurm` before scheduling. The script runs one array replica
at a time (`0-2%1`) and never schedules a successor. Different segment roots prevent
checkpoint/log collisions. Review live throughput and partition limits before any
submission; the four-hour directive is a template request, not measured headroom.

Replication manifest contract: `status: reported`, `worker_receipt_audit: passed`,
three `paired_master_reports`, each carrying a repository-relative `artifact_path`,
SHA-256, and independently verified `hub_artifact`/`wandb_artifact` references. This
is an orchestration handoff manifest pointing to the complete reports, not a new
scientific report schema. Attach all master-pair outcomes, including negative ones;
passing navigation performance is not a prerequisite invented by this pilot.

Run the released checkpoint once on **1031–1060 × all 48 scenarios** through the
canonical benchmark runner and released adapter, on CPU, before segment one.
Between GPU segments, preserve the complete segment, independently read back hub
and W&B, audit worker seeds, then run the same CPU selection pairs. The internal
trainer's single-episode evaluation is diagnostic only; ignore its selected-best
file for pilot checkpoint choice. The externally evaluated segment final export
is the continuation checkpoint. Rollout rounding is recorded as actual steps and
must not silently change the planned 3M budget cap interpretation.

Selection receipt contract: exact `evaluation_seeds`, `scenario_count: 48`,
`fallback_used: false`, `execution_errors: 0`, `completed_segment`,
`checkpoint_sha256`, and a `results_artifact` with the same path/digest/custody
fields as the report manifest. Later segments also require `initialization_seed`,
`checkpoint_artifact` and `previous_success_mean`/`success_mean`. The first receipt
must bind the released checkpoint digest. Every later receipt must bind that
replica's final checkpoint and show **strictly rising selection success** against
the previous selection, starting with the released baseline. Stop that seed when
success fails to rise, or on NaN, execution error or fallback. Keep stopped seeds
in the record and do not replace them. This exploratory stop rule has no claimed
significance threshold. Select the highest selection success, ties to the earliest
checkpoint; retain final and best exports even when stopped.

Only after checkpoint choices are frozen, run selected checkpoints once on
**1101–1130 × all 48 scenarios**, CPU. Those report seeds never select checkpoints.
Report success, static and pedestrian collisions, cap-speed fractions, per-family
outcomes and training-seed variability; publish negative results equally. Preserve
released source pointers, initialization ZIPs, every segment final, selection best,
resolved configs, source SHA, worker receipts, manifests, selection rows, logs and
normalization companions if any to the external hub and W&B. The script copies and
checks hub bytes and waits for W&B commit; an independent readback is still required
before the next segment's selection receipt. Register new checkpoints only with
those durable pointers and hashes. Record all attempts in the extra-tuning ledger;
no release replacement or manuscript claim follows automatically.

Job 11724 used 8h04m per 10M steps on one L40S with 22 workers. For 3 × 3M steps,
the nominal training cost is **7.26 L40S GPU-hours**, or **3.63–10.89 hours** at the
issue's ±50% planning uncertainty; reserve **12 GPU-hours** pending a measured first
segment. Nominal training core-hours are 159.7 (22 workers). Rollout rounding and
startup add cost. These are extrapolations, not a new throughput measurement or an
A30 equivalence claim. CPU evaluation is separate and unmeasured: at most 14,400
selection episodes (one common baseline plus nine checkpoint evaluations) and
4,320 report episodes. Measure the baseline's CPU cost and update the estimate
before admitting the full pilot; do not keep the GPU allocation while waiting for
CPU selection.
