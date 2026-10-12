# Unsubmitted released-PPO fine-tuning pilot

Refs #9628, #10071 and #9609. Author approval: 2026-10-12. This is development
tuning preparation. No GPU training or Slurm submission was performed for this
packet. No frozen artifact, reward implementation, alignment config, or seed-policy
PR file is changed.

The machine-readable [protocol](../../configs/training/ppo/ftplan/pilot_protocol_20261012.yaml)
and [prepared training template](../../configs/training/ppo/ftplan/released_warm_weighted_3m_20261012.template)
bind the released plain-PPO checkpoint `ppo_release_robot_b1002_last_20261001`, three initialization seeds
100001/100002/100003, 3M steps per seed, and a factor-two training oversample of
bottleneck, doorway, exiting elevator/room and frontal approach. Sampling factors
are a predeclared pilot choice, not optimized on released results. Training layouts
use the training-only range under each logged master. Initialization changes future
optimizer/minibatch draws; the initial deterministic actor/value weights remain the
same B1002 policy in all three runs. The template is deliberately named
`.template`: it is dependent preparation, not a canonical runnable `.yaml`
leaf. The stock 149-leaf inventory and its load checks remain unchanged. The
launcher materializes an isolated runtime YAML only after its dependency gates.

The default and only source is **October B1002**, the released plain PPO in 0.0.8:
**2.0 m/s, no reverse, no curriculum, `route_completion_v3`, 18 workers and CNN
32/64/64** ([baseline](../../configs/baselines/ppo_release_robot_0_0_8_cpu.yaml),
[recipe](../../configs/training/ppo/expert_ppo_release_contract_b_seed1002.yaml)).
Its pinned GitHub release and complete W&B artifact are recorded in the model
registry and protocol. The template inherits that recipe and pins 18 workers;
its new training seeds replace the historical source recipe's seed. The April
issue-791 policy is not this comparator, and the March BR06 v3 policy belongs to
guarded PPO. No BR06 arm is retained: this pilot tests improvement of the plain
PPO comparator, and no separate BR06 comparison is needed for that question.

Preserve the B1002 observation/action contract. It uses velocity-delta actions
with serialized bounds `[-2, -1]` to `[2, 1]`; the canonical trainer rejects any
space mismatch. The launcher requires an action-contract receipt bound to the
exact reviewed source head and checkpoint digest, with `mapping: identity`,
`target_semantics: velocity_delta`, `target_action_low: [-2, -1]`,
`target_action_high: [2, 1]`, `max_linear_speed: 2.0`, `allow_backwards: false`,
and a `review_artifact` with path/digest/hub/W&B readback references. It does not
use the generic bounds-migration option in the warm-start API. A new observation
or action schema would require separate reviewed conversion and parity proof.
The inherited source reward remains `route_completion_v3` without curriculum;
this pilot does not incorporate the reward lane's proposed improvements. Record
the resolved settings alongside every segment. The benchmark plant/action parity
receipt remains an admission gate, including time step and acceleration limits.

The corrected five-dev-seed diagnostic for the **April issue-791 checkpoint**
reports **5/5** for the absolute-target signature: all 20 sampled commands target
2 m/s and all five trajectories have static contact. That historical result does
not establish B1002's wall-approach behavior. B1002 requires its own diagnostic
before interpreting or adding an obstacle feature; that follow-up remains on
#9609. The prepared B1002 pilot has no obstacle-observation extension.

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
`FTPLAN_HUB_ROOT`, `FTPLAN_ACTION_CONTRACT_RECEIPT`, and `FTPLAN_SEGMENT` (1–3), using the repository max-time submission
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

Selection receipt contract: `status: completed`, exact `evaluation_seeds`, `scenario_count: 48`,
`fallback_used: false` (JSON boolean), `execution_errors: 0` (JSON integer), `completed_segment`,
`checkpoint_sha256`, and a `results_artifact` with the same path/digest/custody
fields as the report manifest. Later segments also require `initialization_seed`,
`checkpoint_artifact` and `previous_success_mean`/`success_mean`. The first receipt
must bind the released checkpoint digest. Missing, null, empty-string and incorrectly
typed fallback/error fields fail admission, as do failed or incomplete statuses.
Every later receipt must bind that replica's final checkpoint and show
**strictly rising selection success** against
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
normalization companions if any to the external hub and W&B. EXIT, INT and TERM paths preserve partial artifacts as well as successful
segments. A bound `segment_status.json` and W&B metadata record complete/incomplete
status and the original exit code. Training runs in an isolated process group via
`setsid`. On INT or TERM, the EXIT handler sends TERM to that group, allows five
seconds, then sends KILL and reaps the launcher before custody. This also stops a
trainer that inherited ignored INT or ignores TERM. Interruption exits remain
130 (INT) or 143 (TERM). Failed/interrupted segments retain that exit
code even if custody also fails; successful training with failed custody exits
nonzero and cannot continue. An early Slurm TERM request provides a custody window;
uncatchable SIGKILL or unavailable storage can still prevent preservation, and
must leave the segment blocked with local outputs retained. The script copies and
checks hub bytes and waits for W&B commit; an independent readback is still required
before the next segment's selection receipt. Register new checkpoints only with
those durable pointers and hashes. Record all attempts in the extra-tuning ledger;
no release replacement or manuscript claim follows automatically.

There is no measured B1002 fine-tuning throughput in this packet. As a provisional
cross-recipe planning proxy, job 11724 used 8h04m per 10M steps on one L40S with
22 workers. Scaling 3 × 3M steps by that rate and a heuristic **22/18 worker ratio**
gives **8.87 L40S GPU-hours**, or **4.44–13.31 hours** at ±50%; reserve **15 GPU-hours**.
This updates the previous 7.26-hour, 22-worker estimate. B1002's smaller CNN and
changed scenario lengths may change throughput in either direction; neither the
worker scaling nor an A30 equivalence is measured. Re-estimate after the first
1M-step segment, before admitting later segments. Nominal worker core-hours are
159.7 (18 workers); allocated core-hours also include launcher/evaluation overhead.
Rollout rounding and startup add cost.

CPU evaluation is separate and unmeasured: at most 14,400 selection episodes
(one common baseline plus nine checkpoint evaluations) and 4,320 report episodes.
Measure the baseline's CPU cost before admitting the full pilot; release the GPU
allocation while waiting for CPU selection.
