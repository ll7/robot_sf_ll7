# Three-width doorway comparison (issue #9348)

This application freezes a within-planner comparison at 2.2, 2.8 and 3.6 m
free opening widths and 1.0 m wall depth. It uses the #6644 map generator;
the historical 2.0 m map stays unchanged and is outside the comparison.
The [protocol note](../context/issue_9348_three_width_doorway_protocol.md)
records the source-backed geometry and planner-grid distinction.

The intended campaign has `goal` and `social_force`, seeds 225–227, and a
native horizon of 400 steps at 0.1 s per step: 18 rows. The generated
scenarios differ only in their explained geometry identity and map path.
Both planners use `algo_config_path: null` and resolve to `{}`. The current
map-runner policy builders record the canonical empty-mapping runtime identity
`44136fa355b3678a` for both rows; no planner config file is part of this slice.
The producer and row validator reject any other planner config identity.
The planner-free oracle runs first. Its conservative grid search reports no
route at 2.2 and 2.8 m despite positive continuous clearance; the executed
policies do not use that route search. This remains an explicit diagnostic
finding, not a width effect or an ordinary planner failure. Readiness requires
positive continuous clearance, available execution, known geometry, and stable
runtime input identity. The preregistered unavailable `slow_speed_tier`
distributional metric may leave the completed H1 oracle rollout explicitly
fallback/degraded; it is diagnostic only and does not block H1
executability/binding readiness or become comparison evidence. Unknown geometry,
execution failure, or unstable binding keeps the preflight `go` field false.

## Diagnostic commands

```bash
uv run python scripts/validation/run_issue_9348_three_width_doorway_preflight.py \
  --out-json output/benchmarks/issue_9348_preflight.json \
  --variants-dir output/benchmarks/issue_9348_variants
uv run python scripts/validation/run_issue_9348_paired_reset_smoke.py \
  --out-json output/benchmarks/issue_9348_pair_smoke.json \
  --variants-dir output/benchmarks/issue_9348_pair_smoke_variants
```

The first command records geometry, oracle findings, asset SHA-256 digests,
and planner rows marked `not_run`. The second uses the real episode runner
for one step in each of the 18 cells. Its opt-in post-reset hook restores a
portable simulator snapshot and records initial actor, external random-stream,
non-width configuration, and map digests. It fails if any planner/seed pair
does not match across three distinct maps. One-step outcomes are diagnostic
only and cannot establish the 400-step comparison. Keep generated assets and
receipts in durable campaign storage before a confirmation run.

## H400 producer and retrieval check

The 0.0.8 private operations launch packet invokes the public producer below
from a clean worktree at its pinned source commit. The packet and the canonical
`submit_and_record.sh` queue entry own the Slurm resources, scheduler receipt,
remote result root, retrieval, and preservation. Set `DURABLE_CAMPAIGN_ROOT`
to the reviewed, fresh external result directory in that packet; do not use a
worktree-local `output/` directory for confirmation artifacts.

```bash
uv run python scripts/validation/run_issue_9348_three_width_campaign.py \
  --mode run --manifest configs/benchmarks/issue_9348_three_width_doorway_v1.yaml \
  --output-root "$DURABLE_CAMPAIGN_ROOT"
uv run python scripts/validation/run_issue_9348_three_width_campaign.py \
  --mode verify --output-root "$RETRIEVED_CAMPAIGN_ROOT"
```

The producer runs the geometry/oracle preflight, then a separate actor-present
H10 confirmation over all 18 planner/seed/width cells. H400 starts only if
every short probe follows its declared baseline command route, has a nonempty
simulation/action trace and valid typed spawn clearance, is fallback-free,
and is paired by reset and RNG receipts. The oracle's exact `slow_speed_tier`
finding stays diagnostic
and cannot clear this gate. A red confirmation writes a failure receipt and
stops before H400. The producer then runs the 18 H400 episodes serially with
one paired-reset session. It writes validated raw episode JSONL,
line and file SHA-256 digests, copied scientific inputs, generated maps and
scenarios, a six-pair receipt manifest, a report, and full-tree `SHA256SUMS`.
The verify mode rebuilds H10 admission from the preserved raw probes before
checking the H400 report, without modifying the bundle. A failed
run writes `run_failure.json` and returns nonzero; it is not confirmation
evidence. The private launcher must also preserve startup, producer exit,
scheduler, retrieval, and cold-readback receipts outside the hashed producer
tree.

The report gives each planned and observed row, outcome, failure and exclusion
reason, and trace location. Within each planner it compares adjacent widths
and 2.2 versus 3.6 m using complete seed pairs. The 95% percentile interval
resamples whole seeds (10,000 draws, seed 9348); three seeds are descriptive,
not a significance basis. Arrival time uses successful pairs only, with
failures censored at termination. Pedestrian delay or impairment is explicitly
unavailable without a matched no-robot control trace. Human review must tie
each mechanism claim to a recorded trace before promoting the slice. Only rows
with the expected `goal` native-command mode or documented Social Force adapter,
valid typed spawn clearance, nonempty simulation/action steps, and no runtime
fallback marker may enter the complete-pair comparison. These baseline planners
have empty specialized internal-decision arrays by design. The report lists
unavailable reset sampler/route and unrelated paired-effect telemetry as
ancillary coverage gaps. Unexpected adapters, fallback, degraded and incomplete
rows are excluded; no physical doorway or deployment safety claim follows from
this simulator-only comparison.
The report's legacy `native_rows` count means eligible baseline rows under
these arm-specific routes; each row also records its actual command execution
mode, so Social Force adapter execution is not presented as native commands.
