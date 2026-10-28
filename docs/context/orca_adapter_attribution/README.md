# ORCA adapter attribution: where ORCA's request becomes zero

**Diagnostic only, development seeds 1161–1190, new current-version execution.**
This is a new execution of the 0.0.8 release code (`373dbfde4f39667cf9e8732dabe7df5118bdeab1`,
package files unchanged) on three matched-route doorway cells. It is not release
evidence and admits nothing to the thesis. 450 episodes; every row is `native`;
fallback 0 and degraded 0 are counted, not assumed.

**Question.** At 2.40 and 2.60 m, ORCA on the differential-drive robot completes no episode
although the opening is physically feasible. Is the zero request produced by the RVO2
solver, by the adapter's heading-safe projection, or between the executed command and
the measured motion? And does native holonomic execution change the outcome?

## Result

| Configuration | 2.40 m | 2.60 m | 2.80 m |
| --- | ---: | ---: | ---: |
| ORCA, differential drive | 0/30 | 0/30 | 27/30 |
| ORCA, native holonomic | 0/30 | 0/30 | 0/30 |
| Social force, differential drive | 0/30 | 0/30 | 0/30 (30 pedestrian contacts) |
| Social force, native holonomic | 0/30 | 0/30 | 0/30 (30 pedestrian contacts) |
| Hybrid v4 (continuous), differential drive | 0/30 | 24/30 | 24/30 |

* **The immediate RVO2 output is below the stall threshold (rule R3 holds).** In all 60/60
  failed ORCA differential-drive episodes at 2.40/2.60 m, all 6,000 terminal-window steps
  have solver speed below 0.05 m/s; 5,911 are exactly zero, and all 60 final steps are exactly
  zero. Preferred velocity is generally goalward, at about 3 m/s. A1 = 1.00, seed-block 95 %
  interval [1.00, 1.00]; 0/60 failures are dominated by projection suppression (A2). This is
  the immediate output conditional on the executed adapter's inputs, not proof that RVO2 or
  obstacle inflation intrinsically caused the historical doorway failure. Projection does
  transiently suppress goalward requests (8 steps in 4 episodes at 2.40 m; 118 steps in 17
  episodes at 2.60 m); its historical role in reaching the stop is unresolved.
* **The request fades at the mouth of the opening.** The final below-threshold run starts with the robot
  centre at x ≈ 14.2 m (2.40 m) and 14.4 m (2.60 m). The 1.0 m disc then reaches 0.2–0.4 m into
  the 1 m deep opening (wall x = 15–16 m), with no pedestrian nearer than 3.3 m. The median
  solver speed in the 5 s before that is 0.04 m/s (exploratory).
* **The inflated obstacle gap is descriptive.** The free gap at the
  wall between the inflated obstacle squares the adapter passes to RVO2 has median 1.61 m
  (2.40 m cell) and 1.69 m (2.60 m cell) over all steps, against a 2.0 m robot diameter. This
  gap is not a sharp threshold: successful 2.80 m episodes also saw gaps of 1.67–1.95 m.
  No exact-geometry intervention was run, so the gap cannot establish a cause.
* **Native holonomic execution does not remove the deficit and is worse at 2.80 m.** The paired
  success difference holonomic minus differential drive is 0 at 2.40 and 2.60 m (both arms 0/30,
  a shared floor) and −0.90 at 2.80 m (95 % interval [−1.00, −0.80]); holonomic ORCA also ends
  with a below-threshold solver request there (30/30), with median terminal gap 1.83 m against 2.00 m on
  differential drive.

Rules (predeclared in [`contract.yaml`](contract.yaml)): R1 false, R2 false, R3 **true**,
R4 **true** (2.80 m only), R5 true only as a shared 0/30 floor, so it bounds no effect
beyond these cells. Stage 2 (head-on, bottleneck, crossing) was **not run**: its trigger
required neither R2 nor R3 to hold.

## What the design can and cannot separate

* The step trace separates below-threshold solver output from suppression of a nonzero request
  inside this executed differential-drive stack. It does not establish the mechanism of the
  earlier episodes, whose original matched-route packet was unavailable.
* Native holonomic execution changes motion authority (instant velocity, no heading,
  acceleration or turn-rate limit) as well as removing the projection. It still uses the same
  grid-based obstacle inflation of the ORCA adapter. It therefore does not isolate a pure
  projection effect, does not test "ORCA without the adapter", and is not an algorithm ranking.
* The heading still enters the solver input through the corner-inflation rule and the robot's
  current velocity. The trace measures the immediate output given those inputs; earlier
  projected commands may have helped shape them. It does not establish an intrinsic cause.
* The cells reconstruct the matched-route width series (base doorway map, route `[[7,5],[25.5,5]]`,
  gap widened symmetrically) because the original packet was not available. Pilot seeds
  1001–1010 qualified the reconstruction (ORCA differential drive 0/10, 0/10, 10/10; terminal x
  14.20/14.50 m). These are new development-seed runs on reconstructed geometry, not the
  original matched-route episodes. Their 0/30 and 0/30 counts agree with the earlier report;
  at 2.80 m they yield 27/30 against the reported 25/30 on other seeds.

## Packet

* [`contract.yaml`](contract.yaml): frozen preregistration and execution commit
  `a4a31e3575a130bbc0dfccec7fbc173ff114fa42`, before any main episode. Same commit for all
  arms. A later analysis change added a `--steps` filename option. After independent review,
  extraction was repaired as described below; the contract itself remains unchanged.
* [`run_xorca.py`](run_xorca.py): harness that wraps planner and step-loop functions at runtime and
  adds an `xorca` record (preferred velocity, raw solver velocity, occupancy penalty, adapter
  command, obstacle gap, pre/post pose) to each step of the runner's own
  `simulation_step_trace`, which already holds the projected command and environment action.
  [`provenance/neutrality.json`](provenance/neutrality.json): with and without the wrappers, 15 of 15
  episodes are identical step by step.
* [`acquire.py`](acquire.py), [`extract.py`](extract.py), [`compact_steps.py`](compact_steps.py),
  [`analyze.py`](analyze.py), [`compare_reconstruction.py`](compare_reconstruction.py) and
  [`explore.py`](explore.py) (exploratory, outside every rule).
* `evidence/`: `episodes.csv` (one row per episode with `mode`), `orca_steps.csv.gz` (ORCA
  step table, 6 decimals), `cell_table.csv`, `paired_effects.csv`, `summary.json`,
  `exploratory.json`.
* `provenance/`: environment identity, job receipts, pilot table, custody (W&B artifact
  `wandb://ll7/robot_sf/xorca-orca-adapter-attribution:v0` with raw episode JSONL and
  full step traces, raw-bundle checksums).

## Post-registration extraction repair

The original extractor chose a dominant terminal stall from a set, so a tie could depend on
Python's hash seed. The repaired extractor excludes `moving` and `missing`, then chooses the
stall class with the largest count; ties use the lexicographically smallest class label. If no
stall is present, the label is `none`. This is a post-registration analysis repair; no raw data,
step classifications, acquisition, or frozen contract changed. A constructed 98-moving-step
tie fails against the former extractor and passes under different hash seeds with the repair.

Re-extraction of all 450 archived main episodes changed exactly these episode rows:

| Configuration | Cell | Seed | Tied counts | Former dominant | Repaired dominant |
| --- | --- | ---: | --- | --- | --- |
| hybrid_v4c_dd | 2.80 m | 1172 | `plant_zero` 1, `command_zero` 1 | `plant_zero` | `command_zero` |
| sf_dd | 2.40 m | 1180 | `solver_zero` 9, `projection_zero` 9 | `solver_zero` | `projection_zero` |

The corresponding failed-episode cell counts changed: hybrid 2.80 m `plant_zero` 1→0 and
`command_zero` 0→1; social force 2.40 m `solver_zero` 30→29 and `projection_zero` 0→1.
All other episode and cell rows, paired effects, exploratory values and preregistered rule
outcomes are unchanged. The compact step table's uncompressed content is unchanged. The summary
adds the interpretation boundary above; the full accounting is recorded in
[`provenance/analysis_repair.json`](provenance/analysis_repair.json).

## Reproduction

The analysis needs only Python and NumPy/PyYAML. Run it from the repository root and write to a
directory outside the packet:

```bash
P=docs/context/orca_adapter_attribution
uv run python $P/analyze.py --contract $P/contract.yaml --evidence $P/evidence \
  --steps orca_steps.csv.gz --output /tmp/xorca-check
uv run python $P/explore.py --evidence $P/evidence --steps orca_steps.csv.gz \
  --output /tmp/xorca-check
for f in summary.json cell_table.csv paired_effects.csv; do cmp $P/evidence/$f /tmp/xorca-check/$f; done
cmp $P/evidence/exploratory.json /tmp/xorca-check/exploratory.json
uv run python $P/test_packet.py
(cd $P && sha256sum -c SHA256SUMS)
```

Full acquisition (about 40 min of single-core episode time; about 6 min on 14 workers), from a
checkout of the **original execution commit**, with the recorded Python, uv and lockfile in
[`provenance/environment.json`](provenance/environment.json). This command is for future
reconstruction; the review repair did not run it:

```bash
git worktree add /tmp/xorca-source a4a31e3575a130bbc0dfccec7fbc173ff114fa42
cd /tmp/xorca-source
uv sync --frozen
P=docs/context/orca_adapter_attribution
uv run python $P/acquire.py --contract $P/contract.yaml --stage main --out-dir <run_dir> --workers 14
```

### Reconstruction comparison contract

Re-extract both the archived raw main files and the reconstructed raw files using the **same
repaired** `extract.py` from this packet. Compare by `(config, cell, seed)`; the rerun may be a
nonempty subset of the archive, but duplicate, missing or extra keys and incomplete step coverage
are failures. With matching scientific inputs and a deterministic environment, the following
must match exactly, without a numerical tolerance:

* Every extracted episode field except `git_hash` and `source_file`.
* Every column of every full-precision extracted step row, keyed by episode and `step`, and
  the entire raw `algorithm_metadata.simulation_step_trace` JSON value, including reset and all
  instrumentation. Use `steps.csv.gz` from `extract.py`, not the rounded ORCA-only packet table.
* Raw `algo`, `scenario_id`, `seed`, `status`, `termination_reason`, `outcome`, `metrics`,
  `metric_values`, `interaction_exposure`, `safety_predicates`, `spawn_validity`, `steps` and
  `horizon` values.

Source identity is checked separately, not silently ignored: every raw row, its sidecar
`run.repo_commit`, and its extracted `git_hash` must match the declared commit for that run;
the raw `provenance.git_hash`, `result_provenance.repo_commit`, and
`event_ledger.software_commit` must agree too. Each extracted `source_file` must name its actual
raw file. The archived reference must use
`a4a31e3575a130bbc0dfccec7fbc173ff114fa42`. Supply the expected rerun commit explicitly; a
different verified commit is reported as different source identity even when scientific fields
match, and does **not** establish an exact-source rerun.

Raw timestamps, elapsed times, run IDs, paths, filenames, path-dependent configuration hashes,
and other provenance containers can legitimately differ. The comparator counts differences in
these fields separately; it does not assert raw JSON or episode-row byte equality. It also does
not compare the path-bearing `scenario_params`, `selected_map_identity`, `event_ledger`, or
`failure_mechanism` objects as scientific equality; their observed differences are reported.
Inspect provenance and input identities separately before interpreting a match as a reconstruction.

From this repaired packet checkout, after restoring the archive and obtaining a rerun by the
separate acquisition procedure above:

```bash
P=docs/context/orca_adapter_attribution
uv run python $P/extract.py --raw-dir <archive_main_dir> --stage main --contract $P/contract.yaml --out-dir <archive_extract>
uv run python $P/extract.py --raw-dir <run_dir> --stage main --contract $P/contract.yaml --out-dir <rerun_extract>
uv run python $P/compare_reconstruction.py \
  --reference-raw <archive_main_dir> --reference-extract <archive_extract> \
  --rerun-raw <run_dir> --rerun-extract <rerun_extract> \
  --rerun-commit a4a31e3575a130bbc0dfccec7fbc173ff114fa42
```

To rebuild the compact analysis from the full archived extraction, run `compact_steps.py
--steps <archive_extract>/steps.csv.gz --output <derived_dir>/orca_steps.csv.gz`, copy the
re-extracted `episodes.csv` into `<derived_dir>`, and run `analyze.py` and `explore.py` as in
the packet-only commands above with `--evidence <derived_dir>`. Compare the resulting scientific
tables before regenerating `SHA256SUMS`.
