# #10000 pedestrian-stack gate receipt — FAILED

`schema_version`: `pedsweep-gate-receipt.v1`
`evidence_tier`: `diagnostic-only`
`paper_facing`: `false`
`benchmark_promotion`: `false`

No PR passed the gate. #10073, #10075 and #10094 remain **blocked**; #10104 **failed** refute review. All sweep attempts and failure classifications are complete; nine width MPPI config rejections prevent complete valid coverage, and contact projection bypasses an active pedestrian start-delay hold ([#10178](https://github.com/ll7/robot_sf_ll7/issues/10178)). Newly observed empty-world failures are tracked in [#10180](https://github.com/ll7/robot_sf_ll7/issues/10180). No success-rate threshold, model adoption or release admission is asserted. All four remain draft with `blocked` retained.

Landing order read from the pedstack lane: #10073 → #10075 → #10094 → #10104. The tested cumulative runtime is `52634e854076ad7a1da6be4edf17fa40baf93fd4`. This receipt is a docs-only annotation on bottom PR #10073, whose measured runtime source was `6b5d4a442f8ed7ffc9ca6c519be2b9cba76d9ca0`; it does not pretend that the later receipt commit was swept or is in the top head's ancestry.

## Sources, coverage and execution

Baseline: corrected benchmark release 0.0.7 source `07f7e8d43084de748915e1b1eb8b2a1603357c6e`, identified by the repository's release closeout and comparison script. The software tag 0.0.7 is a different commit. Baseline release configs and artifacts were read without modification. The baseline contains the 48-scenario main suite and no width release suite. Head contains main plus the three-width suite. Every configured arm/scenario/map slot was attempted (nine width MPPI slots were rejected before a trajectory), differential-drive only, dev seeds **1001, 1002, 1003**. This is bounded development evidence, not a 30-seed or sealed evaluation campaign.

Slurm **21472** (baseline) exited 0. **21471** supplied a complete, valid head main suite, then exited 1 because 117 width slots failed SNQI finalization (v2 metrics versus legacy v1 anchors) and nine MPPI slots failed the predictor horizon contract. **21473** repaired diagnostic SNQI references, wrote 117 width rows and exited 1 because those nine real config rejections remain. All 126 width slots were attempted; complete valid width coverage is **not** claimed. The original failed width attempt is retained separately and never counted as outcome evidence. Submitted via `sbatch` on the authorized login host after checking the queue, A30 CPU QoS, 24 CPUs/64 GiB each, reserved node excluded, at most two concurrent jobs. No sweep or tests ran on the login host. Initial attempts **21469/21470** exited 1 at dependency preflight with no episode rows: missing stable_baselines3. Those are **harness issues**, preserved rather than counted as outcomes. The verified repair was `uv sync --all-extras`.

The head main suite used its unchanged documented empty-world harness and normal campaign execution. The width-only repair used the same pinned production source and cleared only `snqi_weights`/`snqi_baseline` in the diagnostic derived payload ([width-harness.patch](width-harness.patch)); these are post-loop reporting references, not controller parameters. No anchor was modified/relabelled. The 117 original SNQI failures are classified as harness issues; the nine original and repeated MPPI rejections are real config defects in [harness-failures.json](harness-failures.json). [width-input-identity.json](width-input-identity.json) verifies that the repaired control/scenario payload is unchanged after normalizing materialization roots and removing those two reporting references. The baseline used a compatible copy of that harness, keeping old production byte-identical: retain its fixed H600 horizon; remove authored SVG pedestrian markers into diagnostic copies; ensure `map_id=None` selects those copies; check zero density/population and empty map census. [baseline-harness.patch](baseline-harness.patch) records these changes. [baseline-map-proof.json](baseline-map-proof.json) independently checks all 33 baseline map sources: the 19 diagnostic copies match the original XML exactly after removing only the declared pedestrian circles; walls, routes, spawn/goal markers and attributes remain intact. No planner, simulator, frozen artifact, release config/anchor or success parameter was edited.

| Source / suite | Rows | Success | Collision | Timeout | Runtime execution modes |
| --- | ---: | ---: | ---: | ---: | --- |
| top / main | 2016 | 1714 | 203 | 99 | {"adapter": 1584, "mixed": 144, "native": 288} |
| top / width | 117 | 61 | 43 | 13 | {"adapter": 90, "mixed": 9, "native": 18} |
| baseline / main | 2016 | 1207 | 587 | 222 | {"adapter": 1584, "mixed": 144, "native": 288} |

[runtime-audit.json](runtime-audit.json) checks every raw row using the current canonical runtime-marker scanner bound to the declared algorithm, including typed guarded-PPO telemetry handling. Both main grids are complete and unique. The width grid accounts for 117 written rows plus nine declared MPPI config rejections; their performance outcomes are unavailable. For every written row, source SHA agrees, reset/step actor lists are empty, geometry traces and available velocity/action values are finite and complete, and a scoped runtime view has no forbidden planner fallback/degraded/error markers. The unchanged strict scanner flags missing auxiliary paired-effect measurements; reset route/spawn decisions are also not retained. Those typed measurement/telemetry records alone are excluded from the second scan, with their exact omissions preserved. The strict release metadata audit **does not pass**. Metadata-only negative controls confirm that genuine planner fallback/unavailability remains rejected. These limitations are harness/telemetry issues and do not qualify for paper/release admission. Native, adapter and mixed modes are reported explicitly; mixed execution is not mislabeled native. Loaded PPO checkpoint records retain their null hashes; model-weight byte identity is not inferred from those records. Private checkpoint materialization prefixes are omitted from the public audit and retained in raw custody. The archive hashes, source/harness/lock hashes and environment digest are in [artifacts.json](artifacts.json), [source-hashes.json](source-hashes.json) and the runtime audit. Raw archives/logs and exact derived inputs remain in private lane custody; the report gives their locations. Public records contain no workstation paths or credentials.

## Comparison and classifications

There are **2016** paired main slots and **126** unpaired head width slots. Scenario IDs and four documented v3/v4 replacement arm bindings establish descriptive pairing only. Maps, planner configs/models, horizons and metric definitions differ across releases. Baseline rows have no explicit metric schema marker, while head rows declare v2; comparison uses terminal success and presence of contact descriptively, and does not pool distance/SNQI scores as equivalent measurements; [input-manifest.json](input-manifest.json) hashes those inputs. A baseline success does not prove that a specific pedestrian change caused a head failure.

**57 new failure slots**, 55 success→failure events and 30 new-collision events; event counts can overlap. Every slot is classified in [new-failures.csv](new-failures.csv), with complete baseline/head metrics and current trace witnesses in [comparison.json](comparison.json). Classification counts: `{"real_defect": 57}`; unclassified = **0**.

| Arm | Success→failure events | New collision events |
| --- | ---: | ---: |
| guarded_ppo | 4 | 0 |
| hybrid_rule_v4_fast_progress_static_escape | 7 | 0 |
| hybrid_rule_v4_fast_progress_static_escape_continuous | 10 | 0 |
| ppo | 12 | 13 |
| prediction_planner | 2 | 2 |
| sacadrl | 14 | 15 |
| scenario_adaptive_hybrid_orca_v2_bottleneck_yield_v4 | 3 | 0 |
| scenario_adaptive_hybrid_orca_v2_collision_guard_v4 | 3 | 0 |

New contacts are **real defects** in the executed empty-world safety outcome, never infeasibility exceptions. New contact-free `max_steps` failures outside the declared exception are **real defects / progress deficits under the current release contract**. The classification records the observed deficit; its planner/dynamics/config cause remains unresolved. Changed horizons are an explicit confound. These are not proven pedestrian regressions, and the receipt makes no parameter or safety-threshold repair. [prediction-new-collisions.json](prediction-new-collisions.json) supplies reset and command/geometry evidence for two prediction-planner collisions. The required next step in #10180 is causal isolation against fixed inputs, not success-targeted tuning or forcing motion through a safety veto.

The width suite has no 0.0.7 predecessor; its 65 failed slots (56 recorded physical/progress failures and nine MPPI config rejections) are separately classified in [width-failures.json](width-failures.json), without calling them paired regressions. The 4158 expected slots (4149 actual rows plus nine explicit failed-execution placeholders) are in [all-episodes.csv](all-episodes.csv).

## Checked infeasibility exception

[checked-exceptions.json](checked-exceptions.json) verifies the tracked `francis2023_narrow_doorway` declaration and its map SHA-256: 2.0 m opening, 2.0 m robot diameter, 2.2 m required width including margin. Only **timeout with zero robot-attributable contact** is excepted. Of 42 diagnostic slots, **30** qualify; **12** do not. Their outcomes are `{"collision": 12, "timeout": 30}`. Collisions, missing/error/degraded rows or successful traversal are not silently excused. The release declaration's denominator is 420; the diagnostic denominator is 42. No exception is invented for the three-width suite or other scenarios.

## Refute and safety intervention question

The cumulative adversarial review and source-identity checks are in [refute.md](refute.md) and [production-identity.json](production-identity.json). Reviewed #10094/#10104 production paths remain byte-identical after branch integration; all 104 planner/robot/ped_npc paths match merged main. The [prior review](https://github.com/ll7/robot_sf_ll7/pull/10104#issuecomment-5971040837) is disclosed with its source and limits; new material evidence overrides its positive verdict.

| PR / motion surface | Does this override a robot hard-stop/yield veto? | Additional intervention finding |
| --- | --- | --- |
| #10073 opt-in wall forces | No robot veto code override; default path retained. | Pedestrian acceleration changes when selected; no adoption proof. |
| #10075 radius / validation | No robot command override. | Explicit radius changes geometry; measurement is diagnostic. |
| #10094 valid-start construction | No in-episode veto override. | Refuses invalid starts rather than nudging stopped robots. |
| #10104 contact / wall response | No robot veto code override. | **Contact position projection bypasses an active pedestrian controller hold**, #10178. |

Independent production-binding witness: dev seed 1001, two initially separated pedestrians, 0.28 m radius, dt 0.1 s, first with a 10 s start delay. After binding and one step, 9.9 s remains and both held speed cap and velocity are zero. Contact off displacement = 0; `projection_v1` displacement = **0.05293751034406103 m**. [hold_probe.py](hold_probe.py) and [hold_probe.json](hold_probe.json) preserve it. No robot is instantiated. This proves a start-delay contract defect, not an observed robot safety-veto bypass. The fixed mask currently covers prescribed indices but not held bodies; later velocity capping does not undo positional correction. Do not add recovery motion around a veto to repair this.

## Pedestrian-specific pedstack probes and own tests

The empty-world gate **does not test pedestrian interaction**. [pedestrian-probes.json](pedestrian-probes.json) reports the pedstack lane's real-substrate development probes and original source/driver hashes. Production-byte identity from those intermediate probe heads to current PR heads was independently recomputed. Dev seeds 1001–1003, populations 1/20, dt 0.1 s, 1000 steps, 1.2 m door, desired speed 1.3 m/s:

- #10073, radius 0.40 m, legacy→gradient: lone passage 0/3→3/3, crowd overlap pair-steps 26,640→23,704, stalls 0→6, crossings 0/60→54/60. Six legacy trajectory digests match main and #10075/#10094.
- #10104 calibrated_v2, radius 0.28 m, contact/wall off→both: overlap 18,294→0, stalls 6→6, crossings 54/60→54/60, lone passage 3/3→3/3.
- #10104 legacy_v1, radius 0.35 m, off→both: overlap 31,618→0, stalls 0→1, crossings 0/60→39/60, lone passage 0/3→0/3. Retained coefficients do not resolve lone-pedestrian stand-off.

Reported own-test history, not rerun by this receipt lane:

| PR / measured head | Own tests | Gate passed? / disposition |
| --- | --- | --- |
| #10073 `6b5d4a44` | 91 focused; full 43,055 pass / 2 fail / 69 skip / 7 xfail; both failures later cleared by 16 focused, no second full. | **No; blocked.** Shared-stack negative receipt; individual pass not established. |
| #10075 `6613fd0a` | 112 focused; exact-head full 43,089 pass / 69 skip / 7 xfail, exit 0. | **No; blocked.** Own full passes, shared gate does not. |
| #10094 `5ca04037` | 62 focused plus scoped quality checks; full omitted while another full lane active. | **No; blocked.** Individual pass and fresh full not established. |
| #10104 `52634e85` | 174 focused plus scoped quality checks; full omitted while another full lane active. | **No; failed.** New #10178 contradicts hold preservation. |

Hosted draft CI skips are not a whole-suite pass. No blocked labels were removed, no drafts marked ready, no merges/main pushes/restacks were performed. Applicable domain disposition and real-campaign auditing ([#9952](https://github.com/ll7/robot_sf_ll7/issues/9952)) remain outside this diagnostic receipt.

## Reproduction and revival

Use fresh checkouts at the pinned runtime sources, the documented empty-world entry point, and `uv sync --all-extras`. Submit through Slurm with the recorded resources/exclusion and bounded queue, never execute on the login host. Original source/lock/harness digests and compatibility patch are retained. Exact raw custody paths are in the private lane report; `artifacts.json` provides archive basenames/hashes. Normalize extracted run directories to `top/` (21471, retaining its failed width attempt), `baseline/` (21472), and `width-repaired/` (21473) under a custody root. The reconciliation helpers select main from 21471 and width from 21473; they require both main grids complete and the width missing rows to match exactly the nine explicitly recorded MPPI contract rejections. They never manufacture or admit outcomes for those rejections. From the top runtime checkout, using the receipt helper files by their absolute paths:

```bash
PYTHONPATH=.:fast-pysf uv run python "$RECEIPT_DIR/audit_results.py" "$CUSTODY_ROOT" "$AUDIT_OUTPUT"
python3 "$RECEIPT_DIR/compare_sweeps.py" "$CUSTODY_ROOT" "$REBUILT_RECEIPT"
# Copy the declared checked-exceptions.json into REBUILT_RECEIPT before classification.
python3 "$RECEIPT_DIR/classify_findings.py" "$REBUILT_RECEIPT" "$CUSTODY_ROOT" 10180
PYTHONPATH=.:fast-pysf uv run python "$RECEIPT_DIR/hold_probe.py"
```

These reconciliation helpers read existing rows and never reset/step an environment; only the explicitly dev1001 hold witness simulates. The checked-in summary is the evidence snapshot, not a claim that later dependency resolution produces identical floating-point trajectories. Resolve/disposition #10178 and #10180 without success tuning, review and sweep any changed runtime head, then establish each PR's gate, own validation and domain requirements before readiness. Maintainer owns runtime repairs; the pedsweep lane owns this receipt and custody.
