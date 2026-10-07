# #10180: SACADRL and PPO causal contract dispositions

**Diagnostic-only; no benchmark promotion or gate clearance.** This annotation reconciles
existing committed development evidence. It adds no trajectories and does not reclassify any
contact as acceptable. Actual fallback/degraded execution remains ineligible; the original
strict auxiliary/reset metadata omissions remain. Historical checkpoint weight identity was
not independently reverified here. Confidence is limited to the recorded source-pair and
contract layers; the exact unsafe-motion cause within the retained planner is unresolved.

## Refreshed state and decision

The [original October 6 receipt](https://github.com/ll7/robot_sf_ll7/blob/a7fde2527844c0884214ea08d8d0f63a80ed7287/docs/context/evidence/issue_10000_pedsweep_20261006/README.md)
records 57 failed slots, with 55 success-loss and 30 new-contact labels. The
[committed r2 receipt](https://github.com/ll7/robot_sf_ll7/blob/e711f56c67d4f63fe8c044ada2208da0875f0e36/docs/context/evidence/issue_10000_pedsweep_20261006/r2/README.md)
assigns 75/85 labels to one mechanism class. The
[later INV10180 comment](https://github.com/ll7/robot_sf_ll7/issues/10180#issuecomment-6032651460)
reports the remaining ten as joint map/configuration × planner/arm causes, for 85/85
explained at those layers. That later claim and its raw-custody readback are **reported
evidence**, not newly verified raw evidence in this annotation. Its external
`inv10180/analysis/events/` records were not retrieved here. All four pedestrian-stack PRs
remain draft/blocked; #10180 remains open. The width binding now has its focused owner,
[#10186](https://github.com/ll7/robot_sf_ll7/issues/10186).

The [per-event data](r2/per-event.json) reconcile as follows. Labels overlap within slots.

| Scope | Slots | Success-loss labels | New-contact labels | Total labels |
| --- | ---: | ---: | ---: | ---: |
| All SACADRL failures | 15 | 14 | 15 | 29 |
| All PPO failures | 13 | 12 | 13 | 25 |
| Combined first slice | 28 | 26 | 28 | 54 |
| SACADRL adjacent reset-sampling pair | 5 | 5 | 5 | 10 |
| PPO adjacent action-binding pair | 4 | 3 | 4 | 7 |

**Disposition:** retain the corrected reset-sampling and declared action contracts. The
adjacent source changes causally create 17 retained labels on nine slots, but these witnesses
do not prove that either implementation is erroneous. The observed collisions stay failures;
their exact unsafe-motion mechanism still needs narrower proof. No implementation repair is
justified by these records, and reverting a correction to rescue outcomes would not establish
correctness. This is a contract-retention disposition, not a performance acceptance decision.

## Fixed-input witnesses

The [frozen runtime-pair plan](r2/runtime-pairs-plan.md) and
[manifest](r2/runtime-pair-manifest.json) bind each pair to baseline scenario/map and planner
profiles with the H600 contract, development seeds only. The
[executed input identities](r2/executed-input-identities.json) match scenario and map hashes
and seed lists across each selected pair. The
[compact extraction](learned-arm-witnesses-20261007.json) preserves the exact event IDs,
pre/post source and trajectory hashes, recorded reset poses, input identities, and separate
cumulative-top outcomes. It is derived evidence, not another campaign.

| Mechanism | Source issue | Evidence tier | Fixed configuration | Seeds | Artifacts | Outcome | Verdict | Limit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clearance-aware robot reset sampling | #10180; #9739/#9725 | diagnostic-only | Baseline scenario/map/profile, H600 | 1001 | r2 adjacent-pair records and compact extraction | Five successes become contacts | Retain clearance contract; retain failures | Complete realized goal/route/RNG equality is not recorded in the compact pair evidence |
| Checkpoint-bound PPO action interpretation | #10180; #9995 | diagnostic-only | Baseline scenario/map/profile, H600 | 1001, 1003 | Same, plus existing PPO contract note | Three successes and one timeout become contacts | Retain declared velocity_delta; retain failures | Missing exact canonical training pin; pair weights not newly reverified |

### SACADRL: retain clearance-aware reset sampling

The source pair is parent `9c3452face9f361d764a48f1053eee074c809204` →
[`b8dccb9c9f9ddc1f1cc3bb980ee260f851a727b2`](https://github.com/ll7/robot_sf_ll7/commit/b8dccb9c9f9ddc1f1cc3bb980ee260f851a727b2).
All five slots use dev1001:

| Slot | Scenario | Parent / H600 | Corrected sampling / H600 | Cumulative top / H400 |
| --- | --- | --- | --- | --- |
| S042 | francis2023_intersection_no_gesture | Success, step 234 | Contact, step 43 | Contact, step 42 |
| S044 | francis2023_intersection_proceed | Success, step 234 | Contact, step 43 | Contact, step 42 |
| S046 | francis2023_intersection_wait | Success, step 234 | Contact, step 43 | Contact, step 42 |
| S048 | francis2023_narrow_hallway | Success, step 234 | Contact, step 43 | Contact, step 42 |
| S050 | francis2023_perpendicular_traffic | Success, step 234 | Contact, step 43 | Contact, step 42 |

The recorded reset position changes from `[4.387535640902389, 13.04622710398411]` to
`[4.4698728687849885, 13.391563830167973]`; S048 has the corresponding y coordinates
11 m lower. Recorded velocity and heading remain zero. These closely related map variants
are five witnesses of one cluster, not five independent mechanism discoveries.

The correction explicitly enforces robot radius plus 0.1 m clearance from walls/bounds and
rejects a start when no valid candidate exists. Retain that contract. The causal variable is
the **clearance-aware reset-sampling change**, not proven displacement of the start alone:
sampling a goal follows start sampling, and extra rejected candidate batches can affect later
RNG draws. The compact pair records do not establish equality of complete realized
goal/route state. Neither changed nor unchanged goals are inferred here.

The native [SACADRL model path](https://github.com/ll7/robot_sf_ll7/blob/d90bf2436e24dea33198a8bbaf6bbf4342b0edac/robot_sf/planner/socnav_sacadrl.py)
has no static-map features in its network input. The
[existing adaptation note](https://github.com/ll7/robot_sf_ll7/blob/61cc91877a159e1d08b321543bd860042520b9ed/docs/context/issue_10007_fxs_sacadrl_sampling.md)
also documents the upstream-versus-release motion and robot-size limitations. These are
plausible limitations, not an individually proven cause of this cluster. Adding a heading
hold or recovery motion would require a separately specified method; this disposition adds
neither. The five physical contacts remain blocking failures under the existing gate policy.

### PPO: retain the declared velocity-delta interpretation

The source pair is parent `46809b6bc18caef2a9cd82686e9670c224459c42` →
[`c9102a9c81bb60cebb760b2bac94af695bc087a4`](https://github.com/ll7/robot_sf_ll7/commit/c9102a9c81bb60cebb760b2bac94af695bc087a4).
Recorded initial robot pose and physical velocity are identical within each pair.

| Slot | Scenario | Seed | Parent / H600 | Delta binding / H600 | Cumulative top |
| --- | --- | ---: | --- | --- | --- |
| S025 | classic_station_platform_medium | 1003 | Contact-free timeout, step 600 | Contact, step 119 | Contact, step 230 / H650 |
| S026 | francis2023_blind_corner | 1001 | Success, step 220 | Contact, step 83 | Contact, step 100 / H400 |
| S027 | francis2023_blind_corner | 1003 | Success, step 225 | Contact, step 116 | Contact, step 103 / H400 |
| S031 | francis2023_exiting_room | 1003 | Success, step 181 | Contact, step 48 | Contact, step 37 / H400 |

S025 contributes only a new-contact label. The independent spawn-clearance source pair
creates none of these seven PPO labels; those controls remain in the per-event data.

At inspected main `61cc91877a159e1d08b321543bd860042520b9ed`, the
[decoder](https://github.com/ll7/robot_sf_ll7/blob/61cc91877a159e1d08b321543bd860042520b9ed/robot_sf/baselines/ppo.py)
adds signed raw output to physical `(v, omega)` before target clipping, as required by the
declared `ppo-target-velocity.v2` contract. The
[contract note](https://github.com/ll7/robot_sf_ll7/blob/61cc91877a159e1d08b321543bd860042520b9ed/docs/context/ppo_checkpoint_action_semantics.md)
documents that historical training allowed 3.0 m/s, reverse and instantaneous unscaled deltas,
while the shared release plant retains 2.0 m/s, no reverse and acceleration limits. It also
states that the canonical PPO checkpoint's exact training commit remains null. Declared
semantics are not a fully recovered historical training provenance chain.

The existing [#9995 implementation proof](https://github.com/ll7/robot_sf_ll7/pull/9995)
reports real-checkpoint, matched-input checks against an independent delta/actuation oracle,
and explicitly reports worsened development outcomes. That proof was inspected, not rerun.
The current witness does not refute the declared decoder contract. Retain velocity_delta;
historical absolute-action controls remain diagnostic. The training/plant mismatch and
policy limitations have not been separately isolated as the cause of each contact, so neither
is treated as a safety exemption or a reason to tune release parameters.

## Horizons, remaining scope and next discriminating witness

All 28 selected-arm cumulative-top failures are contacts before their declared horizon. The
nine source-pair post results also contact before H600. These are not contact-free horizon
deficits; extending a horizon cannot erase the observed contact. Pair and cumulative-top
trajectories are separate measurements and have different terminal steps.

The other 37 SACADRL/PPO labels have scenario/profile bundle attribution or the later reported
joint attribution. They are not assigned to the two exact source pairs above. In particular,
S041's scenario-bundle contrast removes contact but does not restore success. The later room
H400/H600 results concern other arms; the reported runner-only H600 attempt retained a simulator
H400 cap and remains a negative contract control, not an extended-horizon performance result.

**Next step: inspect the preserved S042/dev1001 raw reset before scheduling compute.** Retrieve
the adjacent-pair `episodes.json` using the [driver](r2/replay_runtime_pairs.py) and
[custody manifest](r2/artifact-provenance.json). Check whether its existing reset/first-step
records already establish goal/route identity. Verify model bytes, map, configuration and
environment; retain the original failing record. If a new command/geometry proof is needed,
Freeze the complete realized robot pose, velocity, goal/route state and RNG state, then use the
recorded post-clearance reset for a bounded command/geometry witness through first contact.
Compare raw model action, requested heading/velocity, applied motion, active route target and
contact geometry against the declared adapter/plant contract. A numerical contract violation
justifies a minimal repair with a focused failing-before/passing-after assertion. If every
transition conforms, record a narrowly evidenced planner limitation while retaining the failed
outcome. Do not substitute a successful old start or change acceptance policy.

The corresponding PPO continuation is S026/dev1001 with verified checkpoint bytes, full reset,
fixed model observation/raw output and an independent delta/actuation oracle. These are next
proof boundaries, not claims that new probes ran. No new full gate is justified by this
documentation-only disposition. A subsequent causal runtime change requires the existing
focused proof, exact-head refute and applicable #10000 gate.

Only the existing checked 2.0 m doorway **contact-free timeout** exception remains. No contact,
invalid/missing/degraded execution or unlisted scenario is excused. Release configurations,
safety thresholds, success metrics, acceptance bands, frozen artifacts and robot vetoes are
unchanged. The original raw archives, failed attempts and external development custody remain
the owners of replay evidence.

## Reconciliation and provenance

The compact JSON is a projection of existing records, with SHA256 bindings to its source files.
The following read-only check, run from the repository root with Python 3, verifies those
bindings, the selected event counts, copied pair records and matching executed input identities.
It performs no reset, step, model download or release check. This proves documentary consistency,
not runtime regression correctness or raw archive custody.

```python
import hashlib
import json
from collections import Counter
from pathlib import Path

root = Path("docs/context/evidence/issue_10000_pedsweep_20261006")
packet = json.loads((root / "learned-arm-witnesses-20261007.json").read_text())
for path, digest in packet["source_sha256"].items():
    assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
events = json.loads((root / "r2/per-event.json").read_text())
identities = json.loads((root / "r2/executed-input-identities.json").read_text())
expected_counts = {"sacadrl": (15, 14, 15), "ppo": (13, 12, 13)}
for arm, counts in expected_counts.items():
    rows = [e for e in events if e["arm"] == arm]
    labels = Counter(e["event"] for e in rows)
    assert (len({e["slot_id"] for e in rows}), labels["success_to_failure"],
            labels["new_collision"]) == counts
selected = []
for witness in packet["witnesses"]:
    rows = [e for e in events if e["slot_id"] == witness["slot_id"]]
    assert sorted(e["event_id"] for e in rows) == witness["event_ids"]
    event = rows[0]
    for key in ("arm", "scenario", "seed", "mechanism_class"):
        assert witness[key] == event[key]
    pair = witness["runtime_pair"]
    assert all(pair in e["runtime_pair_tests"] for e in rows)
    assert pair["creates_event"] and pair["mechanism_active"]
    assert pair["post"]["termination_reason"] == "collision"
    assert pair["post"]["steps"] < 600 and pair["pre"]["collision_count"] == 0
    for phase in ("pre", "post"):
        found = [x for x in identities if x["mode"] == pair["pair"] + "_" + phase
                 and x["arm"] == event["arm"] and x["scenario"] == event["scenario"]]
        assert len(found) == 1
        assert {k: v for k, v in found[0].items() if k != "mode"} == witness["input_identity"]
        assert event["seed"] in found[0]["seeds"]
    assert witness["cumulative_top"] == event["outcomes"]["contact_fixed_10104"]
    selected.extend(witness["event_ids"])
assert len(packet["witnesses"]) == 9 and len(selected) == len(set(selected)) == 17
assert Counter(w["arm"] for w in packet["witnesses"]) == {"sacadrl": 5, "ppo": 4}
assert packet["simulator_runs"] == 0 and packet["full_gate_runs"] == 0
print("PASS: 9 witnesses, 17 labels; source hashes, input identities and pair outcomes agree")
```

The source-pair input archive's **recorded** SHA256 is
`f18d9c64f8ad36f0445bee2588a72c3ce04297c29e88d497f68289586e5d64f5`.
The [input hash receipt](r2/runtime-input-hashes.json) includes the campaign, scenarios, maps,
profiles and manifest. It does not recover missing per-load checkpoint hashes. The
[runtime audit](r2/runtime-audit.json) reports finite complete zero-actor diagnostic traces
and no actual runtime fallback/error markers, with strict release admission still false.

Delivery is an additive documentation branch based on receipt PR #10073 at
`75787aaf591344ee59ec0e9c55de56ff38b486b6`. Runtime inspection uses the separately stated
fresh-main SHA; neither the annotation commit nor the receipt branch is represented as a new
tested runtime. Historical observation tables, all r2 data and checked exception bytes stay
unchanged. The root README gains an annotation link; its checksum and the two new artifact
checksums are recorded in the updated root `SHA256SUMS`. The prior README and checksum manifest
remain retrievable at the receipt parent commit. Downstream propagation is limited to this
receipt and #10180; no paper, release, model registry or readiness surface is promoted.
