# Refute review for the pedestrian stack

Target: 52634e854076ad7a1da6be4edf17fa40baf93fd4, containing #10073 → #10075 → #10094 → #10104. Diagnostic-only; no model/default or empirical adoption.

Verdict: FAIL for #10104's preservation of controller holds (#10178). Empty-world execution and row classification are reported separately. A prior positive review cannot erase new material evidence.

| Claim | Attempt to refute / evidence | Disposition |
| --- | --- | --- |
| Default profile/contact/radius selection preserves legacy behavior | Inspect constructor InitVar order, constructor/copy/replace/serialization override paths, loaders and substrate builder. Pedstack records 91/112/62/174 focused passes, six exact legacy dev-trajectory digests, and full-suite history with failures and omitted runs explicit. New selectors are absent from release inputs. | Existing default-path proof; no adoption inference. |
| Profiles are empirical calibrations | Profile source marks calibrated_v2 rejected and gradient_v3 dev-only; doorway/crowd probes report penetrations, overlap and stalls separately. | Refuted as an adoption claim; dev-only selection is the bounded contract. |
| Validation/censoring establishes accepted model or useful fitted ranking | Check corrected acceptance/validation files and prior adversarial #10104 review: V3 cannot mark a censored passage PASS; 4/7 shipped passes invariant over settings, law-sensitive subscore 3/11; unresolved source-equivalence and initial-position contracts remain explicit. | No model acceptance; numerical fits stay diagnostics. |
| The initial-start protocol changes release settings | Five #10094 production files identical to reviewed e6d52b55; the production start guard refuses inadmissible starts before a trajectory. | Byte identity recomputed at current exact head. |
| Contact/wall response leaves inactive paths and robot safety decisions unchanged | Six #10104 reviewed production paths identical to b105a8d3. All 104 planner/robot/ped_npc paths are identical to merged main 2d6b19db. Source adds contact only when selected and rejects unsupported HSFM integrators. | No code override of the robot hard-stop/yield veto. No claim that pedestrian interaction is safe. |
| Position/velocity projection honors a held actor | Independent dev1001 witness uses actual SinglePedestrianBehavior.bind_pysf_peds + step, 10s start delay. With 9.9s still remaining, cap=0 and velocity=0, projection_v1 causes 0.05293751034406103m displacement; off stays fixed. | Real defect, #10178. Fixed mask includes prescribed indices but not active holds; velocity capping occurs after position correction. |
| Zero pair overlap guarantees robot progress | Prior b105a8d3 adversarial review re-derives a wall-contact resting pedestrian that makes all 38 DWA candidates unsafe; robot safety stop holds and the hybrid deadlocks. | Known opt-in adoption risk; do not add an escape motion around the veto. |
| Legacy far field is preserved everywhere | Prior far-wall witness and pedstack bound show nearest-wall weight can suppress the aggregate far field. The per-segment bounds are disclosed; no total geometry-independent or near-field bound exists. | Disclosed law limitation; not a repaired/calibrated model. |

Safety question for motion additions:
- #10073 wall-force profiles: no robot hard-stop/yield override; opt-in force changes alter pedestrian acceleration. Retained default path is unchanged.
- #10075 shared radius/validation: no robot command override; explicit radius changes pedestrian placement/force geometry/metric interpretation, and measurement runners are diagnostic.
- #10094 valid-start construction: no in-episode safety override; refuses invalid initial states rather than nudging a stopped robot into motion.
- #10104 contact/wall response: no robot hard-stop/yield code override. **Yes, a pedestrian controller hold is bypassed through position correction in the witnessed contact case.** Velocity remains zero and the hold timer remains active. This is not a safe-stop preservation claim.

The sweep contains no pedestrian interaction. The fresh production-hold witness and reported pedstack probes supplement it but do not replace campaign-row interaction auditing (#9952). No parameters, safety thresholds, frozen artifacts or release configs were changed to obtain these outcomes.
