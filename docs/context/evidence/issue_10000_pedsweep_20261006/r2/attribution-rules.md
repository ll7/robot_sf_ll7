Diagnostic-only classification rules frozen before replay results.

Universe: 85 event labels on 57 distinct slots, 55 success_to_failure and 30 new_collision. Each event label is counted once; overlapping labels share a trajectory but are not collapsed.

Reproduction prerequisite: baseline_reproduction and contact_fixed_10104 must reproduce the relevant original baseline and head event, respectively. Exact source/input/seed identity and complete zero-actor trajectories required; actual runtime fallback/degraded/errors cannot become success evidence.

Class a: a bottom-up pedestrian layer alone creates/removes the relevant event at identical scenario/map/planner inputs, with a matching activated mechanism. Merely adding pedestrian config metadata does not establish causality.
Class b: changing only baseline scenario/map/horizon inputs removes the event, while changing only planner profile does not; physical or terminal witness must support the changed declared contract. Cite source commit/PR and exact changed map/config surface.
Class c: changing only baseline planner/arm profile removes the event, while changing only scenario inputs does not; cite the selected model/profile/arm replacement and its source commit/PR. Code-only planner/adapter attribution additionally requires an isolated executable counterfactual, not a list of intervening commits.
Class d: trajectory and physical/terminal observations unchanged, event caused solely by a schema/label definition; compare metrics to simulator termination and physical collision flags. SNQI failure is harness reporting and cannot establish a physical-event metric explanation.
Unexplained: failed reproduction, missing/degraded replay, jointly changed factors, both single-axis interventions sufficient, neither sufficient, or unsupported runtime attribution. Do not break causal ties arbitrarily. Within-class source bundles can identify a mechanism class without falsely identifying one line as its unique cause.

All six pedestrian heads receive the same top frozen scenario/map/planner payload. base_main 2d6b19db differs from 66df3de19 only in documentation; full robot_sf/fast-pysf diff verified empty, so runtime comparison is equivalent to the pre-stack main despite its later docs commit.
No full sweep, no parameter fitting, no thresholds/safety changes, no protected config/map/anchor writes. Baseline inputs are immutable copied diagnostic fixtures; top/base profile switch is a predeclared causal intervention, never success tuning. Three Slurm crossovers and a baseline-source reproduction each cover the same 57 event slots.
