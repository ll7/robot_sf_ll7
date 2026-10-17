# Measured robustness verdict

Diagnostic-only; thirty development seeds and ten fixed representative cells.
These conclusions assess success and episode collision incidence within the
tested simulator. They do not admit manuscript wording or establish historical
release parity, full-family generalization, SNQI robustness or physical realism.

| Conclusion | Verdict | Measured support / limit |
|---|---|---|
| A strict, configuration-independent five-planner ranking | Depends on design settings | Success tau-b ranges from 0.105 to 1.000 across the nine radius/speed settings. At radius 0.3 m and preferred speed 1.0 m/s, goal moves from 140/300 to 300/300 successes and ties the hybrid, while PPO remains at 244/300. |
| Static hybrid v4 stays competitive in aggregate success | Robust within this sample | It belongs to the leading set in eight of nine settings; in the ninth it scores 299/300 versus goal's 300/300. This does not establish a statistically distinct unique winner. |
| The hybrid uniquely dominates goal | Depends on the large disc / speed setting | At radius 1.0 m, hybrid exceeds goal at all three pedestrian speeds. At 0.3 m and speeds 1.0/1.3, both achieve 300/300; at 0.5 m and speed 1.3, goal achieves 300/300 versus hybrid 299/300. |
| Absolute performance is insensitive to pedestrian speed | Rejected in this sample | At radius 1.0 m, raising preferred speed from 0.65 to 1.3 lowers hybrid success from 94.3% to 74.0%, ORCA from 92.7% to 70.7%, and goal from 46.7% to 36.7%. The full success order is more stable here than its absolute rates. |
| Robot crowding is relatively hard | Robust within the ten selected cells | It has the lowest pooled five-planner success in all nine settings. Its absolute success still ranges from 6.0% to 88.7%, so relative hardness does not imply persistent failure. |
| Bottleneck/doorway/navigation difficulty is an intrinsic family property | Depends on design settings | Bottleneck pooled success ranges 34.7%–100.0%, doorway 10.0%–98.0%, and crowd navigation 24.0%–95.3%. Their hardness ranks vary. Family success-order tau-b ranges 0.483–1.000. |
| Exiting room and circular crossing are comparatively easy | Robust within the selected cells | Pooled success stays 99.3%–100.0% and 96.0%–100.0%, respectively, across all nine settings. Near-ceiling ties limit a strict difficulty ordering. |
| Narrow doorway demonstrates universal planner difficulty | Geometrically confounded by the 1.0 m disc | Its measured 2.0 m gap gives a 2.0 m diameter zero clearance, and cannot meet the release's 0.1 m per-side margin. Radii 0.5/0.3 permit positive clearance. This family is outside Figure 7.3 and was not part of the ten-cell rank sweep. |
| The full-roster success winner is independent of command ceilings | Depends on caps | Three hybrids tie at 283/300 in the control and fall to 273/300 after the swap; sampling moves from 274/300 to 279/300 and becomes the observed leader. Six episodes separate sampling from the tied hybrids; this does not establish a statistically unique winner. |
| The broad fourteen-planner order survives the cap swap | Largely retained, with consequential local changes | Success tau-b is 0.834 (95% paired-seed CI 0.622–0.899); collision tau-b is 0.739 (0.548–0.921). The same six planners occupy the success-leading six positions, while guarded PPO drops from rank 8 to 11 and sampling moves from 5 to 1. |
| Family difficulty is insensitive to this cap swap | Relatively robust in this check | Pooled fourteen-planner family success-order tau-b is 0.956. This supports more stability under the requested cap swap than under the radius/speed grid; it does not establish stability under every fully matched cap regime. |
| The crowd model reproduces known phenomena sufficiently for realism claims | Not established | The release pair kernel gives mainly weak transient lane sorting, zero counterflow crossings through the 1.2 m door, and 0.268 people/s one-way finite-population throughput. Sampled wall overlap remains. There is no measured-reference calibration here. |

The matched uniform radius 1.0 / preferred-speed 0.65 control reproduces all
1,500 success/collision outcomes and episode step counts of the five-planner
legacy control. Consequently the large ranking changes above are not explained
by an unmatched legacy speed assignment. Radius treatments retain planner
thresholds and model weights, while radius-aware placement and goal sampling may
change. This measures robustness to a design change; it does not isolate a
single dynamical mechanism or demonstrate the result of retuning every planner.

Both cap arms are complete, with 300 matched episodes per planner. Pooled
success hardly changes (3,065/4,200 to 3,059/4,200), masking individual changes:
predictive MPPI gains 24 successes, risk-DWA gains 22, guarded PPO loses 25,
and goal loses 17. The requested swap retains two ceilings and conditional
lower caps; it cannot establish a fully equal-cap comparison. Radius/speed
treatments retain the original ceilings, so their large ordering changes are a
separate design dependence rather than an effect of the cap swap.

The author can retain a qualified claim that the static hybrid is competitive,
and that robot crowding is relatively hard while the selected exiting-room and
circular-crossing cells are comparatively easy. A unique universal winner,
configuration-independent strict ordering, universal bottleneck/doorway
hardness, or empirically validated crowd realism is not supported by this
diagnostic. Thesis wording and admission remain author decisions.
