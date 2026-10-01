<!-- AI-GENERATED (#10074) - NEEDS-REVIEW -->
Recommendation: retain the released 0.40 m default while this validation remains blocked. Do not adopt a radius from these diagnostic flow numbers. If a collision-safe setup reproducing V3 density is pursued next, prioritize 0.25 m as a development candidate, not an accepted winner: only it meets the geometric nonoverlap density bound among this grid. Radius, wall law and body/placement fidelity need joint validation before choosing a production value.

| radius m | ideal nonoverlap maximum /m² | V3 initial overlaps at b=1.0 (mean±SD, n=30) | full five-width V4 banks /30 | V8 centre / edge m (mean) |
|---|---|---|---|---|
| 0.25 | 4.618802 | 54 ± 0 | 0 | 1.435953 / 1.185953 |
| 0.30 | 3.207501 | 54 ± 0 | 0 | 1.451685 / 1.151685 |
| 0.35 | 2.356532 | 104 ± 0 | 29 | 1.468265 / 1.118265 |
| 0.40 | 1.804220 | 199.9 ± 1.97135 | 30 | 1.485713 / 1.085713 |

Every radius has 0/210 aperture passages and 0/150 complete 60-person narrow runs. Native V1 speed stays 0.65 m/s. Smaller discs reduce the nominal overlap count and would permit narrower body geometry, but legacy wall standoff still prevents aperture passage. The archived V3 starting grid itself remains overlapping even at 0.25 m (54 overlapping pairs); geometric feasibility is not actual collision-safe placement. At wide width 2.4 m, 0.25/0.30 leave two people uncounted on every seed, 0.35 leaves one on one seed, and 0.40 completes all 30. Shrinking the radius changes downstream goal margins, so the throughput response is nonmonotonic rather than a pure body-area effect.

All dense V3/V4 radii show physical wall penetration approaching the entire radius, with minimum centre-to-segment distance near zero. Larger radius increases the reported penetration for such trajectories; smaller radius does not repair the force integration or wall-contact dynamics. V8 centre distances decrease slightly as radius shrinks, while edge clearance increases; the unresolved source distance origin prevents ranking those changes against 0.4 m. V7 segregation at width 3.0 varies from about 0.182 to 0.214, with no verified target; width 3.6 is unchanged. V5 has no successful obstacle passage and V6 lacks the source-equivalent onset protocol, so neither can select a winner.

The nonoverlap maxima are the ideal hexagonal packing bound 1/(2√3 r²), ignoring boundary losses. They diagnose feasibility, not simulated or empirical density. The larger radii cannot represent 3.3/m² without overlap; 0.30 m is already below it (3.2075/m²).
