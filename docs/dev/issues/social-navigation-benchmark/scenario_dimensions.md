# Scenario Dimensions (Draft)

Dimension | Symbol | Values (initial) | Notes
--------- | ------ | ---------------- | -----
Crowd density | ρ | {low, med, high} | Agents per m² approx: {0.3, 0.8, 1.2}
Flow pattern | F | {unidirectional, bidirectional, crossing, merging} | Initial heading assignment
Obstacle complexity | O | {open, bottleneck, maze-lite} | Maps or layout presets
Group fraction | G_f | {0.0, 0.2, 0.4} | Fraction of agents in groups
Group size dist | G_sz | {singleton, mixed(2-4)} | Distribution model
Speed heterogeneity | σ_v | {low, high} | Std dev multiplier for desired speeds
Goal topology | T_g | {point-to-point, swap, circulate} | Assignment logic
Robot start context | C_r | {ahead, behind, embedded} | Relative placement to crowd flow
Seed | s | integer | RNG seed

## Open Questions
- Add adversarial density spikes? (tag: experimental)
- Continuous density vs. discrete labels? (start discrete for reproducibility)
- Need map scaling dimension? (maybe derived)

## Next
Convert this table into a canonical YAML scenario template set after review.
