# Mean-matched heterogeneous pedestrian speeds (development-seed diagnostic)

**Status: preregistered, acquisition in progress.** New current-version execution on the
0.0.8 release source (`373dbfde`), development seeds only. Not release evidence; this packet
admits nothing into the thesis.

Question: do the four planners' robot outcomes depend on every pedestrian sharing one
preferred speed? Uniform (12 x 0.65 m/s) is compared with mixed (3 x 0.325, 6 x 0.65,
3 x 0.975 m/s, realized mean 0.65 m/s). Identities, starts, goals, groups and radii match.

- [`contract.yaml`](contract.yaml): frozen execution contract and predeclared outcome rule.
- [`driver.py`](driver.py): acquisition through the normal campaign runner with the frozen
  release planner configs, plus a read-only observer for realized inputs and trajectories.
- [`analyze.py`](analyze.py): extraction, realized-input checks, paired effects, decision.
- [`qualification/`](qualification/): pilot (seeds 1001-1010) checks; never pooled.
