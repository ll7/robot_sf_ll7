# fast-pysf wrapper

This document shows how to use the `FastPysfWrapper` to sample forces from the bundled `fast-pysf` simulator and how to plot the force field using the example script.

## 0.1.0 pedestrian wall-law default

From 0.1.0, new simulation settings and new authored scenarios default to the
radius-aligned **physical_margin x10** preset,
`body_edge_exponential_v3_physical_margin10x`. It uses a 0.05 m margin added to
the pedestrian force radius, with ten times the original physical-margin
amplitude (effective factor 100 at the unchanged config factor 10). The original
`body_edge_exponential_v3_physical_margin` selector remains the unscaled variant.

This is a maintainer modelling choice, **not shown to be more realistic**.
[#10376](https://github.com/ll7/robot_sf_ll7/issues/10376) reports 0.085% of
pedestrian-steps within 10 cm of a wall in a development diagnostic. The flow
comparison with Seyfried et al. is not matched in population, width, directionality
or arrival conditions; it cannot establish empirical realism. See also
[#10310](https://github.com/ll7/robot_sf_ll7/issues/10310) and
[#10304](https://github.com/ll7/robot_sf_ll7/issues/10304).

The new default **breaks comparability with 0.0.x results**. Released `release_0_0_*`
and `*_frozen.yaml` scenario inputs retain legacy behavior and unchanged bytes.
Historical unversioned mappings also resolve to legacy. To reproduce legacy
physics explicitly in any new scenario, use:

```yaml
simulation_config:
  obstacle_force_law: legacy_shifted_gradient_v1
```

In Python, use `SimulationSettings(obstacle_force_law="legacy_shifted_gradient_v1")`
or `ObstacleForceConfig(law_version="legacy_shifted_gradient_v1")`. Legacy retains
its original factor 10; the default preset does not change other laws' strengths.
Existing checkpoint comparisons must explicitly select the same law on both sides.

## Legacy obstacle-force semantics

`ObstacleForceConfig.threshold` is an additive distance offset in meters, not an
activation threshold. Before evaluating the potential field, fast-pysf combines it with
`agent_radius * sigma` and subtracts the result from the distance to an obstacle segment:
`d = max(distance_to_segment - (threshold + agent_radius * sigma), 1e-5)`. The default
value of `-0.57` therefore adds 0.57 m to the effective distance and softens near-wall
repulsion; it does not turn the force on or off.

The obstacle potential is inverse-square in `d`; its force uses the corresponding
inverse-cubic distance-gradient term (proportional to `d^-3`). This differs from the
exponential pedestrian-pedestrian social force: both obstacle and pedestrian-robot
repulsion use potential-field forces instead.

## Using the wrapper

```python
from pysocialforce import Simulator
from robot_sf.sim.fast_pysf_wrapper import FastPysfWrapper
import numpy as np

# Create a simulator (small example)
state = ...  # numpy array with shape (N, 7) [x,y,vx,vy,goalx,goaly,tau]
obstacles = [...]  # list of (x1, y1, x2, y2) line segments
sim = Simulator(state=state, obstacles=obstacles)

wrapper = FastPysfWrapper(sim)

# Sample single point
force = wrapper.get_forces_at((0.5, 0.0))
print(force)  # [fx, fy]

# Sample a grid and plot (see examples/plotting/plot_force_field.py for a ready example)
xs = np.linspace(-2, 6, 40)
ys = np.linspace(-3, 3, 40)
field = wrapper.get_force_field(xs, ys)

# Cache grid for fast interpolation
wrapper.build_force_grid_cache(xs, ys)
print(wrapper.interpolate_force(0.1, 0.2))
```

## Example script

See `examples/plotting/plot_force_field.py` for a runnable example that builds a small scene,
computes the force field and plots it using `matplotlib.quiver`.
