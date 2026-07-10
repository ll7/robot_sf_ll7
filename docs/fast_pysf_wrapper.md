# fast-pysf wrapper

This document shows how to use the `FastPysfWrapper` to sample forces from the bundled `fast-pysf` simulator and how to plot the force field using the example script.

## Obstacle-force semantics

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
