# fast-pysf wrapper

This document shows how to use the `FastPysfWrapper` to sample forces from the bundled `fast-pysf` simulator and how to plot the force field using the example script.

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

# Sample a grid and plot (see examples/plot_force_field.py for a ready example)
xs = np.linspace(-2, 6, 40)
ys = np.linspace(-3, 3, 40)
field = wrapper.get_force_field(xs, ys)

# Cache grid for fast interpolation
wrapper.build_force_grid_cache(xs, ys)
print(wrapper.interpolate_force(0.1, 0.2))
```

## Example script

See `examples/plot_force_field.py` for a runnable example that builds a small scene,
computes the force field and plots it using `matplotlib.quiver`.
