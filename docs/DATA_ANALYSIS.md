# Data Analysis

To perform data analysis on a specific episode, you need to follow the steps

## Create JSON File

A JSON File can be created by running the [generate_dataset.py](../robot_sf/data_analysis/generate_dataset.py) script.

By just running the script, the latest recorded `*.pkl` file will be converted.

For a specific file adjust the filename argument.

## Plot the data

Once the JSON File has been created, you can reuse it in different plots.

For this just use the correct function call with your dataset as filename

### Basic Plots

In [plot_dataset.py](../robot_sf/data_analysis/plot_dataset.py) there are these options:

- All npc pedestrian coordinates
- All npc velocities
- Ego pedestrian velocity
- Ego pedestrian acceleration

### Kernel Density

In [plot_kernel_density.py](../robot_sf/data_analysis/plot_kernel_density.py) there are these options:

- Plot the kernel density estimation of the npc pedestrians position
- Plot the kde of the x and y coordinates of the npc pedestrian in comparison to the ego pedestrian

### Trajectory Plots

In [plot_npc_trajectory.py](../robot_sf/data_analysis/plot_npc_trajectory.py) there are these options:

- Plot the multiple trajectories of a single npc pedestrian index
- Plot all trajectories present in the episode
- Plot for a single npc pedestrian index the trajectories, the velocity and the acceleration
- Plot the acceleration distribution of the npc pedestrians
- Plot the velocity distribution of the npc pedestrians
- Plot the comparison of the acceleration distribution of the npc pedestrians and the ego pedestrian
- Plot the comparison of the velocity distribution of the npc pedestrians and the ego pedestrian
- Plot the velocity distribution of the npc pedestrians with the positions colorcoded to the velocity