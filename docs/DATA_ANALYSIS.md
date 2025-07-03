# Data Analysis

To perform data analysis on a specific episode, you need to follow these steps

## Create JSON File (Optional)

A JSON File can be created by running the `save_to_json` function located in the [extract_json_from_pickle.py](../robot_sf/data_analysis/extract_json_from_pickle.py) file.

By running the function, a recorded `*.pkl` file will be converted into a `*.json` file.

Once the `*.json` file has been created, you can reuse it in different plots.

## Plot the data

The data can be plotted via the newly generated `*.json` file or directly via the recording `*.pkl` file.

An easy to use example can be found in the [data_analysis_example.py](../examples/data_analysis_example.py) file.

In-depth for more customization: 
- Json: [extract_json_from_pickle.py](../robot_sf/data_analysis/extract_json_from_pickle.py)
- Directly: [extract_obj_from_pickle.py](../robot_sf/data_analysis/extract_obj_from_pickle.py)

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
- Plot the positions of the npc pedestrians color coded to the velocity

## More Information

A more detailed explanation of each plot can be found in the [PED_METRICS.md](./ped_metrics/PED_METRICS.md) file.