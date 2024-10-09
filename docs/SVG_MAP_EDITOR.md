# robot-sf

## SVG Map Editor

### About

This file explains how to build a map for robot-sf using a svg-editor.
All examples are made with [inkscape](https://inkscape.org/).
![example](./img/inkscape_example.png)

### Setup

These settings should be checked before building a map.

- Use **px** as global unit (File -> Document Properties)

- Use absolute coordinates for your path, marked by the **M**.\
(Edit -> Preferences -> Input/Output -> SVG Output -> Path string format -> Absolute)

*Inkscape version: 1.3.2*

[Further reference](https://github.com/ll7/robot_sf_ll7/issues/40)

### Building the map

Colours can be selected as desired, as the simulation uses its own colour scheme.
However, it is best to remain consistent to make the map easier to understand.

The most important part is setting the label. In Inkscape this can be done by double-clicking the object in the layers-and-objects list on the right side or by right-clicking the object and selecting the object properties.

Use layers to make it clearer.

#### Obstacles

Obstacles should be avoided by the vehicle and the pedestrians.\
Draw them by using the rectangle tool.\
Set the label to **obstacle**

[Obstacle Issue](https://github.com/ll7/robot_sf_ll7/issues/55)

#### Robot

The robot needs a spawn zone to define his starting position and a goal zone he needs to reach to finish the episode.\
Multiple zones can be used.\
Draw them by using the rectangle tool.\
Set the labels to **robot_spawn_zone** and **robot_goal_zone**

The robot path defines the route the robot takes, while reaching the goal zone.\
Use the pen tool for this and perform multiple left clicks to set waypoints along the path.\
The path should not start or end inside the spawn/goal zone, but just before it.\
Set the label to **robot_route_\<spawn\>_\<goal\>**

(e.g. robot_route_1_0 -> Using Spawn 1 and Goal 0.\
The zone numbers are counted from bottom to top in the list on the right-hand side)

#### NPC Pedestrians

The Pedestrians also need a spawn/goal zone. If they reach the goal they will spawn again at the start\
Set the labels to **ped_spawn_zone** and **ped_goal_zone**

For the path you don't need to set specific waypoints, just make sure the path doesn't collide with an obstacle.\
Set the label to **ped_route_\<spawn\>_\<goal\>**

### Colors

The Colors can be found here: [sim_view.py](../robot_sf/render/sim_view.py)

### New Features

If you want to implement new features: [svg_map_parser.py](../robot_sf/nav/svg_map_parser.py)
