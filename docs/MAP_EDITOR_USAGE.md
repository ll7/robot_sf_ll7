# robot-sf

**DEPRECATED**

> [!CAUTION]
> DEPRECATED: This method is no longer supported! Use [svg editor](./SVG_MAP_EDITOR.md) instead.

This Method is deprecated:
```sh
sudo apt-get update && sudo apt-get install -y python3-tk
```

```sh
python3 -m map_editor
```

*Note: See this documentation on how to use the map editor.*

## Map Editor

### About
The map editor module supports assembling maps for the robot_sf simulator
through a graphical interface. It comes as a split-screen design
containing a text editor to the left and a map preview window to the right.

![](./img/map_editor_screenshot.png)

### Map File Format
Following sections outline the different parts of a valid map file.

#### X / Y Margin
The x / y coordinate space bounds have to be specified through margins,
both properties are mandatory.

The ``x_margin`` property specifies the x coordinate space as [``min_x``, ``max_x``].
Same goes for ``y_margin`` spanning the y coordinate space between [``min_y``, ``max_y``].
Both space bounds are continuous floating-point numbers.

```json
{
    "x_margin": [-10, 60],
    "y_margin": [-10, 60],
    ...
}
```

#### Spawn and Goal Zones
The random sampling of spawn and goal positions of the robot and all pedestrians
is specified through rectangular zones.
Each rectangle with points ABCD is given as a tuple of (A, B, C) where |AC|
represents the diagonal; D = A + (C - B).

The spawn zones are defined as ``robot_spawn_zones`` and ``ped_spawn_zones``
and the goal zones as ``robot_goal_zones`` and ``ped_goal_zones``. Moreover,
there's ``ped_crowded_zones`` to model high traffic pedestrian zones.
All five properties are mandatory, but can be empty.

```json
{
    "robot_spawn_zones": [
        [[0, 40], [0, 30], [10, 30]]
    ],
    "ped_spawn_zones": [
        [[20, 40], [20, 30], [30, 30]]
    ],
    "robot_goal_zones": [
        [[40, 10], [40, 0], [50, 0]]
    ],
    "ped_goal_zones": [
        [[40, 10], [40, 0], [50, 0]]
    ],
    "ped_crowded_zones": [
        [[40, 10], [40, 0], [50, 0]]
    ],
    ...
}
```

In the map editor's preview display, robot spawn zones are blue,
pedestrian spawn zones are red, robot goal zones are green,
pedestrian goal zones are magenta and crowded zones are orange.

#### Robot / Pedestrian Routes
For navigating from spawn zones to goal zones, routes have to be specified.
Each route contains a list of waypoints which are used by the simulator
to mock global navigation.

The routes are specified by the ``robot_routes`` and ``ped_routes`` properties.
It's not required to connect all zones with each other (in contrast to
prior versions of robot\_sf). Pedestrians and robots are only spawned at
the origin of actual routes.

Spawn and goal zones with n elements are assigned ids within [0, n-1]
in an ascending order according to their position in the map file.
This is also outlined in the map editor's preview display.

```json
{
    "robot_spawn_zones": [
        [[0, 40], [0, 30], [10, 30]]
    ],
    "robot_goal_zones": [
        [[40, 10], [40, 0], [50, 0]]
    ],
    "robot_routes": [
        {
            "spawn_id": 0,
            "goal_id": 0,
            "waypoints": [
                [10, 30],
                [15, 27],
                [20, 23],
                [25, 19],
                [30, 16],
                [35, 13],
                [40, 10]
            ]
        }
    ],
    ...
}
```

```json
{
    "ped_spawn_zones": [
        [[0, 40], [0, 30], [10, 30]]
    ],
    "ped_goal_zones": [
        [[40, 10], [40, 0], [50, 0]]
    ],
    "ped_routes": [
        {
            "spawn_id": 0,
            "goal_id": 0,
            "waypoints": [
                [10, 30],
                [15, 27],
                [20, 23],
                [25, 19],
                [30, 16],
                [35, 13],
                [40, 10]
            ]
        }
    ],
    ...
}
```

In the map editor, the robot route waypoints are displayed as green circles
and the pedestrian route waypoints are displayed as yellow circles.

#### Obstacles
Last but not least, the map contains a set of obstacles that the robot
and the pedestrians need to navigate around to avoid collisions.

The obstacles are specified as ```obstacles``` property. It contains
a list of polygon where each polygon consists of a list of vertices.
The obstacles list is allowed to be empty, although it doesn't make
a lot of sense to have an empty map.

```json
{
    "obstacles": [
        [
            [0, 10],
            [10, 10],
            [10, 0],
            [0, 0]
        ],
        [
            [40, 50],
            [50, 50],
            [50, 40],
            [40, 40]
        ]
    ],
    ...
}
```

Obstacles are displayed with black lines which is consistent with
the simulation's coloring.
