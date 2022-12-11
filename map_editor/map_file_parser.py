import json
from typing import Union, Tuple, List
from dataclasses import dataclass

import numpy as np


# scalar range of (low, high) tuple
Range2D = Tuple[float, float]


@dataclass
class VisualizableMapConfig:
    x_margin: Range2D
    y_margin: Range2D
    obstacles: np.ndarray


def parse_mapfile_text(text: str) -> Union[VisualizableMapConfig, None]:
    try:
        map_data = json.loads(text)
        all_lines = list()
        for obstacle in map_data['Obstacles']:
            obs_lines: List[List[float]] = map_data['Obstacles'][obstacle]['Edges']
            for line in obs_lines:
                # info: incoming lines are stored as (x1, y1, x2, y2)
                line = [line[0], line[1], line[2], line[3]]
                all_lines.append(line)
            
            if obs_lines:
                start_x, start_y = obs_lines[0][0], obs_lines[0][1]
                end_x, end_y = obs_lines[-1][0], obs_lines[-1][1]
                last_line = [end_x, end_y, start_x, start_y]
                all_lines.append(last_line)

        obstacles = np.array(all_lines)

        x_margin = map_data['x_margin']
        x_margin = (x_margin[0], x_margin[1])

        y_margin = map_data['y_margin']
        y_margin = (y_margin[0], y_margin[1])

        return VisualizableMapConfig(x_margin, y_margin, obstacles)
    except:
        return None
