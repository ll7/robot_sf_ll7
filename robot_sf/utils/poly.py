import math
import random
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon


def clip(val, min_val, max_val):
    if min_val > max_val:
        return val
    elif val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    else:
        return val


@dataclass
class PolygonCreationSettings:
    n_vertex: int
    irregularity: float
    spikeness: float
    radius: float=1


def generate_polygon_points(config: PolygonCreationSettings) -> List[Tuple[float, float]]:
    angle_steps = []
    lower = (2 * math.pi / config.n_vertex) - config.irregularity
    upper = (2 * math.pi / config.n_vertex) - config.irregularity

    summ = 0
    for i in range(config.n_vertex):
        tmp = random.uniform(lower, upper)
        angle_steps.append(tmp)
        summ += tmp

    k = summ / (2 * math.pi)
    for i in range(config.n_vertex):
        angle_steps[i] /= k

    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(config.n_vertex):
        r_i = clip(random.gauss(config.radius, config.spikeness), 0, config.radius)
        pos_x = r_i * math.cos(angle)
        pos_y = r_i * math.sin(angle)
        points.append((pos_x, pos_y))
        angle = angle + angle_steps[i]

    return points


def normalize(vertices: np.ndarray) -> np.ndarray:
    xlim, ylim = [0, 1], [0, 1]
    xmin, xmax = vertices[:, 0].min(), vertices[:, 0].max()
    ymin, ymax = vertices[:, 1].min(), vertices[:, 1].max()

    for i in range(vertices.shape[0]):
        vertices[i, 0] = (xlim[1] - xlim[0]) * (vertices[i, 0] - xmin) / (xmax - xmin) + xlim[0]
        vertices[i, 1] = (ylim[1] - ylim[0]) * (vertices[i, 1] - ymin) / (ymax - ymin) + ylim[0]

    return vertices


def random_polygon(config: PolygonCreationSettings) -> Polygon:
    points = generate_polygon_points(config)
    vertices = np.array(points)

    # TODO: figure out if shapely Polygon handles this already
    last = np.reshape(vertices[0, :], (1, 2))
    vertices = np.concatenate((vertices, last), axis=0)

    vertices = normalize(vertices)
    return Polygon(vertices)


def load_polygon(vertices: List[Tuple[float, float]]) -> Polygon:
    # TODO: think of normalizing the vertices (original behavior didn't normalize, though)
    return Polygon(np.array(vertices))


def move_polygon(poly: Polygon, offset: Tuple[float, float]) -> Polygon:
    points = zip(poly.xy[0], poly.xy[1])
    vertex = np.array(points)
    pos_x, pos_y = offset
    vertex[:, 0] += pos_x
    vertex[:, 1] += pos_y
    return Polygon(vertex)
