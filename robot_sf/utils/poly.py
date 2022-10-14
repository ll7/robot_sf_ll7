# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:32:16 2020

@author: Matteo Caruso
"""

import math
import random
from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
from shapely.geometry import Polygon


def clip(x, min, max):
    if min > max:
        return x
    elif x < min:
        return min
    elif x > max:
        return max
    else:
        return x


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
        x = r_i * math.cos(angle)
        y = r_i * math.sin(angle)
        points.append((x, y))
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
    x, y = offset
    vertex[:,0] += x
    vertex[:,1] += y
    return Polygon(vertex)


# class MyPolygon:
#     def __init__(self, n_vert=5, irregularity=0, spikeness=0):
#         normalized_in = True
#         radius = 1
#         self.n_vertex = n_vert
#         self.normalized = normalized_in
#         self.num_edges = n_vert
#         self.irregularity = clip(irregularity, 0, 1) * 2 * math.pi / self.n_vertex
#         self.spikeness = clip(spikeness, 0, 1) * radius
#         self.vertex = np.array([])
#         self.edges = np.array([])
#         self.centre = np.zeros((1, 2))
#         self.radius = radius
#         self.initialized = False
#         self._initialize()

#     def _initialize(self):
#         if self.initialized:
#             return False

#         self._compute_edges()
#         self._compute_centroid()
#         self.initialized = True

#     def _compute_centroid(self):
#         cx = 0
#         cy = 0
#         A = 0
#         for i in range(self.vertex.shape[0]-1):
#             cx += (self.vertex[i,0] + self.vertex[i+1,0])*(self.vertex[i,0]*self.vertex[i+1,1] - self.vertex[i+1,0]*self.vertex[i,1])
#             cy += (self.vertex[i,1] + self.vertex[i+1,1])*(self.vertex[i,0]*self.vertex[i+1,1] - self.vertex[i+1,0]*self.vertex[i,1])
#             A += (self.vertex[i,0]*self.vertex[i+1,1] - self.vertex[i+1,0]*self.vertex[i,1])
#         A /= 2
#         cx /= 6*A
#         cy /= 6*A

#         self.centre[0,0] = cx
#         self.centre[0,1] = cy
 
#     def _compute_edges(self):
#         #Store the edge in the format [xmin, xmax,ymin,ymax]
#         tmp = []
#         for i in range(self.vertex.shape[0]-1):
#             tmp.append([self.vertex[i,0], self.vertex[i+1,0], self.vertex[i,1],self.vertex[i+1,1]])
#         self.edges = tmp
